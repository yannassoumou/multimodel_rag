"""
PDF → Qwen3-VL Embedding Pipeline
Converts PDF pages to images and sends them to a vLLM pooling server
for multimodal embeddings using the correct OpenAI-style message format.
"""

import os
import base64
import math
import tempfile
from io import BytesIO

import numpy as np
import requests
import torch
from pdf2image import convert_from_path
from PIL import Image

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
EMBEDDING_SERVER_URL = "http://minisforum.tailfe1a8c.ts.net:8888/v1/embeddings"
MODEL_NAME           = "Qwen3-VL-Embedding-8B"
INPUT_PDF            = "./SF_LRN_Web_Svc_Integ.pdf"   # local path or HTTP URL
REQUEST_TIMEOUT      = 90    # seconds per page request
JPEG_QUALITY         = 85

# Qwen3-VL pixel constraints
_FACTOR      = 32                                # IMAGE_BASE_FACTOR(16) * 2
MIN_PIXELS   = 4   * _FACTOR * _FACTOR           #      4 096
MAX_PIXELS   = 1800 * _FACTOR * _FACTOR          # 1 843 200


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def load_pdf(source: str) -> str:
    """
    Accept a local path or HTTP URL and return a local file path to the PDF.
    Temporary files are the caller's responsibility to clean up.
    """
    if source.startswith("http://") or source.startswith("https://"):
        print(f"[pdf] Downloading {source} …")
        r = requests.get(source, timeout=60)
        r.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp.write(r.content)
        tmp.close()
        print(f"[pdf] Saved to {tmp.name}")
        return tmp.name, True          # (path, is_temp)

    if not os.path.isfile(source):
        raise FileNotFoundError(f"PDF not found: {source}")
    return source, False


def pdf_to_images(pdf_path: str) -> list[Image.Image]:
    """Rasterise every page of the PDF to a PIL Image."""
    print(f"[pdf] Rasterising pages …")
    images = convert_from_path(pdf_path)
    print(f"[pdf] {len(images)} page(s) extracted.")
    return images


def resize_for_model(image: Image.Image) -> Image.Image:
    """Scale image so its pixel count sits within [MIN_PIXELS, MAX_PIXELS]."""
    w, h   = image.size
    pixels = w * h

    if pixels > MAX_PIXELS:
        scale     = math.sqrt(MAX_PIXELS / pixels)
        new_w     = max(1, int(w * scale))
        new_h     = max(1, int(h * scale))
        image     = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        print(f"  [resize] {w}×{h} → {new_w}×{new_h}  ({pixels:,} → {new_w*new_h:,} px)")

    elif pixels < MIN_PIXELS:
        scale     = math.sqrt(MIN_PIXELS / pixels)
        new_w     = max(1, int(w * scale))
        new_h     = max(1, int(h * scale))
        image     = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        print(f"  [resize] {w}×{h} → {new_w}×{new_h}  (upscaled to meet MIN_PIXELS)")

    return image


def image_to_data_uri(image: Image.Image) -> str:
    """Encode a PIL Image as a JPEG data URI (base64)."""
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=JPEG_QUALITY)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


# ──────────────────────────────────────────────
# Embedding
# ──────────────────────────────────────────────
def embed_image(image: Image.Image, page_num: int = 0) -> torch.Tensor | None:
    """
    Send one page image to the vLLM /v1/embeddings endpoint.

    vLLM pooling servers for multimodal models expect an OpenAI-style
    messages array inside 'input', not a bare content block.
    """
    image      = resize_for_model(image)
    data_uri   = image_to_data_uri(image)

    payload = {
        "model": MODEL_NAME,
        "encoding_format": "float",
        # vLLM multimodal embeddings: input = list-of-messages
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri}
                    },
                    {
                        "type": "text",
                        "text": "Represent the document page for retrieval."
                    }
                ]
            }
        ]
    }

    try:
        resp = requests.post(
            EMBEDDING_SERVER_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )
    except requests.exceptions.Timeout:
        print(f"  [page {page_num}] ✗  Request timed out after {REQUEST_TIMEOUT}s")
        return None
    except requests.exceptions.ConnectionError as exc:
        print(f"  [page {page_num}] ✗  Connection error: {exc}")
        return None

    if resp.status_code != 200:
        print(f"  [page {page_num}] ✗  HTTP {resp.status_code}: {resp.text[:300]}")
        return None

    data = resp.json()
    items = data.get("data", [])
    if not items:
        print(f"  [page {page_num}] ✗  Empty 'data' field in response.")
        return None

    embedding = torch.tensor(items[0]["embedding"], dtype=torch.float32)
    print(f"  [page {page_num}] ✓  shape={list(embedding.shape)}")
    return embedding


def embed_pages(images: list[Image.Image]) -> torch.Tensor | None:
    """
    Embed each page individually, then return the mean-pooled document vector.
    """
    embeddings = []
    total = len(images)

    for i, img in enumerate(images, start=1):
        print(f"[embed] Page {i}/{total}")
        vec = embed_image(img, page_num=i)
        if vec is not None:
            embeddings.append(vec)

    if not embeddings:
        print("[embed] No embeddings produced — all pages failed.")
        return None

    if len(embeddings) < total:
        print(f"[embed] Warning: {total - len(embeddings)} page(s) failed and were skipped.")

    doc_vec = torch.mean(torch.stack(embeddings), dim=0)
    print(f"\n[embed] Document vector  shape={list(doc_vec.shape)}")
    print(f"[embed]   mean={doc_vec.mean().item():.5f}  std={doc_vec.std().item():.5f}")
    return doc_vec


# ──────────────────────────────────────────────
# Quick connectivity test
# ──────────────────────────────────────────────
def test_server() -> bool:
    """Send a tiny blank image to verify the server is reachable and the format is correct."""
    print("[test] Sending probe request to embedding server …")
    blank = Image.new("RGB", (64, 64), color="white")
    vec   = embed_image(blank, page_num=0)
    if vec is not None:
        print(f"[test] ✓  Server OK — embedding dim={vec.shape[0]}\n")
        return True
    else:
        print("[test] ✗  Server probe failed. Check URL and payload format.\n")
        return False


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
def main():
    # 1. Connectivity check
    if not test_server():
        return

    pdf_path = None
    is_temp  = False

    try:
        # 2. Resolve PDF
        pdf_path, is_temp = load_pdf(INPUT_PDF)

        # 3. Rasterise
        images = pdf_to_images(pdf_path)
        if not images:
            print("[main] No pages found in PDF.")
            return

        # 4. Embed
        print(f"\n[main] Embedding {len(images)} page(s) …")
        doc_embedding = embed_pages(images)

        if doc_embedding is None:
            print("[main] Failed to produce document embedding.")
            return

        # 5. (Optional) persist or use the embedding
        print("\n[main] Done.")
        print(f"  Embedding shape : {list(doc_embedding.shape)}")
        print(f"  Embedding dtype : {doc_embedding.dtype}")
        # e.g. torch.save(doc_embedding, "doc_embedding.pt")

    except Exception as exc:
        import traceback
        print(f"[main] Unhandled exception: {exc}")
        traceback.print_exc()

    finally:
        if is_temp and pdf_path and os.path.exists(pdf_path):
            os.remove(pdf_path)
            print(f"[main] Cleaned up temp file: {pdf_path}")


if __name__ == "__main__":
    main()