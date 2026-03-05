"""
PDF → Qwen3-VL Embedding Pipeline
Converts PDF pages to images and sends them to a vLLM pooling server
for multimodal embeddings using the correct OpenAI-style message format.
"""

import os
import base64
import math
import shutil
import tempfile
import time
from collections.abc import Iterator
from io import BytesIO

import numpy as np
import requests
from pdf2image import convert_from_path
from pymilvus import MilvusClient
from PIL import Image

from file_converter import convert_to_pdf, is_convertible

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
EMBEDDING_SERVER_URL = "http://wsl-windows.tailfe1a8c.ts.net:8888/v1/embeddings"
MODEL_NAME           = "/mnt/f/ai/models/Qwen/Qwen3-VL-Embedding-2B"
# Local path or HTTP URL; supports .pdf, .pptx, .xlsx (pptx/xlsx are converted to PDF first)
INPUT_PDF            = "./R2603/PDF/Bouton unique d'export.pdf"   # or .pptx / .xlsx
REQUEST_TIMEOUT      = 90    # seconds per page request
JPEG_QUALITY         = 85
PDF_DPI              = 150   # lower = less memory per page (pipeline resizes later)

# Self-hosted Milvus
MILVUS_HOST          = "lenovo.tailfe1a8c.ts.net"
MILVUS_PORT          = 19530
MILVUS_COLLECTION    = "doc_embeddings"

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


def pdf_to_images(pdf_path: str) -> Iterator[Image.Image]:
    """
    Rasterise the PDF one page at a time. Yields each page as a PIL Image.
    Keeps at most one page in memory (avoids loading the whole PDF at once).
    """
    print("[pdf] Rasterising pages …")
    page_num = 1
    while True:
        imgs = convert_from_path(
            pdf_path,
            first_page=page_num,
            last_page=page_num,
            dpi=PDF_DPI,
        )
        if not imgs:
            break
        yield imgs[0]
        page_num += 1
    print(f"[pdf] {page_num - 1} page(s) extracted.")


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
        # vLLM multimodal embeddings use "messages", not "input" (input = list of strings/token IDs)
        "messages": [
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


def embed_pages(images: Iterator[Image.Image] | list[Image.Image]) -> torch.Tensor | None:
    """
    Embed each page individually, then return the mean-pooled document vector.
    Accepts an iterable (e.g. generator from pdf_to_images); only one page is in memory at a time.
    """
    embeddings = []
    for i, img in enumerate(images, start=1):
        print(f"[embed] Page {i}")
        vec = embed_image(img, page_num=i)
        if vec is not None:
            embeddings.append(vec)

    if not embeddings:
        print("[embed] No embeddings produced — all pages failed.")
        return None

    n_embedded = len(embeddings)
    print(f"[embed] {n_embedded} page(s) embedded.")

    doc_vec = torch.mean(torch.stack(embeddings), dim=0)
    print(f"\n[embed] Document vector  shape={list(doc_vec.shape)}")
    print(f"[embed]   mean={doc_vec.mean().item():.5f}  std={doc_vec.std().item():.5f}")
    return doc_vec


# ──────────────────────────────────────────────
# Milvus (self-hosted)
# ──────────────────────────────────────────────
def _get_milvus_client() -> MilvusClient | None:
    """Create and return a Milvus client for the configured host/port. Returns None on failure."""
    uri = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
    try:
        return MilvusClient(uri=uri)
    except Exception as e:
        print(f"[milvus] Connection failed: {e}")
        return None


def send_vector_to_milvus(
    vector: torch.Tensor | list[float],
    *,
    doc_id: str | int | None = None,
    source: str | None = None,
    collection_name: str | None = None,
    client: MilvusClient | None = None,
) -> dict | None:
    """
    Insert a single vector into the self-hosted Milvus collection.
    Creates the collection if it does not exist (dimension from vector length).
    Pass an existing client to reuse the connection; otherwise a new one is created.
    Returns the insert result or None on failure.
    """
    collection_name = collection_name or MILVUS_COLLECTION

    if isinstance(vector, torch.Tensor):
        vector = vector.cpu().tolist()
    dim = len(vector)

    if client is None:
        client = _get_milvus_client()
        if client is None:
            return None

    try:
        if not client.has_collection(collection_name):
            client.create_collection(
                collection_name=collection_name,
                dimension=dim,
                primary_field_name="id",
                vector_field_name="vector",
                metric_type="COSINE",
                auto_id=False,
            )
            print(f"[milvus] Created collection {collection_name!r} (dim={dim})")

        pk = doc_id if doc_id is not None else int(time.time() * 1e6)
        data = [{"id": pk, "vector": vector}]

        res = client.insert(collection_name=collection_name, data=data)
        print(f"[milvus] Inserted 1 vector into {collection_name!r}")
        return res
    except Exception as e:
        print(f"[milvus] Insert failed: {e}")
        return None


def send_vectors_to_milvus(
    vectors: list[torch.Tensor | list[float]],
    *,
    ids: list[int | str] | None = None,
    collection_name: str | None = None,
    client: MilvusClient | None = None,
) -> dict | None:
    """
    Insert multiple vectors into the self-hosted Milvus collection in one call.
    Creates the collection if it does not exist. Pass an existing client to reuse
    the connection. Returns the insert result or None on failure.
    """
    if not vectors:
        return None

    collection_name = collection_name or MILVUS_COLLECTION
    converted: list[list[float]] = []
    for v in vectors:
        if isinstance(v, torch.Tensor):
            converted.append(v.cpu().tolist())
        else:
            converted.append(v)
    dim = len(converted[0])
    if any(len(v) != dim for v in converted):
        print("[milvus] Vectors must all have the same dimension.")
        return None

    if client is None:
        client = _get_milvus_client()
        if client is None:
            return None

    try:
        if not client.has_collection(collection_name):
            client.create_collection(
                collection_name=collection_name,
                dimension=dim,
                primary_field_name="id",
                vector_field_name="vector",
                metric_type="COSINE",
                auto_id=False,
            )
            print(f"[milvus] Created collection {collection_name!r} (dim={dim})")

        n = len(converted)
        if ids is not None and len(ids) != n:
            print("[milvus] ids length must match vectors length.")
            return None
        base_ts = int(time.time() * 1e6)
        data = [
            {"id": (ids[i] if ids is not None else base_ts + i), "vector": converted[i]}
            for i in range(n)
        ]
        res = client.insert(collection_name=collection_name, data=data)
        print(f"[milvus] Inserted {n} vectors into {collection_name!r}")
        return res
    except Exception as e:
        print(f"[milvus] Batch insert failed: {e}")
        return None


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
    temp_dir = None

    try:
        # 2. Resolve input to a PDF path (convert pptx/xlsx if needed)
        input_path = INPUT_PDF.strip()
        if input_path.startswith("http://") or input_path.startswith("https://"):
            pdf_path, is_temp = load_pdf(input_path)
        elif is_convertible(input_path):
            pdf_path, is_temp, temp_dir = convert_to_pdf(input_path)
            if pdf_path is None:
                print("[main] Conversion to PDF failed.")
                return
        else:
            pdf_path, is_temp = load_pdf(input_path)

        # 3. Rasterise (one page at a time; generator keeps memory low)
        images = pdf_to_images(pdf_path)

        # 4. Embed
        print("\n[main] Embedding pages …")
        doc_embedding = embed_pages(images)

        if doc_embedding is None:
            print("[main] Failed to produce document embedding.")
            return

        # 5. Persist / use the embedding
        print("\n[main] Done.")
        print(f"  Embedding shape : {list(doc_embedding.shape)}")
        print(f"  Embedding dtype : {doc_embedding.dtype}")

        # 6. Send to Milvus (reuse one client)
        client = _get_milvus_client()
        if client:
            send_vector_to_milvus(
                doc_embedding,
                source=pdf_path,
                client=client,
            )
        else:
            print("[main] Skipping Milvus insert (connection failed).")

    except Exception as exc:
        import traceback
        print(f"[main] Unhandled exception: {exc}")
        traceback.print_exc()

    finally:
        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"[main] Cleaned up temp dir: {temp_dir}")
        elif is_temp and pdf_path and os.path.exists(pdf_path):
            os.remove(pdf_path)
            print(f"[main] Cleaned up temp file: {pdf_path}")


if __name__ == "__main__":
    main()