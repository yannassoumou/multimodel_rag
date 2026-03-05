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
from pymilvus import DataType, MilvusClient
from PIL import Image

from file_converter import convert_to_pdf, is_convertible

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
EMBEDDING_SERVER_URL = "http://wsl-windows.tailfe1a8c.ts.net:8888/v1/embeddings"
MODEL_NAME           = "/mnt/f/ai/models/Qwen/Qwen3-VL-Embedding-2B"
# Local path or HTTP URL, or directory (processed recursively); supports .pdf, .pptx, .xlsx
INPUT_PATH           = "./R2603/"   # or directory path
REQUEST_TIMEOUT      = 90    # seconds per page request
JPEG_QUALITY         = 85
PDF_DPI              = 150   # lower = less memory per page (pipeline resizes later)

# Self-hosted Milvus
MILVUS_HOST          = "lenovo.tailfe1a8c.ts.net"
MILVUS_PORT          = 19530
MILVUS_DOC_COLLECTION  = "doc_embeddings"
MILVUS_PAGE_COLLECTION = "page_embeddings"

# Qwen3-VL pixel constraints
_FACTOR      = 32                                # IMAGE_BASE_FACTOR(16) * 2
MIN_PIXELS   = 4   * _FACTOR * _FACTOR           #      4 096
MAX_PIXELS   = 1800 * _FACTOR * _FACTOR          # 1 843 200


# Processable file extensions (lowercase)
_SUPPORTED_EXTENSIONS = {".pdf", ".pptx", ".xlsx"}


def iter_input_files(path: str):
    """
    Yield all processable files under path. If path is a file, yields that file
    (if its extension is .pdf, .pptx, or .xlsx). If path is a directory, recursively
    walks it and yields every file with a supported extension.
    """
    path = path.strip()
    if os.path.isfile(path):
        if os.path.splitext(path)[1].lower() in _SUPPORTED_EXTENSIONS:
            yield path
        return
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Not a file or directory: {path}")
    for root, _dirs, files in os.walk(path):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in _SUPPORTED_EXTENSIONS:
                yield os.path.join(root, name)


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
def embed_image(image: Image.Image, page_num: int = 0) -> list[float] | None:
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

    embedding = list(items[0]["embedding"])
    print(f"  [page {page_num}] ✓  dim={len(embedding)}")
    return embedding


def embed_pages(images: Iterator[Image.Image] | list[Image.Image]) -> tuple[np.ndarray, list[dict]] | tuple[None, None]:
    """
    Embed each page individually, then return the mean-pooled document vector and
    a list of per-page entries: [{"page_num": 1, "vector": [...]}, ...].
    Accepts an iterable (e.g. generator from pdf_to_images); only one page is in memory at a time.
    """
    page_entries: list[dict] = []
    for i, img in enumerate(images, start=1):
        print(f"[embed] Page {i}")
        vec = embed_image(img, page_num=i)
        if vec is not None:
            page_entries.append({"page_num": i, "vector": vec})

    if not page_entries:
        print("[embed] No embeddings produced — all pages failed.")
        return None, None

    n_embedded = len(page_entries)
    print(f"[embed] {n_embedded} page(s) embedded.")

    embeddings = [e["vector"] for e in page_entries]
    doc_vec = np.mean(embeddings, axis=0).astype(np.float32)
    print(f"\n[embed] Document vector  dim={doc_vec.shape[0]}")
    print(f"[embed]   mean={float(doc_vec.mean()):.5f}  std={float(doc_vec.std()):.5f}")
    return doc_vec, page_entries


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


def ensure_doc_collection(client: MilvusClient, dim: int) -> None:
    """Create doc_embeddings collection if it does not exist. Schema: id, vector, source_file, total_pages, obsoleted, version."""
    if client.has_collection(MILVUS_DOC_COLLECTION):
        return
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field(field_name="source_file", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="total_pages", datatype=DataType.INT64)
    schema.add_field(field_name="obsoleted", datatype=DataType.BOOL, default_value=False)
    schema.add_field(field_name="version", datatype=DataType.VARCHAR, max_length=32, default_value="v1")
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
    client.create_collection(
        collection_name=MILVUS_DOC_COLLECTION,
        schema=schema,
        index_params=index_params,
    )
    print(f"[milvus] Created collection {MILVUS_DOC_COLLECTION!r} (dim={dim})")


def ensure_page_collection(client: MilvusClient, dim: int) -> None:
    """Create page_embeddings collection if it does not exist. Schema: id, vector, doc_id, source_file, page_num."""
    if client.has_collection(MILVUS_PAGE_COLLECTION):
        return
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field(field_name="doc_id", datatype=DataType.INT64)
    schema.add_field(field_name="source_file", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="page_num", datatype=DataType.INT64)
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
    client.create_collection(
        collection_name=MILVUS_PAGE_COLLECTION,
        schema=schema,
        index_params=index_params,
    )
    print(f"[milvus] Created collection {MILVUS_PAGE_COLLECTION!r} (dim={dim})")


def insert_doc_embedding(
    client: MilvusClient,
    doc_id: int,
    vector: np.ndarray | list[float],
    source_file: str,
    total_pages: int,
) -> dict | None:
    """Insert one document row into doc_embeddings. Returns insert result or None on failure."""
    if isinstance(vector, np.ndarray):
        vector = vector.tolist()
    dim = len(vector)
    ensure_doc_collection(client, dim)
    source_file = (source_file or "")[:512]
    data = [{
        "id": doc_id,
        "vector": vector,
        "source_file": source_file,
        "total_pages": total_pages,
        "obsoleted": False,
        "version": "v1",
    }]
    try:
        res = client.insert(collection_name=MILVUS_DOC_COLLECTION, data=data)
        print(f"[milvus] Inserted 1 doc into {MILVUS_DOC_COLLECTION!r} (id={doc_id})")
        return res
    except Exception as e:
        print(f"[milvus] Doc insert failed: {e}")
        return None


def insert_page_embeddings(
    client: MilvusClient,
    doc_id: int,
    source_file: str,
    page_entries: list[dict],
) -> dict | None:
    """
    Batch insert page rows into page_embeddings. Each entry must have "page_num" and "vector".
    Page id = doc_id * 10_000 + page_num to keep uniqueness and link to doc.
    """
    if not page_entries:
        return None
    vectors = [e["vector"] for e in page_entries]
    dim = len(vectors[0])
    vectors = [v.tolist() if isinstance(v, np.ndarray) else list(v) for v in vectors]
    ensure_page_collection(client, dim)
    source_file = (source_file or "")[:512]
    data = [
        {
            "id": doc_id * 10_000 + page_entries[i]["page_num"],
            "vector": vectors[i],
            "doc_id": doc_id,
            "source_file": source_file,
            "page_num": page_entries[i]["page_num"],
        }
        for i in range(len(page_entries))
    ]
    try:
        res = client.insert(collection_name=MILVUS_PAGE_COLLECTION, data=data)
        print(f"[milvus] Inserted {len(data)} pages into {MILVUS_PAGE_COLLECTION!r} (doc_id={doc_id})")
        return res
    except Exception as e:
        print(f"[milvus] Page insert failed: {e}")
        return None


def send_vector_to_milvus(
    vector: np.ndarray | list[float],
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
    collection_name = collection_name or MILVUS_DOC_COLLECTION

    if isinstance(vector, np.ndarray):
        vector = vector.tolist()
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
    vectors: list[np.ndarray | list[float]],
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

    collection_name = collection_name or MILVUS_DOC_COLLECTION
    converted: list[list[float]] = []
    for v in vectors:
        if isinstance(v, np.ndarray):
            converted.append(v.tolist())
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
        print(f"[test] ✓  Server OK — embedding dim={len(vec)}\n")
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

    client = _get_milvus_client()
    if not client:
        print("[main] Milvus connection failed. Exiting.")
        return

    # Use seconds (not microseconds) so doc_id * 10_000 + page_num stays within INT64
    base_ts = int(time.time())
    input_path = INPUT_PATH.strip()

    try:
        files = list(iter_input_files(input_path))
    except FileNotFoundError as e:
        print(f"[main] {e}")
        return
    if not files:
        print("[main] No processable files found.")
        return

    print(f"[main] Processing {len(files)} file(s) …\n")

    for file_index, source_file in enumerate(files):
        pdf_path = None
        is_temp = False
        temp_dir = None
        try:
            # Resolve to PDF (download or convert pptx/xlsx)
            if source_file.startswith("http://") or source_file.startswith("https://"):
                pdf_path, is_temp = load_pdf(source_file)
            elif is_convertible(source_file):
                pdf_path, is_temp, temp_dir = convert_to_pdf(source_file)
                if pdf_path is None:
                    print(f"[main] Conversion failed for {source_file}, skipping.")
                    continue
            else:
                pdf_path, is_temp = load_pdf(source_file)

            # Rasterise and embed
            images = pdf_to_images(pdf_path)
            print(f"\n[main] Embedding: {source_file}")
            doc_embedding, page_entries = embed_pages(images)

            if doc_embedding is None or not page_entries:
                print(f"[main] No embeddings for {source_file}, skipping.")
                continue

            doc_id = base_ts * 1000 + file_index  # unique per run, keeps page_id in INT64 range
            insert_doc_embedding(
                client,
                doc_id,
                doc_embedding,
                source_file=source_file,
                total_pages=len(page_entries),
            )
            insert_page_embeddings(client, doc_id, source_file, page_entries)
            print(f"[main] Done: {source_file}\n")

        except Exception as exc:
            import traceback
            print(f"[main] Error processing {source_file!r}: {exc}")
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