"""
PDF → Qwen3-VL-Embedding-8B Pipeline
--------------------------------------
Two modes:
  1. embed  — rasterise PDF(s), embed pages, save to JSON cache (resumes if interrupted)
              accepts a single PDF file OR a directory (scanned recursively)
  2. insert — read JSON cache(s), insert into Milvus

Usage:
    # Embed a single PDF
    HSA_OVERRIDE_GFX_VERSION=11.5.1 MIOPEN_DISABLE_AI_HEURISTICS=1 \\
        python main.py embed --pdf ./doc.pdf [--author "John Doe"] [--date "2024-01-01"]

    # Embed all PDFs in a directory (recursive)
    HSA_OVERRIDE_GFX_VERSION=11.5.1 MIOPEN_DISABLE_AI_HEURISTICS=1 \\
        python main.py embed --dir ./documents/ [--cache-dir ./cache/]

    # Insert a single cache into Milvus
    python main.py insert --cache ./doc_cache.json --host 192.168.1.100 --port 19530

    # Insert all caches in a directory into Milvus
    python main.py insert --cache-dir ./cache/ --host 192.168.1.100 --port 19530

Requirements:
    pip install transformers>=4.57.0 qwen-vl-utils>=0.0.14 torch pdf2image pillow pymilvus pypdf
"""

import argparse
import base64
import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

import requests
import torch
import torch.nn.functional as F
from pdf2image import convert_from_path
from PIL import Image

# ── Import the official embedder ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.qwen3_vl_embedding import Qwen3VLEmbedder

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
MODEL_PATH    = "/mnt/nvme_storage/.lmstudio/models/lmstudio-community/Qwen3-VL-Embedding-8B"
INSTRUCTION   = "Represent the document page for retrieval."
BATCH_SIZE    = 4
EMBEDDING_DIM = 4096
JPEG_QUALITY  = 85
COLLECTION    = "pdf_pages"


# ──────────────────────────────────────────────────────────────────────────────
# PDF helpers
# ──────────────────────────────────────────────────────────────────────────────
def find_pdfs(directory: str) -> list[str]:
    """Recursively find all PDF files under a directory."""
    pdfs = sorted(Path(directory).rglob("*.pdf"))
    # also catch uppercase .PDF
    pdfs += sorted(p for p in Path(directory).rglob("*.PDF") if p not in pdfs)
    return [str(p) for p in sorted(set(pdfs))]


def load_pdf(source: str) -> tuple[str, bool]:
    if source.startswith("http://") or source.startswith("https://"):
        print(f"[pdf] Downloading {source} ...")
        r = requests.get(source, timeout=60)
        r.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp.write(r.content)
        tmp.close()
        return tmp.name, True
    if not os.path.isfile(source):
        raise FileNotFoundError(f"PDF not found: {source}")
    return source, False


def extract_pdf_metadata(pdf_path: str) -> dict:
    """Extract author and date from PDF metadata using pypdf."""
    meta = {"author": None, "date": None}
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        info   = reader.metadata
        if info:
            meta["author"] = info.get("/Author") or info.get("author")
            raw_date = info.get("/CreationDate") or info.get("creation_date")
            if raw_date:
                raw = str(raw_date).replace("D:", "")[:8]
                try:
                    meta["date"] = datetime.strptime(raw, "%Y%m%d").strftime("%Y-%m-%d")
                except ValueError:
                    meta["date"] = str(raw_date)
    except Exception as e:
        print(f"[pdf] Could not extract metadata: {e}")
    return meta


def pdf_to_images(pdf_path: str) -> list[Image.Image]:
    print("[pdf] Rasterising pages ...")
    images = convert_from_path(pdf_path)
    print(f"[pdf] {len(images)} page(s) extracted.")
    return images


def image_to_base64(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode()


# ──────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ──────────────────────────────────────────────────────────────────────────────
def cache_path_for(pdf_path: str, cache_dir: str | None = None) -> str:
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"{base}_cache.json")
    return os.path.join(os.path.dirname(os.path.abspath(pdf_path)), f"{base}_cache.json")


def load_cache(path: str) -> dict:
    if os.path.exists(path):
        print(f"[cache] Resuming from: {path}")
        with open(path, "r") as f:
            return json.load(f)
    return {"metadata": {}, "pages": {}}


def save_cache(cache: dict, path: str):
    with open(path, "w") as f:
        json.dump(cache, f)


# ──────────────────────────────────────────────────────────────────────────────
# Core embed logic for a single PDF
# ──────────────────────────────────────────────────────────────────────────────
def embed_single_pdf(
    pdf_path: str,
    embedder: Qwen3VLEmbedder,
    cache_dir: str | None = None,
    author_override: str | None = None,
    date_override: str | None = None,
) -> str:
    """
    Embed all pages of one PDF, save/update JSON cache.
    Returns the path to the cache file.
    """
    pdf_filename = os.path.basename(pdf_path)
    out_cache    = cache_path_for(pdf_path, cache_dir)

    print(f"\n{'='*60}")
    print(f"[pdf] {pdf_filename}")
    print(f"{'='*60}")

    # Metadata
    auto_meta = extract_pdf_metadata(pdf_path)
    author    = author_override or auto_meta["author"] or "Unknown"
    date      = date_override   or auto_meta["date"]   or "Unknown"
    print(f"[pdf] author={author}  date={date}")

    # Rasterise
    images = pdf_to_images(pdf_path)
    total  = len(images)

    # Load cache (resume support)
    cache = load_cache(out_cache)
    cache["metadata"] = {
        "pdf_filename" : pdf_filename,
        "pdf_path"     : os.path.abspath(pdf_path),
        "author"       : author,
        "date"         : date,
        "total_pages"  : total,
        "embedded_at"  : datetime.now().isoformat(),
    }

    already_done = set(cache["pages"].keys())
    remaining    = [i for i in range(total) if str(i) not in already_done]

    if already_done:
        print(f"[cache] {len(already_done)} pages already done, {len(remaining)} remaining.")
    if not remaining:
        print("[embed] All pages already cached — skipping.")
        return out_cache

    # Embed in batches
    n_batches = math.ceil(len(remaining) / BATCH_SIZE)
    t_start   = time.time()

    for b in range(n_batches):
        batch_indices = remaining[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
        batch_images  = [images[i] for i in batch_indices]

        i_start = batch_indices[0] + 1
        i_end   = batch_indices[-1] + 1
        print(f"[embed] Batch {b+1}/{n_batches}  pages {i_start}–{i_end} ...", end=" ", flush=True)
        t_batch = time.time()

        try:
            vecs = embedder.process(
                [{"image": img, "instruction": INSTRUCTION} for img in batch_images],
                normalize=True,
            )

            for page_idx, img, vec in zip(batch_indices, batch_images, vecs):
                cache["pages"][str(page_idx)] = {
                    "page_id"      : f"{pdf_filename}::page_{page_idx + 1}",
                    "page_number"  : page_idx + 1,
                    "pdf_filename" : pdf_filename,
                    "pdf_path"     : os.path.abspath(pdf_path),
                    "author"       : author,
                    "date"         : date,
                    "image_b64"    : image_to_base64(img),
                    "embedding"    : vec.cpu().tolist(),
                }

            save_cache(cache, out_cache)

            elapsed       = time.time() - t_batch
            done_so_far   = len(already_done) + (b + 1) * BATCH_SIZE
            total_elapsed = time.time() - t_start
            pages_per_s   = min(done_so_far, total) / max(total_elapsed, 0.1)
            eta_s         = max(0, (len(remaining) - (b + 1) * BATCH_SIZE) / pages_per_s)
            eta_min       = eta_s / 60

            print(f"OK  ({elapsed:.1f}s)  {pages_per_s:.1f} p/s  ETA={eta_min:.1f}min")

        except Exception as exc:
            print(f"FAILED — {exc}")

    total_time = time.time() - t_start
    n_done = len(cache["pages"])
    print(f"[embed] {n_done}/{total} pages cached in {total_time/60:.1f} min  →  {out_cache}")
    return out_cache


# ──────────────────────────────────────────────────────────────────────────────
# Embed command
# ──────────────────────────────────────────────────────────────────────────────
def cmd_embed(args):
    # Collect PDFs to process
    if args.dir:
        pdf_paths = find_pdfs(args.dir)
        if not pdf_paths:
            print(f"[embed] No PDF files found in: {args.dir}")
            return
        print(f"[embed] Found {len(pdf_paths)} PDF(s) in '{args.dir}':")
        for p in pdf_paths:
            print(f"  {p}")
        cache_dir = args.cache_dir or os.path.join(args.dir, ".cache")
    else:
        pdf_path, is_temp = load_pdf(args.pdf)
        pdf_paths = [pdf_path]
        cache_dir = args.cache_dir or None

    # Load model once — reused across all PDFs
    print(f"\n[model] Loading Qwen3VLEmbedder from {MODEL_PATH} ...")
    embedder = Qwen3VLEmbedder(
        model_name_or_path=MODEL_PATH,
        torch_dtype=torch.bfloat16,
        default_instruction=INSTRUCTION,
    )
    print("[model] Ready.\n")

    # Process each PDF
    t_global = time.time()
    caches   = []

    for idx, pdf_path in enumerate(pdf_paths, start=1):
        print(f"\n[progress] PDF {idx}/{len(pdf_paths)}")
        try:
            cache_file = embed_single_pdf(
                pdf_path     = pdf_path,
                embedder     = embedder,
                cache_dir    = cache_dir,
                author_override = args.author if hasattr(args, "author") else None,
                date_override   = args.date   if hasattr(args, "date")   else None,
            )
            caches.append(cache_file)
        except Exception as exc:
            import traceback
            print(f"[embed] FAILED for {pdf_path}: {exc}")
            traceback.print_exc()

    total_time = time.time() - t_global
    print(f"\n{'='*60}")
    print(f"[done] {len(caches)}/{len(pdf_paths)} PDF(s) embedded in {total_time/60:.1f} min")
    print(f"[done] Cache files:")
    for c in caches:
        print(f"  {c}")


# ──────────────────────────────────────────────────────────────────────────────
# Insert command
# ──────────────────────────────────────────────────────────────────────────────
def load_cache_files(args) -> list[str]:
    """Return list of cache JSON file paths from --cache or --cache-dir."""
    if hasattr(args, "cache") and args.cache and os.path.isfile(args.cache):
        return [args.cache]
    if hasattr(args, "cache_dir") and args.cache_dir:
        files = sorted(Path(args.cache_dir).rglob("*_cache.json"))
        return [str(f) for f in files]
    return []


def cmd_insert(args):
    from pymilvus import MilvusClient, DataType

    cache_files = load_cache_files(args)
    if not cache_files:
        print("[insert] No cache files found.")
        return

    print(f"[insert] Found {len(cache_files)} cache file(s):")
    for c in cache_files:
        print(f"  {c}")

    # Connect to Milvus
    uri = f"http://{args.host}:{args.port}"
    print(f"\n[milvus] Connecting to {uri} ...")
    client = MilvusClient(uri=uri)
    print("[milvus] Connected.\n")

    # Create collection if needed
    if not client.has_collection(COLLECTION):
        print(f"[milvus] Creating collection '{COLLECTION}' ...")
        schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id",           DataType.VARCHAR,      max_length=512,   is_primary=True)
        schema.add_field("embedding",    DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
        schema.add_field("page_number",  DataType.INT64)
        schema.add_field("pdf_filename", DataType.VARCHAR,      max_length=512)
        schema.add_field("pdf_path",     DataType.VARCHAR,      max_length=1024)
        schema.add_field("author",       DataType.VARCHAR,      max_length=256)
        schema.add_field("date",         DataType.VARCHAR,      max_length=64)
        schema.add_field("image_b64",    DataType.VARCHAR,      max_length=65535)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name  = "embedding",
            index_type  = "HNSW",
            metric_type = "COSINE",
            params      = {"M": 16, "efConstruction": 200},
        )
        client.create_collection(
            collection_name = COLLECTION,
            schema          = schema,
            index_params    = index_params,
        )
        print(f"[milvus] Collection '{COLLECTION}' created.\n")
    else:
        print(f"[milvus] Collection '{COLLECTION}' already exists.\n")

    # Insert each cache
    INSERT_BATCH   = 100
    total_inserted = 0
    total_pages    = 0

    for cache_file in cache_files:
        print(f"[insert] Processing: {cache_file}")
        with open(cache_file, "r") as f:
            cache = json.load(f)

        pages      = cache["pages"]
        meta       = cache["metadata"]
        n_pages    = len(pages)
        total_pages += n_pages
        page_list  = list(pages.values())
        n_batches  = math.ceil(n_pages / INSERT_BATCH)
        inserted   = 0

        print(f"  PDF: {meta.get('pdf_filename')}  |  pages: {n_pages}  |  author: {meta.get('author')}  |  date: {meta.get('date')}")

        for b in range(n_batches):
            batch = page_list[b * INSERT_BATCH : (b + 1) * INSERT_BATCH]
            rows  = [
                {
                    "id"           : p["page_id"],
                    "embedding"    : p["embedding"],
                    "page_number"  : p["page_number"],
                    "pdf_filename" : p["pdf_filename"],
                    "pdf_path"     : p.get("pdf_path", ""),
                    "author"       : p["author"],
                    "date"         : p["date"],
                    "image_b64"    : p["image_b64"],
                }
                for p in batch
            ]

            print(f"  Batch {b+1}/{n_batches} ({len(rows)} records) ...", end=" ", flush=True)
            try:
                client.insert(collection_name=COLLECTION, data=rows)
                inserted       += len(rows)
                total_inserted += len(rows)
                print(f"OK")
            except Exception as exc:
                print(f"FAILED — {exc}")

        print(f"  Inserted {inserted}/{n_pages} pages from {meta.get('pdf_filename')}\n")

    print(f"[done] Total inserted: {total_inserted}/{total_pages} pages into '{COLLECTION}'")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="PDF embedding pipeline with Milvus")
    sub    = parser.add_subparsers(dest="command", required=True)

    # ── embed ──
    p_embed = sub.add_parser("embed", help="Embed PDF pages and save to JSON cache")
    src = p_embed.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdf",  help="Path or URL to a single PDF file")
    src.add_argument("--dir",  help="Directory to scan recursively for PDFs")
    p_embed.add_argument("--cache-dir", default=None, help="Directory to store cache files (default: next to each PDF, or <dir>/.cache/)")
    p_embed.add_argument("--author",    default=None, help="Override author metadata (applied to all PDFs)")
    p_embed.add_argument("--date",      default=None, help="Override date metadata YYYY-MM-DD (applied to all PDFs)")

    # ── insert ──
    p_insert = sub.add_parser("insert", help="Insert cached embeddings into Milvus")
    src2 = p_insert.add_mutually_exclusive_group(required=True)
    src2.add_argument("--cache",     help="Path to a single JSON cache file")
    src2.add_argument("--cache-dir", help="Directory containing *_cache.json files (scanned recursively)")
    p_insert.add_argument("--host", default="localhost", help="Milvus host")
    p_insert.add_argument("--port", default=19530, type=int, help="Milvus port")

    args = parser.parse_args()

    if args.command == "embed":
        cmd_embed(args)
    elif args.command == "insert":
        cmd_insert(args)


if __name__ == "__main__":
    main()