import numpy as np
import torch
from io import BytesIO
from PIL import Image

from scripts.qwen3_vl_embedding import Qwen3VLEmbedder
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

embedder = Qwen3VLEmbedder("Qwen/Qwen3-VL-Embedding-2B")

document_inputs = []
for idx, img in enumerate(document_images):
    img_path = f"temp_page_{idx}.png"
    img.save(img_path)
    document_inputs.append({"image": img_path})

document_embeddings = embedder.process(document_inputs)
query_embeddings = embedder.process(queries)
print(f"Query embeddings shape: {query_embeddings.shape}")