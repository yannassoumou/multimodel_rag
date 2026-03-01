
from pdf2image import convert_from_path
import requests


def save_video_temp(video_data):
    """Save video bytes to temporary file."""
    if isinstance(video_data, bytes):
        video_bytes = video_data
    elif hasattr(video_data, 'read'):
        video_bytes = video_data.read()
    elif isinstance(video_data, dict):
        video_bytes = video_data.get('bytes', video_data)
    else:
        video_bytes = video_data
    
    temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    temp_file.write(video_bytes)
    temp_file.close()
    return temp_file.name

def retrieve_topk(query_embeddings, corpus_embeddings, k=10):
    """Retrieve top-k results based on cosine similarity."""
    similarity_scores = torch.mm(query_embeddings, corpus_embeddings.T)
    results = []
    for i in range(len(query_embeddings)):
        scores = similarity_scores[i].cpu().float().numpy()
        ranked_indices = np.argsort(scores)[::-1][:k]
        ranked_scores = scores[ranked_indices]
        results.append({
            "ranked_indices": ranked_indices.tolist(),
            "ranked_scores": ranked_scores.tolist()
        })
    return results

def encode_corpus(inputs, batch_size=8):
    """Encode corpus in batches."""
    embeddings_list = []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i+batch_size]
        batch_emb = embedder.process(batch)
        embeddings_list.append(batch_emb)
    return torch.cat(embeddings_list, dim=0)

TOP_K = 10


def download_pdf(url, save_path="document.pdf"):
    response = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(response.content)
    print(f"PDF saved to {save_path}")
    return save_path

def pdf_to_images(pdf_path):
    images = convert_from_path(pdf_path)
    return images