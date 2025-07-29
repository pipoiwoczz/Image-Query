import torch
import faiss
import numpy as np
from transformers import CLIPProcessor, CLIPModel

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load faiss
index = faiss.read_index("faiss_index.bin")
with open("image_paths.txt", "r") as f:
    image_paths = [line.strip() for line in f.readlines()]

def search_by_text(query, top_k=5):
    inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features /= text_features.norm(p=2, dim=-1, keepdim=True)
        text_vector = text_features.cpu().numpy().astype('float32')
    distances, indices = index.search(text_vector, top_k)
    return [image_paths[i] for i in indices[0]]
