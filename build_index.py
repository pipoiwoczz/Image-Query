# build_index.py
import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import faiss
import numpy as np
from utils import load_images_from_folder, preprocess_image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load images and extract embeddings
image_paths = load_images_from_folder("images")
image_embeddings = []

for path in image_paths:
    image = preprocess_image(path)
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
        image_embeddings.append(image_features.cpu().numpy())

# Stack to numpy array
image_embeddings = np.vstack(image_embeddings).astype('float32')

# Create FAISS index
dim = image_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(image_embeddings)

# Store index and mapping into .bin file
faiss.write_index(index, "faiss_index.bin")
with open("image_paths.txt", "w") as f:
    for path in image_paths:
        f.write(path + "\n")

print("FAISS index built and saved.")
