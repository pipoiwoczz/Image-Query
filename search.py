# TESTING FILE

import faiss
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt

# -------- CONFIG --------
INDEX_PATH = "faiss_index.bin"
IMAGE_PATHS_FILE = "image_paths.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 5  # Numbers of images return

# -------- LOAD CLIP --------
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# -------- LOAD INDEX & IMAGE PATHS --------
index = faiss.read_index(INDEX_PATH)
with open(IMAGE_PATHS_FILE, "r", encoding="utf-8") as f:
    image_paths = [line.strip() for line in f]

# -------- FEATURE EXTRACTION --------
def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        embeddings = clip_model.get_image_features(**inputs)
    features = embeddings.cpu().numpy()
    faiss.normalize_L2(features)  
    return features

# -------- SEARCH --------
def search_similar_images(query_image_path):
    query_vector = extract_features(query_image_path)
    distances, indices = index.search(query_vector, TOP_K)
    return [image_paths[i] for i in indices[0]], distances[0]

# -------- VISUALIZE --------
def show_results(query_img, results):
    plt.figure(figsize=(15, 4))
    plt.subplot(1, TOP_K + 1, 1)
    plt.imshow(Image.open(query_img))
    plt.title("Query Image")
    plt.axis("off")

    for i, img_path in enumerate(results):
        plt.subplot(1, TOP_K + 1, i + 2)
        plt.imshow(Image.open(img_path))
        plt.title(f"Result {i+1}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# -------- MAIN --------
if __name__ == "__main__":
    query_image = input("Input path: ").strip()
    try:
        results, dists = search_similar_images(query_image)
        print("Top kết quả gần nhất:")
        for path, dist in zip(results, dists):
            print(f"{path} (distance={dist:.4f})")
        show_results(query_image, results)
    except Exception as e:
        print(f"❌ Lỗi: {e}")
