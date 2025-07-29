# extract_preprocess_deepfashion.py
import os
from PIL import Image
from tqdm import tqdm
import shutil

# Directory of downloaded img from DeepFashio
SRC_IMG_DIR = "images/img"
DST_IMG_DIR = "images"

# Make sure destination directory exist
os.makedirs(DST_IMG_DIR, exist_ok=True)

# Read images list list_eval_partition.txt
with open("images/list_eval_partition.txt", "r") as f:
    lines = f.readlines()[2:]  # Bỏ 2 dòng header
    image_list = [line.strip().split() for line in lines]

# Preprocess image
def preprocess_and_save(src_path, dst_path):
    try:
        image = Image.open(src_path).convert("RGB")
        image = image.resize((224, 224))  # Resize để phù hợp với CLIP
        image.save(dst_path)
    except Exception as e:
        print(f"Error when preprocess image: {src_path} - {e}")

# 
for rel_path, _, _ in tqdm(image_list):
    src_path = os.path.join(DST_IMG_DIR, rel_path)
    file_name = rel_path.replace("/", "_")  # EX: 'men/Shirts/img1.jpg' → 'men_Shirts_img1.jpg'
    dst_path = os.path.join(DST_IMG_DIR, file_name)
    preprocess_and_save(src_path, dst_path)

print(f"Save images successfully")
