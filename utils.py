# utils.py
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def load_images_from_folder(folder):
    exts = (".jpg", ".jpeg", ".png")
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]

def preprocess_image(path):
    return Image.open(path).convert("RGB")

def show_image(image_paths, num_cols=5, title='Search Results'):
    num_images = len(image_paths)
    num_rows = (num_images + num_cols - 1) // num_cols

    plt.figure(figsize=(15, 3 * num_rows))
    plt.suptitle(title, fontsize=16)

    for idx, image_path in enumerate(image_paths):
        plt.subplot(num_rows, num_cols, idx + 1)
        if os.path.exists(image_path):
            img = mpimg.imread(image_path)
            plt.imshow(img)
            plt.axis('off')
            plt.title(f'#{idx+1}')
        else:
            plt.text(0.5, 0.5, 'Image Not Found', ha='center', va='center')
            plt.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

