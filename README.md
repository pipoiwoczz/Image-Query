# AI Fashion Search Engine

[![Watch the demo](https://img.youtube.com/vi/pmll65e1jqk/maxresdefault.jpg)](https://youtu.be/pmll65e1jqk)

A multimodal search engine for fashion items that allows users to find visually similar products using either images or natural language descriptions. Powered by state-of-the-art computer vision and NLP models.

## Project Structure

```graphql
├── images/ # Folder containing images
├── app.py # Gradio app to run the web demo
├── build_index.py # Builds the FAISS index from image embeddings
├── captioner.py # LLM-based caption generator
├── extract_metadata.py # Extracts metadata from DeepFashion dataset
├── extract_preprocess_deepfashion.py # Handles image preprocessing
├── faiss_index.bin # Serialized FAISS index
├── image_metadata.txt # Metadata (captions/descriptions) per image
├── image_paths.txt # Corresponding paths to each image
├── query.py # Main function for running CLIP-based queries
├── README.md # Project documentation
├── search.py # Test core search logic (image/caption → vector → top-k)
├── utils.py # Utility functions
```

## Features
- **Image-to-Image Search**: Upload a fashion item photo to find visually similar products
- **Text-to-Image Search**: Describe fashion items in natural language (e.g., "red floral summer dress")
- **Automatic Captioning**: BLIP model generates descriptions for image queries
- **Lightning Fast**: Sub-second query response time using FAISS indexing
- **Responsive UI**: Gradio-based interface with gallery previews
- **Dataset Processing**: Tools for preparing and preprocessing fashion datasets

## Technology Stack 

| Component          | Technology                                  |
|--------------------|---------------------------------------------|
| Core Models        | CLIP-ViT-B/32, BLIP-image-captioning-base   |
| Similarity Search  | FAISS (Facebook AI Similarity Search)       |
| Backend Framework  | PyTorch, Hugging Face Transformers          |
| Frontend           | Gradio                                      |
| Image Processing   | PIL, OpenCV                                 |
| Utilities          | NumPy, tqdm, Matplotlib                     |

##  Dataset

We use the [DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) dataset for experimentation.

> 🔖 Please refer to the original [DeepFashion License](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) before using it in commercial applications.

---

## Getting Started
### 1. Install dependencies
```bash
pip install -r requirements.txt
```
### 2. Download dataset
- Download the `Img directory` and `list_eval_partition.txt` in `Eval directory` from [DeepFashion](https://drive.google.com/drive/folders/0B7EVK8r0v71pekpRNUlMS3Z5cUk?resourcekey=0-GHiFnJuDTvzzGuTj6lE6og)
- Extract the `Img directory` and put it in your repository, rename this folder to `images`, then copy `list_eval_partition.txt` and put it in directory `images`.
### 3. Preprocess images
```bash
python extract_preprocess_deepfashion.py
```
### 4. Build the FAISS index
```bash
python build_faiss_index.py
```
This script extracts features from all fashion images and builds a FAISS index saved as faiss_index.bin. This step will take lots of time, please be patient.
### 5. Run the Web Demo
```bash
python app.py
```

## Usage
- Input a natural language query.
- Adjust the number of top-k results using the dropdown.
- View the top-matching fashion items with images and descriptions.

## Limitations & Future Work
- ❌ Does not handle real-time dataset updates (needs re-indexing).
- 📉 Accuracy may vary with low-res or blurry images.
- ⚙️ No user login/authentication for the web app.
- 🧠 Future work:
    - Add shopping platform integration (e.g., Amazon or Zalando links)
    - Train domain-specific CLIP/BLIP models

## 🙏 Acknowledgments

This project would not be possible without the following open-source contributions and datasets:

- [OpenAI CLIP](https://github.com/openai/CLIP) – for enabling multimodal (image-text) representation learning.
- [BLIP (Bootstrapped Language Image Pretraining)](https://github.com/salesforce/BLIP) – for high-quality image captioning.
- [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss) – for efficient vector similarity search at scale.
- [Hugging Face Transformers](https://github.com/huggingface/transformers) – for easy access to state-of-the-art models and tokenizers.
- [DeepFashion Dataset](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) – for high-quality fashion product images and annotations.

