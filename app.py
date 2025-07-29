import gradio as gr
from query import search_by_text
from captioner import generate_caption

def search_with_image(image):
    # generate catopn from image
    caption = generate_caption(image)
    # search in faiss with caption       
    results = search_by_text(caption)

    return results, caption

def search_with_text(text):
    results = search_by_text(text)
    return results, text

# Basic website interface
iface = gr.Interface(
    fn = search_with_image,
    inputs = gr.Image(type="filepath", label="Upload Image"),
    outputs = [gr.Gallery(label="Top 5 Results"), gr.Text(label="Generated Caption")],
    title = "AI Fashion Search",
    description = ""
)

text_iface = gr.Interface(
    fn = search_with_text,
    inputs = gr.Textbox(label="Enter your query"),
    outputs = [gr.Gallery(label="Top 5 Results"), gr.Text(label="Query Used")],
    title = "Text-to-Image Search"
)

gr.TabbedInterface([iface, text_iface], ["Search with image", "Search with text"]).launch()
