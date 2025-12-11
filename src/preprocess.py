# preprocess.py
from PIL import Image
from transformers import CLIPProcessor

def load_and_preprocess_image(image_path):
    """
    Loads and preprocesses an image using CLIP's processor.
    """
    image = Image.open(image_path).convert("RGB")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    inputs = processor(images=image, return_tensors="pt")
    return inputs, processor
