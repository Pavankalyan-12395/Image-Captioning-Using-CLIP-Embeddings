# model_utils.py
import torch
from transformers import CLIPModel

def generate_image_embeddings(inputs):
    """
    Generates image embeddings using CLIP.
    """
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features, model
