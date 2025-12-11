# caption_generator.py
import torch
from sklearn.metrics.pairwise import cosine_similarity

def match_captions(image_features, captions, clip_model, processor):
    """
    Computes similarity between image embeddings and caption embeddings,
    then returns captions sorted by similarity.
    """
    # Get text embeddings
    text_inputs = processor(text=captions, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**text_inputs)

    # Convert to numpy arrays for cosine similarity
    image_features = image_features.detach().cpu().numpy()
    text_features = text_features.detach().cpu().numpy()

    similarities = cosine_similarity(image_features, text_features)

    # Sort captions by similarity
    best_indices = similarities.argsort(axis=1)[0][::-1]
    best_captions = [captions[i] for i in best_indices]

    return best_captions, similarities[0][best_indices].tolist()


def image_captioning(image_path, candidate_captions, load_fn, embed_fn):
    """
    Pipeline function to handle preprocessing, embedding and caption matching.
    """
    inputs, processor = load_fn(image_path)
    image_features, clip_model = embed_fn(inputs)

    best_captions, similarities = match_captions(
        image_features,
        candidate_captions,
        clip_model,
        processor
    )

    return best_captions, similarities
