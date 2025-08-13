import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

class CLIPEmbedder:
    def __init__(self, model_name: str = "clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)

    def encode(self, image: Image.Image) -> np.ndarray:
        emb = self.model.encode(image, convert_to_numpy=True, normalize_embeddings=True)
        return emb.astype("float32")
