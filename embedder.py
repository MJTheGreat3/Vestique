from __future__ import annotations
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from config import CLIP_MODEL_NAME
from utils import normalize_embedding


class QueryEncoder:
    """
    Maps a cropped query image → L2-normalised CLIP visual embedding.
    BLIP is NOT used at query time; it runs post-retrieval in the reranker.
    """

    def __init__(self) -> None:
        device_str       = "cuda" if torch.cuda.is_available() else "cpu"
        self.device      = torch.device(device_str)
        self.model       = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(self.device).eval()
        self.processor   = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    @torch.no_grad()
    def encode(self, image: Image.Image) -> np.ndarray:
        """Returns shape (512,), float32, L2-normalised."""
        inputs   = self.processor(images=image, return_tensors="pt").to(self.device)
        features = self.model.get_image_features(**inputs)
        return normalize_embedding(features.cpu().detach().numpy().flatten()).astype(np.float32)