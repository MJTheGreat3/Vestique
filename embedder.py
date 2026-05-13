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
        self.model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)

        # ckpt = torch.load(
        #     "models/best_clip.pt",
        #     map_location=self.device
        # )
        # state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        # try:
        #     self.model.load_state_dict(state_dict)
        # except RuntimeError:
        #     print(
        #         "Skipping local CLIP checkpoint because it is not compatible "
        #         "with transformers CLIPModel. Using base CLIP weights."
        #     )
        self.model = self.model.to(self.device).eval()
        self.processor   = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    @torch.no_grad()
    def encode(self, image: Image.Image) -> np.ndarray:
        """Returns shape (512,), float32, L2-normalised."""
        inputs   = self.processor(images=image, return_tensors="pt").to(self.device)
        features = self.model.get_image_features(**inputs)

        if hasattr(features, "pooler_output") and features.pooler_output is not None:
            features = features.pooler_output
        elif hasattr(features, "last_hidden_state"):
            features = features.last_hidden_state[:, 0, :]

        return normalize_embedding(features.cpu().detach().numpy().flatten()).astype(np.float32)
