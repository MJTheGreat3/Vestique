from __future__ import annotations
import numpy as np
import torch
from PIL import Image

import clip

from utils import normalize_embedding


class QueryEncoder:
    """
    Maps a cropped query image → L2-normalised CLIP visual embedding.
    BLIP is NOT used at query time; it runs post-retrieval in the reranker.
    """

    def __init__(self) -> None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)

        print("Loading CLIP model...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        ckpt_path = "models/clip.pt"
        print(f"Loading fine-tuned weights from: {ckpt_path}")
        try:
            ckpt = torch.load(ckpt_path, map_location=self.device)
            state_dict = (
                ckpt["model_state_dict"]
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt
                else ckpt
            )
            self.model.load_state_dict(state_dict)
            print("Fine-tuned weights loaded successfully.")
        except RuntimeError as e:
            print("Failed to load fine-tuned weights:")
            print(e)
            print("Using base CLIP weights instead.")

    @torch.no_grad()
    def encode(self, image: Image.Image) -> np.ndarray:
        """Returns shape (512,), float32, L2-normalised."""
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        features = self.model.encode_image(image_input)
        return normalize_embedding(
            features.cpu().detach().numpy().flatten()
        ).astype(np.float32)
