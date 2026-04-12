from __future__ import annotations
import numpy as np
import torch
from PIL import Image
from transformers import (
    CLIPModel, CLIPProcessor,
    BlipProcessor, BlipForConditionalGeneration,
)

from config import CLIP_MODEL_NAME, BLIP_MODEL_NAME, BLIP_MAX_TOKENS, EMBEDDING_DIM
from utils import normalize_embedding


class FashionEmbedder:
    """
    Two-stage embedding pipeline:
      1. BLIP  → natural-language caption of the clothing crop.
      2. CLIP  → image embedding + text embedding of the caption.
      Combined embedding = average(image_emb, text_emb), L2-normalised.
    """

    def __init__(self) -> None:
        device_str       = "cuda" if torch.cuda.is_available() else "cpu"
        self.device      = torch.device(device_str)

        # ── CLIP ──────────────────────────────────────────────────────────────
        self.clip_model     = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(self.device).eval()
        self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

        # ── BLIP ──────────────────────────────────────────────────────────────
        self.blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
        self.blip_model     = (
            BlipForConditionalGeneration
            .from_pretrained(BLIP_MODEL_NAME)
            .to(self.device)
            .eval()
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    @torch.no_grad()
    def _image_embedding(self, image: Image.Image) -> np.ndarray:
        inputs   = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        features = self.clip_model.get_image_features(**inputs)
        return features.cpu().numpy().flatten()

    @torch.no_grad()
    def _text_embedding(self, text: str) -> np.ndarray:
        inputs   = self.clip_processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        features = self.clip_model.get_text_features(**inputs)
        return features.cpu().numpy().flatten()

    @torch.no_grad()
    def _caption(self, image: Image.Image) -> str:
        inputs  = self.blip_processor(image, return_tensors="pt").to(self.device)
        ids     = self.blip_model.generate(**inputs, max_new_tokens=BLIP_MAX_TOKENS)
        return self.blip_processor.decode(ids[0], skip_special_tokens=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def embed(self, image: Image.Image) -> tuple[np.ndarray, str]:
        """
        Returns
        -------
        embedding : np.ndarray, shape (EMBEDDING_DIM,), L2-normalised
        caption   : str  – BLIP-generated description
        """
        caption   = self._caption(image)
        image_emb = normalize_embedding(self._image_embedding(image))
        text_emb  = normalize_embedding(self._text_embedding(caption))
        combined  = normalize_embedding((image_emb + text_emb) / 2.0)
        return combined.astype(np.float32), caption