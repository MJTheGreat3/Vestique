from __future__ import annotations
import torch
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForImageTextRetrieval


ITM_MODEL = "Salesforce/blip-itm-base-coco"   # BLIP ITM head; swap for -large if VRAM allows


class Reranker:
    """
    BLIP-2 ITM re-ranker.

    For each (query_image, candidate_caption) pair, computes an
    image-text matching score. Candidates are re-sorted descending.

    Note: transformers exposes the ITM head via BlipForImageTextRetrieval.
    The 'use_itm_head=True' flag returns the binary ITM logit (match vs no-match);
    we take the softmax probability of the "match" class as the re-rank score.
    """

    def __init__(self) -> None:
        device_str    = "cuda" if torch.cuda.is_available() else "cpu"
        self.device   = torch.device(device_str)
        self.processor = BlipProcessor.from_pretrained(ITM_MODEL)
        self.model     = (
            BlipForImageTextRetrieval
            .from_pretrained(ITM_MODEL)
            .to(self.device)
            .eval()
        )

    @torch.no_grad()
    def _itm_score(self, image: Image.Image, caption: str) -> float:
        inputs = self.processor(
            images=image,
            text=caption,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(self.device)
        itm_out = self.model(**inputs, use_itm_head=True)
        # itm_out.itm_score shape: (1, 2)  →  [no-match, match]
        prob = torch.softmax(itm_out.itm_score, dim=-1)[0, 1].item()
        return prob

    def rerank(self, query_image: Image.Image, candidates: list[dict]) -> list[dict]:
        """
        Parameters
        ----------
        query_image : PIL.Image
        candidates  : list of dicts from FashionSearch.query()
                      each must contain a 'caption' key (pre-stored in Pinecone metadata)

        Returns
        -------
        Same list, re-sorted by ITM score descending. Each dict gains an 'itm_score' key.
        """
        for item in candidates:
            caption         = item.get("caption", item.get("title", ""))
            item["itm_score"] = self._itm_score(query_image, caption)

        return sorted(candidates, key=lambda x: x["itm_score"], reverse=True)