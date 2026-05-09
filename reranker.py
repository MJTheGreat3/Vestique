from __future__ import annotations

import torch
from PIL import Image

from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration
)

ITM_MODEL = "Salesforce/blip2-opt-2.7b"


class Reranker:
    """
    BLIP-2 reranker.

    Scores how well a candidate caption matches
    the query image using conditional likelihood.
    """

    def __init__(self) -> None:

        device_str = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device_str)

        self.processor = Blip2Processor.from_pretrained(
            ITM_MODEL
        )

        self.model = (
            Blip2ForConditionalGeneration
            .from_pretrained(
                ITM_MODEL,
                torch_dtype=torch.float16
                if torch.cuda.is_available()
                else torch.float32
            )
            .to(self.device)
            .eval()
        )

    @torch.no_grad()
    def _score_caption(
        self,
        image: Image.Image,
        caption: str
    ) -> float:

        inputs = self.processor(
            images=image,
            text=caption,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model(
            **inputs,
            labels=inputs["input_ids"]
        )

        loss = outputs.loss.item()

        return -loss

    def rerank(
        self,
        query_image: Image.Image,
        candidates: list[dict]
    ) -> list[dict]:

        for item in candidates:

            caption = item.get(
                "caption",
                item.get("title", "")
            )

            item["itm_score"] = self._score_caption(
                query_image,
                caption
            )

        return sorted(
            candidates,
            key=lambda x: x["itm_score"],
            reverse=True
        )