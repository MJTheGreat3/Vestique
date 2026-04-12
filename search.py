from __future__ import annotations
import numpy as np
from pinecone import Pinecone

from config import PINECONE_API_KEY, PINECONE_INDEX, RETRIEVAL_K


class FashionSearch:
    """
    Thin wrapper around a Pinecone index.
    Each vector in the index should have metadata like:
      { "image_url": "...", "title": "...", "price": "...", "brand": "..." }
    """

    def __init__(self) -> None:
        pc          = Pinecone(api_key=PINECONE_API_KEY)
        self.index  = pc.Index(PINECONE_INDEX)

    def query(self, embedding: np.ndarray) -> list[dict]:
        response = self.index.query(
            vector=embedding.tolist(),
            top_k=RETRIEVAL_K,          # over-fetch
            include_metadata=True,
        )

        results = []
        for match in response.matches:
            meta = match.metadata or {}
            results.append({
                "id":        match.id,
                "score":     round(float(match.score), 4),   # ANN cosine similarity
                "image_url": meta.get("image_url", ""),
                "title":     meta.get("title", "Unknown"),
                "price":     meta.get("price", "—"),
                "brand":     meta.get("brand", "—"),
                "caption":   meta.get("caption", ""),         # ← needed by reranker
            })
        return results