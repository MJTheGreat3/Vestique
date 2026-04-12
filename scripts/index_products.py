"""
Usage:
    python scripts/index_products.py --csv products.csv
    python scripts/index_products.py --image-dir ./product_images

CSV expected columns:
    id, image_path, image_url, title, brand, price
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
from pinecone import Pinecone, ServerlessSpec

from config import PINECONE_API_KEY, PINECONE_INDEX, EMBEDDING_DIM
from embedder import FashionEmbedder


def get_or_create_index(pc: Pinecone) -> object:
    existing = [i.name for i in pc.list_indexes()]
    if PINECONE_INDEX not in existing:
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"Created index '{PINECONE_INDEX}'")
    return pc.Index(PINECONE_INDEX)


def index_from_csv(csv_path: str, embedder: FashionEmbedder, index) -> None:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        batch  = []
        for i, row in enumerate(reader):
            try:
                image = Image.open(row["image_path"]).convert("RGB")
                emb, _ = embedder.embed(image)
                batch.append((
                    row["id"],
                    emb.tolist(),
                    {
                        "image_url": row.get("image_url", ""),
                        "title":     row.get("title", ""),
                        "brand":     row.get("brand", ""),
                        "price":     row.get("price", ""),
                    },
                ))
                if len(batch) >= 50:
                    index.upsert(vectors=batch)
                    print(f"  Upserted {i+1} records…")
                    batch = []
            except Exception as e:
                print(f"  Skipping {row.get('id','?')}: {e}")

        if batch:
            index.upsert(vectors=batch)
    print("Indexing complete.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="Path to products CSV")
    args = parser.parse_args()

    if not args.csv:
        parser.print_help()
        sys.exit(1)

    pc       = Pinecone(api_key=PINECONE_API_KEY)
    index    = get_or_create_index(pc)
    embedder = FashionEmbedder()

    print(f"Indexing into '{PINECONE_INDEX}'…")
    if args.csv:
        index_from_csv(args.csv, embedder, index)


if __name__ == "__main__":
    main()