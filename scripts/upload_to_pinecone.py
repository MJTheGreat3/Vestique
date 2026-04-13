import numpy as np
import json
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
load_dotenv()

# Load fused embeddings
embs = np.load("embeddings/gallery_fused.npy")

# Load IDs
with open("embeddings/gallery_ids.json") as f:
    ids = json.load(f)

# Load captions (optional but recommended)
with open("blip_captions/gallery_captions_crop.json") as f:
    captions = json.load(f)

print("Embeddings:", embs.shape)

# Init Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API"))

index_name = "visual-search-index"

# Create index if needed
if index_name not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=embs.shape[1],
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)

# Upload
batch_size = 100

for i in range(0, len(embs), batch_size):
    batch = []

    for j in range(i, min(i + batch_size, len(embs))):
        id_str = str(ids[j])

        batch.append({
            "id": id_str,
            "values": embs[j].tolist(),
            "metadata": {
                "caption": captions[j] if j< len(captions) else ""
            }
        })

    index.upsert(vectors=batch)
    print(f"Uploaded {i + len(batch)} / {len(embs)}")

print("Upload complete")
