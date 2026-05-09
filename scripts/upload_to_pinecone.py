import numpy as np
import json
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# LOAD DATA
embs = np.load(
    "embeddings/gallery_fused.npy"
)

with open("embeddings/gallery_ids.json") as f:
    ids = json.load(f)

with open("embeddings/gallery_caps.json") as f:
    captions = json.load(f)

with open("embeddings/image_urls.json") as f:
    image_urls = json.load(f)

print("Embeddings:", embs.shape)
print("IDs       :", len(ids))
print("Captions  :", len(captions))

assert len(ids) == len(captions)
assert len(ids) == embs.shape[0]


# INIT PINECONE


pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

index_name = "visual-search-index"

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


# DELETE OLD VECTORS
print("Clearing old vectors...")
index.delete(delete_all=True)

# UPLOAD
batch_size = 100
for i in range(0, len(embs), batch_size):
    batch = []
    for j in range(i, min(i + batch_size, len(embs))):
        batch.append({
            "id": str(j),
            "values": embs[j].tolist(),
            "metadata": {
                "image_name": ids[j],
                "caption": captions[j],
                "image_url": image_urls.get(str(j), "")
            }
        })

    index.upsert(vectors=batch)
    print(f"Uploaded {i + len(batch)} / {len(embs)}")

print("Upload complete")