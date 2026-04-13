import numpy as np
import json

# Load image embeddings
img = np.load("embeddings/gallery_embs_crop.npy")

# Load text embeddings
txt = np.load("blip_captions/gallery_embs_crop.npy")

# Load IDs
with open("embeddings/gallery_ids.json") as f:
    ids = json.load(f)

# Check
print("Image:", img.shape)
print("Text :", txt.shape)
print("IDs  :", len(ids))

assert img.shape == txt.shape
assert len(ids) == img.shape[0]

# Fusion
alpha = 0.7
fused = alpha * img + (1 - alpha) * txt

# Normalize
fused = fused / np.linalg.norm(fused, axis=1, keepdims=True)

# Save
np.save("embeddings/gallery_fused.npy", fused)

print(" Saved gallery_fused.npy")
