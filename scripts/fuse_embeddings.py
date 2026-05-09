import os
import json
import numpy as np


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

EMB_DIR = os.path.join(
    BASE_DIR,
    "embeddings"
)

img = np.load(
    os.path.join(
        EMB_DIR,
        "gallery_embs_crop.npy"
    )
)

txt = np.load(
    os.path.join(
        EMB_DIR,
        "gallery_text_embs.npy"
    )
)


with open(
    os.path.join(
        EMB_DIR,
        "gallery_ids.json"
    )
) as f:
    ids = json.load(f)

print("Image:", img.shape)
print("Text :", txt.shape)
print("IDs  :", len(ids))
assert img.shape == txt.shape
assert len(ids) == img.shape[0]

alpha = 0.7
fused = alpha * img + (1 - alpha) * txt


fused = fused / np.linalg.norm(
    fused,
    axis=1,
    keepdims=True
)
save_path = os.path.join(
    EMB_DIR,
    "gallery_fused.npy"
)
np.save(save_path, fused)
print(f"Saved: {save_path}")