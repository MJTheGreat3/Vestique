import os
import json
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader

load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# Load IDs (ORDER IS IMPORTANT)
with open("embeddings/gallery_ids.json") as f:
    ids = json.load(f)

DATASET_ROOT = "/Users/mj/IIITB/Sem_6/VR/Proj_2/Img/Img/img"

# Build id -> sorted images
id_to_images = {}

for root, dirs, files in os.walk(DATASET_ROOT):
    if os.path.basename(root).startswith("id_"):
        id_name = os.path.basename(root)
        imgs = sorted([f for f in files if f.endswith(".jpg")])
        full_paths = [os.path.join(root, f) for f in imgs]
        id_to_images[id_name] = full_paths

# Track usage count per ID
id_counters = {k: 0 for k in id_to_images}

image_urls = {}

# MAIN LOOP (aligned with embeddings)
for j, id_name in enumerate(ids):

    img_list = id_to_images.get(id_name, [])

    if len(img_list) == 0:
        image_urls[str(j)] = ""
        continue

    idx = id_counters[id_name]

    if idx >= len(img_list):
        idx = len(img_list) - 1

    img_path = img_list[idx]

    id_counters[id_name] += 1

    # Upload
    res = cloudinary.uploader.upload(
        img_path,
        folder="visual-search",
        public_id=f"{id_name}_{idx}",
        overwrite=False
    )

    image_urls[str(j)] = res["secure_url"]

    if j % 100 == 0:
        print("Uploaded images:", j)

# Save mapping
with open("embeddings/image_urls.json", "w") as f:
    json.dump(image_urls, f)

print("Image upload complete")