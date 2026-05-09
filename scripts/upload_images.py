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

DATASET_ROOT = "/Users/mj/IIITB/Sem_6/VR/Proj_2/Img/Img"

with open("embeddings/gallery_ids.json") as f:
    ids = json.load(f)

image_urls = {}
for j, img_name in enumerate(ids):

    img_path = os.path.join(
        DATASET_ROOT,
        img_name
    )
    if not os.path.exists(img_path):
        print(f"Missing: {img_path}")
        image_urls[str(j)] = ""
        continue
    try:

        res = cloudinary.uploader.upload(
            img_path,
            folder="visual-search",
            public_id=f"img_{j}",
            overwrite=True
        )

        image_urls[str(j)] = res["secure_url"]

        if j % 100 == 0:
            print(f"Uploaded {j}")

    except Exception as e:

        print(f"Upload failed: {img_path}")
        print(e)

        image_urls[str(j)] = ""

# SAVE URLS
with open("embeddings/image_urls.json", "w") as f:
    json.dump(image_urls, f)

print("Image upload complete")