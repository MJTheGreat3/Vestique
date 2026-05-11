import os
import math
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from dotenv import load_dotenv
from ultralytics import YOLO
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration
)
from pinecone import Pinecone
from huggingface_hub import hf_hub_download

import clip

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"

QUERY_DIR = "queries"
DATASET_ROOT = "Img"

GALLERY_IDS_PATH = "embeddings/gallery_ids.json"

TOP_K = 15
ALPHA = 0.7

VALID_EXTS = (".jpg", ".jpeg", ".png", ".webp")

print("\nLoading gallery ids...")

with open(GALLERY_IDS_PATH) as f:
    gallery_ids = json.load(f)

vectorid_to_path = {}

for idx, path in enumerate(gallery_ids):

    vectorid_to_path[str(idx)] = path

print("Gallery ids loaded.")

print("\nLoading YOLO...")

yolo_model = YOLO("yolov8n.pt")

print("YOLO loaded.")

print("\nLoading CLIP...")

clip_model, preprocess = clip.load(
    "ViT-B/32",
    device=device
)

checkpoint_path = hf_hub_download(
    repo_id="matt-terofact/vestique-clip",
    filename="best_clip.pt"
)

checkpoint = torch.load(
    checkpoint_path,
    map_location=device
)

clip_model.load_state_dict(checkpoint)

clip_model.eval()

print("CLIP loaded.")

print("\nLoading BLIP2...")

blip_processor = Blip2Processor.from_pretrained(
    "Salesforce/blip2-opt-2.7b"
)

blip_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)

blip_model.eval()

print("BLIP2 loaded.")

print("\nConnecting to Pinecone...")

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

index = pc.Index("visual-search-index")

print("Pinecone connected.")

query_paths = []

for fname in sorted(os.listdir(QUERY_DIR)):

    if fname.lower().endswith(VALID_EXTS):

        query_paths.append(
            os.path.join(QUERY_DIR, fname)
        )

print(f"\nTotal query images: {len(query_paths)}")


def extract_item_id(path):

    parts = path.replace("\\", "/").split("/")

    for p in parts:

        if p.startswith("id_"):
            return p

    return None


def crop_main_object(image_path):

    image = Image.open(image_path).convert("RGB")

    results = yolo_model(image_path)

    boxes = results[0].boxes

    if len(boxes) == 0:
        return image

    largest_area = -1
    best_box = None

    for box in boxes:

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        area = (x2 - x1) * (y2 - y1)

        if area > largest_area:

            largest_area = area

            best_box = (x1, y1, x2, y2)

    x1, y1, x2, y2 = map(int, best_box)

    cropped = image.crop((x1, y1, x2, y2))

    return cropped


def generate_caption(image):

    inputs = blip_processor(
        images=image,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():

        generated_ids = blip_model.generate(
            **inputs,
            max_new_tokens=40
        )

    caption = blip_processor.decode(
        generated_ids[0],
        skip_special_tokens=True
    )

    return caption


def encode_image(image):

    img_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():

        emb = clip_model.encode_image(img_tensor)

    emb = emb / emb.norm(
        dim=-1,
        keepdim=True
    )

    return emb.cpu().numpy()


def encode_text(text):

    tokens = clip.tokenize(
        [text],
        truncate=True
    ).to(device)

    with torch.no_grad():

        emb = clip_model.encode_text(tokens)

    emb = emb / emb.norm(
        dim=-1,
        keepdim=True
    )

    return emb.cpu().numpy()


def create_fused_embedding(image_path):

    cropped = crop_main_object(image_path)

    caption = generate_caption(cropped)

    image_emb = encode_image(cropped)

    text_emb = encode_text(caption)

    fused = (
        ALPHA * image_emb
        +
        (1 - ALPHA) * text_emb
    )

    fused = fused / np.linalg.norm(
        fused,
        axis=1,
        keepdims=True
    )

    return fused[0], caption, cropped


def retrieve(query_embedding):

    results = index.query(
        vector=query_embedding.tolist(),
        top_k=TOP_K,
        include_metadata=True
    )

    return results["matches"]


def recall_at_k(relevant, k):

    retrieved_k = relevant[:k]

    return 1.0 if any(retrieved_k) else 0.0


def ndcg_at_k(relevant, k):

    dcg = 0.0

    for idx, rel in enumerate(relevant[:k]):

        if rel:

            dcg += 1.0 / math.log2(idx + 2)

    total_rel = sum(relevant)

    idcg = 0.0

    for idx in range(min(total_rel, k)):

        idcg += 1.0 / math.log2(idx + 2)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def average_precision_at_k(relevant, k):

    hits = 0
    precision_sum = 0.0

    for idx, rel in enumerate(relevant[:k]):

        if rel:

            hits += 1

            precision_sum += hits / (idx + 1)

    total_rel = sum(relevant)

    if total_rel == 0:
        return 0.0

    return precision_sum / min(total_rel, k)


metrics = {
    5: {
        "Recall": [],
        "NDCG": [],
        "mAP": []
    },
    10: {
        "Recall": [],
        "NDCG": [],
        "mAP": []
    },
    15: {
        "Recall": [],
        "NDCG": [],
        "mAP": []
    }
}

print("\nRunning batch evaluation...")

for query_path in query_paths:

    query_name = os.path.basename(query_path)

    query_item_id = extract_item_id(query_name)

    if query_item_id is None:

        print(f"Skipping {query_name}")
        continue

    query_embedding, caption, cropped = create_fused_embedding(
        query_path
    )

    matches = retrieve(query_embedding)

    relevant = []

    for match in matches:

        retrieved_path = match["metadata"]["image_name"]

        retrieved_item_id = extract_item_id(
            retrieved_path
        )

        relevant.append(
            retrieved_item_id == query_item_id
        )

    for K in [5, 10, 15]:

        metrics[K]["Recall"].append(
            recall_at_k(relevant, K)
        )

        metrics[K]["NDCG"].append(
            ndcg_at_k(relevant, K)
        )

        metrics[K]["mAP"].append(
            average_precision_at_k(relevant, K)
        )

print("\n===== FINAL METRICS =====\n")

print(
    f"{'K':<5}"
    f"{'Recall':<12}"
    f"{'NDCG':<12}"
    f"{'mAP':<12}"
)

print("-" * 45)

for K in [5, 10, 15]:

    recall = np.mean(
        metrics[K]["Recall"]
    )

    ndcg = np.mean(
        metrics[K]["NDCG"]
    )

    mAP = np.mean(
        metrics[K]["mAP"]
    )

    print(
        f"{K:<5}"
        f"{recall:<12.4f}"
        f"{ndcg:<12.4f}"
        f"{mAP:<12.4f}"
    )

print("\nDisplaying sample retrievals...")

num_examples = min(5, len(query_paths))

for i in range(num_examples):

    query_path = query_paths[i]

    query_embedding, caption, cropped = create_fused_embedding(
        query_path
    )

    matches = retrieve(query_embedding)

    fig, axes = plt.subplots(
        1,
        TOP_K + 1,
        figsize=(20, 5)
    )

    axes[0].imshow(cropped)

    axes[0].set_title("QUERY")

    axes[0].axis("off")

    for j, match in enumerate(matches):

        metadata = match["metadata"]

        image_rel_path = metadata["image_name"]

        gallery_path = os.path.join(
            DATASET_ROOT,
            image_rel_path
        )

        score = match["score"]

        if os.path.exists(gallery_path):

            img = Image.open(
                gallery_path
            ).convert("RGB")

            axes[j + 1].imshow(img)

            axes[j + 1].set_title(
                f"Top {j+1}\n{score:.3f}"
            )

        else:

            axes[j + 1].text(
                0.5,
                0.5,
                "Missing",
                ha="center",
                va="center"
            )

        axes[j + 1].axis("off")

    plt.tight_layout()

    plt.show()

print("\nDone.")