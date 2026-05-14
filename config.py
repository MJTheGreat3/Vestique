import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Pinecone ──────────────────────────────────────────────────────────────────
PINECONE_API_KEY   = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX     = os.getenv("PINECONE_INDEX_NAME", "vestique-fashion")

# ── Model paths ───────────────────────────────────────────────────────────────
YOLO_MODEL_PATH    = os.getenv("YOLO_MODEL_PATH", "models/yolo11s.pt")
YOLO_POSE_MODEL    = os.getenv("YOLO_POSE_MODEL", "yolo11s-pose.pt")
YOLO_HF_REPO       = os.getenv("YOLO_HF_REPO", "")

CLIP_MODEL_PATH    = os.getenv("CLIP_MODEL_PATH", "")
CLIP_HF_REPO       = os.getenv("CLIP_HF_REPO", "")

CLIP_MODEL_NAME    = "openai/clip-vit-base-patch32"
BLIP_MODEL_NAME    = "Salesforce/blip-image-captioning-base"

# ── Search ────────────────────────────────────────────────────────────────────
RETRIEVAL_K        = 25
FINAL_TOP_K        = 10
EMBEDDING_DIM      = 512
BLIP_MAX_TOKENS    = 50

# ── Detection ─────────────────────────────────────────────────────────────────
YOLO_CONF_THRESHOLD = 0.25
CROP_PAD_FRACTION   = 0.05
HIP_CONF_THRESHOLD  = 0.3


# ── Weight downloaders ────────────────────────────────────────────────────────
def ensure_yolo_weights() -> None:
    if not Path(YOLO_MODEL_PATH).exists():
        from huggingface_hub import hf_hub_download

        print("Downloading YOLO weights from HuggingFace Hub...")

        hf_hub_download(
            repo_id=YOLO_HF_REPO,
            filename="yolo11s.pt",
            local_dir=".",
            local_dir_use_symlinks=False,
        )

        print("YOLO download complete.")


def ensure_clip_weights() -> None:
    if not Path(CLIP_MODEL_PATH).exists():
        from huggingface_hub import hf_hub_download

        print("Downloading clip.pt from HuggingFace Hub...")

        hf_hub_download(
            repo_id=CLIP_HF_REPO,
            filename="clip.pt",
            local_dir=".",
            local_dir_use_symlinks=False,
        )

        print("CLIP download complete.")