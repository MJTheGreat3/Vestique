import os
from dotenv import load_dotenv

load_dotenv()

# ── Pinecone ──────────────────────────────────────────────────────────────────
PINECONE_API_KEY   = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX     = os.getenv("PINECONE_INDEX_NAME", "vestique-fashion")

# ── Model paths ───────────────────────────────────────────────────────────────
YOLO_MODEL_PATH    = os.getenv("YOLO_MODEL_PATH", "best.pt")
CLIP_MODEL_NAME    = "openai/clip-vit-base-patch32"
BLIP_MODEL_NAME    = "Salesforce/blip-image-captioning-base"

# ── Search ────────────────────────────────────────────────────────────────────
RETRIEVAL_K        = 50           # how many ANN candidates to fetch
FINAL_TOP_K        = 10           # how many to show after re-ranking
EMBEDDING_DIM      = 512          # CLIP ViT-B/32 output dim
BLIP_MAX_TOKENS    = 50

# ── Detection ─────────────────────────────────────────────────────────────────
YOLO_CONF_THRESHOLD = 0.25        # minimum confidence to accept a detection
CROP_PAD_FRACTION   = 0.05        # add 5% padding around the detected bbox

# ── Weight downloader ─────────────────────────────────────────────────────────
def ensure_clip_weights() -> None:
    """Download clip.pt from HuggingFace Hub if not already present locally."""
    if not Path(CLIP_MODEL_PATH).exists():
        from huggingface_hub import hf_hub_download
        print("Downloading clip.pt from HuggingFace Hub…")
        hf_hub_download(
            repo_id=CLIP_HF_REPO,
            filename="clip.pt",
            local_dir=".",
        )
        print("Download complete.")