from __future__ import annotations

import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper

from detector import ClothingDetector
from embedder import QueryEncoder
from reranker import Reranker
from config import FINAL_TOP_K, ensure_clip_weights
from search import FashionSearch
from utils import pil_to_rgb, resize_for_display

ensure_clip_weights()

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Vestique – Visual Fashion Search",
    page_icon="🪡",
    layout="wide",
)

# ── Rustic / antique CSS ──────────────────────────────────────────────────────
RUSTIC_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=Crimson+Text:ital,wght@0,400;0,600;1,400&display=swap');

/* ── Global palette ─────────────────────────────────────────────────── */
:root {
    --parchment:   #F5E6C8;
    --parchment2:  #EDD9A3;
    --ink:         #2C1A0E;
    --ink-light:   #5C3D1E;
    --rust:        #8B3A0F;
    --gold:        #C9923A;
    --gold-light:  #E8B96A;
    --sepia:       #9E7B4F;
    --cream:       #FDF6E3;
}

/* ── App background – aged parchment ────────────────────────────────── */
.stApp {
    background-color: var(--parchment);
    background-image:
        repeating-linear-gradient(
            0deg, transparent, transparent 28px,
            rgba(160,120,60,0.04) 28px, rgba(160,120,60,0.04) 29px
        );
    font-family: 'Crimson Text', Georgia, serif;
    color: var(--ink);
}

/* ── Sidebar ─────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #3D2B1F;
    border-right: 2px solid var(--gold);
}
[data-testid="stSidebar"] * { color: var(--parchment) !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: 'Playfair Display', Georgia, serif;
    color: var(--gold-light) !important;
}

/* ── Headings ─────────────────────────────────────────────────────────── */
h1, h2, h3, h4 {
    font-family: 'Playfair Display', Georgia, serif;
    color: var(--rust);
    letter-spacing: 0.02em;
}

/* ── Decorative divider ─────────────────────────────────────────────── */
hr {
    border: none;
    border-top: 1px solid var(--gold);
    margin: 1.5rem 0;
    opacity: 0.6;
}

/* ── Buttons ─────────────────────────────────────────────────────────── */
.stButton > button {
    font-family: 'Playfair Display', Georgia, serif !important;
    background-color: var(--rust) !important;
    color: var(--parchment) !important;
    border: 1.5px solid var(--gold) !important;
    border-radius: 4px !important;
    letter-spacing: 0.05em;
    transition: background 0.2s, color 0.2s;
}
.stButton > button:hover {
    background-color: var(--gold) !important;
    color: var(--ink) !important;
}

/* ── File uploader ───────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--gold) !important;
    border-radius: 4px;
    background: var(--cream);
}

/* ── Info / success / warning boxes ─────────────────────────────────── */
[data-testid="stAlert"] {
    font-family: 'Crimson Text', Georgia, serif;
    border-radius: 4px;
    border-left: 4px solid var(--gold);
    background: var(--cream) !important;
    color: var(--ink) !important;
}

/* ── Image containers ────────────────────────────────────────────────── */
[data-testid="stImage"] img {
    border: 2px solid var(--sepia);
    border-radius: 2px;
    box-shadow: 4px 4px 12px rgba(44, 26, 14, 0.2);
}

/* ── Result cards ────────────────────────────────────────────────────── */
.result-card {
    background: var(--cream);
    border: 1.5px solid var(--sepia);
    border-radius: 4px;
    padding: 10px;
    box-shadow: 2px 2px 8px rgba(44,26,14,0.12);
    text-align: center;
    font-family: 'Crimson Text', Georgia, serif;
    color: var(--ink);
    margin-bottom: 8px;
}
.result-card img {
    width: 100%;
    height: 180px;
    object-fit: cover;
    border: 1px solid var(--parchment2);
    margin-bottom: 6px;
    display: block;
}
.result-card .item-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--rust);
    margin-bottom: 2px;
}
.result-card .item-meta {
    font-size: 0.82rem;
    color: var(--sepia);
}
.result-card .score-badge {
    display: inline-block;
    font-size: 0.72rem;
    background: var(--rust);
    color: var(--parchment);
    padding: 1px 6px;
    border-radius: 2px;
    margin-top: 4px;
    letter-spacing: 0.04em;
}

/* ── Caption box ─────────────────────────────────────────────────────── */
.caption-box {
    background: var(--cream);
    border: 1px solid var(--gold);
    border-left: 4px solid var(--rust);
    padding: 10px 14px;
    font-style: italic;
    font-size: 1rem;
    color: var(--ink-light);
    border-radius: 2px;
    margin: 8px 0;
}

/* ── Spinner ─────────────────────────────────────────────────────────── */
.stSpinner > div { border-top-color: var(--rust) !important; }
</style>
"""

st.markdown(RUSTIC_CSS, unsafe_allow_html=True)

# ── Cached model loaders (only instantiated once per session) ─────────────────

@st.cache_resource(show_spinner="Warming up the YOLO lookout…")
def load_detector() -> ClothingDetector:
    return ClothingDetector()

@st.cache_resource(show_spinner="Loading CLIP encoder…")
def load_encoder() -> QueryEncoder:
    return QueryEncoder()

@st.cache_resource(show_spinner="Loading BLIP-2 re-ranker…")
def load_reranker() -> Reranker:
    return Reranker()

@st.cache_resource(show_spinner="Opening the Pinecone catalogue…")
def load_search() -> FashionSearch:
    return FashionSearch()

# ── Session-state initialiser ────────────────────────────────────────────────

def _init_state() -> None:
    defaults = {
        "stage":          "upload",   # upload | confirm | manual | results
        "original_image": None,
        "auto_crop":      None,
        "final_crop":     None,
        "caption":        None,
        "results":        None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("# 🪡 Vestique")
    st.markdown(
        "*Search the world's garments by image — an antiquary's eye for modern fashion.*"
    )
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown(
        "1. **Upload** a photo containing clothing\n"
        "2. **Confirm** the auto-detected garment crop\n"
        "3. **Search** our catalogue with AI vision\n"
        "4. **Browse** the top 10 visual matches"
    )
    st.markdown("---")
    if st.button("↺  Start over"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown(
    "<h1 style='text-align:center;letter-spacing:0.08em;'>✦ VESTIQUE ✦</h1>"
    "<p style='text-align:center;font-style:italic;color:#9E7B4F;margin-top:-8px;'>"
    "A visual search engine for the discerning clothier</p>",
    unsafe_allow_html=True,
)
st.markdown("<hr/>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# STAGE 1 – Upload
# ═════════════════════════════════════════════════════════════════════════════

if st.session_state.stage == "upload":
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown("### Upload a garment photograph")
        uploaded = st.file_uploader(
            "Drag & drop or click to browse",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )
        if uploaded:
            image = pil_to_rgb(Image.open(uploaded))
            st.session_state.original_image = image

            with st.spinner("Inspecting your garment…"):
                detector = load_detector()
                crop, box, conf = detector.detect(image)

            if crop is not None:
                st.session_state.auto_crop = crop
                st.session_state.stage     = "confirm"
                st.rerun()
            else:
                st.warning(
                    "No clothing article detected automatically. "
                    "You may crop manually."
                )
                st.session_state.stage = "manual"
                st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
# STAGE 2 – Confirm auto crop
# ═════════════════════════════════════════════════════════════════════════════

elif st.session_state.stage == "confirm":
    st.markdown("### Is this the garment you wish to search?")
    col_orig, col_crop = st.columns(2)

    with col_orig:
        st.markdown("**Original photograph**")
        st.image(resize_for_display(st.session_state.original_image), use_container_width=True)

    with col_crop:
        st.markdown("**Detected garment**")
        st.image(resize_for_display(st.session_state.auto_crop), use_container_width=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    col_yes, col_no, _ = st.columns([1, 1, 3])

    with col_yes:
        if st.button("✓  Yes, search this", use_container_width=True):
            st.session_state.final_crop = st.session_state.auto_crop
            st.session_state.stage      = "searching"
            st.rerun()

    with col_no:
        if st.button("✗  No, let me crop", use_container_width=True):
            st.session_state.stage = "manual"
            st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
# STAGE 3 – Manual crop
# ═════════════════════════════════════════════════════════════════════════════

elif st.session_state.stage == "manual":
    st.markdown("### Select the garment region manually")
    st.info("Drag the handles to frame the clothing article, then confirm.")

    image = st.session_state.original_image
    cropped = st_cropper(
        image,
        realtime_update=True,
        box_color="#C9923A",
        aspect_ratio=None,
    )

    st.markdown("<br/>", unsafe_allow_html=True)
    col_btn, _ = st.columns([1, 4])
    with col_btn:
        if st.button("✓  Search this crop", use_container_width=True):
            st.session_state.final_crop = pil_to_rgb(cropped)
            st.session_state.stage      = "searching"
            st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
# STAGE 4 – Embedding + search
# ═════════════════════════════════════════════════════════════════════════════

elif st.session_state.stage == "searching":
    # Step 2: CLIP query encoding
    with st.spinner("Encoding your garment with CLIP…"):
        encoder  = load_encoder()                              # was load_embedder()
        embedding = encoder.encode(st.session_state.final_crop)

    # Step 3: ANN retrieval
    with st.spinner("Searching the HNSW catalogue…"):
        searcher   = load_search()
        candidates = searcher.query(embedding)                 # returns RETRIEVAL_K results

    # Step 4: BLIP-2 ITM re-ranking
    with st.spinner(f"Re-ranking {len(candidates)} candidates with BLIP-2 ITM…"):
        reranker = load_reranker()
        results  = reranker.rerank(st.session_state.final_crop, candidates)[:FINAL_TOP_K]
        st.session_state.results = results

    st.session_state.stage = "results"
    st.rerun()
    
# ═════════════════════════════════════════════════════════════════════════════
# STAGE 5 – Results
# ═════════════════════════════════════════════════════════════════════════════

elif st.session_state.stage == "results":
    col_query, col_res = st.columns([1, 3])

    with col_query:
        st.markdown("### Your search")
        st.image(resize_for_display(st.session_state.final_crop, 300), use_container_width=True)
        if st.session_state.caption:
            st.markdown(
                f"<div class='caption-box'>&ldquo;{st.session_state.caption}&rdquo;</div>",
                unsafe_allow_html=True,
            )

    with col_res:
        st.markdown("### Top 10 matches")
        results = st.session_state.results or []

        if not results:
            st.warning("No matches found in the catalogue.")
        else:
            # 5 columns × 2 rows
            for row_start in range(0, 10, 5):
                row_results = results[row_start : row_start + 5]
                cols = st.columns(5)
                for col, item in zip(cols, row_results):
                    with col:
                        img_html = (
                            f"<img src='{item['image_url']}' alt='{item['title']}'/>"
                            if item["image_url"]
                            else "<div style='width:100%;height:180px;background:#EDD9A3;'></div>"
                        )
                        st.markdown(
                            f"""
                            <div class="result-card">
                                {img_html}
                                <div class="item-title">{item['title']}</div>
                                <div class="item-meta">{item['brand']} · {item['price']}</div>
                                <span class="score-badge">cosine {item['score']} · itm {item['itm_score']:.3f}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    st.markdown("<hr/>", unsafe_allow_html=True)
    if st.button("← Search another garment"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()