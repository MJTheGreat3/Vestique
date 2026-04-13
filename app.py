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
FASHION_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Light mode ──────────────────────────────────────────────────────── */
:root {
    --ivory:       #F7F6F2;
    --ivory2:      #EDECEA;
    --ink:         #0C0C0C;
    --ink-muted:   #4A4A4A;
    --champagne:   #B8975A;
    --champ-light: #D4B478;
    --surface:     #FFFFFF;
    --border:      #E0DFDB;
}

/* ── Dark mode ───────────────────────────────────────────────────────── */
@media (prefers-color-scheme: dark) {
    :root {
        --ivory:       #111110;
        --ivory2:      #1C1C1B;
        --ink:         #F0EFE9;
        --ink-muted:   #A8A8A0;
        --champagne:   #C9A86C;
        --champ-light: #E0C080;
        --surface:     #171716;
        --border:      #2E2E2C;
    }
}

/* ── Global ──────────────────────────────────────────────────────────── */
.stApp {
    background-color: var(--ivory);
    font-family: 'DM Sans', system-ui, sans-serif;
    font-weight: 300;
    color: var(--ink);
    letter-spacing: 0.01em;
}

/* ── Headings ─────────────────────────────────────────────────────────── */
h1, h2, h3, h4 {
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-weight: 300;
    color: var(--ink);
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

/* ── Sidebar ─────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: var(--ink);
    border-right: 1px solid #222;
}
[data-testid="stSidebar"] * {
    color: var(--ivory) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 300 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: 'Cormorant Garamond', serif !important;
    color: var(--champ-light) !important;
    letter-spacing: 0.14em;
}

/* ── Divider ─────────────────────────────────────────────────────────── */
hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 2rem 0;
}

/* ── Buttons ─────────────────────────────────────────────────────────── */
.stButton > button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 400 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    background-color: var(--ink) !important;
    color: var(--ivory) !important;
    border: 1px solid var(--ink) !important;
    border-radius: 0 !important;
    padding: 0.6rem 1.8rem !important;
    transition: background 0.2s, color 0.2s !important;
}
.stButton > button:hover {
    background-color: var(--champagne) !important;
    border-color: var(--champagne) !important;
    color: var(--surface) !important;
}

/* ── File uploader ───────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border: 1px solid var(--border) !important;
    border-radius: 0 !important;
    background: var(--surface) !important;
}

/* ── Alert / info boxes ──────────────────────────────────────────────── */
[data-testid="stAlert"] {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.88rem;
    border-radius: 0 !important;
    border-left: 3px solid var(--champagne) !important;
    background: var(--surface) !important;
    color: var(--ink-muted) !important;
}

/* ── Images ──────────────────────────────────────────────────────────── */
[data-testid="stImage"] img {
    border: 1px solid var(--border);
    border-radius: 0;
}

/* ── Result cards ────────────────────────────────────────────────────── */
.result-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 0;
    padding: 0;
    overflow: hidden;
    text-align: left;
    font-family: 'DM Sans', sans-serif;
    color: var(--ink);
    margin-bottom: 12px;
    transition: border-color 0.2s;
}
.result-card:hover {
    border-color: var(--champagne);
}
.result-card img {
    width: 100%;
    height: 200px;
    object-fit: cover;
    display: block;
}
.result-card .card-body {
    padding: 10px 12px 12px;
}
.result-card .item-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 0.95rem;
    font-weight: 400;
    letter-spacing: 0.04em;
    color: var(--ink);
    margin-bottom: 3px;
}
.result-card .item-meta {
    font-size: 0.75rem;
    font-weight: 300;
    color: var(--ink-muted);
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.result-card .score-badge {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 400;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--champagne);
    margin-top: 6px;
}

/* ── Caption box ─────────────────────────────────────────────────────── */
.caption-box {
    background: var(--surface);
    border-left: 3px solid var(--champagne);
    padding: 12px 16px;
    font-family: 'Cormorant Garamond', serif;
    font-style: italic;
    font-size: 1.05rem;
    font-weight: 300;
    color: var(--ink-muted);
    letter-spacing: 0.02em;
    margin: 10px 0;
}

/* ── Spinner ─────────────────────────────────────────────────────────── */
.stSpinner > div {
    border-top-color: var(--champagne) !important;
}
</style>
"""

RUSTIC_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=Crimson+Text:ital,wght@0,400;0,600;1,400&display=swap');

/* ── Light mode palette ──────────────────────────────────────────────── */
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

/* ── Dark mode palette ───────────────────────────────────────────────── */
@media (prefers-color-scheme: dark) {
    :root {
        --parchment:   #2A1F14;
        --parchment2:  #3A2A18;
        --ink:         #F0DEB8;
        --ink-light:   #D4B483;
        --rust:        #E8935A;
        --gold:        #D4A55A;
        --gold-light:  #F0C878;
        --sepia:       #C4976A;
        --cream:       #1E1610;
    }
}

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

st.markdown(FASHION_CSS, unsafe_allow_html=True)

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
    st.markdown("# Vestique")
    st.markdown(
        "*Search by image. Find the garment.*"
    )
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown(
        "01 &nbsp; Upload a photograph\n\n"
        "02 &nbsp; Confirm the detected garment\n\n"
        "03 &nbsp; We search the catalogue\n\n"
        "04 &nbsp; Browse your matches"
    )
    st.markdown("---")
    if st.button("↺  Start over"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown(
    "<h1 style='text-align:center;font-size:2.8rem;font-weight:300;letter-spacing:0.22em;'>"
    "VESTIQUE</h1>"
    "<p style='text-align:center;font-family:DM Sans,sans-serif;font-size:0.78rem;"
    "letter-spacing:0.22em;text-transform:uppercase;color:var(--ink-muted);margin-top:-12px;'>"
    "Visual Fashion Search</p>",
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

    with col_res:
        st.markdown("### Top 10 matches")
        results = st.session_state.results or []

        if not results:
            st.warning("No matches found in the catalogue.")
        else:
            for row_start in range(0, min(10, len(results)), 5):
                row_results = results[row_start : row_start + 5]
                cols = st.columns(5)
                for col, item in zip(cols, row_results):
                    with col:
                        st.markdown(
                            f"""
                            <div class="result-card">
                                <div class="card-body">
                                    <div class="item-title">{item['id']}</div>
                                    <div class="item-meta" style="font-style:italic;text-transform:none;margin-top:4px;">
                                        {item.get('caption', '')}
                                    </div>
                                    <div class="score-badge">
                                        cosine {item['score']}
                                        · itm {item.get('itm_score', 0):.3f}
                                    </div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    st.markdown("<hr/>", unsafe_allow_html=True)
    if st.button("← Search another garment"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()