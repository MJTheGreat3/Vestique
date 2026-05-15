# Vestique — Multimodal Fashion Retrieval System

A multimodal visual product search engine for fashion e-commerce that enables **query-by-image retrieval** using semantic understanding, multimodal embeddings, ANN vector search, and semantic reranking.

---

## Overview

Vestique is an end-to-end fashion retrieval pipeline designed to retrieve visually and semantically similar clothing products from a large gallery database.

The system combines:

- YOLO-based garment localization
- BLIP2 semantic caption generation
- CLIP multimodal embedding fusion
- FAISS/Pinecone vector retrieval
- BLIP2 semantic reranking
- Streamlit-based interactive deployment

The project was evaluated on the **DeepFashion In-Shop Clothes Retrieval Benchmark** using:

- Recall@K
- NDCG@K
- mAP@K

The best-performing configuration used:

- Fine-tuned CLIP ViT-B/32
- BLIP2 semantic caption generation
- Multimodal fusion with α = 0.70

---

# System Architecture

## Offline Indexing Pipeline

For each gallery image, the system performs:

1. **Garment Localization (YOLO11s)**  
   Detects and crops the primary clothing region.

2. **Semantic Caption Generation (BLIP2)**  
   Generates fashion-aware captions describing:
   - clothing type
   - texture
   - color
   - style

3. **Multimodal Embedding Generation (CLIP)**  
   Generates:
   - image embeddings
   - text embeddings

4. **Embedding Fusion**

```math
v_i = \alpha \phi_V(\hat{x}_i) + (1-\alpha)\phi_T(c_i)
```

5. **Vector Indexing**
   - FAISS for offline evaluation
   - Pinecone for deployed ANN retrieval

6. **Cloudinary Upload**
   Stores gallery images for scalable retrieval visualization.

---

## Online Query Pipeline

For a user-uploaded image:

1. YOLO-based clothing localization
2. CLIP query embedding generation
3. ANN retrieval using Pinecone
4. BLIP2 semantic reranking
5. Retrieval visualization in Streamlit

---

# Features

- Query-by-image fashion retrieval
- Multimodal image-text embeddings
- BLIP2 semantic caption fusion
- YOLO-based garment localization
- CLIP fine-tuning using supervised contrastive learning
- FAISS cosine-similarity retrieval
- Pinecone vector database integration
- Cloudinary image hosting
- Interactive Streamlit UI
- Semantic reranking using BLIP2

---

# Dataset

The project uses the **DeepFashion In-Shop Clothes Retrieval Benchmark**.

| Property | Value |
|---|---|
| Total Images | 52,712 |
| Item IDs | 7,982 |
| Training Images | 25,882 |
| Query Images | 14,218 |
| Gallery Images | 12,612 |

The dataset introduces realistic retrieval challenges including:

- pose variation
- scale variation
- illumination changes
- background clutter
- viewpoint changes
- visually similar clothing categories

---

# Tech Stack

## Core Frameworks

- Python
- PyTorch
- HuggingFace Transformers
- OpenAI CLIP
- Ultralytics YOLO
- OpenCV

## Retrieval Infrastructure

- FAISS
- Pinecone
- Cloudinary

## Deployment

- Streamlit

---

# Fine-Tuning Strategy

The CLIP visual encoder was fine-tuned using **Supervised Contrastive Learning**.

### Training Details

| Parameter | Value |
|---|---|
| CLIP Variant | ViT-B/32 |
| Optimizer | AdamW |
| Learning Rate | 1e-5 |
| Weight Decay | 1e-4 |
| Scheduler | CosineAnnealingLR |
| Epochs | 15 |
| Loss Function | SupConLoss |
| Temperature | 0.07 |

Only the final four transformer blocks of the CLIP visual encoder were unfrozen during training.

---

# Evaluation Metrics

The retrieval system was evaluated using:

- Recall@K
- NDCG@K
- mAP@K

for:

```math
K \in \{5, 10, 15\}
```


# Streamlit Demo Application

The deployed application supports:

- image upload
- automatic garment detection
- crop visualization
- ANN retrieval
- semantic reranking
- interactive retrieval display

Each retrieved product displays:

- product image
- similarity score
- semantic caption

---

# Repository Structure

```bash
Vestique/
│
├── app/                        # Streamlit application
├── models/                     # YOLO, CLIP, BLIP utilities
├── retrieval/                  # FAISS & Pinecone retrieval logic
├── training/                   # Fine-tuning scripts
├── preprocessing/              # Dataset preprocessing
├── evaluation/                 # Metric computation
├── indexing/                   # Embedding generation & indexing
├── utils/                      # Helper utilities
├── assets/                     # Demo assets/images
├── requirements.txt
└── README.md
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/MJTheGreat3/Vestique.git
cd Vestique
```

## Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Application

## Start Streamlit App

```bash
streamlit run app.py
```

---

# Retrieval Workflow

```text
Input Image
     ↓
YOLO Localization
     ↓
BLIP2 Caption Generation
     ↓
CLIP Embedding Generation
     ↓
Multimodal Fusion
     ↓
Pinecone / FAISS Retrieval
     ↓
BLIP2 Semantic Reranking
     ↓
Retrieved Fashion Products
```

---

# Limitations

- Retrieval quality depends on BLIP2 caption quality.
- BLIP2 reranking introduces computational overhead.
- Incorrect YOLO localization can reduce embedding quality.
- Evaluated only on the DeepFashion benchmark.

---

# Future Improvements

- Real-world fashion dataset evaluation
- Faster reranking pipelines
- Cross-attention fusion architectures
- Mobile deployment support
- Better garment segmentation
- Multi-garment retrieval support

---

# Authors

- Avancha Deedepya
- Mathew Joseph
- Ayush Tiwari
- MS Dheeraj Murthy

---

# References

- OpenAI CLIP
- BLIP / BLIP2
- DeepFashion Dataset
- FAISS
- Pinecone
- Ultralytics YOLO
