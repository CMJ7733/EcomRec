# EcomRec — E-commerce Recommendation System

A full-stack personalized recommendation system based on Amazon Reviews 2023 (Beauty), implementing the industry-standard "Recall + Ranking" two-tower architecture.

## Quick Start

```bash
make install   # Install dependencies
make data      # Download & preprocess Amazon Beauty dataset
make train     # Train all recall & ranking models
make app       # Launch Streamlit dashboard at localhost:8501
```

## Architecture

Multi-route Recall (Top-Pop / ItemCF / BPR-MF / ALS) → DeepFM Ranking → MMR Reranking

## Key Results

See [README.md](README.md) for benchmark tables (populated after training).

## Tech Stack

Python 3.10+ | PyTorch 2.4 | LightGBM | Polars | Streamlit | MLflow | Hydra
