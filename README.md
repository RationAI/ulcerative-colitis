# Automated Grading of Ulcerative Colitis from H&E Whole Slide Images

This repository contains the machine learning pipeline developed for automated histological grading of ulcerative colitis (UC) biopsies using the Nancy Histological Index (NHI). The approach frames NHI grading as a hierarchical classification problem: a binary neutrophil-detection task routes each whole-slide image through either a low-grade (NHI 0–2) or a high-grade (NHI 2–4) classifier. All models are based on pretrained pathology foundation models and use attention-based multiple instance learning (MIL) over tile embeddings, making them applicable to gigapixel slides without requiring tile-level annotations. Experiments evaluate multiple foundation models (prov-gigapath, UNI, UNI2, Virchow, Virchow2), MIL against tile-level approaches, ordered regression, and ensembling/confidence strategies over a multi-institution cohort (IKEM, FTN, KNL).

> **Note: This code is not runnable as-is.** The underlying H&E whole-slide image data and derived embeddings are sensitive patient data from clinical institutions and are not included in this repository. The code is provided for reference and reproducibility of the methodology only.

---

## Repository Structure

```
.
├── configs/                    # Hydra configuration tree
│   ├── dataset/                # Dataset split URIs (embeddings, slides, tiles)
│   ├── checkpoints/            # Checkpoint paths for test/inference runs
│   ├── predictions/            # MLflow prediction artifact URIs (per institution/fold)
│   └── experiment/
│       ├── ml/                 # ML training & test experiment configs (Exp I–VI + final)
│       └── postprocessing/     # Postprocessing experiment configs (Exp VII–VIII + final)
├── ml/                         # PyTorch Lightning modules & entry point
│   ├── __main__.py             # Entry point: `python -m ml`
│   ├── base.py                 # BaseModule (tile-level training loop)
│   ├── classification.py       # Tile-level softmax classification
│   ├── ordered_regression.py   # Cumulative link ordered regression
│   ├── mil.py                  # Attention MIL (bag-level)
│   ├── neutrophils.py          # Binary neutrophil detection (tile-level)
│   └── modeling/               # Building blocks (heads, normalization)
├── preprocessing/              # Data preparation pipeline
│   ├── tiling.py               # Tile extraction from WSIs
│   ├── tissue_masks.py         # Tissue segmentation
│   ├── quality_control.py      # Tile quality filtering
│   ├── embeddings.py           # Foundation model feature extraction
│   ├── neutrophils.py          # Neutrophil pseudo-label extraction
│   ├── create_dataset.py       # Assemble HuggingFace dataset
│   └── split_dataset.py        # Train/val/test split assignment
├── postprocessing/             # Slide-level aggregation & confidence
│   ├── ensembling.py           # Soft majority vote + hierarchical ensembling
│   └── markov_chain_confidence.py  # Markov chain absorption confidence scores
├── scripts/                    # Kubernetes job submission scripts
│   ├── ml/                     # train.py, test.py, neutrophils.py
│   ├── preprocessing/          # Preprocessing job scripts
│   └── postprocessing/         # Postprocessing job scripts
├── pyproject.toml
└── thesis.pdf                  # Thesis manuscript
```

---

## Setup

Requires Python 3.12 and [`uv`](https://github.com/astral-sh/uv).

```bash
uv sync --frozen
```

---

## Pipeline Overview

The full pipeline runs in four stages:

### 1. Preprocessing

Tiles are extracted from WSIs, tissue-masked, quality-filtered, and encoded by a foundation model into fixed-dimensional embeddings. Datasets are assembled as HuggingFace Datasets and registered in MLflow.

```bash
# Tissue segmentation masks
uv run --active -m preprocessing.tissue_masks +dataset=processed/...

# Tile extraction from WSIs
uv run --active -m preprocessing.tiling +dataset=processed_w_masks/... +experiment=preprocessing/tiling/...

# Tile quality filtering
uv run -m preprocessing.quality_control +dataset=processed/...

# Foundation model feature extraction
uv run --active -m preprocessing.embeddings +dataset=tiled/...

# Assemble HuggingFace dataset
uv run -m preprocessing.create_dataset +dataset=raw/...

# Train/val/test split assignment
uv run -m preprocessing.split_dataset +experiment=preprocessing/split_dataset/...
```

### 2. ML Training

Models are trained via Hydra experiment configs. The entry point is `python -m ml` and an experiment config must be selected:

```bash
# Example: train Experiment IV (MIL) fold 1
uv run -m ml +experiment=ml/experiment_iv_mil_and_hierarchical_modeling/nancy_high/train/fold_1

# Example: train final model (all institutions combined)
uv run -m ml +experiment=ml/final/nancy_high/train
```

### 3. ML Testing / Inference

```bash
# Example: test Experiment IV fold 1
uv run -m ml +experiment=ml/experiment_iv_mil_and_hierarchical_modeling/nancy_high/test/fold_1

# Neutrophil detection (produces binary slide-level predictions)
uv run -m ml.neutrophils +experiment=ml/experiment_vi_neutrophil_detection/run
```

### 4. Postprocessing

Slide-level predictions from the three hierarchical tasks are combined into final NHI grades and confidence scores:

```bash
# Ensembling (soft majority vote + hierarchical routing)
uv run -m postprocessing.ensembling +postprocessing=ensembling

# Markov chain absorption confidence
uv run -m postprocessing.markov_chain_confidence +postprocessing=markov_chain_confidence
```

---

## Experiments

| # | Name | Config path | Description |
|---|------|-------------|-------------|
| I | Baseline Classification | `experiment/ml/experiment_i_baseline_classification` | VGG16 tile-level classifier trained on raw pixel tiles |
| II | Ordered Regression | `experiment/ml/experiment_ii_ordered_regression` | Cumulative link loss for NHI ordinal structure |
| III | Pathology Foundation Models | `experiment/ml/experiment_iii_pathology_foundation_model` | Linear probe on frozen prov-gigapath embeddings |
| IV | MIL & Hierarchical Modeling | `experiment/ml/experiment_iv_mil_and_hierarchical_modeling` | Attention MIL; three-task hierarchical decomposition |
| V | More Foundation Models | `experiment/ml/experiment_v_more_foundation_models` | MIL sweep over all foundation models |
| VI | Neutrophil Detection | `experiment/ml/experiment_vi_neutrophil_detection` | Binary tile-level neutrophil classifier |
| VII | Ensembling | `experiment/postprocessing/experiment_vii_ensembling` | Soft majority vote vs. hierarchical routing |
| VIII | Markov Chain Confidence | `experiment/postprocessing/experiment_viii_markov_chain_model_aggregation` | Absorption distribution confidence (entropy / herfindahl / std) |
| — | Final Model | `experiment/ml/final` | Attention MIL (virchow2) trained on all three institutions |

All cross-validation experiments run 5 folds over the IKEM institution. Final evaluation uses the held-out test split across IKEM, FTN, and KNL.
