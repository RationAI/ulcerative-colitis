# Weakly supervised Nancy Histological Index scoring

This repository contains a weakly supervised computational pathology pipeline that grades
histological inflammatory activity in colonic biopsies from patients with ulcerative colitis (UC)
and primary sclerosing cholangitis-associated IBD (PSC-IBD) on the full five-grade
**Nancy Histological Index (NHI 0–4)**. It learns from slide- or biopsy-level NHI labels only,
without any region- or cell-level annotation, and outputs for each H&E whole-slide image (WSI):

- a five-grade NHI prediction, and
- a continuous **confidence score** (0–1) that flags predictions likely to need human review.

> [!IMPORTANT]
> This is a research tool. It is not a medical device and must not be used for clinical
> decision-making. It was designed to grade inflammatory activity in colonic biopsies with an
> established diagnosis of UC or PSC-IBD; it does not diagnose IBD and does not detect dysplasia.

## Method overview

```
WSI (H&E, CZI)
  │  tissue detection (Otsu on HSV saturation)
  │  quality control (blur: PIQE-based, artifacts: colour-deconvolution residual; tile excluded if >25 % affected)
  │  tiling: 224 × 224 px at ~1.55 µm/px (pyramid level 2), 50 % overlap
  ▼
Virchow2 tile embeddings (frozen, 2,560-d)
  ▼
Three attention-based MIL branches (instance-first: tile-level classifier + attention pooling)
  ├─ Neutrophils   : NHI 0–1 vs NHI 2–4
  ├─ Nancy-low     : NHI 0 / NHI 1 / active
  └─ Nancy-high    : inactive / NHI 2 / NHI 3 / NHI 4
  ▼
Post-processing
  ├─ Ensembling: mean P(active) of the three branches ≥ 0.5 → Nancy-high branch, else Nancy-low;
  │              final grade = most probable grade within the selected branch
  └─ Confidence: branches combined as an 8-state absorbing Markov chain → unified P(NHI 0–4);
                 confidence = max(0, 1 − σ), σ = standard deviation of that distribution
```

The confidence score is ordinal-aware: probability shared between adjacent grades lowers it less
than probability spread across distant grades.

## Repository structure

| Path | Contents |
|------|----------|
| `preprocessing/` | Dataset creation, tissue masks, quality control, biopsy-level stratified splitting, tiling, Virchow2 embeddings |
| `ml/` | MIL model (`ml/mil.py`), bag datasets and balanced sampler, training/testing/prediction entry point (`python -m ml`) |
| `postprocessing/` | Ensembling (`ensembling.py`) and Markov-chain confidence (`markov_chain_confidence.py`); `*_predict.py` variants for unlabelled cohorts |
| `configs/` | [Hydra](https://hydra.cc) configuration; the final configurations are under `configs/experiment/**/final*` and `configs/checkpoints/final/`, the unlabelled IKEM validation cohort under `configs/experiment/**/*ikem_validation*` |
| `scripts/` | Job-submission wrappers for the RationAI Kubernetes cluster (internal) |

## Installation

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/RationAI/ulcerative-colitis.git
cd ulcerative-colitis
uv sync --frozen
```

Training and inference of the MIL branches need a CUDA GPU; embeddings are precomputed, so a small GPU
(~10 GB) is sufficient.

## Running the pipeline

Every step is a Hydra application that logs its outputs (masks, tiles, embeddings, checkpoints,
predictions, metrics) to an [MLflow](https://mlflow.org) tracking server; later steps read the
artifacts of earlier steps through the `mlflow_uris` entries in `configs/dataset/**`.
Institution-specific settings (data folder, label files, file-name pattern) are in
`configs/dataset/raw/{ikem,ftn,knl_patos}.yaml`.

Run the steps in this order, once per institution where applicable (`<inst>` = `ikem`, `ftn`, or `knl_patos`):

```bash
# 1. Preprocessing
uv run -m preprocessing.create_dataset   +dataset=raw/<inst>
uv run -m preprocessing.tissue_masks     +dataset=processed/<inst>
uv run -m preprocessing.quality_control  +dataset=processed/<inst>
uv run -m preprocessing.split_dataset    +experiment=preprocessing/split_dataset/<inst>
uv run -m preprocessing.tiling           +dataset=processed_w_masks/<inst>
uv run -m preprocessing.embeddings       +dataset=tiled/<inst>

# 2. Training and testing of the three branches (<task> = neutrophils, nancy_low, nancy_high)
uv run -m ml +experiment=ml/final/<task>/train
uv run -m ml +experiment=ml/final/<task>/test

# 3. Post-processing (per institution)
uv run -m postprocessing.ensembling              +experiment=postprocessing/final_ensembling/<inst>
uv run -m postprocessing.markov_chain_confidence +experiment=postprocessing/final_markov_chain_confidence/<inst>

# 4. Prediction on the unlabelled IKEM validation cohort (no metrics, predictions only)
uv run -m ml +experiment=ml/ikem_validation/<task>
uv run -m postprocessing.ensembling_predict              +experiment=postprocessing/predict_ensembling/ikem_validation
uv run -m postprocessing.markov_chain_confidence_predict +experiment=postprocessing/predict_markov_chain_confidence/ikem_validation
```

After each step, update the corresponding `mlflow_uris` in `configs/dataset/**`,
`configs/checkpoints/final/**`, and `configs/predictions/{final,ikem_validation}.yaml` to point to the new run.

### Final configuration

| Setting | Value |
|---------|-------|
| Split | biopsy-level (`case_id`), stratified by NHI; IKEM and FTN: train/tuning/test target 70/15/15; KNL: 0/50/50 |
| Training data | IKEM + FTN training sets; tuning set (all three centers) used only for early stopping |
| Optimizer | Adam, learning rate 1 × 10⁻⁵, batch size 4 bags, no dropout or weight decay |
| Class balance | slide-level weighted over-/under-sampling |
| Stopping | max 300 epochs, early stopping on tuning-set Cohen's κ (patience 20), best-κ checkpoint kept |
| Losses | binary cross-entropy (Neutrophils), cross-entropy (Nancy-low, Nancy-high) |
| Tile filters | blur score ≤ 0.25 and artifact score ≤ 0.25 (tiles of slides where QC failed have no score and are kept) |

The random seed is drawn per run and stored in the run's logged configuration (`configs/config-resolved.yaml`).

## Using the code outside RationAI infrastructure

The code is published to document the method, and it runs end to end on RationAI infrastructure.
To run it elsewhere, note that:

- **Data are not included.** The WSIs and labels cannot be shared publicly. All `mlflow-artifacts:/…` URIs and `/mnt/…` paths in `configs/` refer
  to internal storage and must be replaced with your own.
- **Quality control and embeddings are computed by remote services.** `preprocessing/quality_control.py`
  and `preprocessing/embeddings.py` call RationAI's QC and model-serving services through
  [`rationai-sdk`](https://github.com/RationAI/rationai-sdk-python). Outside RationAI, replace these calls
  with local implementations; Virchow2 weights are available from its developers
  ([paige-ai/Virchow2](https://huggingface.co/paige-ai/Virchow2)) under their own license.
- **Some dependencies are hosted on the RationAI GitLab** (`rationai-masks`, `rationai-tiling`), and the
  optional `job` dependency group and `scripts/` target the internal Kubernetes cluster.

## License

The code is released under the [MIT License](LICENSE). Virchow2 is subject to its own license
terms, which apply to any use of the model weights.

## Acknowledgements

Developed by the [RationAI](https://github.com/RationAI) group at the Faculty of Informatics, Masaryk University,
in collaboration with the Institute for Clinical and Experimental Medicine (IKEM), Thomayer University
Hospital (FTN), and Regional Hospital Liberec (KNL). Computational resources were provided by the
e-INFRA CZ project (ID: 90140), supported by the Ministry of Education, Youth and Sports of the Czech Republic.