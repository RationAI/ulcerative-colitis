"""Which half of h_i = [z_i; m_i] (CLS vs mean-pooled patch tokens) actually drives each head?

Motivating question (asked before committing further to patch-level concept
explainability at all): if a head's predictions are almost entirely driven
by z_i (the CLS token), there is little point explaining m_i's patch
concepts for that head - concept_mil.tex's whole machinery (NMF dictionary,
sigma tables, R^2 checks) only ever touches the m_i pathway. This script
answers "how much, per head and per class" via three methods of increasing
rigor, sharing one pass over the real corpus:

1. **IQR-weighted weight contribution** (weights only, no attention, no
   redundancy accounting): `|Theta_j| * IQR_j` per dimension, summed per
   pathway. Correction vs. `theta_m_check.py`'s use of IQR: `Theta_m` acts on
   `m_i` (mean-pooled, tile-count-sized), not on raw patch tokens
   (patch_statistics.py's IQR, patch-count-sized) - averaging shrinks
   variance, so this script computes its own IQR of `z_i`/`m_i` directly
   from the same tile-level features the other two methods use, rather than
   reusing `patch_statistics.py`'s output. Cheap either way (tile-count, not
   patch-count, sized).
2. **Variance decomposition** of the tile logit `Theta_z.z_i + Theta_m.m_i`
   into its two terms plus their covariance, on real tiles - the first
   data-grounded (not weights-only) number, but still at the observed,
   fixed attention (concept_mil.tex's "operating point, not counterfactual"
   remark applies here exactly as it does to concept ablation). Also broken
   out per (model-predicted, per-tile - never per-slide, same reasoning as
   `grade_split.py`) grade, to catch a pathway that is unimportant on
   average but decisive within one grade.
3. **Real counterfactual ablation**: reconstructs the actual attention +
   classifier forward pass directly from each checkpoint's `attention.0`/
   `attention.2`/`classifier` weights (confirmed to all live in the same
   `state_dict` - no dependency on any external model class), then
   mean-imputes z or m at the attention step, the classifier step, or both,
   and re-runs real slide-level inference against real ground-truth labels.
   This is a special case of concept_mil.tex's already-planned faithfulness
   ablation (Sec. "Faithfulness", eq:ablate) with the entire m-pathway (or
   z-pathway) removed instead of one concept - it needs no concept
   dictionary at all, so it can run before any NMF/K decision.

**Ground-truth labels**: `slides.parquet` already carries the real,
tiling-time-joined `nancy_index` (0-4) per slide (see
`preprocessing/tiling.py`'s `add_nancy_index` + `embeddings_xai.py`'s
`slides.groupby("nancy_index")` subsampling) - not a model prediction. Per-
head targets are derived from it using `explainability.postprocessing.
route_grade`'s documented class order:
    nancy_low  (3-class): {0->0, 1->1, >=2->2}
    nancy_high (4-class): {<2->0, 2->1, 3->2, 4->3}
**Unverified assumption**: `neutrophils` (binary) has no separate ground-
truth column anywhere in this repo - its positive class is inferred as
`nancy_index >= 2` (neutrophils first appear at Nancy grade 2 per the
clinical criteria, and `route_grade`'s own high/low routing logic treats
grade>=2 as one branch) rather than read off an independent label. Revisit
this first if the neutrophils AUC numbers look implausible.

**Same full-corpus mean-pooling as grade_split.py, independently** (not
reused from it): grade_split.py only persists small per-tile grade *labels*
downstream, not the pooled `m_i`/`z_i` features themselves, so there is no
artifact to share - this script re-streams the full patch corpus the same
way (`ray.init(num_cpus=8, ...)`, same oversized-parquet-row-group reasoning
documented in explainability-status memory).

**Session discipline**: written and validated only against small synthetic
data plus `ruff`/`mypy strict` - never run against the real corpus in this
session (see explainability-status memory's standing constraint).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import mlflow.artifacts
import numpy as np
import pandas as pd
import ray
import ray.data
import torch
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from scipy.special import expit, softmax
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from explainability.grade_split import (
    build_tile_features,
    compute_tile_grades,
    mean_pool_patches,
)
from explainability.tiles import load_slides, load_tokens_dataset, resolve_token_dirs


@dataclass
class ModelWeights:
    """One head's full MIL model, pulled straight from its checkpoint's `state_dict`.

    `attn_w1`/`attn_b1` is `attention.0` (`U`), `attn_w2`/`attn_b2` is
    `attention.2` (`q`) - confirmed present in the same checkpoint as
    `classifier.weight`/`.bias` by directly inspecting a downloaded
    checkpoint's `state_dict` keys, so no external model class is needed to
    reconstruct `u_i = q^T tanh(U h_i)` (concept_mil.tex eq. 2.1).
    """

    attn_w1: np.ndarray  # (hidden, 2*embed_dim)
    attn_b1: np.ndarray  # (hidden,)
    attn_w2: np.ndarray  # (1, hidden)
    attn_b2: np.ndarray  # (1,)
    cls_w: np.ndarray  # (num_classes, 2*embed_dim)
    cls_b: np.ndarray  # (num_classes,)


def load_full_model(checkpoint_uri: str, embed_dim: int) -> ModelWeights:
    """Download a MIL checkpoint and pull out both the attention module and the classifier."""
    checkpoint_path = mlflow.artifacts.download_artifacts(checkpoint_uri)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["state_dict"]
    weight = state_dict["classifier.weight"].numpy()
    if weight.shape[1] != 2 * embed_dim:
        raise ValueError(
            f"classifier.weight has {weight.shape[1]} input columns, expected {2 * embed_dim}"
        )
    return ModelWeights(
        attn_w1=state_dict["attention.0.weight"].numpy(),
        attn_b1=state_dict["attention.0.bias"].numpy(),
        attn_w2=state_dict["attention.2.weight"].numpy(),
        attn_b2=state_dict["attention.2.bias"].numpy(),
        cls_w=weight,
        cls_b=state_dict["classifier.bias"].numpy(),
    )


def nancy_to_target(nancy_index: np.ndarray, head: str) -> np.ndarray:
    """Map real ground-truth `nancy_index` (0-4) to one head's class index, per `route_grade`."""
    if head == "neutrophils":
        return (nancy_index >= 2).astype(np.int64)
    if head == "nancy_low":
        return np.minimum(nancy_index, 2)
    if head == "nancy_high":
        return np.where(nancy_index < 2, 0, nancy_index - 1)
    raise ValueError(f"Unknown head: {head}")


def dimension_iqr(x: np.ndarray) -> np.ndarray:
    """Per-column IQR (q75-q25), shape `(x.shape[1],)`."""
    q75, q25 = np.percentile(x, [75, 25], axis=0)
    return q75 - q25


def weight_contribution_table(
    models: dict[str, ModelWeights], z_iqr: np.ndarray, m_iqr: np.ndarray, embed_dim: int
) -> pd.DataFrame:
    """Method 1: `sum_j |Theta_j| * IQR_j` per pathway, per head/class - weights + IQR only."""
    rows = []
    for head, model in models.items():
        theta_z = model.cls_w[:, :embed_dim]
        theta_m = model.cls_w[:, embed_dim:]
        contrib_z = np.abs(theta_z) @ z_iqr
        contrib_m = np.abs(theta_m) @ m_iqr
        for c in range(model.cls_w.shape[0]):
            rows.append(
                {
                    "head": head,
                    "class": c,
                    "contrib_z": float(contrib_z[c]),
                    "contrib_m": float(contrib_m[c]),
                    "share_m": float(contrib_m[c] / (contrib_z[c] + contrib_m[c])),
                }
            )
    return pd.DataFrame(rows)


def variance_decomposition_table(
    z: np.ndarray, m: np.ndarray, models: dict[str, ModelWeights], embed_dim: int, grade: np.ndarray | None
) -> pd.DataFrame:
    """Method 2: `Var(Theta_z.z)`, `Var(Theta_m.m)`, `Cov` per head/class, pooled and per grade.

    `share_m_adj` gives half the covariance to each pathway (a fair,
    Shapley-style split of the shared/redundant variance) rather than
    assigning it entirely to one side.
    """
    rows = []
    grade_values: list[int | None] = [None] + (
        sorted(np.unique(grade).tolist()) if grade is not None else []
    )
    for head, model in models.items():
        theta_z = model.cls_w[:, :embed_dim]
        theta_m = model.cls_w[:, embed_dim:]
        z_term = z @ theta_z.T  # (n, C)
        m_term = m @ theta_m.T  # (n, C)
        for grade_value in grade_values:
            if grade_value is None:
                mask: np.ndarray = np.ones(z.shape[0], dtype=bool)
            else:
                assert grade is not None  # grade_values only holds non-None entries when grade is
                mask = np.asarray(grade == grade_value)
            if mask.sum() < 2:
                continue
            for c in range(model.cls_w.shape[0]):
                zt, mt = z_term[mask, c], m_term[mask, c]
                var_z, var_m = float(np.var(zt)), float(np.var(mt))
                cov_zm = float(np.cov(zt, mt)[0, 1])
                var_total = var_z + var_m + 2 * cov_zm
                corr_zm = cov_zm / np.sqrt(var_z * var_m) if var_z > 0 and var_m > 0 else float("nan")
                share_m_adj = (var_m + cov_zm) / var_total if var_total > 0 else float("nan")
                rows.append(
                    {
                        "head": head,
                        "class": c,
                        "grade": grade_value,
                        "n_tiles": int(mask.sum()),
                        "var_z": var_z,
                        "var_m": var_m,
                        "cov_zm": cov_zm,
                        "corr_zm": corr_zm,
                        "share_m_adj": share_m_adj,
                    }
                )
    return pd.DataFrame(rows)


# (ablate_z_attn, ablate_z_cls, ablate_m_attn, ablate_m_cls)
CONDITIONS: dict[str, tuple[bool, bool, bool, bool]] = {
    "baseline": (False, False, False, False),
    "z_attn": (True, False, False, False),
    "z_cls": (False, True, False, False),
    "z_both": (True, True, False, False),
    "m_attn": (False, False, True, False),
    "m_cls": (False, False, False, True),
    "m_both": (False, False, True, True),
    "null_both": (True, True, True, True),
}


def forward_slide(
    z: np.ndarray,
    m: np.ndarray,
    model: ModelWeights,
    ablate_z_attn: bool,
    ablate_z_cls: bool,
    ablate_m_attn: bool,
    ablate_m_cls: bool,
    z_bar: np.ndarray,
    m_bar: np.ndarray,
) -> np.ndarray:
    """Real attention-pooled slide logit under one (pathway, injection-point) ablation.

    Mean-imputing at "attn" vs "cls" independently is only meaningful
    because `u_i` is *nonlinear* in `h_i` (tanh) - unlike the linear
    classifier step, `u_i` genuinely changes when one pathway is replaced by
    its population mean, so attention can redistribute (concept_mil.tex's
    own distinction between exact tile-dropping, eq:drop, and the nonlinear
    concept-removal counterfactual, eq:ablate - this is the latter, applied
    to a whole pathway instead of one concept).
    """
    z_attn_in = np.broadcast_to(z_bar, z.shape) if ablate_z_attn else z
    m_attn_in = np.broadcast_to(m_bar, m.shape) if ablate_m_attn else m
    h_attn = np.concatenate([z_attn_in, m_attn_in], axis=1)
    u = (np.tanh(h_attn @ model.attn_w1.T + model.attn_b1) @ model.attn_w2.T + model.attn_b2)[:, 0]
    a = softmax(u)

    z_cls_in = np.broadcast_to(z_bar, z.shape) if ablate_z_cls else z
    m_cls_in = np.broadcast_to(m_bar, m.shape) if ablate_m_cls else m
    h_cls = np.concatenate([z_cls_in, m_cls_in], axis=1)
    s_i = h_cls @ model.cls_w.T + model.cls_b  # (n, num_classes)
    return a @ s_i  # (num_classes,)


def logits_to_prob(logits: np.ndarray, num_classes: int) -> np.ndarray:
    return expit(logits) if num_classes == 1 else softmax(logits)


def run_ablation(
    tile_features: pd.DataFrame,
    z: np.ndarray,
    m: np.ndarray,
    models: dict[str, ModelWeights],
    nancy_index_by_slide: pd.Series,
    z_bar: np.ndarray,
    m_bar: np.ndarray,
) -> pd.DataFrame:
    """Method 3: real forward pass per slide, per head, per ablation condition.

    Loops over slides in plain Python (a few hundred to a couple thousand,
    each with a few hundred tiles - `pd.factorize` groups them, matching
    `tile_r2_check.py`'s own in-driver-loop convention over
    `ray.data.groupby`, since the per-slide data is small by this point).
    """
    slide_codes, slide_ids = pd.factorize(tile_features["slide_id"])
    # Targets must be derived from this same slide_ids order (factorize's
    # first-appearance order), not a separately-sorted one - slide_id is an
    # opaque row_hash, so a `groupby(...).first()`'s default sort=True order
    # silently disagrees with it, scrambling predictions against the wrong
    # slide's label and flattening AUC to chance.
    targets = {
        head: nancy_to_target(nancy_index_by_slide.loc[slide_ids].to_numpy(), head) for head in models
    }
    predictions: dict[tuple[str, str], list[np.ndarray]] = {}
    for head in models:
        for condition in CONDITIONS:
            predictions[(head, condition)] = [None] * len(slide_ids)  # type: ignore[list-item]

    for slide_idx in range(len(slide_ids)):
        rows = slide_codes == slide_idx
        z_s, m_s = z[rows], m[rows]
        for head, model in models.items():
            for condition, flags in CONDITIONS.items():
                logits = forward_slide(z_s, m_s, model, *flags, z_bar, m_bar)
                predictions[(head, condition)][slide_idx] = logits

    result_rows = []
    for head, model in models.items():
        num_classes = model.cls_w.shape[0]
        target = targets[head]
        baseline_probs = np.stack(
            [logits_to_prob(logits, num_classes) for logits in predictions[(head, "baseline")]]
        )
        for condition in CONDITIONS:
            probs = np.stack(
                [logits_to_prob(logits, num_classes) for logits in predictions[(head, condition)]]
            )
            for c in range(num_classes if num_classes > 1 else 1):
                y_true = target if num_classes == 1 else (target == c).astype(int)
                y_prob = probs[:, 0] if num_classes == 1 else probs[:, c]
                y_prob_baseline = baseline_probs[:, 0] if num_classes == 1 else baseline_probs[:, c]
                try:
                    auc = float(roc_auc_score(y_true, y_prob))
                except ValueError:
                    auc = float("nan")
                try:
                    fidelity_corr = float(pearsonr(y_prob, y_prob_baseline)[0])
                except ValueError:
                    fidelity_corr = float("nan")
                result_rows.append(
                    {
                        "head": head,
                        "class": c,
                        "condition": condition,
                        "n_slides": len(slide_ids),
                        "auc": auc,
                        "fidelity_corr": fidelity_corr,
                    }
                )
    ablation_df = pd.DataFrame(result_rows)
    baseline_auc = ablation_df[ablation_df["condition"] == "baseline"].set_index(["head", "class"])["auc"]
    ablation_df["delta_auc"] = ablation_df.apply(
        lambda row: row["auc"] - baseline_auc.loc[(row["head"], row["class"])], axis=1
    )
    return ablation_df


def summarize_ablation(ablation_df: pd.DataFrame) -> pd.DataFrame:
    """One row per head/class: which pathway dominates, and how much of the achievable range it covers."""
    rows = []
    for (head, c), group in ablation_df.groupby(["head", "class"]):
        by_condition = group.set_index("condition")["delta_auc"]
        delta_z, delta_m, delta_null = by_condition["z_both"], by_condition["m_both"], by_condition["null_both"]
        dominant = "m" if delta_m <= delta_z else "z"
        pct_from_m = delta_m / delta_null if delta_null != 0 else float("nan")
        rows.append(
            {
                "head": head,
                "class": c,
                "delta_auc_z_both": delta_z,
                "delta_auc_m_both": delta_m,
                "delta_auc_null_both": delta_null,
                "dominant_pathway": dominant,
                "pct_of_range_from_m": pct_from_m,
            }
        )
    return pd.DataFrame(rows)


@with_cli_args(["+explainability=embedding_importance"])
@hydra.main(config_path="../configs", config_name="explainability", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patch_dirs = resolve_token_dirs(
        config.sources, config.get("local_embeddings_xai_dir"), kind="patch", split=config.split
    )
    cls_dirs = resolve_token_dirs(
        config.sources, config.get("local_embeddings_xai_dir"), kind="cls", split=config.split
    )

    print("Mean-pooling patch tokens by tile (full corpus, one streaming pass)...", flush=True)
    patch_means = mean_pool_patches(load_tokens_dataset(patch_dirs), config.embed_dim)
    cls_ds = load_tokens_dataset(cls_dirs)
    tile_features = build_tile_features(patch_means, cls_ds)
    print(f"Pooled {len(tile_features)} tiles.", flush=True)

    h = np.stack(tile_features["h"].to_numpy())
    embed_dim = config.embed_dim
    z, m = h[:, :embed_dim], h[:, embed_dim:]

    # slides.parquet's slide-identifier column is "id", not "slide_id" (same
    # mismatch explainability/concept_masks.py already works around via
    # .set_index("id")) - tiling.py's per-tile rows are the only place this
    # value is actually called "slide_id".
    slides = load_slides(config.sources, config.get("local_embeddings_xai_dir"), split=config.split)
    tile_features = tile_features.merge(
        slides[["id", "nancy_index"]], left_on="slide_id", right_on="id", how="inner"
    ).drop(columns="id")
    if len(tile_features) != len(h):
        raise ValueError(
            f"{len(h) - len(tile_features)} tiles had no matching slide-level nancy_index "
            "- check slides.parquet coverage before trusting any result below."
        )

    models = {
        head: load_full_model(config.checkpoints[head].checkpoint, embed_dim)
        for head in ("neutrophils", "nancy_low", "nancy_high")
    }

    # Method 1
    print("Method 1: IQR-weighted weight contribution...", flush=True)
    z_iqr, m_iqr = dimension_iqr(z), dimension_iqr(m)
    weight_contribution_df = weight_contribution_table(models, z_iqr, m_iqr, embed_dim)
    print(weight_contribution_df.to_string(index=False), flush=True)

    # Method 2 (also broken out per model-predicted per-tile grade - same
    # tile-level route_grade call grade_split.py uses, computed fresh here
    # since this script doesn't depend on a prior grade_split.py run)
    print("Method 2: variance decomposition...", flush=True)
    tile_grade = compute_tile_grades(
        tile_features,
        (models["neutrophils"].cls_w, models["neutrophils"].cls_b),
        (models["nancy_low"].cls_w, models["nancy_low"].cls_b),
        (models["nancy_high"].cls_w, models["nancy_high"].cls_b),
    )
    variance_decomposition_df = variance_decomposition_table(
        z, m, models, embed_dim, grade=tile_grade["grade"].to_numpy()
    )
    print(variance_decomposition_df[variance_decomposition_df["grade"].isna()].to_string(index=False), flush=True)

    # Method 3
    print("Method 3: real counterfactual ablation...", flush=True)
    nancy_index_by_slide = tile_features.drop_duplicates("slide_id").set_index("slide_id")["nancy_index"]
    z_bar, m_bar = z.mean(axis=0), m.mean(axis=0)
    ablation_df = run_ablation(tile_features, z, m, models, nancy_index_by_slide, z_bar, m_bar)
    summary_df = summarize_ablation(ablation_df)
    print(summary_df.to_string(index=False), flush=True)

    weight_contribution_path = output_dir / "weight_contribution.parquet"
    variance_decomposition_path = output_dir / "variance_decomposition.parquet"
    ablation_path = output_dir / "ablation.parquet"
    summary_path = output_dir / "ablation_summary.parquet"
    weight_contribution_df.to_parquet(weight_contribution_path, index=False)
    variance_decomposition_df.to_parquet(variance_decomposition_path, index=False)
    ablation_df.to_parquet(ablation_path, index=False)
    summary_df.to_parquet(summary_path, index=False)

    manifest: dict[str, Any] = {
        "split": config.split,
        "n_tiles": len(tile_features),
        "n_slides": tile_features["slide_id"].nunique(),
        "ablation_summary": summary_df.to_dict(orient="records"),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    logger.log_artifact(str(weight_contribution_path))
    logger.log_artifact(str(variance_decomposition_path))
    logger.log_artifact(str(ablation_path))
    logger.log_artifact(str(summary_path))
    logger.log_artifact(str(manifest_path))
    for row in summary_df.to_dict(orient="records"):
        tag = f"{row['head']}_class{row['class']}"
        logger.log_metrics(
            {
                f"delta_auc_m_both/{tag}": row["delta_auc_m_both"],
                f"delta_auc_z_both/{tag}": row["delta_auc_z_both"],
                f"pct_of_range_from_m/{tag}": row["pct_of_range_from_m"],
            }
        )


if __name__ == "__main__":
    ctx = ray.data.DataContext.get_current()
    ctx.enable_rich_progress_bars = True
    ctx.use_ray_tqdm = False

    # num_cpus=8: same oversized-parquet-row-group root cause as
    # patch_statistics.py/nmf_fit.py/grade_split.py - mean_pool_patches reads
    # the full patch corpus. Keep in sync with cpu= in
    # scripts/explainability/embedding_importance.py.
    with ray.init(num_cpus=8, runtime_env={"excludes": [".git", ".venv"]}):
        main()
