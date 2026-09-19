"""Slide-level AUC of a fully concept-native (z_i-only) surrogate MIL model.

Extends `tile_r2_check_cls.py`'s tile-level `r2_ols_vs_z` finding (the shared
k=12 `kind=cls` dictionary's OLS fit is a good surrogate for the classifier
readout) up to real slide-level predictions - the check that was originally
proposed but scoped down to tile-level-only because only a grade-specific
dictionary existed (see explainability-status memory's 2026-09-19 entries):
MIL attention pools over a slide's *entire* tile bag, so a slide-level
prediction needs every tile in the bag to have a concept representation, and
a grade-specific dictionary only ever covered one grade's tiles. Now that a
single shared (grade-unsplit) dictionary is the adopted choice - validated on
two independent grades (0 and 4), both showing no benefit from
grade-splitting - every tile has a `varphi`, so this is possible.

**Two surrogate models, both OLS-fitted from `varphi` directly (no analytic
plug-in - `tile_r2_check_cls.py` already showed the plug-in formula
`varphi @ sigma_z` is far worse than a fitted regression on the same
`varphi`, so this only ever uses the fitted approach)**:
    s_hat_i  = OLS fit of varphi onto Theta_z . z_i        (classifier)
    u_hat_i  = OLS fit of varphi onto the real attention score u_i (attention)
Per slide: `a_hat = softmax(u_hat)` over the slide's tiles,
`surrogate_logits = a_hat @ s_hat` - the same attention-pooled-logit
aggregation the real MIL model uses (`embedding_importance.py`'s
`forward_slide`), but built entirely from `varphi` - no `m_i`, no raw `z_i`,
everything through the K=12 concept bottleneck.

**No m_i anywhere in the surrogate itself** (per user decision, 2026-09-19 -
"everything through the bottleneck"). But `u_i = q^T tanh(U [z_i; m_i])`
genuinely depends on `m_i` (nonlinear - can't be dropped the way the linear
classifier term can), so the *real* attention score this script's `u_hat` is
fit against uses real `z_i` with `m_i` mean-ablated (`m_bar`, the corpus-wide
mean) - the same convention `embedding_importance.py`'s `ablate_m_attn`
condition already established, reused directly rather than invented fresh.
This still requires real `m_i` for every tile to compute `m_bar` in the first
place, and for the real (unablated) baseline forward pass below - so despite
the surrogate being z_i-only, this script is NOT cheap like
`tile_r2_check_cls.py`: it needs the same full patch-corpus streaming pass
`embedding_importance.py` does (`mean_pool_patches`, `num_cpus=8`), not a
light local diagnostic.

**Two evaluation targets per head/class, both against the real model's own
baseline (unablated, exact) forward pass**:
    auc_vs_groundtruth - real ground-truth `nancy_index` (clinical usefulness)
    auc_vs_predicted   - the real model's own predicted label (fidelity: does
                          the surrogate agree with what the model says, right
                          or wrong)
Plus `fidelity_corr` (Pearson correlation between surrogate and real
predicted probabilities - a continuous companion to the binarized
`auc_vs_predicted`) and `real_auc_vs_groundtruth` (the real model's own AUC,
logged alongside as the ceiling the surrogate is being measured against).
"""

import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import ray
import ray.data
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from scipy.special import softmax
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from explainability.embedding_importance import (
    ModelWeights,
    forward_slide,
    load_full_model,
    logits_to_prob,
    nancy_to_target,
)
from explainability.grade_split import build_tile_features, mean_pool_patches
from explainability.tile_r2_check_cls import load_varphi, merge_or_warn
from explainability.tile_r2_ols_check import ols_fit
from explainability.tiles import load_slides, load_tokens_dataset, resolve_token_dirs


def real_attention_score(z: np.ndarray, m_bar: np.ndarray, model: ModelWeights) -> np.ndarray:
    """u_i = q^T tanh(U [z_i; m_bar]) for every tile - real z, mean-ablated m.

    See module docstring for why m_bar (not each tile's real m) is used here:
    attention is nonlinear in m, so it can't just be dropped the way the
    linear classifier term can, and this matches
    `embedding_importance.py`'s own `ablate_m_attn` convention rather than
    inventing a new one.
    """
    m_in = np.broadcast_to(m_bar, z.shape)
    h_attn = np.concatenate([z, m_in], axis=1)
    u = np.tanh(h_attn @ model.attn_w1.T + model.attn_b1) @ model.attn_w2.T + model.attn_b2
    return u[:, 0]


def slide_level_auc(
    s_hat: dict[str, np.ndarray],
    u_hat: dict[str, np.ndarray],
    z: np.ndarray,
    m: np.ndarray,
    models: dict[str, ModelWeights],
    slide_ids_col: pd.Series,
    nancy_index_by_slide: pd.Series,
    z_bar: np.ndarray,
    m_bar: np.ndarray,
) -> pd.DataFrame:
    """Per slide: surrogate (varphi-only) prediction vs. real (exact, unablated) forward pass.

    Loops over slides in plain Python, same convention as
    `embedding_importance.py`'s `run_ablation` (a few hundred to ~1000
    slides, each with a few hundred tiles - cheap by this point).

    Args:
        s_hat: Per head, OLS-fitted surrogate tile logits, shape
            `(n_tiles, num_classes)`.
        u_hat: Per head, OLS-fitted surrogate attention score, shape
            `(n_tiles,)`.
        z: Real CLS tokens, shape `(n_tiles, embed_dim)`.
        m: Real mean-pooled patch tokens, shape `(n_tiles, embed_dim)` - only
            used for the real baseline forward pass, never by the surrogate.
        models: Head -> `ModelWeights`.
        slide_ids_col: `slide_id` per tile, same row order as `z`/`m`/`s_hat`/`u_hat`.
        nancy_index_by_slide: Real ground-truth `nancy_index`, indexed by `slide_id`.
        z_bar: Corpus-wide mean of `z`, shape `(embed_dim,)`.
        m_bar: Corpus-wide mean of `m`, shape `(embed_dim,)`.

    Returns:
        One row per (head, class): `auc_vs_groundtruth`, `auc_vs_predicted`,
        `fidelity_corr`, `real_auc_vs_groundtruth`.
    """
    slide_codes, slide_ids = pd.factorize(slide_ids_col)
    targets_gt = {
        head: nancy_to_target(nancy_index_by_slide.loc[slide_ids].to_numpy(), head) for head in models
    }

    surrogate_probs: dict[str, list[np.ndarray]] = {
        head: [None] * len(slide_ids)  # type: ignore[list-item]
        for head in models
    }
    real_probs: dict[str, list[np.ndarray]] = {
        head: [None] * len(slide_ids)  # type: ignore[list-item]
        for head in models
    }

    for slide_idx in range(len(slide_ids)):
        rows = slide_codes == slide_idx
        z_s, m_s = z[rows], m[rows]
        for head, model in models.items():
            num_classes = model.cls_w.shape[0]
            a_hat = softmax(u_hat[head][rows])
            surrogate_logits = a_hat @ s_hat[head][rows]
            surrogate_probs[head][slide_idx] = logits_to_prob(surrogate_logits, num_classes)
            real_logits = forward_slide(z_s, m_s, model, False, False, False, False, z_bar, m_bar)
            real_probs[head][slide_idx] = logits_to_prob(real_logits, num_classes)

    rows_out = []
    for head, model in models.items():
        num_classes = model.cls_w.shape[0]
        target_gt = targets_gt[head]
        surrogate = np.stack(surrogate_probs[head])
        real = np.stack(real_probs[head])
        real_pred = np.argmax(real, axis=1) if num_classes > 1 else (real[:, 0] >= 0.5).astype(int)
        for c in range(num_classes if num_classes > 1 else 1):
            y_surrogate = surrogate[:, 0] if num_classes == 1 else surrogate[:, c]
            y_real = real[:, 0] if num_classes == 1 else real[:, c]
            y_gt = target_gt if num_classes == 1 else (target_gt == c).astype(int)
            y_pred_label = real_pred if num_classes == 1 else (real_pred == c).astype(int)
            try:
                auc_vs_groundtruth = float(roc_auc_score(y_gt, y_surrogate))
            except ValueError:
                auc_vs_groundtruth = float("nan")
            try:
                auc_vs_predicted = float(roc_auc_score(y_pred_label, y_surrogate))
            except ValueError:
                auc_vs_predicted = float("nan")
            try:
                fidelity_corr = float(pearsonr(y_surrogate, y_real)[0])
            except ValueError:
                fidelity_corr = float("nan")
            try:
                real_auc_vs_groundtruth = float(roc_auc_score(y_gt, y_real))
            except ValueError:
                real_auc_vs_groundtruth = float("nan")
            rows_out.append(
                {
                    "head": head,
                    "class": c,
                    "n_slides": len(slide_ids),
                    "auc_vs_groundtruth": auc_vs_groundtruth,
                    "auc_vs_predicted": auc_vs_predicted,
                    "fidelity_corr": fidelity_corr,
                    "real_auc_vs_groundtruth": real_auc_vs_groundtruth,
                }
            )
    return pd.DataFrame(rows_out)


@with_cli_args(["+explainability=slide_auc_check_cls"])
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

    # See embedding_importance.py's main() for the same "id" vs "slide_id"
    # column-name mismatch note.
    slides = load_slides(config.sources, config.get("local_embeddings_xai_dir"), split=config.split)
    tile_features = tile_features.merge(
        slides[["id", "nancy_index"]], left_on="slide_id", right_on="id", how="inner"
    ).drop(columns="id")

    varphi_df = load_varphi(Path(config.w_dir), config.n_components)
    print(f"Loaded {len(varphi_df)} tiles' varphi from {config.w_dir}", flush=True)

    merged = merge_or_warn(tile_features, varphi_df, "varphi")

    embed_dim = config.embed_dim
    h = np.stack(merged["h"].to_numpy())
    z, m = h[:, :embed_dim], h[:, embed_dim:]
    varphi = np.stack(merged["varphi"].to_numpy())
    nancy_index_by_slide = merged.drop_duplicates("slide_id").set_index("slide_id")["nancy_index"]
    z_bar, m_bar = z.mean(axis=0), m.mean(axis=0)

    models = {
        head: load_full_model(config.checkpoints[head].checkpoint, embed_dim)
        for head in ("neutrophils", "nancy_low", "nancy_high")
    }

    s_hat: dict[str, np.ndarray] = {}
    u_hat: dict[str, np.ndarray] = {}
    for head, model in models.items():
        theta_z = model.cls_w[:, :embed_dim]
        z_target = z @ theta_z.T  # (n, num_classes) - what the classifier surrogate fits
        num_classes = model.cls_w.shape[0]
        s_hat[head] = np.stack(
            [ols_fit(varphi, z_target[:, c]) for c in range(num_classes)], axis=1
        )
        u_real = real_attention_score(z, m_bar, model)
        u_hat[head] = ols_fit(varphi, u_real)
        print(f"Fit {head}'s z-only classifier + attention surrogate from varphi.", flush=True)

    print("Computing per-slide surrogate vs. real forward passes...", flush=True)
    result_df = slide_level_auc(
        s_hat, u_hat, z, m, models, merged["slide_id"], nancy_index_by_slide, z_bar, m_bar
    )
    print(result_df.to_string(index=False), flush=True)

    result_path = output_dir / "auc.parquet"
    result_df.to_parquet(result_path, index=False)

    manifest = {
        "split": config.split,
        "n_tiles": len(merged),
        "n_slides": merged["slide_id"].nunique(),
        "n_components": config.n_components,
        "w_dir": config.w_dir,
        "results": result_df.to_dict(orient="records"),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    logger.log_artifact(str(result_path))
    logger.log_artifact(str(manifest_path))
    for row in result_df.to_dict(orient="records"):
        tag = f"{row['head']}_class{row['class']}"
        logger.log_metrics(
            {
                f"auc_vs_groundtruth/{tag}": row["auc_vs_groundtruth"],
                f"auc_vs_predicted/{tag}": row["auc_vs_predicted"],
                f"fidelity_corr/{tag}": row["fidelity_corr"],
                f"real_auc_vs_groundtruth/{tag}": row["real_auc_vs_groundtruth"],
            }
        )


if __name__ == "__main__":
    ctx = ray.data.DataContext.get_current()
    ctx.enable_rich_progress_bars = True
    ctx.use_ray_tqdm = False

    # num_cpus=8: same oversized-parquet-row-group root cause as
    # token_statistics.py/nmf_fit.py/grade_split.py/embedding_importance.py -
    # mean_pool_patches reads the full patch corpus. Keep in sync with cpu=
    # in scripts/explainability/slide_auc_check_cls.py.
    with ray.init(num_cpus=8, runtime_env={"excludes": [".git", ".venv"]}):
        main()
