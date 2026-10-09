"""Tile-level R^2 and slide-level AUC of concept surrogates of the real MIL model.

Replaces `tile_r2_check_cls.py` + `slide_auc_check_cls.py`. The NMF
dictionary is now fit on the full tile embedding `h_i` - exactly what the MIL
model consumes (`s_i = Theta h_i + b`, `u_i = q^T tanh(U h_i + b1) + b2`) - so
there's no z/m split and no mean-ablated `m_bar` stand-in anywhere: every
target here is the real model's own readout of the real embedding.

Two surrogates, both built only from the tile's concept weights `phi_i` (W):

    recon  the real model applied to the dictionary reconstruction
           h_hat_i = shift + phi_i @ H (nmf_fit.py's own transform target -
           it fits shift-only embeddings against the recovered, gauge-fixed
           H). No fitting: the concept-to-logit map is exactly
           H Theta^T, the closed-form sigma, with attention through the
           real (nonlinear) attention module.
    ols    s_i and u_i each OLS-regressed (with intercept) on phi_i - the
           best linear readout of the concepts, fitted per head/class.

Tile level: R^2 (and Pearson r for recon) of each surrogate's `s` per
head/class and `u` per head against the real values. Slide level: per slide,
`softmax(u) @ s` for real vs. each surrogate, scored as AUC against
ground-truth `nancy_index` and against the real model's predicted label, plus
the Pearson correlation of predicted probabilities (fidelity).

One streaming pass over the embeddings computes each tile's real `s`/`u` for
every head; everything after that works on small per-tile arrays, aligned
to W by (slide_id, x, y).
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

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
    load_full_model,
    logits_to_prob,
    nancy_to_target,
)
from explainability.tile_r2_ols_check import r2_score
from explainability.tiles import (
    load_embedding_slides,
    load_embeddings_dataset,
    resolve_embedding_split_dirs,
)


KEYS = ["slide_id", "x", "y"]


def tile_logits(h: np.ndarray, model: ModelWeights) -> np.ndarray:
    """s_i = Theta h_i + b, shape `(n, num_classes)`."""
    return h @ model.cls_w.T + model.cls_b


def attention_scores(h: np.ndarray, model: ModelWeights) -> np.ndarray:
    """u_i = q^T tanh(U h_i + b1) + b2, shape `(n,)`."""
    return (np.tanh(h @ model.attn_w1.T + model.attn_b1) @ model.attn_w2.T + model.attn_b2)[:, 0]


def compute_readouts(
    dataset: ray.data.Dataset, models: dict[str, ModelWeights], batch_size: int
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Stream every tile embedding once and keep only the real model's per-tile readouts.

    Returns:
        `(keys, s, u)`: `keys` is the (slide_id, x, y) of each row; `s[head]`
        has shape `(n, num_classes)` and `u[head]` shape `(n,)`, in the same
        row order.
    """
    key_chunks: list[pd.DataFrame] = []
    s_chunks: dict[str, list[np.ndarray]] = {head: [] for head in models}
    u_chunks: dict[str, list[np.ndarray]] = {head: [] for head in models}
    n_rows = 0
    for batch in dataset.select_columns([*KEYS, "embedding"]).iter_batches(
        batch_size=batch_size, batch_format="numpy"
    ):
        h = np.stack(batch["embedding"]).astype(np.float32, copy=False)
        for head, model in models.items():
            if h.shape[1] != model.cls_w.shape[1]:
                raise ValueError(
                    f"embedding width {h.shape[1]} != {head} classifier input width "
                    f"{model.cls_w.shape[1]}"
                )
            s_chunks[head].append(tile_logits(h, model).astype(np.float32))
            u_chunks[head].append(attention_scores(h, model).astype(np.float32))
        key_chunks.append(pd.DataFrame({key: batch[key] for key in KEYS}))
        n_rows += h.shape[0]
        print(f"compute_readouts: {n_rows} tiles", flush=True)

    keys = pd.concat(key_chunks, ignore_index=True)
    s = {head: np.concatenate(chunks) for head, chunks in s_chunks.items()}
    u = {head: np.concatenate(chunks) for head, chunks in u_chunks.items()}
    return keys, s, u


def load_dictionary(
    w_dir: Path, n_components: int
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Load one nmf_fit.py run's `(w_metadata, W, H, shift)` from its output dir."""
    w = np.load(w_dir / "w.f32.npy", mmap_mode="r")
    metadata = pd.read_parquet(w_dir / "w_metadata.parquet")
    h = pd.read_parquet(w_dir / "h.parquet").sort_index(axis=0).sort_index(axis=1)
    shift = np.load(w_dir / "shift.npy")
    if w.shape[1] != n_components or len(h) != n_components:
        raise ValueError(
            f"{w_dir}: W width {w.shape[1]} / H rows {len(h)}, expected {n_components}"
        )
    if len(metadata) != w.shape[0]:
        raise ValueError(f"{w_dir}: {len(metadata)} metadata rows but W has {w.shape[0]}")
    return metadata, np.asarray(w), h.to_numpy(dtype=np.float32), shift.astype(np.float32)


def align(readout_keys: pd.DataFrame, w_keys: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Row indices into (readouts, W) for every tile present in both, warning on a mismatch."""
    merged = readout_keys.assign(_r=np.arange(len(readout_keys))).merge(
        w_keys.assign(_w=np.arange(len(w_keys))), on=KEYS, how="inner"
    )
    if len(merged) != len(readout_keys) or len(merged) != len(w_keys):
        print(
            f"WARNING: {len(readout_keys)} streamed tiles vs {len(w_keys)} W rows merged to "
            f"{len(merged)} - check for missing or duplicate (slide_id, x, y) keys.",
            flush=True,
        )
    return merged["_r"].to_numpy(), merged["_w"].to_numpy()


def iter_chunks(n: int, chunk_size: int) -> Iterator[slice]:
    for start in range(0, n, chunk_size):
        yield slice(start, min(start + chunk_size, n))


def recon_readouts(
    phi: np.ndarray, h: np.ndarray, shift: np.ndarray, model: ModelWeights, chunk_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Real model's `(s, u)` on h_hat = shift + phi @ H, chunked to bound memory."""
    s = np.empty((len(phi), model.cls_w.shape[0]), dtype=np.float32)
    u = np.empty(len(phi), dtype=np.float32)
    for rows in iter_chunks(len(phi), chunk_size):
        h_hat = shift + phi[rows] @ h
        s[rows] = tile_logits(h_hat, model)
        u[rows] = attention_scores(h_hat, model)
    return s, u


def ols_predict(phi: np.ndarray, target: np.ndarray) -> np.ndarray:
    """OLS fit of `target` (`(n,)` or `(n, c)`) on `[phi, 1]`, returning in-sample predictions."""
    design = np.hstack([phi, np.ones((len(phi), 1), dtype=phi.dtype)])
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    return design @ beta


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(pearsonr(a, b)[0])


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:  # only one class present
        return float("nan")


def slide_probs(
    s: np.ndarray, u: np.ndarray, slide_codes: np.ndarray, n_slides: int
) -> np.ndarray:
    """Per slide, `logits_to_prob(softmax(u) @ s)`, shape `(n_slides, num_classes)`."""
    num_classes = s.shape[1]
    order = np.argsort(slide_codes, kind="stable")
    bounds = np.searchsorted(slide_codes[order], np.arange(n_slides + 1))
    probs = np.empty((n_slides, num_classes))
    for slide in range(n_slides):
        rows = order[bounds[slide] : bounds[slide + 1]]
        probs[slide] = logits_to_prob(softmax(u[rows]) @ s[rows], num_classes)
    return probs


@with_cli_args(["+explainability=concept_surrogate_check"])
@hydra.main(config_path="../configs", config_name="explainability", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = {
        head: load_full_model(config.checkpoints[head].checkpoint, config.embed_dim)
        for head in ("neutrophils", "nancy_low", "nancy_high")
    }

    split_dirs = resolve_embedding_split_dirs(
        config.sources, config.get("local_embeddings_dir"), split=config.split
    )
    readout_keys, s_real, u_real = compute_readouts(
        load_embeddings_dataset(split_dirs), models, config.batch_size
    )

    w_keys, w, h, shift = load_dictionary(Path(config.w_dir), config.n_components)
    r_idx, w_idx = align(readout_keys, w_keys)
    phi = w[w_idx]
    keys = readout_keys.iloc[r_idx].reset_index(drop=True)
    print(f"Aligned {len(keys)} tiles with W from {config.w_dir}", flush=True)

    slides = load_embedding_slides(split_dirs).set_index("id")
    slide_codes, slide_ids = pd.factorize(keys["slide_id"])
    nancy_index = slides.loc[slide_ids, "nancy_index"].to_numpy()
    if pd.isna(nancy_index).any():
        raise ValueError(f"{int(pd.isna(nancy_index).sum())} slides lack nancy_index in {config.split}")

    tile_rows: list[dict[str, Any]] = []
    slide_rows: list[dict[str, Any]] = []
    for head, model in models.items():
        s, u = s_real[head][r_idx], u_real[head][r_idx]
        s_recon, u_recon = recon_readouts(phi, h, shift, model, config.chunk_size)
        s_ols, u_ols = ols_predict(phi, s), ols_predict(phi, u)
        num_classes = s.shape[1]

        for c in range(num_classes):
            tile_rows.append(
                {
                    "head": head,
                    "target": f"s_class{c}",
                    "r2_recon": r2_score(s[:, c], s_recon[:, c]),
                    "pearson_recon": safe_pearson(s[:, c], s_recon[:, c]),
                    "r2_ols": r2_score(s[:, c], s_ols[:, c]),
                }
            )
        tile_rows.append(
            {
                "head": head,
                "target": "u",
                "r2_recon": r2_score(u, u_recon),
                "pearson_recon": safe_pearson(u, u_recon),
                "r2_ols": r2_score(u, u_ols),
            }
        )

        probs = {
            name: slide_probs(s_, u_, slide_codes, len(slide_ids))
            for name, (s_, u_) in {
                "real": (s, u),
                "recon": (s_recon, u_recon),
                "ols": (s_ols, u_ols),
            }.items()
        }
        target = nancy_to_target(nancy_index, head)
        real_pred = (
            (probs["real"][:, 0] >= 0.5).astype(int)
            if num_classes == 1
            else np.argmax(probs["real"], axis=1)
        )
        for c in range(num_classes):
            y_gt = target if num_classes == 1 else (target == c).astype(int)
            y_pred = real_pred if num_classes == 1 else (real_pred == c).astype(int)
            row: dict[str, Any] = {
                "head": head,
                "class": c,
                "n_slides": len(slide_ids),
                "auc_real_vs_groundtruth": safe_auc(y_gt, probs["real"][:, c]),
            }
            for name in ("recon", "ols"):
                row[f"auc_{name}_vs_groundtruth"] = safe_auc(y_gt, probs[name][:, c])
                row[f"auc_{name}_vs_predicted"] = safe_auc(y_pred, probs[name][:, c])
                row[f"fidelity_corr_{name}"] = safe_pearson(probs["real"][:, c], probs[name][:, c])
            slide_rows.append(row)
        print(f"Done {head}.", flush=True)

    tile_df = pd.DataFrame(tile_rows)
    slide_df = pd.DataFrame(slide_rows)
    print(tile_df.to_string(index=False), flush=True)
    print(slide_df.to_string(index=False), flush=True)

    tile_path = output_dir / "tile_r2.parquet"
    slide_path = output_dir / "slide_auc.parquet"
    tile_df.to_parquet(tile_path, index=False)
    slide_df.to_parquet(slide_path, index=False)
    manifest = {
        "split": config.split,
        "w_dir": config.w_dir,
        "n_components": config.n_components,
        "n_tiles": len(keys),
        "n_slides": len(slide_ids),
        "tile_r2": tile_df.to_dict(orient="records"),
        "slide_auc": slide_df.to_dict(orient="records"),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    logger.log_artifact(str(tile_path))
    logger.log_artifact(str(slide_path))
    logger.log_artifact(str(manifest_path))
    for tile_row in tile_rows:
        tag = f"{tile_row['head']}_{tile_row['target']}"
        logger.log_metrics(
            {
                f"tile_{metric}/{tag}": float(tile_row[metric])
                for metric in ("r2_recon", "pearson_recon", "r2_ols")
            }
        )
    for slide_row in slide_rows:
        tag = f"{slide_row['head']}_class{slide_row['class']}"
        logger.log_metrics(
            {
                f"slide_{k}/{tag}": float(v)
                for k, v in slide_row.items()
                if k not in ("head", "class", "n_slides")
            }
        )


if __name__ == "__main__":
    ctx = ray.data.DataContext.get_current()
    ctx.enable_rich_progress_bars = True
    ctx.use_ray_tqdm = False

    # num_cpus=8: same conservative cap as token_statistics.py/nmf_fit.py
    # (full embeddings corpus read). Keep in sync with cpu= in
    # scripts/explainability/concept_surrogate_check.py.
    with ray.init(num_cpus=8, runtime_env={"excludes": [".git", ".venv"]}):
        main()
