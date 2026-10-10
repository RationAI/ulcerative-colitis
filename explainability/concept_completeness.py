"""Q1 - how good are the concept encoder and decoder? Completeness metrics of one dictionary.

Errors follow arXiv:2609.34750 (RMSE over the data distribution), at every
level of the attention-MIL model (notation in explainability/concepts.py):

    RE     RMSE(z_i, D(E(z_i)))                tiles, in Z
    FE_s   RMSE(g_s(z_i), g_s(D(E(z_i))))      tiles, per output logit
    FE_u   RMSE(g_u(z_i), g_u(D(E(z_i))))      tiles, attention score per model
    FE_L   RMSE(L(Z_s), L_A(Z_s))              slides, per output logit - f_A
                                               recomputes attention from the
                                               reconstructions too
    MCE    RMSE(f, eta o C) for eta = the best linear head on the concepts
           (OLS on [v, 1], per tile logit and per attention score, then pooled
           like the real model for slides). eta o C is one admissible head, so
           this is an upper estimate of the paper's MCE; MCE <= FE always.

Tile-logit FE is bounded by the head's Lipschitz constant: per output,
|Theta_c (z - z')| <= ||Theta_c|| ||z - z'||, so FE_s <= ||Theta_c|| RE, and
g_u is ||q|| ||U||_2-Lipschitz (tanh is 1-Lipschitz). `rho_fe` = FE / bound is
the paper's tightness ratio: small means most of the reconstruction error
lies in directions the head doesn't read.

Each error is also given relative to the target's spread (`r2` = 1 -
error^2 / variance), since raw RMSEs aren't comparable across outputs.

Slide-level predictions are compared both with the real model's own
prediction (fidelity: label agreement, AUC against its predicted label,
probability correlation) and with ground truth (AUC of f, f_A and eta).
Multiclass logits are centred across the head's classes throughout (softmax
ignores a shared offset).

In-sample: the dictionary and the OLS heads are fit on the same split they
are evaluated on - upper bounds until this is run on a held-out split.
"""

import json
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
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from explainability.concepts import (
    SlideIndex,
    iter_tiles,
    load_autoencoder,
    ols_predict,
    r2_score,
    rmse,
)
from explainability.model import (
    ModelWeights,
    attention_scores,
    centred_classifier,
    class_mask,
    load_full_model,
    logits_to_prob,
    nancy_to_target,
    predicted_labels,
    tile_logits,
)
from explainability.tiles import (
    load_embedding_slides,
    load_embeddings_dataset,
    resolve_embedding_split_dirs,
)


HEADS = ("neutrophils", "nancy_low", "nancy_high")


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:  # only one class present
        return float("nan")


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(pearsonr(a, b)[0])


def error_row(true: np.ndarray, approx: np.ndarray, eta: np.ndarray) -> dict[str, float]:
    return {
        "fe": rmse(true, approx),
        "mce_ols": rmse(true, eta),
        "std": float(np.std(true)),
        "r2_fe": r2_score(true, approx),
        "r2_mce_ols": r2_score(true, eta),
    }


@with_cli_args(["+explainability=concept_completeness"])
@hydra.main(config_path="../configs", config_name="explainability", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models: dict[str, ModelWeights] = {
        head: load_full_model(config.checkpoints[head].checkpoint, config.embed_dim)
        for head in HEADS
    }
    autoencoder = load_autoencoder(Path(config.w_dir), config.nnls_iter)
    split_dirs = resolve_embedding_split_dirs(
        config.sources, config.get("local_embeddings_dir"), split=config.split
    )

    # One pass: encode/decode every tile, keep only small per-tile readouts.
    key_chunks, v_chunks = [], []
    readouts: dict[str, dict[str, list[np.ndarray]]] = {
        head: {"s": [], "s_A": [], "u": [], "u_A": []} for head in HEADS
    }
    sq_err = sum_z = sum_sq = 0.0
    sum_z_vec: np.ndarray | None = None
    for keys, z in iter_tiles(load_embeddings_dataset(split_dirs), config.batch_size):
        v = autoencoder.encode(z)
        z_hat = autoencoder.decode(v)
        sq_err += float(np.sum((z - z_hat) ** 2))
        sum_sq += float(np.sum(z.astype(np.float64) ** 2))
        batch_sum = z.astype(np.float64).sum(axis=0)
        sum_z_vec = batch_sum if sum_z_vec is None else sum_z_vec + batch_sum
        sum_z += len(z)
        for head, model in models.items():
            readouts[head]["s"].append(tile_logits(z, model).astype(np.float32))
            readouts[head]["s_A"].append(tile_logits(z_hat, model).astype(np.float32))
            readouts[head]["u"].append(attention_scores(z, model).astype(np.float32))
            readouts[head]["u_A"].append(attention_scores(z_hat, model).astype(np.float32))
        key_chunks.append(keys)
        v_chunks.append(v)
    assert sum_z_vec is not None
    n_tiles = int(sum_z)
    keys = pd.concat(key_chunks, ignore_index=True)
    v = np.concatenate(v_chunks)
    re = float(np.sqrt(sq_err / n_tiles))
    total_var = sum_sq / n_tiles - float(np.sum((sum_z_vec / n_tiles) ** 2))
    re_rel = re / float(np.sqrt(total_var))
    print(f"RE = {re:.4f} (relative {re_rel:.4f}) over {n_tiles} tiles, K={v.shape[1]}", flush=True)

    slides = SlideIndex(keys["slide_id"])
    nancy_index = (
        load_embedding_slides(split_dirs).set_index("id").loc[slides.ids, "nancy_index"].to_numpy()
    )
    if pd.isna(nancy_index).any():
        raise ValueError(f"{int(pd.isna(nancy_index).sum())} slides lack nancy_index")

    tile_rows: list[dict[str, Any]] = []
    slide_rows: list[dict[str, Any]] = []
    agreement_rows: list[dict[str, Any]] = []
    for head, model in models.items():
        r = {name: np.concatenate(chunks) for name, chunks in readouts[head].items()}
        s_eta, u_eta = ols_predict(v, r["s"]), ols_predict(v, r["u"])
        theta, _ = centred_classifier(model)
        num_classes = theta.shape[0]

        for c in range(num_classes):
            bound = float(np.linalg.norm(theta[c])) * re
            row = error_row(r["s"][:, c], r["s_A"][:, c], s_eta[:, c])
            tile_rows.append(
                {"head": head, "target": f"s_class{c}", **row, "fe_bound": bound,
                 "rho_fe": row["fe"] / bound if bound > 0 else float("nan")}
            )
        lipschitz_u = float(np.linalg.norm(model.attn_w2) * np.linalg.norm(model.attn_w1, ord=2))
        row = error_row(r["u"], r["u_A"], u_eta)
        tile_rows.append(
            {"head": head, "target": "u", **row, "fe_bound": lipschitz_u * re,
             "rho_fe": row["fe"] / (lipschitz_u * re)}
        )

        logits = {
            "real": slides.attention_pool(r["s"], r["u"]),
            "A": slides.attention_pool(r["s_A"], r["u_A"]),
            "eta": slides.attention_pool(s_eta, u_eta),
        }
        probs = {name: logits_to_prob(value, num_classes) for name, value in logits.items()}
        real_pred = predicted_labels(probs["real"])
        target = nancy_to_target(nancy_index, head)
        for name in ("A", "eta"):
            agreement_rows.append(
                {"head": head, "surrogate": name,
                 "label_agreement": float(np.mean(predicted_labels(probs[name]) == real_pred))}
            )
        for c in range(num_classes):
            slide_row: dict[str, Any] = {"head": head, "class": c, "n_slides": len(slides)}
            err = error_row(logits["real"][:, c], logits["A"][:, c], logits["eta"][:, c])
            slide_row.update({f"logit_{k}": val for k, val in err.items()})
            gt = class_mask(target, c, num_classes)
            pred = class_mask(real_pred, c, num_classes)
            slide_row["auc_real_vs_groundtruth"] = safe_auc(gt, probs["real"][:, c])
            for name in ("A", "eta"):
                slide_row[f"auc_{name}_vs_groundtruth"] = safe_auc(gt, probs[name][:, c])
                slide_row[f"auc_{name}_vs_predicted"] = safe_auc(pred, probs[name][:, c])
                slide_row[f"prob_corr_{name}"] = safe_pearson(probs["real"][:, c], probs[name][:, c])
            slide_rows.append(slide_row)
        print(f"Done {head}.", flush=True)

    tile_df, slide_df, agreement_df = (
        pd.DataFrame(tile_rows), pd.DataFrame(slide_rows), pd.DataFrame(agreement_rows)
    )
    for df in (tile_df, slide_df, agreement_df):
        print(df.to_string(index=False), flush=True)

    tile_df.to_parquet(output_dir / "tile_fidelity.parquet", index=False)
    slide_df.to_parquet(output_dir / "slide_fidelity.parquet", index=False)
    agreement_df.to_parquet(output_dir / "slide_agreement.parquet", index=False)
    manifest = {
        "w_dir": config.w_dir,
        "split": config.split,
        "n_components": int(v.shape[1]),
        "n_tiles": n_tiles,
        "n_slides": len(slides),
        "re": re,
        "re_relative": re_rel,
        "tile_fidelity": tile_df.to_dict(orient="records"),
        "slide_fidelity": slide_df.to_dict(orient="records"),
        "slide_agreement": agreement_df.to_dict(orient="records"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    for name in ("tile_fidelity.parquet", "slide_fidelity.parquet", "slide_agreement.parquet",
                 "manifest.json"):
        logger.log_artifact(str(output_dir / name))
    logger.log_metrics({"re": re, "re_relative": re_rel})
    for tile_row in tile_rows:
        tag = f"{tile_row['head']}_{tile_row['target']}"
        logger.log_metrics({f"tile_fe/{tag}": tile_row["fe"], f"tile_mce_ols/{tag}": tile_row["mce_ols"]})
    for logged_row in slide_rows:
        tag = f"{logged_row['head']}_class{logged_row['class']}"
        logger.log_metrics(
            {f"slide_{k}/{tag}": float(val) for k, val in logged_row.items()
             if k not in ("head", "class", "n_slides")}
        )


if __name__ == "__main__":
    ctx = ray.data.DataContext.get_current()
    ctx.enable_rich_progress_bars = True
    ctx.use_ray_tqdm = False

    # num_cpus=8: same conservative cap as nmf_fit.py (full embeddings corpus
    # read). Keep in sync with cpu= in scripts/explainability/concept_completeness.py.
    with ray.init(num_cpus=8, runtime_env={"excludes": [".git", ".venv"]}):
        main()
