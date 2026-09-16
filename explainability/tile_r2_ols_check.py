"""Extends tile_r2_check.py's Q4 check: OLS gap + how much flows through the CLS token.

Two additions to concept_mil.tex's Q4 table (post-2026-09-04 revision, see
Remark "A purely weight-space coverage does not bound R^2" and Sec.
"Relation to the fitted surrogate"), both
computed on top of an *already-finished* `tile_r2_check.py` run's
`tile_features.parquet` - no patch re-stream, the only new read is one
grade's CLS token partition (one row per tile, ~350MB for grade=4, not the
~9M-row patch stream `tile_r2_check.py` itself needs), light enough to run
in-session.

**1. R^2 gap to OLS (eq:ols).** `tile_r2_check.py` only checks the
closed-form plug-in prediction `varphi @ sigma` against the m-driven target
`m @ Theta_m^T`. This script additionally fits an actual OLS regression of
that same target on the same tile-level `varphi` (no `sigma` involved, just
whatever linear combination of the K concept weights best predicts the
target on this data) and reports both R^2s side by side. Since OLS is the
best possible linear fit of the target from exactly these K numbers,
R^2(OLS) >= R^2(plug-in) always; the gap separates two failure modes for a
low plug-in R^2 that look identical from `tile_r2_check.py` alone:
    - gap ~ 0: no linear combination of varphi explains the target either -
      the information genuinely isn't in the concept weights -> raise K.
    - gap large: varphi *does* carry enough information, just not via the
      "honest" sigma coefficients - the NMF reconstruction residual
      `r_i = m_i - sum_k varphi_ik H_k` happens to correlate with which
      concepts a tile uses, and OLS exploits that correlation to reassign
      credit for off-basis signal onto whichever concepts co-occur with it
      ("residual leakage" - concept_mil.tex's "Relation to the fitted
      surrogate" section). Not a K problem; a trustworthiness-of-attribution
      problem.

**2. Full-head target (Theta . h_i, not just Theta_m . m_i).** Both
predictions above are also checked against the FULL tile logit - the real
Theta_z . z_i term plus the (plug-in- or OLS-)approximated m term - instead
of just the m-driven term in isolation. Because Theta_z . z_i is *exact* on
both sides of every comparison (never approximated, straight from the CLS
token), it cancels out of every residual identically to the m-only check;
only the R^2 *denominator* (total target variance) changes, from
Var(Theta_m . m_i) to Var(Theta_z . z_i + Theta_m . m_i). A third baseline -
predicting the full target using *only* the exact CLS term, i.e. pretending
the m/concept pathway contributes nothing at all - calibrates how much of
that denominator the CLS pathway already accounts for by itself: if
R^2(CLS-only) is already close to 1, most of the tile logit's variance flows
through the CLS token and the concept basis was never going to explain much
of the *total* logit regardless of its own quality on the m-only target.

**No shift/scale transform for z_i**: concept_mil.tex's non-negativity
transform (Sec. "Non-negativity transform (NMF only)") applies only to the
patch-token/m_i pathway fed to NMF (requirement R1: the concept dictionary
lives in patch-token space, never the concatenated h_i space) - Theta_z acts
on the raw CLS token embedding directly, no shift/scale needed.

**`m` inherits tile_r2_check.py's shift-only space**: `tile_features.parquet`'s
`m` column is `mean_p max(t_ip - shift, 0)`, not the raw mean patch token -
see that script's own module docstring ("Uses raw (uncentered) sigma") for
why this is the self-consistent choice (matches `h.parquet`'s own space,
since `h` is only scale-recovered, not shift-recovered - eq:recover). Every
target/prediction pair here inherits that same per-class constant offset
consistently, so - exactly as in `tile_r2_check.py`'s own m-only check - it
drops out of every R^2 (variance-based, offset-invariant) without needing to
be undone.
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

from explainability.sigma_table import load_h
from explainability.theta_m_check import load_classifier
from explainability.tiles import resolve_grade_token_dir


def load_cls_features(token_dir: str) -> pd.DataFrame:
    """Load one grade's CLS token partition as `slide_id, x, y, z` (one row per tile already).

    Unlike patch tokens, no pooling/aggregation is needed - `grade_split.py`
    already wrote exactly one CLS row per tile - so this reads straight to
    pandas rather than deferring to `ray.data`'s lazy/streaming machinery
    `tile_r2_check.py` needs for the much larger patch tables.
    """
    df = ray.data.read_parquet(token_dir).to_pandas()
    return df[["slide_id", "x", "y", "embedding"]].rename(columns={"embedding": "z"})


def r2_score(true: np.ndarray, pred: np.ndarray) -> float:
    """R^2 of `pred` against `true`, NaN (not just numerically unstable) when `true` has zero variance."""
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    if ss_tot <= 0:
        return float("nan")
    ss_res = float(np.sum((true - pred) ** 2))
    return 1.0 - ss_res / ss_tot


def ols_fit(varphi: np.ndarray, target: np.ndarray) -> np.ndarray:
    """OLS coefficients of `target` on `varphi` (eq:ols's beta^OLS, computed directly rather than via eq:ols's closed form)."""
    beta, *_ = np.linalg.lstsq(varphi, target, rcond=None)
    return varphi @ beta


@with_cli_args(["+explainability=tile_r2_ols_check"])
@hydra.main(config_path="../configs", config_name="explainability", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tile_df = pd.read_parquet(config.tile_features_path)
    print(f"Loaded {len(tile_df)} tiles from {config.tile_features_path}", flush=True)

    cls_dir = resolve_grade_token_dir(
        config.get("local_grade_split_dir"), config.grade_split.mlflow_uri, kind="cls", grade=config.grade
    )
    cls_df = load_cls_features(cls_dir)
    print(f"Loaded {len(cls_df)} grade={config.grade} CLS tokens from {cls_dir}", flush=True)

    merged = tile_df.merge(cls_df, on=["slide_id", "x", "y"], how="inner")
    if len(merged) != len(tile_df):
        # A *smaller* merged count means some tile lacked a matching CLS row;
        # a *larger* one means a duplicate (slide_id, x, y) key fanned the
        # join out on one side - both are surprising enough to flag rather
        # than silently reconcile, but neither noticeably moves the R^2
        # numbers below at this row count.
        print(
            f"WARNING: tile_features.parquet had {len(tile_df)} tiles, merged with CLS tokens "
            f"gave {len(merged)} rows - check for missing or duplicate (slide_id, x, y) keys.",
            flush=True,
        )

    h = load_h(config.h.mlflow_uri)
    if h.shape[0] != config.n_components:
        raise ValueError(
            f"h.parquet at {config.h.mlflow_uri} has {h.shape[0]} rows, "
            f"expected n_components={config.n_components}"
        )

    m = np.stack(merged["m"].to_numpy())
    varphi = np.stack(merged["varphi"].to_numpy())
    z = np.stack(merged["z"].to_numpy())

    rows = []
    for head, checkpoint_cfg in config.checkpoints.items():
        weight, _bias = load_classifier(checkpoint_cfg.checkpoint, config.embed_dim)
        theta_z = weight[:, : config.embed_dim]
        theta_m = weight[:, config.embed_dim :]
        sigma = h @ theta_m.T  # (K, C), eq. 2.9

        m_target = m @ theta_m.T  # (n, C)
        z_target = z @ theta_z.T  # (n, C) - exact, never approximated
        full_target = m_target + z_target
        plugin_pred = varphi @ sigma

        num_classes = theta_m.shape[0]
        for c in range(num_classes):
            ols_pred_c = ols_fit(varphi, m_target[:, c])
            # CLS-only baseline: predicts the *average* m-contribution, not
            # literally zero - a hard zero would penalize this baseline for
            # m_target's own systematic offset (m lives in tile_r2_check.py's
            # shift-space, not raw units - see module docstring), which
            # plugin/ols never pay because NMF's own fit already reconstructs
            # that offset on average. mean(m_target) is the fair "I know
            # nothing about which concepts this tile has" stand-in for that
            # same offset - the intercept a real model's bias would supply.
            clsonly_pred_c = z_target[:, c] + m_target[:, c].mean()
            row = {
                "head": head,
                "class": c,
                "n_tiles": len(merged),
                "r2_plugin_vs_m": r2_score(m_target[:, c], plugin_pred[:, c]),
                "r2_ols_vs_m": r2_score(m_target[:, c], ols_pred_c),
                "r2_clsonly_vs_full": r2_score(full_target[:, c], clsonly_pred_c),
                "r2_plugin_vs_full": r2_score(full_target[:, c], z_target[:, c] + plugin_pred[:, c]),
                "r2_ols_vs_full": r2_score(full_target[:, c], z_target[:, c] + ols_pred_c),
            }
            row["ols_gap_vs_m"] = row["r2_ols_vs_m"] - row["r2_plugin_vs_m"]
            rows.append(row)

        print(f"=== {head} ===", flush=True)
        head_df = pd.DataFrame([r for r in rows if r["head"] == head])
        print(head_df.to_string(index=False), flush=True)

    result_df = pd.DataFrame(rows)
    result_path = output_dir / "ols_check.parquet"
    result_df.to_parquet(result_path, index=False)

    manifest = {
        "grade": config.grade,
        "n_components": config.n_components,
        "n_tiles": len(merged),
        "n_tiles_dropped": len(tile_df) - len(merged),
        "rows": result_df.to_dict(orient="records"),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    logger.log_artifact(str(result_path))
    logger.log_artifact(str(manifest_path))
    for row in rows:
        tag = f"{row['head']}_class{row['class']}"
        logger.log_metrics(
            {
                f"r2_plugin_vs_m/{tag}": row["r2_plugin_vs_m"],
                f"r2_ols_vs_m/{tag}": row["r2_ols_vs_m"],
                f"ols_gap_vs_m/{tag}": row["ols_gap_vs_m"],
                f"r2_clsonly_vs_full/{tag}": row["r2_clsonly_vs_full"],
                f"r2_plugin_vs_full/{tag}": row["r2_plugin_vs_full"],
                f"r2_ols_vs_full/{tag}": row["r2_ols_vs_full"],
            }
        )


if __name__ == "__main__":
    # No num_cpus pin (unlike tile_r2_check.py/nmf_fit.py/patch_statistics.py):
    # this reads one grade's small CLS partition (one row per tile), not the
    # oversized-row-group patch tables that OOM without a concurrency cap -
    # only the working_dir upload exclusion is needed here, same as those.
    with ray.init(runtime_env={"excludes": [".git", ".venv"]}):
        main()
