"""z_i analogue of tile_r2_check.py plus tile_r2_ols_check.py, for a kind=cls NMF fit.

Same base question as `tile_r2_check.py`: does the concept basis, on real
tiles, actually carry what the classifier reads -

    R^2 of  varphi @ sigma_z   vs.   z @ Theta_z^T        (tile level)

- but neither `tile_r2_check.py` nor `tile_r2_ols_check.py` fits this case
directly, hence one merged script covering both rather than a `kind=` flag
on either:

- `tile_r2_check.py`'s whole job is streaming ~10^7 patch tokens and
  mean-pooling their NMF activations into one tile-level `varphi` per tile
  (`P` patch rows -> 1 tile row). A `kind=cls` `nmf_fit.py` run has already
  done the equivalent work: exactly one CLS token per tile means its saved
  `w.f32.npy` *is already* tile-level `varphi` directly (see
  `explainability/nmf_fit.py`'s comment at its own transform pass) - no
  patch stream, no pooling, nothing to re-derive. Reusing `tile_r2_check.py`
  as-is would silently redo an expensive no-op; giving it a `kind=` flag
  would just be two unrelated code paths sharing a file.
- `tile_r2_ols_check.py` is structurally closer (no streaming, just load +
  merge + regress), and its `r2_score`/`ols_fit` helpers are fully generic
  (reused here directly, unchanged) - but it's an *extension* specifically
  of the m-pathway: it treats `m` as the approximated-by-concepts term and
  `z` as the exact, never-approximated one. Here the roles are reversed -
  `z` is what this script's NMF fit approximates, `m` is read exactly (no
  concept decomposition of `m` happens in this script at all) - different
  enough throughout the main loop (which target is "approximated" changes
  which comparisons even make sense) that swapping which script is the
  "base" and which is the "extension" isn't just a config change.

**OLS gap** (eq:ols): `tile_r2_check.py`'s plug-in prediction only. This adds
an actual OLS regression of `z @ Theta_z^T` on the same `varphi` (no `sigma_z`
involved) and reports both R^2s side by side - R^2(OLS) >= R^2(plug-in)
always, so the gap separates "the information isn't in varphi at all" (gap
~0, raise K) from "residual leakage" (gap large - varphi carries the
information but sigma_z isn't the reason, see concept_mil.tex's "Relation to
the fitted surrogate").

**m-only baseline / pathway share**: the z_i-side symmetric question to
`tile_r2_ols_check.py`'s CLS-share check. Both predictions are also checked
against the FULL tile logit `Theta_m . m_i + Theta_z . z_i` (not just the
z-driven term in isolation). Because `Theta_m . m_i` is *exact* here (`m` is
read directly, never approximated - the reverse of the m-pathway script,
where `Theta_z . z_i` was the exact side), it cancels out of every residual
identically; only the R^2 *denominator* changes. An "m-only" baseline -
predicting the full target using *only* the exact m term, pretending the
z-concept pathway contributes nothing - calibrates how much of that
denominator the m pathway already accounts for by itself.

**"Concept-only" full-target check** (`r2_zonly_vs_full`,
`r2_plugin_conceptonly_vs_full`, `r2_ols_conceptonly_vs_full`,
`conceptonly_gap_vs_full`): answers a different question than
`r2_plugin_vs_full`/`r2_ols_vs_full` above, which keep the *real* `m_target`
and so only measure the cost of the z-approximation *given the model still
has full access to m*. Here `mean(m_target)` (a single constant over the
whole sample, carrying zero per-tile information - nothing about `m` leaks
through it) stands in for `m` entirely, isolating "how good is the model if
*all* information has to flow through the concept bottleneck, with no real
signal from the m pathway at all." `r2_zonly_vs_full` (exact `z`, not the
concept approximation, still with `m` blanked out) is the ceiling any
concept-only approach could reach; `conceptonly_gap_vs_full` is how much of
that ceiling `plugin_pred`'s K-concept bottleneck actually recovers - the
same gap relationship as `ols_gap_vs_z`, one level up.

**No shift/scale transform for z_i**: `Theta_z` acts on the raw CLS
embedding directly - concept_mil.tex's non-negativity transform (Sec.
"Non-negativity transform (NMF only)") applies only because the concept
dictionary must live in patch-token space (R1); it says nothing about how
`Theta_z` reads `z_i` itself. The saved `varphi` (`w.f32.npy`) is already in
the right space too - it's `nmf_fit.py`'s own final-pass transform output,
computed once against the shift-only (non-negativity-transformed) CLS tokens
and the fully-recovered, gauge-fixed `H` - nothing here re-derives or
re-transforms it.

**`m` reused from an existing kind=patch `tile_r2_check.py` run**, not
recomputed: `m` (mean-pooled, shift-only patch embedding) doesn't depend on
this script's cls dictionary at all, only on grade and the patch-side shift -
re-streaming the patch corpus just to get `m` again would defeat the entire
point of this script being cheap. Inherits that run's shift-only space (see
its own module docstring, "Uses raw (uncentered) sigma") - the resulting
per-class constant offset is consistent across every target/prediction pair
here too, so it drops out of every R^2 (variance-based, offset-invariant)
without needing to be undone, exactly as in the m-pathway script.
"""

import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import ray.data
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

from explainability.sigma_table import load_h
from explainability.theta_m_check import load_classifier
from explainability.tile_r2_ols_check import load_cls_features, ols_fit, r2_score
from explainability.tiles import resolve_grade_token_dir


def load_varphi(w_dir: Path, n_components: int) -> pd.DataFrame:
    """Load a kind=cls nmf_fit.py run's saved per-tile W as `slide_id, x, y, varphi`.

    Unlike kind=patch, no streaming/mean-pooling is needed here - a kind=cls
    run's `w.f32.npy` already has exactly one row per tile (see module
    docstring), so this is a direct read of two small-ish files already on
    the project mount, not a `ray.data` pass over anything.

    Args:
        w_dir: A specific `nmf_fit.py` (kind=cls) run's `output_dir` -
            holds `w.f32.npy` and `w_metadata.parquet`.
        n_components: Expected width of `w.f32.npy`'s second axis - checked
            against the file, not assumed.

    Returns:
        One row per tile: `slide_id`, `x`, `y`, `varphi` (list of
        `n_components` floats).
    """
    w = np.asarray(np.load(w_dir / "w.f32.npy", mmap_mode="r"))
    metadata = pd.read_parquet(w_dir / "w_metadata.parquet")
    if w.shape[1] != n_components:
        raise ValueError(f"{w_dir / 'w.f32.npy'} has width {w.shape[1]}, expected {n_components}")
    if len(metadata) != w.shape[0]:
        raise ValueError(
            f"w_metadata.parquet has {len(metadata)} rows but w.f32.npy has {w.shape[0]}"
        )
    return pd.DataFrame(
        {"slide_id": metadata["slide_id"], "x": metadata["x"], "y": metadata["y"], "varphi": list(w)}
    )


def load_m_features(tile_features_path: str) -> pd.DataFrame:
    """Load just `slide_id, x, y, m` from an existing kind=patch tile_r2_check.py run.

    See module docstring ("m reused from an existing kind=patch
    tile_r2_check.py run") for why this is a read, not a recomputation -
    that run's own `varphi` column (its m-pathway concept weights) is
    irrelevant here and dropped.
    """
    return pd.read_parquet(tile_features_path, columns=["slide_id", "x", "y", "m"])


def merge_or_warn(base: pd.DataFrame, other: pd.DataFrame, other_name: str) -> pd.DataFrame:
    """Inner-join on (slide_id, x, y), warning (not failing) on a row-count surprise.

    A smaller merged count means some `base` tile lacked a matching row in
    `other`; a larger one means a duplicate (slide_id, x, y) key fanned the
    join out on one side - both are surprising enough to flag rather than
    silently reconcile, matching `tile_r2_ols_check.py`'s own merge check.
    """
    merged = base.merge(other, on=["slide_id", "x", "y"], how="inner")
    if len(merged) != len(base):
        print(
            f"WARNING: {len(base)} tiles merged with {other_name} gave {len(merged)} rows - "
            "check for missing or duplicate (slide_id, x, y) keys.",
            flush=True,
        )
    return merged


@with_cli_args(["+explainability=tile_r2_check_cls"])
@hydra.main(config_path="../configs", config_name="explainability", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    h = load_h(config.h.mlflow_uri)
    if h.shape[0] != config.n_components:
        raise ValueError(
            f"h.parquet at {config.h.mlflow_uri} has {h.shape[0]} rows, "
            f"expected n_components={config.n_components}"
        )

    varphi_df = load_varphi(Path(config.w_dir), config.n_components)
    print(f"Loaded {len(varphi_df)} tiles' varphi from {config.w_dir}", flush=True)

    cls_dir = resolve_grade_token_dir(
        config.get("local_grade_split_dir"), config.grade_split.mlflow_uri, kind="cls", grade=config.grade
    )
    cls_df = load_cls_features(cls_dir)
    print(f"Loaded {len(cls_df)} grade={config.grade} CLS tokens from {cls_dir}", flush=True)

    m_df = load_m_features(config.m_features_path)
    print(f"Loaded {len(m_df)} tiles' m from {config.m_features_path}", flush=True)

    merged = merge_or_warn(varphi_df, cls_df, "CLS tokens")
    merged = merge_or_warn(merged, m_df, "m features")

    varphi = np.stack(merged["varphi"].to_numpy())
    z = np.stack(merged["z"].to_numpy())
    m = np.stack(merged["m"].to_numpy())

    rows = []
    for head, checkpoint_cfg in config.checkpoints.items():
        weight, _bias = load_classifier(checkpoint_cfg.checkpoint, config.embed_dim)
        theta_z = weight[:, : config.embed_dim]
        theta_m = weight[:, config.embed_dim :]
        sigma_z = h @ theta_z.T  # z_i-side analogue of eq. 2.9

        z_target = z @ theta_z.T  # (n, C) - the term this script's NMF approximates
        m_target = m @ theta_m.T  # (n, C) - exact, never approximated here
        full_target = z_target + m_target
        plugin_pred = varphi @ sigma_z

        num_classes = theta_z.shape[0]
        for c in range(num_classes):
            ols_pred_c = ols_fit(varphi, z_target[:, c])
            mean_m_c = m_target[:, c].mean()
            # m-only baseline: predicts the full target using only the exact
            # m term, plus mean(z_target) standing in for the unknown
            # z-concept contribution - the fair "I know nothing about which
            # concepts this tile has" stand-in, symmetric to
            # tile_r2_ols_check.py's clsonly_pred.
            monly_pred_c = m_target[:, c] + z_target[:, c].mean()
            # "Concept-only" predictions: mean(m_target) is a single constant
            # over the whole sample, carrying zero per-tile information, so
            # nothing about m leaks through it - what's left in each residual
            # is exactly (a) how wrong the z-side prediction is, and (b) how
            # much m naturally varies that a concept-only model has no way to
            # know, which is exactly "how good is the model if everything
            # flows through concepts". zonly (exact z) is the ceiling any
            # concept-only approach could reach; plugin/ols measure how close
            # the actual K-concept bottleneck / OLS fit get to it.
            zonly_pred_c = z_target[:, c] + mean_m_c
            plugin_conceptonly_pred_c = plugin_pred[:, c] + mean_m_c
            ols_conceptonly_pred_c = ols_pred_c + mean_m_c
            true_c, pred_c = z_target[:, c], plugin_pred[:, c]
            ss_tot = float(np.sum((true_c - true_c.mean()) ** 2))
            pearson_r_plugin = (
                float(np.corrcoef(true_c, pred_c)[0, 1]) if ss_tot > 0 else float("nan")
            )
            row = {
                "head": head,
                "class": c,
                "n_tiles": len(merged),
                "r2_plugin_vs_z": r2_score(z_target[:, c], plugin_pred[:, c]),
                "pearson_r_plugin_vs_z": pearson_r_plugin,
                "r2_ols_vs_z": r2_score(z_target[:, c], ols_pred_c),
                "r2_monly_vs_full": r2_score(full_target[:, c], monly_pred_c),
                "r2_plugin_vs_full": r2_score(full_target[:, c], m_target[:, c] + plugin_pred[:, c]),
                "r2_ols_vs_full": r2_score(full_target[:, c], m_target[:, c] + ols_pred_c),
                "r2_zonly_vs_full": r2_score(full_target[:, c], zonly_pred_c),
                "r2_plugin_conceptonly_vs_full": r2_score(
                    full_target[:, c], plugin_conceptonly_pred_c
                ),
                "r2_ols_conceptonly_vs_full": r2_score(full_target[:, c], ols_conceptonly_pred_c),
            }
            row["ols_gap_vs_z"] = row["r2_ols_vs_z"] - row["r2_plugin_vs_z"]
            row["conceptonly_gap_vs_full"] = (
                row["r2_zonly_vs_full"] - row["r2_plugin_conceptonly_vs_full"]
            )
            rows.append(row)

        print(f"=== {head} ===", flush=True)
        head_df = pd.DataFrame([r for r in rows if r["head"] == head])
        print(head_df.to_string(index=False), flush=True)

    result_df = pd.DataFrame(rows)
    print("\n=== pooled plug-in R^2 across heads/classes ===", flush=True)
    print(
        f"mean R^2 = {result_df['r2_plugin_vs_z'].mean():.4f}  "
        f"min R^2 = {result_df['r2_plugin_vs_z'].min():.4f}",
        flush=True,
    )

    result_path = output_dir / "r2.parquet"
    result_df.to_parquet(result_path, index=False)

    manifest = {
        "grade": config.grade,
        "n_components": config.n_components,
        "n_tiles": len(merged),
        "r2_by_head_class": result_df.to_dict(orient="records"),
        "mean_r2_plugin_vs_z": float(result_df["r2_plugin_vs_z"].mean()),
        "min_r2_plugin_vs_z": float(result_df["r2_plugin_vs_z"].min()),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    logger.log_artifact(str(result_path))
    logger.log_artifact(str(manifest_path))
    for row in rows:
        tag = f"{row['head']}_class{row['class']}"
        logger.log_metrics(
            {
                f"r2_plugin_vs_z/{tag}": row["r2_plugin_vs_z"],
                f"pearson_r_plugin_vs_z/{tag}": row["pearson_r_plugin_vs_z"],
                f"r2_ols_vs_z/{tag}": row["r2_ols_vs_z"],
                f"ols_gap_vs_z/{tag}": row["ols_gap_vs_z"],
                f"r2_monly_vs_full/{tag}": row["r2_monly_vs_full"],
                f"r2_plugin_vs_full/{tag}": row["r2_plugin_vs_full"],
                f"r2_ols_vs_full/{tag}": row["r2_ols_vs_full"],
                f"r2_zonly_vs_full/{tag}": row["r2_zonly_vs_full"],
                f"r2_plugin_conceptonly_vs_full/{tag}": row["r2_plugin_conceptonly_vs_full"],
                f"r2_ols_conceptonly_vs_full/{tag}": row["r2_ols_conceptonly_vs_full"],
                f"conceptonly_gap_vs_full/{tag}": row["conceptonly_gap_vs_full"],
            }
        )


if __name__ == "__main__":
    # No num_cpus pin needed (unlike tile_r2_check.py/nmf_fit.py/
    # token_statistics.py): the only ray.data use here is load_cls_features's
    # read of one grade's small CLS partition (one row per tile), not a
    # streaming pass over the oversized-row-group patch tables - same
    # lightweight shape as tile_r2_ols_check.py's own __main__.
    with ray.init(runtime_env={"excludes": [".git", ".venv"]}):
        main()
