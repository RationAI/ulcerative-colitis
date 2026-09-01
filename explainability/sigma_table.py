"""Step 1 of concept_mil.tex's order-of-work table: sigma tables + readout coverage.

Both quantities are "weights only" (concept_mil.tex Sec. 4, step 1 of the
order-of-work table in Sec. 8): they need only a fitted dictionary `H` and a
trained classifier's `Theta_m`, no patch data read and no NMF re-run. This
script sweeps every `H` in the n_components x scale_power grid
(`configs/explainability/nmf_fit.yaml`'s sweep) against all three task heads.

**Sigma table** (eq. 2.9, "boxed" in the note):
    sigma = H @ Theta_m^T in R^{K x C}
per-unit-concept response of each class logit, read directly off the trained
weights - no regression, no labels, no data. `H` here is already the fully
recovered (scale multiplied back in) and gauge-fixed dictionary `nmf_fit.py`
writes to `h.parquet` - i.e. exactly the `H_k` of eq. 2.24, not the
scaled-fit `H~_k`. That means sigma_table needs neither `shift` nor `scale`
(patch_statistics' output) itself: the non-negativity transform's `-c` term
was already absorbed into the classifier bias per eq. 2.24, which doesn't
appear in sigma at all.

**Readout coverage** (eq. 2.10 in Sec. 4, "Q4"): for class c, the fraction of
`Theta_m[c,:]`'s squared norm that lies inside span(H) -
    cov_c = || Theta_m[c,:] @ Q Q^T ||^2 / || Theta_m[c,:] ||^2  in [0,1]
with Q an orthonormal basis of span(H) (via SVD of H, not just H's own unit-
norm rows - H's rows are gauge-fixed but not orthogonal to each other, so a
proper orthonormal basis is needed for a projection). This is the note's own
"appropriate criterion for choosing K" (Sec. 4): low coverage means the
concept basis cannot express what the classifier reads, regardless of how
patches actually land on it, and K should be increased. Row-centering (the
note's remark below eq. 3.10, needed before *sigma*'s signs are meaningful
for the two multi-class heads) does not apply to coverage - projection is
invariant to which direction within R^C is examined.

**Assumed, not verified** (same caveat as theta_m_check.py, whose
`load_theta_m` this script reuses): `classifier.weight`'s columns are
ordered `[z_i; m_i]`, and its rows follow whatever class-index convention
each head's own training config used - this script reports plain integer
class indices `0..C-1`, not the note's own {Lo,Hi}/{0,1,Hi}/{Lo,2,3,4} class
labels (Sec. 3.6), since that label ordering was never checked against this
repo's training code.
"""

import json
from pathlib import Path

import hydra
import mlflow.artifacts
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

from explainability.theta_m_check import load_theta_m


def load_h(mlflow_uri: str) -> np.ndarray:
    """Download and load an nmf_fit.py h.parquet as a plain (n_components, embed_dim) array.

    `h.parquet` is written by `pd.DataFrame(h).rename_axis("component").to_parquet`
    (see explainability/nmf_fit.py), so the row index is "component" and the
    columns are the raw integer feature dimensions - sorted on both axes here
    since parquet round-tripping doesn't guarantee either stays ordered.
    """
    path = mlflow.artifacts.download_artifacts(mlflow_uri)
    df = pd.read_parquet(path).sort_index(axis=0).sort_index(axis=1)
    return df.to_numpy(dtype=np.float64)


def orthonormal_basis(h: np.ndarray) -> tuple[np.ndarray, int]:
    """Orthonormal basis of span(H) via SVD, plus the numerically effective rank used.

    H's gauge-fixed rows are unit norm but not mutually orthogonal, so they
    aren't themselves a valid basis for a projection. Right singular vectors
    with a near-zero singular value don't actually lie in span(H) (SVD only
    guarantees they complete an orthonormal basis of R^D) - dropping them
    matters when concepts are near-collinear (a real possibility at higher
    K, see Q5), since including them would inflate coverage with directions
    the dictionary doesn't really span.

    Args:
        h: Dictionary matrix, shape (n_components, embed_dim).

    Returns:
        (q, rank): q has shape (embed_dim, rank), orthonormal columns
        spanning the same space as H's rows; rank <= n_components, equal to
        n_components unless H is numerically rank-deficient.
    """
    _, s, vt = np.linalg.svd(h, full_matrices=False)
    tol = s.max() * max(h.shape) * np.finfo(h.dtype).eps
    keep = s > tol
    return vt[keep].T, int(keep.sum())


def readout_coverage(theta_m: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Per-class projection of Theta_m onto span(H) (eq. 2.10): ||Theta_m[c,:] Q||^2 / ||Theta_m[c,:]||^2.

    ||v @ Q @ Q.T||^2 == ||v @ Q||^2 for orthonormal-column Q (Q.T @ Q = I),
    so projecting onto Q's column space directly avoids ever forming the
    (embed_dim, embed_dim) projector Q @ Q.T.

    Args:
        theta_m: Shape (num_classes, embed_dim).
        q: Orthonormal basis of span(H), shape (embed_dim, rank) - see
            `orthonormal_basis`.

    Returns:
        Shape (num_classes,), each entry in [0, 1].
    """
    projected = theta_m @ q
    return np.sum(projected**2, axis=1) / np.sum(theta_m**2, axis=1)


def row_center(sigma: np.ndarray) -> np.ndarray:
    """Center each concept's sigma row across classes (remark below eq. 3.10).

    Only the multi-class heads need this before signs are interpretable -
    for a softmax head, only *differences* between rows of Theta are
    identified, so a constant per-concept shift is otherwise read as signal.
    The binary head has one effective direction and is left uncentered by
    the caller (num_classes == 2 skips this).
    """
    return sigma - sigma.mean(axis=1, keepdims=True)


@with_cli_args(["+explainability=sigma_table"])
@hydra.main(config_path="../configs", config_name="explainability", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Theta_m per head - loaded once, reused across every (n_components,
    # scale_power) combo in the sweep below (unlike H, it doesn't depend on
    # either knob).
    theta_ms = {
        name: load_theta_m(checkpoint_cfg.checkpoint, config.embed_dim)
        for name, checkpoint_cfg in config.checkpoints.items()
    }

    sigma_rows: list[dict[str, float | int | str | None]] = []
    coverage_rows: list[dict[str, float | int | str]] = []

    for n_components_str, scale_powers in config.h.items():
        n_components = int(n_components_str)
        for scale_power_str, entry in scale_powers.items():
            scale_power = float(scale_power_str)
            h = load_h(entry.mlflow_uri)
            if h.shape[0] != n_components:
                raise ValueError(
                    f"h.parquet at {entry.mlflow_uri} has {h.shape[0]} rows, "
                    f"expected n_components={n_components}"
                )
            q, rank = orthonormal_basis(h)
            if rank < n_components:
                print(
                    f"WARNING k={n_components} scale_power={scale_power}: "
                    f"H's numerical rank is {rank} < {n_components} "
                    "(near-collinear concepts) - coverage below is computed "
                    "against the reduced span.",
                    flush=True,
                )

            combo_coverage: list[float] = []
            for head, theta_m in theta_ms.items():
                sigma = h @ theta_m.T  # eq. 2.9: (n_components, num_classes)
                num_classes = theta_m.shape[0]
                centered = row_center(sigma) if num_classes > 2 else None

                for k in range(n_components):
                    for c in range(num_classes):
                        sigma_rows.append(
                            {
                                "n_components": n_components,
                                "scale_power": scale_power,
                                "head": head,
                                "concept": k,
                                "class": c,
                                "sigma": float(sigma[k, c]),
                                "sigma_centered": (
                                    float(centered[k, c]) if centered is not None else None
                                ),
                            }
                        )

                coverage = readout_coverage(theta_m, q)
                combo_coverage.extend(coverage.tolist())
                for c, cov in enumerate(coverage):
                    coverage_rows.append(
                        {
                            "n_components": n_components,
                            "scale_power": scale_power,
                            "head": head,
                            "class": c,
                            "coverage": float(cov),
                            "h_rank": rank,
                        }
                    )

            print(
                f"k={n_components:2d}  scale_power={scale_power:.1f}  "
                f"mean coverage={np.mean(combo_coverage):.3f}  "
                f"min coverage={np.min(combo_coverage):.3f}",
                flush=True,
            )

    sigma_df = pd.DataFrame(sigma_rows)
    coverage_df = pd.DataFrame(coverage_rows)

    sigma_path = output_dir / "sigma.parquet"
    coverage_path = output_dir / "coverage.parquet"
    sigma_df.to_parquet(sigma_path, index=False)
    coverage_df.to_parquet(coverage_path, index=False)

    # Pooled per-combo mean coverage (across heads and classes) - the single
    # number the note's own criterion for choosing K (Sec. 4) reduces to:
    # low pooled coverage means the concept basis can't express what the
    # classifiers read, regardless of scale_power, and K should be raised.
    pooled = (
        coverage_df.groupby(["n_components", "scale_power"])["coverage"]
        .agg(["mean", "min"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    print("\n=== pooled coverage across heads/classes, best first ===", flush=True)
    print(pooled.to_string(index=False), flush=True)

    manifest = {
        "n_sigma_rows": len(sigma_df),
        "n_coverage_rows": len(coverage_df),
        "pooled_coverage_by_combo": pooled.to_dict(orient="records"),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    logger.log_artifact(str(sigma_path))
    logger.log_artifact(str(coverage_path))
    logger.log_artifact(str(manifest_path))
    for row in pooled.to_dict(orient="records"):
        tag = f"k{int(row['n_components'])}_sp{row['scale_power']:.1f}"
        logger.log_metrics(
            {f"coverage_mean/{tag}": row["mean"], f"coverage_min/{tag}": row["min"]}
        )


if __name__ == "__main__":
    main()
