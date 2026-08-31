"""One-off diagnostic: does nmf_fit.py's IQR scaling suppress classifier-relevant dims?

`nmf_fit.py` divides each patch-token dimension by its IQR before fitting NMF,
so no single high-variance dimension (e.g. dimension 1, observed spanning
~-53..41 vs. a typical ~-4..4) dominates the unweighted Frobenius loss. That's
safe if high-IQR dimensions are mostly non-discriminative activation
artifacts (a documented phenomenon in ViT/transformer token embeddings - see
e.g. "massive activations" / high-norm "register" tokens in the literature),
but risky if they're actually dimensions the trained classifier relies on -
in that case scaling would suppress exactly the structure NMF should be
capturing.

This script checks that empirically rather than by assumption: for each
trained MIL classifier checkpoint, it splits `classifier.weight` (shape
`(num_classes, 2*embed_dim)`) into `Theta_z` (acts on the CLS token `z_i`)
and `Theta_m` (acts on the mean-pooled patch tokens `m_i`) per
concept_mil.tex's `h_i = [z_i; m_i]` notation, and correlates each patch
dimension's *contribution to the logit* against that dimension's IQR from
`patch_statistics.py`'s output. Contribution is `|Theta_m_j| * IQR_j`, not
raw `|Theta_m_j|` alone: a dimension's actual pull on the logit depends on
both its weight and how much it varies across real patches (IQR is
linear-equivariant, so this product is exactly the IQR of that dimension's
own raw term `Theta_m_j * m_j`), and comparing raw weights alone across
dimensions of very different native scale isn't apples-to-apples - see
`theta_m_contribution`'s docstring.

**Assumed, not verified**: that `classifier.weight`'s columns are ordered
`[z_i; m_i]` (CLS first, mean-pooled patches second) - matches the
convention already used in concept_mil.tex and explainability/nmf_fit.py,
but the model server that pools tokens into this checkpoint's training
embeddings (`preprocessing/embeddings.py` on branch `thesis`, `pool_tokens`
left at its server-side default) lives outside both repos, so this ordering
was never directly confirmed against its source. If results here look
inverted or nonsensical, check this first.

**Also approximate, not exact**: `Theta_m`'s classifiers were trained on
embeddings from the `thesis` branch's own tiling/embedding run, not on this
repo's `embeddings_xai.py` output that `patch_statistics.py`'s IQR was
computed from. Both use the same Virchow2 model over largely the same
slides, so per-dimension IQR should transfer closely, but this is a quick
diagnostic, not a from-first-principles guarantee.
"""

import json
from pathlib import Path
from typing import Any

import hydra
import mlflow.artifacts
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from scipy.stats import pearsonr, spearmanr

from explainability.nmf_fit import load_scale, resolve_percentile_stats_path


def load_theta_m(checkpoint_uri: str, embed_dim: int) -> np.ndarray:
    """Download a MIL checkpoint and pull out the mean-pooled-token classifier weight.

    Args:
        checkpoint_uri: mlflow artifact URI for a lightning `checkpoint.ckpt`
            (e.g. from `configs/checkpoints/final/*.yaml`).
        embed_dim: Width of `z_i`/`m_i` each (1280 for Virchow2 patch
            tokens) - `classifier.weight` must be `(num_classes, 2*embed_dim)`.

    Returns:
        `Theta_m`, shape `(num_classes, embed_dim)`: the second half of
        `classifier.weight`'s columns, per `h_i = [z_i; m_i]`.
    """
    checkpoint_path = mlflow.artifacts.download_artifacts(checkpoint_uri)
    # weights_only=False: lightning checkpoints bundle optimizer/callback
    # state alongside the tensors, so a strict weights-only unpickle can't
    # load them - trusted source (our own mlflow-logged checkpoints).
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)[
        "state_dict"
    ]
    weight = state_dict["classifier.weight"].numpy()
    if weight.shape[1] != 2 * embed_dim:
        raise ValueError(
            f"classifier.weight has {weight.shape[1]} input columns, expected "
            f"2*embed_dim={2 * embed_dim}"
        )
    return weight[:, embed_dim:]


def theta_m_contribution(theta_m: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Per-dimension contribution to logit variation: weight x how much that dim varies.

    Raw |Theta_m_j| alone isn't comparable across dimensions of different
    natural scale: a dimension with std 20 and weight 1 moves the logit
    ~20x more, across real patches, than a dimension with std 1 and weight
    1, even though both have "the same weight". IQR is linear-equivariant
    under monotonic rescaling (IQR(a*X) = |a|*IQR(X)), so
    `|Theta_m_j| * scale_j` is exactly the IQR of that one dimension's raw
    contribution `Theta_m_j * m_j` to the logit - a fair, same-units
    "importance" across dimensions regardless of their native scale, unlike
    the weight alone.

    Args:
        theta_m: Shape `(num_classes, embed_dim)`.
        scale: Per-dimension IQR, shape `(embed_dim,)` (see `load_scale`).

    Returns:
        Shape `(embed_dim,)`: L2 norm across classes/logits of
        `theta_m * scale` - for a binary head (num_classes=1, as in the
        neutrophils checkpoint) this is just `|Theta_m_j| * scale_j`.
    """
    return np.linalg.norm(theta_m * scale[None, :], axis=0)


def correlate(scale: np.ndarray, contribution: np.ndarray) -> dict[str, float]:
    """Pearson (linear) and Spearman (monotonic/rank) correlation, IQR vs. contribution.

    Both are reported: Pearson alone can be dominated by the handful of
    extreme-IQR dimensions this whole check is about, so Spearman (rank-
    based, insensitive to their exact magnitude) is the more trustworthy
    read on whether "high IQR" and "high contribution" tend to co-occur.
    """
    pearson_r, pearson_p = pearsonr(scale, contribution)
    spearman_r, spearman_p = spearmanr(scale, contribution)
    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
    }


@with_cli_args(["+explainability=theta_m_check"])
@hydra.main(config_path="../configs", config_name="explainability", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    stats_path = resolve_percentile_stats_path(config.shift.mlflow_uri)
    scale = load_scale(stats_path)  # IQR per patch-token dimension, shape (embed_dim,)

    results: dict[str, Any] = {}
    contributions: dict[str, np.ndarray] = {}
    for name, checkpoint_cfg in config.checkpoints.items():
        theta_m = load_theta_m(checkpoint_cfg.checkpoint, config.embed_dim)
        contribution = theta_m_contribution(theta_m, scale)
        contributions[name] = contribution
        stats = correlate(scale, contribution)

        contribution_rank = pd.Series(contribution).rank(pct=True).to_numpy()
        iqr_rank = pd.Series(scale).rank(pct=True).to_numpy()
        top_iqr_dims = np.argsort(scale)[::-1][:10]
        top_contribution_dims = np.argsort(contribution)[::-1][:10]

        print(f"=== {name} ===", flush=True)
        print(
            f"pearson(IQR, contribution)  r={stats['pearson_r']:+.3f}  p={stats['pearson_p']:.2e}",
            flush=True,
        )
        print(
            f"spearman(IQR, contribution) r={stats['spearman_r']:+.3f}  p={stats['spearman_p']:.2e}",
            flush=True,
        )
        print("top-10 highest-IQR dims -> their contribution percentile rank:", flush=True)
        for dim in top_iqr_dims:
            print(
                f"  dim {dim:4d}  IQR={scale[dim]:8.3f}  "
                f"contribution percentile={contribution_rank[dim] * 100:5.1f}",
                flush=True,
            )
        print(
            "top-10 highest-contribution dims (|Theta_m|*IQR) -> their IQR percentile rank:",
            flush=True,
        )
        for dim in top_contribution_dims:
            print(
                f"  dim {dim:4d}  contribution={contribution[dim]:8.3f}  "
                f"IQR percentile={iqr_rank[dim] * 100:5.1f}",
                flush=True,
            )

        results[name] = {
            **stats,
            "top_iqr_dims": [
                {
                    "dim": int(dim),
                    "iqr": float(scale[dim]),
                    "contribution_percentile": float(contribution_rank[dim] * 100),
                }
                for dim in top_iqr_dims
            ],
            "top_contribution_dims": [
                {
                    "dim": int(dim),
                    "contribution": float(contribution[dim]),
                    "iqr_percentile": float(iqr_rank[dim] * 100),
                }
                for dim in top_contribution_dims
            ],
        }
        logger.log_metrics(
            {f"{name}/pearson_r": stats["pearson_r"], f"{name}/spearman_r": stats["spearman_r"]}
        )

    # Pooled across all three tasks: a dimension that's high-IQR *and*
    # consistently high-contribution across nancy_high/nancy_low/neutrophils
    # is a much stronger case for revisiting the scaling than one that only
    # matters to a single task.
    pooled_contribution = np.mean(np.stack(list(contributions.values())), axis=0)
    pooled_stats = correlate(scale, pooled_contribution)
    print("=== pooled (mean contribution across checkpoints) ===", flush=True)
    print(
        f"pearson  r={pooled_stats['pearson_r']:+.3f}  p={pooled_stats['pearson_p']:.2e}",
        flush=True,
    )
    print(
        f"spearman r={pooled_stats['spearman_r']:+.3f}  p={pooled_stats['spearman_p']:.2e}",
        flush=True,
    )
    results["pooled"] = pooled_stats
    print(
        "\nRule of thumb: contribution_j = |Theta_m_j| * IQR_j is the IQR of "
        "dimension j's own raw pull on the logit (Theta_m_j * m_j), so it's "
        "comparable across dimensions of different native scale, unlike "
        "|Theta_m_j| alone. A strong *positive* correlation here means "
        "high-IQR dims also pull the logit around the most - i.e. the IQR "
        "scaling is suppressing exactly what the classifier relies on, and "
        "the scaling scheme is worth revisiting (e.g. importance-aware "
        "weighting) per the discussion in explainability-status memory. "
        "Weak/no correlation supports the current scaling being safe.",
        flush=True,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "theta_m_iqr_correlation.json"
    output_path.write_text(json.dumps(results, indent=2))
    logger.log_artifact(str(output_path))


if __name__ == "__main__":
    main()
