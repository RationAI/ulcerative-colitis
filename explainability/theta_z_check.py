"""One-off diagnostic: is z_i's classifier-relevant structure IQR-heavy too?

Sibling to `theta_m_check.py` (see that module's docstring for the full
rationale - shared here, not repeated): both check whether a dimension's
*contribution to the logit* (`|Theta| * IQR`, not raw `|Theta|` alone, since
a dimension's actual pull on the logit depends on both its weight and how
much it varies across real tokens) correlates with that dimension's IQR.
`theta_m_check.py` asks this for the mean-pooled patch tokens `m_i`; this one
asks it for the CLS token `z_i` instead, using `Theta_z`
(`classifier.weight`'s first `embed_dim` columns, per concept_mil.tex's
`h_i = [z_i; m_i]`) and CLS-kind `token_statistics.py` percentiles.

Motivation (see explainability-status memory, 2026-09-17 entry):
`embedding_importance.py` found `z_i` dominates almost every head/class -
`m_i` only carries real, non-`z`-redundant signal for nancy_low class 1. If
`z_i` is doing most of the work, whether *its* IQR-heavy dimensions are also
the classifier-relevant ones matters more here than the equivalent question
for `m_i` - `theta_m_check.py` doesn't answer it, since `Theta_z`/`z_i`'s IQR
are a different half of the weight matrix and a different token table
entirely.

**Note: this diagnostic doesn't feed a scaling decision the way
`theta_m_check.py`'s does.** Nothing in this pipeline currently IQR-scales
`z_i` the way `nmf_fit.py` scales `m_i` before NMF (`z_i` isn't NMF'd at
all). This is read-only signal for whether that would ever be worth doing,
not a check on an existing scaling choice.

**Assumed, not verified**: same `classifier.weight` column ordering
(`[z_i; m_i]`, CLS first) as `theta_m_check.py` - see that module's docstring
for why this hasn't been directly confirmed.
"""

import json
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

from explainability.nmf_fit import load_scale, resolve_percentile_stats_path
from explainability.theta_m_check import (
    correlate,
    load_classifier,
    theta_m_contribution,
)


def load_theta_z(checkpoint_uri: str, embed_dim: int) -> np.ndarray:
    """Download a MIL checkpoint and pull out the CLS-token classifier weight.

    Args:
        checkpoint_uri: mlflow artifact URI for a lightning `checkpoint.ckpt`
            (e.g. from `configs/checkpoints/final/*.yaml`).
        embed_dim: Width of `z_i`/`m_i` each (1280 for Virchow2) -
            `classifier.weight` must be `(num_classes, 2*embed_dim)`.

    Returns:
        `Theta_z`, shape `(num_classes, embed_dim)`: the first half of
        `classifier.weight`'s columns, per `h_i = [z_i; m_i]`.
    """
    weight, _ = load_classifier(checkpoint_uri, embed_dim)
    return weight[:, :embed_dim]


@with_cli_args(["+explainability=theta_z_check"])
@hydra.main(config_path="../configs", config_name="explainability", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    stats_path = resolve_percentile_stats_path(config.shift.mlflow_uri)
    scale = load_scale(stats_path)  # IQR per CLS-token dimension, shape (embed_dim,)

    results: dict[str, Any] = {}
    contributions: dict[str, np.ndarray] = {}
    for name, checkpoint_cfg in config.checkpoints.items():
        theta_z = load_theta_z(checkpoint_cfg.checkpoint, config.embed_dim)
        contribution = theta_m_contribution(theta_z, scale)
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
            "top-10 highest-contribution dims (|Theta_z|*IQR) -> their IQR percentile rank:",
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
    # is a much stronger case for treating z_i's IQR-heavy dims as
    # classifier-relevant than one that only matters to a single task.
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
        "\nRule of thumb: contribution_j = |Theta_z_j| * IQR_j is the IQR of "
        "dimension j's own raw pull on the logit (Theta_z_j * z_j), comparable "
        "across dimensions of different native scale, unlike |Theta_z_j| alone. "
        "A strong positive correlation here means z_i's high-IQR dimensions are "
        "also the ones the classifier leans on most - i.e. z_i's dominance "
        "(embedding_importance.py) is concentrated in a few high-variance "
        "dimensions rather than spread evenly, which would matter for any "
        "future attempt to decompose or scale z_i the way m_i is scaled for "
        "NMF. Weak/no correlation means classifier-relevant z_i dimensions "
        "aren't the high-IQR ones - importance is spread more evenly, or "
        "concentrated in dims that don't stand out by IQR alone.",
        flush=True,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "theta_z_iqr_correlation.json"
    output_path.write_text(json.dumps(results, indent=2))
    logger.log_artifact(str(output_path))


if __name__ == "__main__":
    main()
