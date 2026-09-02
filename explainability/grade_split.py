"""Split every xai patch/cls token by its *own tile's* model-predicted Nancy grade (0-4).

First step of grading NMF concepts per Nancy grade level (basic or
hierarchical NMF, decided later - this only prepares the 5 per-grade
datasets, it doesn't fit anything itself).

**Predictions are per-tile, not per-slide, and computed directly from the
trained classifiers - not read from any mlflow-logged MIL prediction
table.** A slide-level MIL prediction (attention-pooled over all of a
slide's tiles) would be *wrong* to broadcast onto every one of that slide's
tiles: a slide with e.g. Nancy grade 4 still has plenty of normal-looking
tissue tiles, and tagging all of them "grade 4" would badly mislabel the
patches NMF later has to organize by grade (explicit user instruction).

Instead, this script uses exactly the tile-level linear read `c_psi` that
concept_mil.tex's whole closed-form argument is built on (`s_i = Theta h_i +
b`, `h_i = [z_i; m_i]`): each tile's own CLS token `z_i` and mean-pooled
patch tokens `m_i` (already present in the xai token tables) are run through
each of the three trained classifiers' raw weights (`explainability.
theta_m_check.load_classifier`, same checkpoints `sigma_table.py`/
`theta_m_check.py` already use) to get a tile-level probability per task -
no MIL attention pooling, no mlflow prediction artifacts, no dependency on
predictions ever having been run for this exact slide set.

Those three per-tile probabilities (neutrophils, nancy_low, nancy_high) are
then combined into one 0-4 grade via `explainability.postprocessing.
route_grade` - the `pred_ensembling` (soft majority vote) rule ported from
the `thesis` branch's `postprocessing/ensembling_predict.py` (user's
explicit choice over `pred_hierarchical` or the markov-chain soft
distribution).

**Pipeline:**
1. Stream every patch token once, mean-pooling by tile (`slide_id, x, y`)
   via a custom `ray.data` `AggregateFn` - the same oversized-row-group OOM
   risk as `patch_statistics.py`/`nmf_fit.py` applies here (this reads the
   full patch corpus), so `ray.init(num_cpus=8, ...)` is used the same way.
2. Load cls tokens (one row per tile already - no aggregation needed) and
   merge them against the per-tile patch means on `(slide_id, x, y)` with a
   plain pandas merge, not `ray.data`'s `Dataset.join()` - both sides are
   per-*tile* sized (thousands, not tens of millions of rows) by this point,
   so there's nothing left for a distributed join to buy; see
   `build_tile_features`'s docstring for why `Dataset.join()` was tried
   first and dropped.
3. Compute each task's tile-level logits/probabilities in plain numpy (small
   data by this point) and route through `route_grade`.
4. Broadcast-join the small per-tile grade table back onto the *full* patch
   (and cls) token datasets via `map_batches` - deliberately avoids ever
   materializing the full patch corpus for this lookup (unlike a
   `Dataset.join()` would require) - then `write_parquet(partition_cols=
   ["grade"])`, same partitioned-write pattern `preprocessing/
   embeddings_xai.py` already uses for `kind`.

**Never run against the real corpus in this session** (standing constraint -
see explainability-status memory): only validated with small synthetic
`ray.data` datasets, plus `ruff`/`mypy strict`.
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
from ray.data.aggregate import AggregateFn
from scipy.special import expit, softmax

from explainability.postprocessing import route_grade
from explainability.theta_m_check import load_classifier
from explainability.tiles import load_tokens_dataset, resolve_token_dirs


def mean_pool_patches(patches_ds: ray.data.Dataset, embed_dim: int) -> ray.data.Dataset:
    """Stream every patch token once, mean-pooling `embedding` by tile `(slide_id, x, y)`.

    This is `m_i` (concept_mil.tex notation): the mean of a tile's non-CLS
    patch tokens. A custom `AggregateFn` (not `Dataset.groupby(...).mean()`,
    which only handles scalar columns) is needed because `embedding` is a
    `list<double>` column, not a scalar - verified directly against this
    repo's installed ray (2.53.0) with a synthetic dataset before relying on
    it here.

    Args:
        patches_ds: Patch token dataset (`kind="patch"`), as returned by
            `explainability.tiles.load_tokens_dataset`.
        embed_dim: Width of one patch token's `embedding` (1280 for
            Virchow2).

    Returns:
        One row per tile: `slide_id`, `x`, `y`, and `m` (the mean-pooled
        embedding, as a plain list - not yet a numpy array; small enough by
        this point that the caller materializes it eagerly).
    """

    def init(_key: tuple[Any, ...]) -> tuple[np.ndarray, int]:
        return np.zeros(embed_dim, dtype=np.float64), 0

    def accumulate_row(
        acc: tuple[np.ndarray, int], row: dict[str, Any]
    ) -> tuple[np.ndarray, int]:
        total, count = acc
        return total + np.asarray(row["embedding"], dtype=np.float64), count + 1

    def merge(
        a: tuple[np.ndarray, int], b: tuple[np.ndarray, int]
    ) -> tuple[np.ndarray, int]:
        return a[0] + b[0], a[1] + b[1]

    def finalize(acc: tuple[np.ndarray, int]) -> list[float]:
        total, count = acc
        return (total / count).tolist()

    agg = AggregateFn(
        init=init, accumulate_row=accumulate_row, merge=merge, finalize=finalize, name="m"
    )
    return patches_ds.select_columns(["slide_id", "x", "y", "embedding"]).groupby(
        ["slide_id", "x", "y"]
    ).aggregate(agg)


def build_tile_features(patch_means: ray.data.Dataset, cls_ds: ray.data.Dataset) -> pd.DataFrame:
    """Join per-tile mean-pooled patches with per-tile CLS tokens into `h_i = [z_i; m_i]`.

    A plain pandas merge, not `Dataset.join()`: both sides are already
    per-*tile* (thousands of rows, not the raw patch corpus), so the whole
    point of a distributed join (avoiding materializing everything at once)
    doesn't apply here - `mean_pool_patches` already reduced the expensive
    side down to one small table. `Dataset.join()` was tried first and hung
    indefinitely on a tiny synthetic dataset in this sandbox even after
    minutes (not merely slow - confirmed via a raised timeout); root cause
    not tracked down (plausibly actor/resource contention between its
    `HashShuffleAggregator` actors and an upstream `map_batches` rename
    stage on a CPU-starved local Ray instance), but since it buys nothing at
    this scale anyway, the simpler and more robust pandas merge is used
    instead - no ray join operator, no actor scheduling risk.

    Args:
        patch_means: Output of `mean_pool_patches` (columns `slide_id, x, y, m`).
        cls_ds: CLS token dataset (`kind="cls"`), as returned by
            `explainability.tiles.load_tokens_dataset`.

    Returns:
        One row per tile: `slide_id`, `x`, `y`, `h` (shape `(2*embed_dim,)`
        numpy array, `[z_i; m_i]` concatenated per concept_mil.tex's own
        column-ordering convention - matches `theta_m_check.load_classifier`'s
        `weight` layout).
    """
    means_df = patch_means.to_pandas()
    cls_df = cls_ds.select_columns(["slide_id", "x", "y", "embedding"]).to_pandas()
    df = means_df.merge(cls_df, on=["slide_id", "x", "y"], how="inner", suffixes=("", "_cls"))
    df["h"] = [
        np.concatenate([np.asarray(z, dtype=np.float64), np.asarray(m, dtype=np.float64)])
        for z, m in zip(df["embedding"], df["m"], strict=True)
    ]
    return df[["slide_id", "x", "y", "h"]]


def compute_tile_grades(
    tile_features: pd.DataFrame,
    neutrophils: tuple[np.ndarray, np.ndarray],
    nancy_low: tuple[np.ndarray, np.ndarray],
    nancy_high: tuple[np.ndarray, np.ndarray],
) -> pd.DataFrame:
    """Run every tile's `h_i` through the three tile-level classifiers and route to a grade.

    `s_i = Theta h_i + b` (concept_mil.tex's `c_psi`) computed directly here,
    not read from any bag-level MIL prediction - see this module's docstring
    for why a slide-level prediction can't be broadcast onto individual tiles.

    Args:
        tile_features: Output of `build_tile_features`.
        neutrophils: `(weight, bias)` for the neutrophils head, from
            `explainability.theta_m_check.load_classifier` - `weight` shape
            `(1, 2*embed_dim)` (binary, one logit).
        nancy_low: Same, `weight` shape `(3, 2*embed_dim)`.
        nancy_high: Same, `weight` shape `(4, 2*embed_dim)`.

    Returns:
        `tile_features` with `slide_id, x, y` plus a new `grade` column
        (int, 0-4) and the raw per-task probabilities (`neut_prob`,
        `low_prob_0..2`, `high_prob_0..3`) for downstream diagnostics.
    """
    h = np.stack(tile_features["h"].to_numpy())

    neut_weight, neut_bias = neutrophils
    neut_prob = expit(h @ neut_weight.T + neut_bias)[:, 0]

    low_weight, low_bias = nancy_low
    low_probs = softmax(h @ low_weight.T + low_bias, axis=1)

    high_weight, high_bias = nancy_high
    high_probs = softmax(h @ high_weight.T + high_bias, axis=1)

    grade = route_grade(neut_prob, low_probs, high_probs)

    result = tile_features[["slide_id", "x", "y"]].copy()
    result["grade"] = grade
    result["neut_prob"] = neut_prob
    for c in range(low_probs.shape[1]):
        result[f"low_prob_{c}"] = low_probs[:, c]
    for c in range(high_probs.shape[1]):
        result[f"high_prob_{c}"] = high_probs[:, c]
    return result


class AssignGrade:
    """Batch-level broadcast join: tag every token row with its tile's `grade`.

    Deliberately a `map_batches` merge against a small in-memory table, not
    another `Dataset.join()` - the left side here is the *full* patch (or
    cls) token corpus (tens of millions of rows), and `Dataset.join()`
    requires materializing both inputs; broadcasting the small (per-tile)
    `tile_grade` table into each batch's own pandas merge avoids ever
    holding the full corpus in the object store at once, the same way
    `preprocessing/embeddings_xai.py`'s `EmbedTiles` broadcasts small
    per-call state into a `flat_map` rather than joining datasets.
    """

    def __init__(self, tile_grade: pd.DataFrame) -> None:
        self.tile_grade = tile_grade[["slide_id", "x", "y", "grade"]]

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        return batch.merge(self.tile_grade, on=["slide_id", "x", "y"], how="inner")


def write_grade_split(
    token_dirs: list[str], tile_grade: pd.DataFrame, output_dir: Path, kind: str
) -> None:
    """Tag one token table (patch or cls) with `grade` and write it partitioned by grade.

    Args:
        token_dirs: Per-institution token directories for this `kind`, as
            returned by `explainability.tiles.resolve_token_dirs`.
        tile_grade: Output of `compute_tile_grades` (needs at least
            `slide_id, x, y, grade`).
        output_dir: Directory to write `kind=<kind>/grade=<0..4>/*.parquet` under.
        kind: "patch" or "cls" - only used to name the output subdirectory
            (input dirs already resolved separately per kind by the caller).
    """
    ds = load_tokens_dataset(token_dirs)
    tagged = ds.map_batches(AssignGrade, batch_format="pandas", fn_constructor_args=(tile_grade,))
    tagged.write_parquet(str(output_dir / f"kind={kind}"), partition_cols=["grade"])


@with_cli_args(["+explainability=grade_split"])
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

    neutrophils = load_classifier(config.checkpoints.neutrophils.checkpoint, config.embed_dim)
    nancy_low = load_classifier(config.checkpoints.nancy_low.checkpoint, config.embed_dim)
    nancy_high = load_classifier(config.checkpoints.nancy_high.checkpoint, config.embed_dim)

    tile_grade = compute_tile_grades(tile_features, neutrophils, nancy_low, nancy_high)
    counts = tile_grade["grade"].value_counts().sort_index()
    print("Tile counts per predicted grade:", flush=True)
    print(counts.to_string(), flush=True)

    tile_grade_path = output_dir / "tile_grade.parquet"
    tile_grade.to_parquet(tile_grade_path, index=False)

    for kind, token_dirs in (("patch", patch_dirs), ("cls", cls_dirs)):
        print(f"Splitting {kind} tokens by grade...", flush=True)
        write_grade_split(token_dirs, tile_grade, output_dir / "tokens", kind)

    manifest = {
        "split": config.split,
        "n_tiles": len(tile_grade),
        "tile_counts_per_grade": {int(k): int(v) for k, v in counts.items()},
        "output_dir": str(output_dir / "tokens"),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Whole output_dir (tile_grade.parquet, manifest.json, and the
    # grade-partitioned tokens/kind=patch|cls/grade=0..4 tables) logged to
    # mlflow in one shot - same convention as preprocessing/embeddings_xai.py
    # (which logs its own, same-scale tokens/kind=patch|cls output wholesale)
    # and explainability/concept_masks.py, *not* nmf_fit.py's W memmap (which
    # deliberately stays project-mount-only, but W has no downstream reader
    # that ever falls back to mlflow for it - explainability.tiles.
    # resolve_token_dirs, unlike nmf_fit.py's W loader, does have exactly
    # that mlflow fallback, so token data has to actually be there for a job
    # without the project mount to work at all).
    logger.log_artifacts(str(output_dir))
    logger.log_metrics({f"tile_count/grade_{k}": v for k, v in manifest["tile_counts_per_grade"].items()})


if __name__ == "__main__":
    ctx = ray.data.DataContext.get_current()
    ctx.enable_rich_progress_bars = True
    ctx.use_ray_tqdm = False

    # num_cpus=8: same oversized-parquet-row-group root cause as
    # patch_statistics.py/nmf_fit.py (see their comments + explainability-
    # status memory) - mean_pool_patches reads the full patch corpus, so the
    # same fix applies. Keep in sync with cpu= in
    # scripts/explainability/grade_split.py.
    with ray.init(num_cpus=8, runtime_env={"excludes": [".git", ".venv"]}):
        main()
