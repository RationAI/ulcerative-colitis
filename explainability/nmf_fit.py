import json
from collections.abc import Iterator
from pathlib import Path

import hydra
import mlflow.artifacts
import numpy as np
import pandas as pd
import ray
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from sklearn.decomposition import MiniBatchNMF

from explainability.tiles import load_tokens_dataset, resolve_token_dirs


def resolve_percentile_stats_path(mlflow_uri: str) -> Path:
    """Download patch_statistics' percentile_stats.parquet from mlflow.

    Unlike the patch/cls token parquet (huge - see `explainability.tiles.
    resolve_token_dirs`'s local-mount-preferred fast path) or W/H (also
    memmap-sized), this is a small per-dimension summary - a few hundred KB
    at most - so there's no large-artifact exception here: it's always
    fetched from mlflow, the one source of truth, rather than assuming a
    particular pod happens to have a local copy sitting around.

    Args:
        mlflow_uri: mlflow artifact URI for a specific patch_statistics
            run's percentile_stats.parquet, e.g.
            "mlflow-artifacts:/86/<run_id>/artifacts/percentile_stats.parquet".

    Returns:
        Local filesystem path to the downloaded file (mlflow's own cache).
    """
    return Path(mlflow.artifacts.download_artifacts(mlflow_uri))


def load_shift(percentile_stats_path: Path, percentile_column: str) -> np.ndarray:
    """Load the per-dimension shift constant picked from patch_statistics' output.

    Args:
        percentile_stats_path: Path to the percentile_stats.parquet produced by
            `explainability.patch_statistics`.
        percentile_column: Which percentile column to use as the shift, e.g.
            "p0.0001" for the 1e-4 quantile.

    Returns:
        A 1D array of shape (embed_dim,), one shift value per dimension.
    """
    stats = pd.read_parquet(percentile_stats_path).sort_index()
    return stats[percentile_column].to_numpy(dtype=np.float32)


def load_scale(
    percentile_stats_path: Path, low_column: str = "p0.25", high_column: str = "p0.75"
) -> np.ndarray:
    """Load the per-dimension IQR scale (p0.75 - p0.25) from patch_statistics' output.

    A handful of embedding dimensions carry far larger magnitude than the
    rest (observed: dimension 1 spans roughly -53..41 vs. a typical ~-4..4 -
    see explainability-status memory), and NMF's (unweighted, Frobenius) loss
    would otherwise let those few dimensions dominate what the dictionary
    fits. IQR is used rather than std since it's robust to exactly the
    outliers being downweighted (std would itself be inflated by them), and
    it comes for free from the same per-dimension percentiles already being
    computed.

    Args:
        percentile_stats_path: Path to the percentile_stats.parquet produced by
            `explainability.patch_statistics` (must include the `low_column`
            and `high_column` percentiles).
        low_column: Percentile column for the IQR's lower bound.
        high_column: Percentile column for the IQR's upper bound.

    Returns:
        A 1D array of shape (embed_dim,), one scale value per dimension.
        Dimensions with a zero (or negative, shouldn't happen) IQR fall back
        to a scale of 1 rather than dividing by zero.
    """
    stats = pd.read_parquet(percentile_stats_path).sort_index()
    iqr = (stats[high_column] - stats[low_column]).to_numpy(dtype=np.float32)
    return np.where(iqr <= 0, 1.0, iqr)


def iter_patch_batches(
    patches_ds: ray.data.Dataset,
    batch_size: int,
    shift: np.ndarray,
    scale: np.ndarray,
    with_metadata: bool = False,
    shuffle_seed: int | None = None,
    shuffle_buffer_size: int | None = None,
) -> Iterator[tuple[np.ndarray, pd.DataFrame | None]]:
    """Yield shifted, scaled, non-negative patch batches with optional provenance.

    `patches_ds` already holds one row per patch token (see
    `preprocessing/embeddings_xai.py`, which explodes and Hive-partitions by
    `kind` at write time), so this reads and transforms patches directly - no
    per-tile reshaping or CLS stripping needed here any more.

    Args:
        patches_ds: `ray.data.Dataset` of patch tokens (e.g. from
            `explainability.tiles.load_tokens_dataset` with `kind="patch"`),
            in whatever order the caller wants read (see `shuffle_seed`).
        batch_size: Number of patches to read per yielded batch.
        shift: Per-dimension shift constant `c`, shape (embed_dim,).
        scale: Per-dimension scale constant `d` (the IQR, see `load_scale`),
            shape (embed_dim,) - matches concept_mil.tex's non-negativity
            transform t~ = (t + c) / d (here `shift` plays the role of `-c`).
        with_metadata: If True, also yield a DataFrame of (slide_id, x, y,
            patch_index) rows aligned with the yielded patch batch, so each
            row of W can be traced back to the patch it came from.
        shuffle_seed: If given, patches are read in a locally-shuffled order
            (a cheap, per-worker approximate shuffle - see `Dataset.iter_batches`'s
            `local_shuffle_buffer_size`, no cross-node data movement) - used
            for the per-epoch NMF training passes. Leave as None (read order
            preserved) for the final transform pass, since its output rows
            must line up 1:1 with the yielded metadata.
        shuffle_buffer_size: Row buffer size for the local shuffle; required
            together with `shuffle_seed`, ignored otherwise.

    Yields:
        Tuples of (patches, metadata), where patches has shape
        (n_rows_in_batch, embed_dim) and metadata is None unless
        `with_metadata` is set.
    """
    columns = ["slide_id", "x", "y", "patch_index", "embedding"] if with_metadata else ["embedding"]

    for batch in patches_ds.select_columns(columns).iter_batches(
        batch_size=batch_size,
        batch_format="numpy",
        local_shuffle_seed=shuffle_seed,
        local_shuffle_buffer_size=shuffle_buffer_size,
    ):
        tokens = np.stack(batch["embedding"]).astype(np.float32, copy=False)
        patches = np.maximum((tokens - shift) / scale, 0.0)

        metadata = None
        if with_metadata:
            metadata = pd.DataFrame(
                {
                    "slide_id": batch["slide_id"],
                    "x": batch["x"],
                    "y": batch["y"],
                    "patch_index": batch["patch_index"],
                }
            )
        yield patches, metadata


def gauge_fix_dictionary(h: np.ndarray) -> np.ndarray:
    """Fix the WH scale ambiguity: rescale H to unit rows.

    For any positive diagonal S, W @ H == (W @ S^-1) @ (S @ H), so component
    magnitudes carry no meaning until this is fixed. No corresponding W
    correction is needed from the caller: `main` re-transforms every patch
    against this gauge-fixed H directly (by pointing `model.components_` at
    it before the transform pass) rather than computing W against the
    pre-fix H and rescaling it after the fact.

    Args:
        h: Dictionary/components matrix, shape (n_components, n_features).

    Returns:
        The gauge-fixed H (unit L2-norm rows).
    """
    norms = np.linalg.norm(h, axis=1)
    norms = np.where(norms == 0, 1.0, norms)
    return h / norms[:, None]


@with_cli_args(["+explainability=nmf_fit"])
@hydra.main(config_path="../configs", config_name="explainability", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_path = resolve_percentile_stats_path(config.shift.mlflow_uri)
    shift = load_shift(stats_path, config.shift.percentile_column)
    scale = load_scale(stats_path)

    token_dirs = resolve_token_dirs(
        config.sources, config.get("local_embeddings_xai_dir"), kind="patch"
    )
    patches_ds = load_tokens_dataset(token_dirs)
    # A plain, unfiltered read_parquet count is metadata-only (row counts come
    # from the parquet footers, no column data decoded) - unlike
    # patch_statistics.py's sampled count, nothing here forces a full read.
    n_patches = patches_ds.count()

    model = MiniBatchNMF(
        n_components=config.n_components,
        init=config.nmf.init,
        beta_loss=config.nmf.beta_loss,
        random_state=config.nmf.random_state,
    )

    # Fit: several shuffled passes over the full corpus, updating H only.
    # W is deliberately not collected here - the H seen by an early batch in
    # a later epoch is already better than the H an early epoch started
    # with, so W from mid-training batches would reflect a moving target
    # rather than the final dictionary.
    for epoch in range(config.nmf.epochs):
        for patches, _ in iter_patch_batches(
            patches_ds,
            config.nmf.batch_size,
            shift,
            scale,
            shuffle_seed=config.nmf.random_state + epoch,
            shuffle_buffer_size=config.nmf.shuffle_buffer_size,
        ):
            model.partial_fit(patches)

    # Recover H_k = H~_k * d (concept_mil.tex eq 2.24): the dictionary was
    # fit on scaled patches, so its raw coefficients are per unit of
    # (dimension j / scale[j]), not per unit of dimension j directly - this
    # multiplies that back out into the original (shifted-only) token space.
    # Must happen *before* gauge-fixing: gauge-fixing normalizes row norms,
    # and this recovery changes those norms (scaling each column by a
    # different amount).
    h = model.components_ * scale[None, :]
    if config.nmf.gauge_fix:
        h = gauge_fix_dictionary(h)

    # Point the model at the final (recovered, possibly gauge-fixed) H and
    # transform *shift-only* patches (scale=1 - h is no longer in the
    # scaled-fit space, so the input mustn't be either) against it: W then
    # comes out of transform() already correct, with no separate rescale
    # needed the way leaving model.components_ unchanged would have required.
    model.components_ = h
    unscaled = np.ones_like(scale)

    # Transform: one clean pass with the now-final H to get every patch's W.
    w_path = output_dir / "w.f32.npy"
    w = np.lib.format.open_memmap(
        w_path, mode="w+", dtype=np.float32, shape=(n_patches, config.n_components)
    )
    metadata_chunks = []
    offset = 0
    for patches, metadata in iter_patch_batches(
        patches_ds,
        config.nmf.batch_size,
        shift,
        unscaled,
        with_metadata=True,
    ):
        w_batch = model.transform(patches)
        w[offset : offset + w_batch.shape[0]] = w_batch
        metadata_chunks.append(metadata)
        offset += w_batch.shape[0]
    w.flush()
    assert offset == n_patches

    pd.concat(metadata_chunks, ignore_index=True).to_parquet(
        output_dir / "w_metadata.parquet", index=False
    )

    h_df = pd.DataFrame(h).rename_axis("component")
    h_df.to_parquet(output_dir / "h.parquet")

    manifest = {
        "w": {"path": str(w_path), "shape": list(w.shape), "dtype": str(w.dtype)},
        "n_components": config.n_components,
        "percentile_column": config.shift.percentile_column,
        "scale_columns": "p0.75 - p0.25 (IQR)",
        "gauge_fixed": config.nmf.gauge_fix,
        "epochs": config.nmf.epochs,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # W and its per-patch metadata stay on the project mount (too large for
    # mlflow, same treatment as patch_sample.f32.npy in patch_statistics.py);
    # only H and the manifest are small enough to log directly.
    logger.log_artifact(str(output_dir / "h.parquet"))
    logger.log_artifact(str(manifest_path))


if __name__ == "__main__":
    ctx = ray.data.DataContext.get_current()
    ctx.enable_rich_progress_bars = True
    ctx.use_ray_tqdm = False

    # num_cpus set deliberately *low* - same fix, same root cause as
    # explainability/patch_statistics.py (see that file's comment and
    # explainability-status memory): every patch token parquet file is a
    # single ~1.6GB row group, so even Ray's own automatic per-file metadata
    # sampling has to materialize close to the whole file - measured at
    # 2-5GB per file. With no cap, Ray schedules up to num_cpus of those
    # concurrently on this single local Ray instance, which is what
    # OOM-killed this job. Keep in sync with cpu= in
    # scripts/explainability/nmf_fit.py.
    with ray.init(num_cpus=8, runtime_env={"excludes": [".git", ".venv"]}):
        main()
