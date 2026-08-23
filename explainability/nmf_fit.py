import json
from collections.abc import Iterator
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import ray
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from sklearn.decomposition import MiniBatchNMF

from explainability.tiles import load_tokens_dataset, resolve_token_dirs


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


def iter_patch_batches(
    patches_ds: ray.data.Dataset,
    batch_size: int,
    shift: np.ndarray,
    with_metadata: bool = False,
    shuffle_seed: int | None = None,
    shuffle_buffer_size: int | None = None,
) -> Iterator[tuple[np.ndarray, pd.DataFrame | None]]:
    """Yield shifted, non-negative patch batches with optional provenance.

    `patches_ds` already holds one row per patch token (see
    `preprocessing/embeddings_xai.py`, which explodes and Hive-partitions by
    `kind` at write time), so this reads and shifts patches directly - no
    per-tile reshaping or CLS stripping needed here any more.

    Args:
        patches_ds: `ray.data.Dataset` of patch tokens (e.g. from
            `explainability.tiles.load_tokens_dataset` with `kind="patch"`),
            in whatever order the caller wants read (see `shuffle_seed`).
        batch_size: Number of patches to read per yielded batch.
        shift: Per-dimension shift constant, shape (embed_dim,).
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
        patches = np.maximum(tokens - shift, 0.0)

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


def gauge_fix_dictionary(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fix the WH scale ambiguity: rescale H to unit rows.

    For any positive diagonal S, W @ H == (W @ S^-1) @ (S @ H), so component
    magnitudes carry no meaning until this is fixed. The matching rescale of
    W (multiplying column k by `norms[k]`) must be applied by the caller,
    since W here is typically too large to hold in memory alongside H.

    Args:
        h: Dictionary/components matrix, shape (n_components, n_features).

    Returns:
        A tuple of (gauge-fixed H, the per-component norms used to fix it).
    """
    norms = np.linalg.norm(h, axis=1)
    norms = np.where(norms == 0, 1.0, norms)
    return h / norms[:, None], norms


@with_cli_args(["+explainability=nmf_fit"])
@hydra.main(config_path="../configs", config_name="explainability", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shift = load_shift(Path(config.shift.percentile_stats_path), config.shift.percentile_column)

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
            shuffle_seed=config.nmf.random_state + epoch,
            shuffle_buffer_size=config.nmf.shuffle_buffer_size,
        ):
            model.partial_fit(patches)

    h = model.components_
    norms = None
    if config.nmf.gauge_fix:
        h, norms = gauge_fix_dictionary(h)

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
        with_metadata=True,
    ):
        w_batch = model.transform(patches)
        if norms is not None:
            w_batch = w_batch * norms[None, :]
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

    with ray.init(runtime_env={"excludes": [".git", ".venv"]}):
        main()
