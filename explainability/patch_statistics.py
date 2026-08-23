import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import ray
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

from explainability.tiles import load_tokens_dataset, resolve_token_dirs


def build_patch_sample(
    dataset: ray.data.Dataset,
    memmap_path: Path,
    patches_fraction: float,
    random_state: int,
    embed_dim: int,
    batch_size: int,
) -> tuple[np.memmap, pd.DataFrame]:
    """Randomly sample patch tokens and write them into a memmap.

    `dataset` already holds one row per patch token (see
    `preprocessing/embeddings_xai.py`, which explodes and Hive-partitions by
    `kind` at write time), so this samples and writes patches directly - no
    per-tile reshaping or CLS stripping needed here any more. The result ends
    up in a memmap since the pooled sample is far larger than fits in memory.

    This makes exactly one pass over the sampled patches (see the internal
    `scratch_path` below for why one pass, rather than the two an
    upfront-sized memmap would need). `random_sample` draws each row
    independently at probability `patches_fraction` in one streaming pass -
    no shuffle, no cross-node data movement - so the sampled count is only
    approximately `patches_fraction * len(dataset)` and isn't known until
    this pass finishes.

    Args:
        dataset: Pooled `ray.data.Dataset` of patch tokens (e.g. from
            `load_tokens_dataset` with `kind="patch"`).
        memmap_path: Where to persist the sampled patch matrix.
        patches_fraction: Fraction of all patch tokens to keep.
        random_state: Seed for the sampling, for reproducibility.
        embed_dim: Dimensionality of a single patch token.
        batch_size: Number of patches to read per batch while filling the
            memmap.

    Returns:
        A tuple of the memmap of shape (n_patches, embed_dim) and a DataFrame
        with the metadata (slide_id, x, y, patch_index) of the sampled patches.
    """
    # Column names come from the *unsampled* dataset - sampling doesn't change
    # them, and this avoids a redundant execution of the random_sample plan
    # just to peek at its schema (`sampled.columns()` would trigger its own
    # separate mini-run of the same lazy pipeline `iter_batches` below runs
    # again for real).
    metadata_columns = dataset.columns()
    metadata_columns.remove("embedding")

    sampled = dataset.random_sample(patches_fraction, seed=random_state)

    memmap_path.parent.mkdir(parents=True, exist_ok=True)

    # The final array must be Fortran-ordered (see compute_percentiles), which
    # needs the row count fixed before a single element is written - but that
    # count isn't known until sampling has run. Getting it via a separate
    # `sampled.count()` pass would mean decoding the `embedding` column for
    # the entire corpus a second time (in practice this stalled the job
    # indefinitely - see the 2026-08-22 19:01-19:16 job log, stuck in
    # ray.data's AggregateNumRows step). A cheap count from a
    # column-projected dataset isn't a safe substitute either: `random_sample`
    # seeds its RNG per Ray *task index*, not per row, so dropping the
    # `embedding` column changes the block/task boundaries and therefore
    # samples a different set of rows than the full read does.
    #
    # So instead: one pass, writing row-major into a growable scratch file
    # (append-only, no upfront shape needed), then a second, purely local
    # disk-to-disk pass that lays the now-exactly-sized data out into the
    # final Fortran-order memmap. That local pass never touches ray or the
    # source parquet again - it costs extra local disk I/O, but replaces a
    # second full read of the *remote* embedding corpus with a cheap
    # sequential local read.
    scratch_path = memmap_path.with_suffix(".scratch")
    offset = 0
    metadata_chunks = []
    with open(scratch_path, "wb") as scratch:
        for batch in sampled.iter_batches(batch_size=batch_size, batch_format="numpy"):
            tokens = np.stack(batch["embedding"]).astype(np.float32, copy=False)
            scratch.write(np.ascontiguousarray(tokens).tobytes())
            offset += tokens.shape[0]

            metadata_chunks.append(pd.DataFrame({col: batch[col] for col in metadata_columns}))
    n_patches = offset
    patch_metadata = pd.concat(metadata_chunks, ignore_index=True)

    # Fortran order: compute_percentiles reads one dimension (column) at a
    # time, 1280 times over. In C order a row (5120 bytes) spans more than
    # one page, so a column read touches nearly the whole file on disk, once
    # per dimension. In F order each column is one contiguous run, so the
    # 1280 reads together add up to a single sequential pass over the file
    # instead.
    patches = np.lib.format.open_memmap(
        memmap_path, mode="w+", dtype=np.float32, shape=(n_patches, embed_dim), fortran_order=True
    )
    scratch_patches = np.memmap(scratch_path, dtype=np.float32, mode="r", shape=(n_patches, embed_dim))
    # Copy in row-chunks (sequential reads off the scratch file) rather than
    # one `patches[:] = scratch_patches[:]`, so this step doesn't need to
    # hold the whole sample in memory at once either.
    chunk_rows = max(batch_size, 1)
    for start in range(0, n_patches, chunk_rows):
        stop = min(start + chunk_rows, n_patches)
        patches[start:stop] = scratch_patches[start:stop]
    patches.flush()
    del scratch_patches
    scratch_path.unlink()

    return patches, patch_metadata


def compute_percentiles(patches: np.memmap, percentiles: list[float]) -> pd.DataFrame:
    """Compute per-dimension percentiles without loading the full sample.

    Each dimension is pulled out of the memmap on its own, so this stays
    memory-light regardless of how large the sample is (the result of one
    such slice is only tens of MB even for a ~90GB sample). This relies on
    `patches` being Fortran-ordered (see `build_patch_sample`) so that each
    of those column reads is a contiguous, cheap disk read rather than a
    scan touching most of the file.

    Args:
        patches: Fortran-ordered memmap of shape (n_patches, embed_dim).
        percentiles: Quantile levels in [0, 1] to compute for each dimension.

    Returns:
        A DataFrame indexed by dimension, one column per requested percentile.
    """
    n_dims = patches.shape[1]
    stats = np.empty((n_dims, len(percentiles)), dtype=np.float64)
    for dim in range(n_dims):
        stats[dim] = np.quantile(patches[:, dim], percentiles)

    columns = [f"p{p:g}" for p in percentiles]
    return pd.DataFrame(stats, columns=columns).rename_axis("dimension")


@with_cli_args(["+explainability=patch_statistics"])
@hydra.main(config_path="../configs", config_name="explainability", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    memmap_path = output_dir / "patch_sample.f32.npy"

    if memmap_path.exists() and not config.sample.overwrite:
        patches = np.lib.format.open_memmap(memmap_path, mode="r")
        patch_metadata = pd.read_parquet(output_dir / "sampled_patches.parquet")
    else:
        token_dirs = resolve_token_dirs(
            config.sources, config.get("local_embeddings_xai_dir"), kind="patch"
        )
        dataset = load_tokens_dataset(token_dirs, read_task_memory=config.read_task_memory)
        patches, patch_metadata = build_patch_sample(
            dataset=dataset,
            memmap_path=memmap_path,
            patches_fraction=config.sample.patches_fraction,
            random_state=config.sample.random_state,
            embed_dim=config.embed_dim,
            batch_size=config.sample.batch_size,
        )
        patch_metadata.to_parquet(output_dir / "sampled_patches.parquet", index=False)

    percentiles = OmegaConf.to_object(config.percentiles)
    stats = compute_percentiles(patches, percentiles)
    stats.to_parquet(output_dir / "percentile_stats.parquet")

    manifest = {
        "patch_sample": {
            "path": str(memmap_path),
            "shape": list(patches.shape),
            "dtype": str(patches.dtype),
        },
        "n_patches_sampled": len(patch_metadata),
        "patches_fraction": config.sample.patches_fraction,
        "random_state": config.sample.random_state,
        "percentiles": percentiles,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # The sample itself stays on the project mount (too large for mlflow);
    # only the lightweight derived artifacts are logged.
    logger.log_artifact(str(output_dir / "sampled_patches.parquet"))
    logger.log_artifact(str(output_dir / "percentile_stats.parquet"))
    logger.log_artifact(str(manifest_path))


if __name__ == "__main__":
    ctx = ray.data.DataContext.get_current()
    ctx.enable_rich_progress_bars = True
    ctx.use_ray_tqdm = False

    # Without num_cpus, ray.init() auto-detects the CPU count from
    # /sys/fs/cgroup - but falls back to the *node's* full core count
    # (observed: 128) whenever no hard cgroup CPU limit is set, which is the
    # case here (the kube job only sends a request, cpu=32 in
    # scripts/explainability/patch_statistics.py - keep these in sync). Left
    # unset, that mismatch overschedules concurrent tasks far beyond what the
    # pod can actually run, which is what stalled this job pinned at its real
    # CPU/RAM budget with ~zero throughput (see explainability-status memory).
    #
    # NOTE: an object_store_memory pin was tried here too (same cgroup-fallback
    # theory, for memory instead of CPU) and reverted - it was a *fixed*
    # literal, so it became an artificial ceiling independent of whatever pod
    # memory is actually given, which is likely what caused the stall to
    # persist even after bumping the pod to 64 CPU / 128GB RAM. Left unset for
    # now so Ray's own (auto-detected, possibly node-wide) sizing applies
    # instead - revisit only with a value that scales with the real pod size,
    # not a hardcoded one.
    with ray.init(num_cpus=32, runtime_env={"excludes": [".git", ".venv"]}):
        main()
