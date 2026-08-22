import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import ray
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

from explainability.tiles import load_tiles_dataset, resolve_tile_dirs


def build_patch_sample(
    dataset: ray.data.Dataset,
    memmap_path: Path,
    tiles_fraction: float,
    random_state: int,
    tokens_per_tile: int,
    cls_token_index: int,
    embed_dim: int,
    batch_size: int,
) -> tuple[np.memmap, pd.DataFrame]:
    """Randomly sample whole tiles and flatten their non-CLS patch tokens.

    Tiles (not individual patches) are the sampling unit: at the target
    fraction the sample already covers roughly half of all patches, so
    per-patch reservoir sampling would add complexity for no real gain in
    representativeness. The result is written directly into a memmap since
    the pooled sample is far larger than fits in memory.

    `random_sample` draws each tile independently at probability
    `tiles_fraction` in one streaming pass - no shuffle, no cross-node data
    movement - so the sampled count is only approximately
    `tiles_fraction * len(dataset)`. The sample is deliberately left
    un-`materialize()`d: at this fraction it can be tens of GB larger than a
    job pod's RAM, so caching it in the object store risks the exact
    out-of-memory problem the memmap exists to avoid. Getting an exact count
    to size the memmap means running the same seeded draw twice - once via
    `count()`, once via the `iter_batches` write loop below - which
    reproduces the same tiles both times, just at the cost of reading the
    source data an extra time.

    Args:
        dataset: Pooled `ray.data.Dataset` of tiles (e.g. from
            `load_tiles_dataset`).
        memmap_path: Where to persist the sampled patch matrix.
        tiles_fraction: Fraction of all tiles to keep.
        random_state: Seed for the tile sampling, for reproducibility.
        tokens_per_tile: Number of tokens per tile embedding (patches + CLS).
        cls_token_index: Index of the CLS/summary token to drop.
        embed_dim: Dimensionality of a single patch token.
        batch_size: Number of tiles to read per batch while filling the memmap.

    Returns:
        A tuple of the memmap of shape (n_patches, embed_dim) and a DataFrame
        with the metadata (slide_id, x, y, ...) of the tiles that were sampled.
    """
    sampled = dataset.random_sample(tiles_fraction, seed=random_state)
    n_tiles = sampled.count()

    patches_per_tile = tokens_per_tile - 1
    n_patches = n_tiles * patches_per_tile

    memmap_path.parent.mkdir(parents=True, exist_ok=True)
    # Fortran order: compute_percentiles reads one dimension (column) at a
    # time, 1280 times over. In C order a row (5120 bytes) spans more than
    # one page, so a column read touches nearly the whole file on disk, once
    # per dimension. In F order each column is one contiguous run, so the
    # 1280 reads together add up to a single sequential pass over the file
    # instead. Writing row-batches is somewhat less contiguous this way, but
    # that happens once, unlike the 1280 reads.
    patches = np.lib.format.open_memmap(
        memmap_path, mode="w+", dtype=np.float32, shape=(n_patches, embed_dim), fortran_order=True
    )

    embedding_columns = sampled.columns()
    embedding_columns.remove("embedding")

    offset = 0
    metadata_chunks = []
    # Reads embedding + metadata columns together so this is the only other
    # pass over the sample besides the count() above (rather than a further,
    # separate to_pandas() pass just for metadata).
    for batch in sampled.iter_batches(batch_size=batch_size, batch_format="numpy"):
        # Arrow list columns come back as an object array of per-row 1D
        # arrays (all the same length here); stack before reshaping.
        tokens = np.stack(batch["embedding"]).astype(np.float32, copy=False)
        tokens = tokens.reshape(-1, tokens_per_tile, embed_dim)
        tokens = np.delete(tokens, cls_token_index, axis=1).reshape(-1, embed_dim)
        patches[offset : offset + tokens.shape[0]] = tokens
        offset += tokens.shape[0]

        metadata_chunks.append(pd.DataFrame({col: batch[col] for col in embedding_columns}))
    patches.flush()
    assert offset == n_patches

    tile_metadata = pd.concat(metadata_chunks, ignore_index=True)
    return patches, tile_metadata


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
        tile_metadata = pd.read_parquet(output_dir / "sampled_tiles.parquet")
    else:
        tile_dirs = resolve_tile_dirs(config.sources, config.get("local_embeddings_xai_dir"))
        dataset = load_tiles_dataset(tile_dirs)
        patches, tile_metadata = build_patch_sample(
            dataset=dataset,
            memmap_path=memmap_path,
            tiles_fraction=config.sample.tiles_fraction,
            random_state=config.sample.random_state,
            tokens_per_tile=config.tokens_per_tile,
            cls_token_index=config.cls_token_index,
            embed_dim=config.embed_dim,
            batch_size=config.sample.batch_size,
        )
        tile_metadata.to_parquet(output_dir / "sampled_tiles.parquet", index=False)

    percentiles = OmegaConf.to_object(config.percentiles)
    stats = compute_percentiles(patches, percentiles)
    stats.to_parquet(output_dir / "percentile_stats.parquet")

    manifest = {
        "patch_sample": {
            "path": str(memmap_path),
            "shape": list(patches.shape),
            "dtype": str(patches.dtype),
        },
        "n_tiles_sampled": len(tile_metadata),
        "tiles_fraction": config.sample.tiles_fraction,
        "random_state": config.sample.random_state,
        "percentiles": percentiles,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # The sample itself stays on the project mount (too large for mlflow);
    # only the lightweight derived artifacts are logged.
    logger.log_artifact(str(output_dir / "sampled_tiles.parquet"))
    logger.log_artifact(str(output_dir / "percentile_stats.parquet"))
    logger.log_artifact(str(manifest_path))


if __name__ == "__main__":
    ctx = ray.data.DataContext.get_current()
    ctx.enable_rich_progress_bars = True
    ctx.use_ray_tqdm = False

    with ray.init(runtime_env={"excludes": [".git", ".venv"]}):
        main()
