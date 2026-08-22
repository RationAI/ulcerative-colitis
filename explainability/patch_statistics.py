import json
from pathlib import Path

import datasets
import hydra
import numpy as np
import pandas as pd
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


def resolve_tile_globs(sources: DictConfig) -> list[str]:
    """Download each institution's tiles and return one glob per institution.

    Args:
        sources: Mapping of institution name to its embeddings_xai dataset
            config (as produced by `configs/dataset/embeddings_xai/*.yaml`).

    Returns:
        Glob patterns pointing at each institution's tile parquet directory,
        ready to hand straight to `datasets.load_dataset`.
    """
    globs = []
    for institution, source in sources.items():
        folder = Path(download_artifacts(source.mlflow_uris.embeddings_xai.train))
        tiles_dir = folder / "tiles"
        if not tiles_dir.is_dir():
            raise FileNotFoundError(f"No tiles directory found for {institution} under {folder}")
        globs.append(str(tiles_dir / "*.parquet"))
    return globs


def build_patch_sample(
    tile_globs: list[str],
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

    Args:
        tile_globs: Per-institution glob patterns over tile parquet files.
        memmap_path: Where to persist the sampled patch matrix.
        tiles_fraction: Fraction of all tiles to keep.
        random_state: Seed for the tile shuffle, for reproducibility.
        tokens_per_tile: Number of tokens per tile embedding (patches + CLS).
        cls_token_index: Index of the CLS/summary token to drop.
        embed_dim: Dimensionality of a single patch token.
        batch_size: Number of tiles to read per batch while filling the memmap.

    Returns:
        A tuple of the memmap of shape (n_patches, embed_dim) and a DataFrame
        with the metadata (slide_id, x, y, ...) of the tiles that were sampled.
    """
    dataset = datasets.load_dataset("parquet", data_files=tile_globs, split="train")

    n_tiles = round(len(dataset) * tiles_fraction)
    sampled = dataset.shuffle(seed=random_state).select(range(n_tiles))

    patches_per_tile = tokens_per_tile - 1
    n_patches = n_tiles * patches_per_tile

    memmap_path.parent.mkdir(parents=True, exist_ok=True)
    patches = np.lib.format.open_memmap(
        memmap_path, mode="w+", dtype=np.float32, shape=(n_patches, embed_dim)
    )

    offset = 0
    embedding_columns = sampled.column_names
    embedding_columns.remove("embedding")
    for batch in sampled.select_columns(["embedding"]).iter(batch_size=batch_size):
        tokens = np.array(batch["embedding"], dtype=np.float32).reshape(
            -1, tokens_per_tile, embed_dim
        )
        tokens = np.delete(tokens, cls_token_index, axis=1).reshape(-1, embed_dim)
        patches[offset : offset + tokens.shape[0]] = tokens
        offset += tokens.shape[0]
    patches.flush()
    assert offset == n_patches

    tile_metadata = sampled.select_columns(embedding_columns).to_pandas()
    return patches, tile_metadata


def compute_percentiles(patches: np.memmap, percentiles: list[float]) -> pd.DataFrame:
    """Compute per-dimension percentiles without loading the full sample.

    Each dimension is pulled out of the memmap on its own (a single column of
    even a ~90GB sample is only tens of MB), so this stays memory-light
    regardless of how large the sample is.

    Args:
        patches: Memmap of shape (n_patches, embed_dim).
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
        tile_globs = resolve_tile_globs(config.sources)
        patches, tile_metadata = build_patch_sample(
            tile_globs=tile_globs,
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
    main()
