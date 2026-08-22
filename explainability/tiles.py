import logging
from pathlib import Path

import ray.data
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig


log = logging.getLogger(__name__)


def resolve_tile_dirs(sources: DictConfig, local_embeddings_xai_dir: str | None) -> list[str]:
    """Locate each institution's tiles directory.

    Prefers the pre-existing local copy under `local_embeddings_xai_dir`
    (matching `output_dir` in `configs/preprocessing/embeddings_xai.yaml`,
    i.e. `<local_embeddings_xai_dir>/<institution>/train`) since it's the
    same data `download_artifacts` would otherwise fetch, just without the
    slow mlflow round-trip. Falls back to `download_artifacts` when no local
    copy is found (e.g. on a fresh kube job that isn't on the project mount).

    Args:
        sources: Mapping of institution name to its embeddings_xai dataset
            config (as produced by `configs/dataset/embeddings_xai/*.yaml`).
        local_embeddings_xai_dir: Root directory to look for a local copy
            under, or None to always go through `download_artifacts`.

    Returns:
        Directories, one per institution, holding that institution's tile
        parquet files - ready to hand straight to `ray.data.read_parquet`.
    """
    tile_dirs = []
    for institution, source in sources.items():
        tiles_dir = None
        if local_embeddings_xai_dir is not None:
            candidate = Path(local_embeddings_xai_dir) / source.institution / "train" / "tiles"
            if candidate.is_dir():
                log.info("Using local tiles for %s: %s", institution, candidate)
                tiles_dir = candidate

        if tiles_dir is None:
            folder = Path(download_artifacts(source.mlflow_uris.embeddings_xai.train))
            tiles_dir = folder / "tiles"
            if not tiles_dir.is_dir():
                raise FileNotFoundError(f"No tiles directory found for {institution} under {folder}")

        tile_dirs.append(str(tiles_dir))
    return tile_dirs


def load_tiles_dataset(tile_dirs: list[str]) -> ray.data.Dataset:
    """Load and pool tile parquet files (across institutions) into one dataset.

    Uses `ray.data.read_parquet` rather than HF `datasets`: it reads straight
    through PyArrow's native parquet reader - the same path `embeddings_xai.py`
    used to write these files in the first place - with no separate row-by-row
    Arrow-conversion pass. That conversion is what makes `datasets.load_dataset`
    slow (tens of examples/s) for the deeply-nested `embedding` columns here;
    `ray.data` doesn't pay it, and scales across the job's whole cluster rather
    than one node's cores.

    Args:
        tile_dirs: Per-institution directories of tile parquet files, as
            returned by `resolve_tile_dirs`. A single `read_parquet` call
            pools all of them into one dataset.

    Returns:
        A single pooled, lazy `ray.data.Dataset` over all tiles.
    """
    return ray.data.read_parquet(tile_dirs)
