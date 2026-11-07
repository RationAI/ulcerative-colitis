import logging
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd
import ray.data
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from report_fast import XopatSession


log = logging.getLogger(__name__)

XOPAT_PROTOCOL = "wsi_service"  # the slide protocol registered on the deployment


def resolve_embedding_split_dirs(
    sources: DictConfig, local_embeddings_dir: str | None, split: str = "train"
) -> list[Path]:
    """Locate each institution's `preprocessing/embeddings.py` split directory.

    Each holds `tiles/*.parquet` (one row per tile: `slide_id, x, y, tissue,
    blur, artifacts, embedding`, where `embedding` is the full tile embedding
    `h_i` the MIL model consumes) and `slides.parquet`. Prefers the local copy
    under `local_embeddings_dir` (`<local_embeddings_dir>/<institution>/<split>`,
    matching `output_dir` in `configs/preprocessing/embeddings.yaml`), falling
    back to `download_artifacts` on `configs/dataset/embeddings/*.yaml`'s
    `mlflow_uris.embeddings[split]` - same data, without the slow mlflow
    round-trip when the project mount is available.

    Args:
        sources: Mapping of institution name to its embeddings dataset config
            (as produced by `configs/dataset/embeddings/*.yaml`).
        local_embeddings_dir: Root directory to look for a local copy under, or
            None to always go through `download_artifacts`.
        split: Which split to resolve - "train", "test_preliminary", etc.

    Returns:
        One split directory per institution.
    """
    split_dirs = []
    for institution, source in sources.items():
        institution = str(institution)
        split_dir = None
        if local_embeddings_dir is not None:
            candidate = Path(local_embeddings_dir) / source.institution / split
            # Require actual parquet files, not just the folder: an interrupted
            # copy can leave an empty tiles/ behind (knl_patos/test_preliminary
            # did), which would otherwise crash the read with no clear error.
            if any((candidate / "tiles").glob("*.parquet")):
                log.info("Using local embeddings for %s: %s", institution, candidate)
                split_dir = candidate
            elif (candidate / "tiles").is_dir():
                print(
                    f"Local {candidate / 'tiles'} has no parquet files - downloading from mlflow",
                    flush=True,
                )

        if split_dir is None:
            split_dir = Path(download_artifacts(source.mlflow_uris.embeddings[split]))
            if not (split_dir / "tiles").is_dir():
                raise FileNotFoundError(f"No tiles directory found for {institution} under {split_dir}")

        split_dirs.append(split_dir)
    return split_dirs


def select_sources(sources: DictConfig, institutions: str) -> DictConfig:
    """The `+`-joined subset of `sources` to use, e.g. "ftn+ikem" or "knl_patos"."""
    names = institutions.split("+")
    missing = [name for name in names if name not in sources]
    if missing:
        raise ValueError(f"Unknown institutions {missing}; configured sources: {list(sources)}")
    return DictConfig({name: sources[name] for name in names})


def load_embeddings_dataset(split_dirs: list[Path]) -> ray.data.Dataset:
    """Lazily pool every institution's per-tile embeddings into one `ray.data.Dataset`."""
    return ray.data.read_parquet([str(split_dir / "tiles") for split_dir in split_dirs])


def load_embedding_slides(split_dirs: list[Path]) -> pd.DataFrame:
    """Pool every institution's `slides.parquet` (one small row per slide, read eagerly)."""
    return pd.concat(
        [pd.read_parquet(split_dir / "slides.parquet") for split_dir in split_dirs],
        ignore_index=True,
    )


def tile_viewport(
    x: int, y: int, tile_extent_x: int, tile_extent_y: int, extent_x: int, tiles_across: float
) -> dict[str, Any]:
    """XOpat viewport centred on a tile, `tiles_across` tiles wide.

    `x`, `y` and both extents are in pixels of the same pyramid level
    (slides.parquet's `level`), so their ratios are the OpenSeadragon viewport
    coordinates, where the slide's width is 1 at any level - `y` too is divided
    by the width. Zoom z shows 1/z of that width.
    """
    return {
        "zoomLevel": extent_x / (tiles_across * tile_extent_x),
        "point": {
            "x": (x + tile_extent_x / 2) / extent_x,
            "y": (y + tile_extent_y / 2) / extent_x,
        },
    }


def xopat_url(
    path: str,
    x: int,
    y: int,
    tile_extent_x: int,
    tile_extent_y: int,
    extent_x: int,
    tiles_across: float,
) -> str:
    """XOpat v3 link opening the slide centred on a tile, `tiles_across` tiles wide."""
    viewport = tile_viewport(x, y, tile_extent_x, tile_extent_y, extent_x, tiles_across)
    session = XopatSession.from_slide(
        path, name=PurePosixPath(path).stem, protocol=XOPAT_PROTOCOL, params={"viewport": viewport}
    )
    return session.url()


def with_slide_links(
    tiles: pd.DataFrame, slides: pd.DataFrame, tiles_across: float
) -> pd.DataFrame:
    """`tiles` with each tile's `slide_name` and `xopat_url`, from the slides table."""
    info = slides.set_index("id").loc[tiles["slide_id"]]
    return tiles.assign(
        slide_name=[PurePosixPath(path).stem for path in info["path"]],
        xopat_url=[
            xopat_url(path, x, y, width, height, extent_x, tiles_across)
            for path, x, y, width, height, extent_x in zip(
                info["path"],
                tiles["x"],
                tiles["y"],
                info["tile_extent_x"],
                info["tile_extent_y"],
                info["extent_x"],
                strict=True,
            )
        ],
    )
