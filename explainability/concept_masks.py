"""Concept masks for xOpat: where each concept sits on the gallery's slides.

A mask holds concept k's share of the code, v_k / sum_l v_l, per tile, the same
number that ranks the gallery's tiles, averaged where tiles overlap. Its
resolution is the embeddings': one value per tile stride. Stored as 8 bits on a
scale fixed per concept across slides, so a colour means the same on every
slide: 0 is no tile, 1 the `low_quantile` of the concept's share over all tiles
or less, 255 the `high_quantile` or more.

Each concept row of the gallery becomes one xOpat session holding its slides, so
xOpat's slide switching moves along the row. Every slide carries all K masks,
with only the row's concept visible.
"""

import colorsys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from ratiopath.masks import write_big_tiff
from ratiopath.masks.mask_builders import MaskBuilder
from report_fast import XopatSession, heatmap_layer
from report_fast.mlflow import artifact_data_id

from explainability.tiles import XOPAT_PROTOCOL, tile_viewport


MASKS_ARTIFACT = "masks"
MASK_THREADS = 8
# Hues red through blue: H&E's purples and pinks would hide an overlay.
HUE_START = 0.98
HUE_RANGE = 0.72
GOLDEN_RATIO = 0.618034


def mask_artifact(concept: int, slide_name: str) -> str:
    """Path of a mask under the run's artifacts, and under the local masks directory's parent."""
    return f"{MASKS_ARTIFACT}/concept_{concept:02d}/{slide_name}.tiff"


def concept_colour(concept: int) -> tuple[float, float, float]:
    """RGB in 0..1 of a concept's overlay, the same on every slide.

    Consecutive concepts land far apart in hue, and alternate strong and pale.
    """
    hue = (HUE_START + HUE_RANGE * (concept * GOLDEN_RATIO % 1)) % 1
    return colorsys.hsv_to_rgb(hue, 1.0 if concept % 2 == 0 else 0.55, 1.0)


def share_scale(
    share: np.ndarray, low_quantile: float, high_quantile: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-concept `(low, high)` of the 8-bit scale, shapes `(K,)`."""
    return np.quantile(share, low_quantile, axis=0), np.quantile(
        share, high_quantile, axis=0
    )


def to_mask_bytes(
    values: np.ndarray, covered: np.ndarray, low: np.ndarray, high: np.ndarray
) -> np.ndarray:
    """`values` `(K, H, W)` on the 8-bit scale, 0 where no tile covers the pixel."""
    span = np.maximum(high - low, 1e-12)[:, None, None]
    scaled = np.clip((values - low[:, None, None]) / span, 0, 1)
    return np.where(covered, 1 + np.rint(254 * scaled), 0).astype(np.uint8)


def write_slide_masks(
    share: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    slide: pd.Series,
    slide_name: str,
    low: np.ndarray,
    high: np.ndarray,
    downsample: int,
    output_dir: Path,
) -> None:
    """Write all K masks of one slide, `downsample` times smaller than its tiles' level."""
    builder = MaskBuilder(
        source_extents=(int(slide["extent_y"]), int(slide["extent_x"])),
        source_tile_extent=(int(slide["tile_extent_y"]), int(slide["tile_extent_x"])),
        output_tile_extent=1,
        stride=(int(slide["stride_y"]), int(slide["stride_x"])),
        n_channels=share.shape[1],
    )
    builder.update_batch(
        share.astype(np.float32), np.column_stack([y, x]).astype(np.int64)
    )
    result = builder.finalize()
    masks = to_mask_bytes(result["mask"], result["overlap_counter"] > 0, low, high)
    for concept in range(share.shape[1]):
        image = builder.resize_to_source(masks[[concept]], kernel="nearest")
        image = image.resize(1 / downsample, kernel="nearest")
        path = output_dir / mask_artifact(concept, slide_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_big_tiff(
            image,
            path,
            mpp_x=float(slide["mpp_x"]) * downsample,
            mpp_y=float(slide["mpp_y"]) * downsample,
        )
    builder.cleanup()


def write_masks(
    share: np.ndarray,
    keys: pd.DataFrame,
    selected: pd.DataFrame,
    slides: pd.DataFrame,
    low: np.ndarray,
    high: np.ndarray,
    downsample: int,
    output_dir: Path,
) -> None:
    """Write the masks of every slide in `selected`, a `with_slide_links` table."""
    table = slides.set_index("id")
    rows = keys.groupby("slide_id").indices
    names = selected.drop_duplicates("slide_id").set_index("slide_id")["slide_name"]

    def write(slide_id: str) -> None:
        tiles = keys.iloc[rows[slide_id]]
        write_slide_masks(
            share[rows[slide_id]],
            tiles["x"].to_numpy(),
            tiles["y"].to_numpy(),
            table.loc[slide_id],
            names[slide_id],
            low,
            high,
            downsample,
            output_dir,
        )

    with ThreadPoolExecutor(MASK_THREADS) as pool:
        list(pool.map(write, names.index))


def row_urls(
    row: pd.DataFrame,
    concept: int,
    n_concepts: int,
    slides: pd.DataFrame,
    run_id: str,
    experiment_id: str,
    tiles_across: float,
) -> list[str]:
    """One link per tile of a concept row, into a session of the row's slides with masks.

    The links differ only in the slide they open and where they centre.
    """
    table = slides.set_index("id")
    session = XopatSession()
    viewports = []
    for tile in row.itertuples():
        slide = table.loc[tile.slide_id]
        visualization = session.add_visualization(str(tile.slide_name))
        session.add_background(
            slide["path"],
            name=str(tile.slide_name),
            visualization_index=visualization,
            protocol=XOPAT_PROTOCOL,
        )
        for k in range(n_concepts):
            mask = artifact_data_id(
                mask_artifact(k, str(tile.slide_name)), run_id, experiment_id
            )
            index = session.add_layer(
                {
                    **heatmap_layer(mask, name=f"Concept {k}", color=concept_colour(k)),
                    "visible": k == concept,
                },
                visualization=visualization,
            )
            # The deployment's default for overlays, iipimage, refuses mlflow artifacts.
            session.data[cast("int", index)]["protocol"] = XOPAT_PROTOCOL
        viewports.append(
            tile_viewport(
                tile.x,
                tile.y,
                slide["tile_extent_x"],
                slide["tile_extent_y"],
                slide["extent_x"],
                tiles_across,
            )
        )

    urls = []
    for index, viewport in enumerate(viewports):
        session.params = {"viewport": viewport, "activeBackgroundIndex": index}
        urls.append(session.url())
    return urls
