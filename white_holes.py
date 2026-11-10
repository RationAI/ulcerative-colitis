"""One-off check: white holes over tissue at pyramid levels 1 and 2 of CZI slides.

Some CZI slides show white rectangles over real tissue at some pyramid levels while
other levels show the tissue. This hurts the models in two ways: tissue tiled at a
level with a hole reaches the model as white, and tissue under a level-3 hole is
left out of the tissue mask (made at level 3) and so never tiled at all.

Each checked level is compared with every other level from 1 to 4 (downsamples 3,
9, 27, 81; slides without level 4 use the others): a cell is a hole where another
level shows content there and the checked level is blank. Level 0 is left out, it
costs 81 times level 2 to read; a hole present in all of levels 1 to 4 is missed.

Levels are compared on a grid of cells 81 level-0 px wide (27 level-1 px, 9
level-2 px, 1 level-4 px), about 14 um. In each level a cell is
    blank      at least 90% near-white (all channels >= 250) or transparent
    content    under half blank: tissue, glass inside a scanned area, or pen
    stained    content whose mean colour has saturation >= 0.1 - tissue, not
               glass (on six slides: >= 99% of mask cells, <= 0.6% of glass)
Tissue is a cell stained at any level, so it does not depend on the mask. Holes
narrower than 3 cells are dropped: they are the coarse levels' blur at edges.
Levels 3 and 4 are read whole, levels 1 and 2 only near stained cells of levels 3
and 4 or the mask.

Slides and tiles come from the tiling tables, so the tiles counted are the ones the
models got: every Nancy slide (all institutions and splits) at level 2, and every
tiled dysplasia slide at levels 1 and 2, each level with its own tiles. A tile is
the 25 x 25 cells (level 2) or 8 x 8 cells (level 1) under it; it is touched when
any of them is a hole and hit when at least half are.

Output under `--output-dir`:
    rows/       one parquet per slide, so a restarted run skips finished slides
    grids/      one npz per slide: the mask, tissue, per-level content and stained
                cells, and per checked level the holes, for later questions
    tissue_masks/, tiling/   the downloaded masks and tiling tables
    holes.parquet   one row per slide and checked level, columns:
        tissue_cells, hole_cells, hole_tissue_cells, tissue_lost
            tissue, all holes, holes over tissue, and their share of tissue
        largest_tissue_hole_cells, n_tissue_holes
        seen_at_L*      hole tissue cells stained at each other level
        mask_cells, hole_mask_cells     the mask, and holes over tissue inside it
        mask_missed_cells, mask_missed_fraction
            tissue outside the mask (grown by 2 cells), same for both levels
        n_tiles, tiles_touched, tiles_hit
    previews/   for slides with a large tissue hole or large tissue the mask
                missed, top to bottom: level 3, the checked level, and the checked
                level with holes over tissue in red and missed tissue in blue

Delete this script once the result is recorded in TODO.md.
"""

import argparse
import os
import time
import traceback
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import pyvips
from openslide import OpenSlide
from PIL import Image
from scipy import ndimage


DYSPLASIA = Path("/mnt/data/IKEM/colon/IBD_AI/dysplasia")
OUTPUT = Path("/mnt/projects/inflammatory_bowel_disease/ulcerative_colitis/white_holes")
TRACKING_URI = "http://mlflow.rationai-mlflow:5000/"
# Tiling tables (slides.parquet and tiles.parquet per split), by dataset.
TILINGS = {
    "nancy": [
        "mlflow-artifacts:/86/854bd20a112f450cb31843e5c436854b/artifacts",  # ikem
        "mlflow-artifacts:/86/b82e5dcc6bf840cf92bea0e0cfdc7603/artifacts",  # ftn
        "mlflow-artifacts:/86/d57c09bc02914ff58ba659b91f34c469/artifacts",  # knl_patos
        "mlflow-artifacts:/86/ef20b95b9e9348de9e4ca81597b3447a/artifacts",  # ikem_validation
    ],
    "dysplasia": [
        "mlflow-artifacts:/111/ca279f80074a4d3db7526209ee989d58/artifacts",  # level 1
        "mlflow-artifacts:/111/a054520c7e354e3eb71b1015e57d6a87/artifacts",  # level 2
    ],
}
# Tissue masks of the preprocessing, by slide folder.
MASKS = {
    "/mnt/data/IKEM/colon/IBD_AI/20x": "mlflow-artifacts:/86/94481ac59246471fb874bfb4dccb5e67/artifacts/tissue_masks",
    "/mnt/data/FTN/colon/IBD_AI": "mlflow-artifacts:/86/17149d1de7014112aba3a252a76d10bc/artifacts/tissue_masks",
    "/mnt/data/KNL_PATOS/colon/IBD_AI": "mlflow-artifacts:/86/6cdfa5ce4f7242a9b2bb394bfd5ed705/artifacts/tissue_masks",
    "/mnt/data/IKEM/colon/IBD_AI/validation": "mlflow-artifacts:/86/bad4715238634f929f3787b4e757a683/artifacts/tissue_masks",
    str(
        DYSPLASIA
    ): "mlflow-artifacts:/111/97084241311949189445f864d42e9d4e/artifacts/tissue_masks",
}

CELL = 81  # level-0 px per cell side
FINE = (3, 9)  # downsamples of levels 1 and 2, read around content only
COARSE = (27, 81)  # downsamples of levels 3 and 4, read whole
WHITE = 250
CONTENT_MAX_BLANK = 0.5  # a level shows content in a cell
HOLE_MIN_BLANK = 0.9  # the checked level is blank in a cell
STAIN_MIN_SATURATION = 0.1  # content that is tissue, not glass
MASK_SLACK_CELLS = 2  # mask edge tolerance when looking for tissue it missed
BLOCK_CELLS = 64  # cells per side of one read of a fine level
MARGIN_CELLS = 10  # read fine levels this far around tissue, about 140 um
PREVIEW_MIN_CELLS = 100  # largest region that earns a preview, about 140 um square
PREVIEW_WIDTH = 1600
TILE_HIT = 0.5  # share of a tile's area over a hole that counts it as hit


def blank(rgba: np.ndarray) -> np.ndarray:
    return (rgba[..., 3] == 0) | (rgba[..., :3].min(axis=2) >= WHITE)


def colour(rgba: np.ndarray) -> np.ndarray:
    """RGB with transparent pixels, which openslide returns black, drawn white."""
    return np.where(rgba[..., 3:] == 0, 255, rgba[..., :3]).astype(np.float32)


def cell_mean(
    values: np.ndarray, size: int, shape: tuple[int, int], pad: float
) -> np.ndarray:
    """Mean of `values` over `size`-px cells, padded to `shape` cells with `pad`."""
    h, w = shape[0] * size, shape[1] * size
    full = np.full((h, w, *values.shape[2:]), pad, dtype=np.float32)
    full[: min(h, values.shape[0]), : min(w, values.shape[1])] = values[:h, :w]
    return full.reshape(shape[0], size, shape[1], size, *values.shape[2:]).mean(
        axis=(1, 3)
    )


def saturation(rgb: np.ndarray) -> np.ndarray:
    top = rgb.max(axis=-1)
    return (top - rgb.min(axis=-1)) / np.maximum(top, 1.0)


def largest_region(cells: np.ndarray) -> tuple[int, int]:
    """(number of connected regions, cells in the largest)."""
    labels, n = ndimage.label(cells)
    return n, int(np.bincount(labels.ravel())[1:].max()) if n else 0


def level_index(slide: OpenSlide, downsample: float) -> int:
    for index, value in enumerate(slide.level_downsamples):
        if abs(value - downsample) < 0.01:
            return index
    raise ValueError(
        f"no level with downsample {downsample}: {slide.level_downsamples}"
    )


def read_whole(
    slide: OpenSlide, downsample: int, grid: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell blank share and mean RGB of a whole level."""
    level = level_index(slide, downsample)
    rgba = np.asarray(slide.read_region((0, 0), level, slide.level_dimensions[level]))
    size = CELL // downsample
    return (
        cell_mean(blank(rgba).astype(np.float32), size, grid, 1.0),
        cell_mean(colour(rgba), size, grid, 255.0),
    )


def read_blocks(
    slide: OpenSlide, downsample: int, grid: tuple[int, int], blocks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell blank share and mean RGB of a level, read only in `blocks`.

    `blocks` is a boolean grid of BLOCK_CELLS-sized blocks; cells outside them
    stay blank.
    """
    level = level_index(slide, downsample)
    width, height = slide.level_dimensions[level]
    size = CELL // downsample
    share = np.ones(grid, dtype=np.float32)
    rgb = np.full((*grid, 3), 255.0, dtype=np.float32)
    for row, column in zip(*np.nonzero(blocks), strict=True):
        by, bx = row * BLOCK_CELLS, column * BLOCK_CELLS
        block = (min(BLOCK_CELLS, grid[0] - by), min(BLOCK_CELLS, grid[1] - bx))
        x, y = bx * size, by * size
        w = min(block[1] * size, width - x)
        h = min(block[0] * size, height - y)
        if w <= 0 or h <= 0:
            continue
        rgba = np.asarray(
            slide.read_region((x * downsample, y * downsample), level, (w, h))
        )
        share[by : by + block[0], bx : bx + block[1]] = cell_mean(
            blank(rgba).astype(np.float32), size, block, 1.0
        )
        rgb[by : by + block[0], bx : bx + block[1]] = cell_mean(
            colour(rgba), size, block, 255.0
        )
    return share, rgb


def to_blocks(cells: np.ndarray) -> np.ndarray:
    """Blocks of BLOCK_CELLS cells holding any of `cells`, as a boolean block grid."""
    rows = -(-cells.shape[0] // BLOCK_CELLS)
    columns = -(-cells.shape[1] // BLOCK_CELLS)
    padded = np.zeros((rows * BLOCK_CELLS, columns * BLOCK_CELLS), dtype=bool)
    padded[: cells.shape[0], : cells.shape[1]] = cells
    return np.asarray(
        padded.reshape(rows, BLOCK_CELLS, columns, BLOCK_CELLS).any(axis=(1, 3))
    )


def to_cells(blocks: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """The cells of `blocks`, on the cell grid."""
    cells = np.repeat(np.repeat(blocks, BLOCK_CELLS, axis=0), BLOCK_CELLS, axis=1)
    return cells[: grid[0], : grid[1]]


def read_tissue_mask(task: dict[str, Any], grid: tuple[int, int]) -> np.ndarray | None:
    """The slide's tissue mask on the cell grid, or None when it has none."""
    folder = str(Path(task["path"]).parent)
    if folder not in MASKS:
        return None
    name = f"{Path(task['path']).stem}.tiff"
    cache = task["output_dir"] / "tissue_masks" / Path(folder).name
    path = cache / name
    if not path.exists():
        try:
            mlflow.artifacts.download_artifacts(
                artifact_uri=f"{MASKS[folder]}/{name}", dst_path=str(cache)
            )
        except Exception:
            return None
    image = pyvips.Image.new_from_file(str(path))
    mask = np.ndarray(
        buffer=image.extract_band(0).write_to_memory(),
        dtype=np.uint8,
        shape=(image.height, image.width),
    )
    size = round(grid[1] * CELL / image.width)
    if size != 27:
        raise ValueError(f"mask {path} is {image.width} px wide, expected level 3")
    return cell_mean((mask > 0).astype(np.float32), CELL // size, grid, 0.0) >= 0.5


def save_preview(
    path: Path,
    reference: np.ndarray,
    checked: np.ndarray,
    holes: np.ndarray,
    missed: np.ndarray,
) -> None:
    marked = checked.copy()
    marked[missed] = (0, 90, 255)
    marked[holes] = (255, 0, 0)
    gap = np.zeros((4, reference.shape[1], 3), dtype=np.float32)
    image = Image.fromarray(
        np.vstack([reference, gap, checked, gap, marked]).astype(np.uint8)
    )
    if image.width > PREVIEW_WIDTH:
        image = image.resize(
            (PREVIEW_WIDTH, max(1, round(image.height * PREVIEW_WIDTH / image.width))),
            Image.Resampling.BOX,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def tile_hits(
    holes: np.ndarray, tiles: pd.DataFrame, tile_extent: int, downsample: float
) -> tuple[int, int]:
    """(tiles with any hole, tiles at least TILE_HIT over a hole); tiles in level px."""
    summed = np.pad(holes.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    x0 = (tiles["x"].to_numpy() * downsample / CELL).astype(int)
    y0 = (tiles["y"].to_numpy() * downsample / CELL).astype(int)
    span = max(1, round(tile_extent * downsample / CELL))
    x1 = np.minimum(x0 + span, holes.shape[1])
    y1 = np.minimum(y0 + span, holes.shape[0])
    x0, y0 = np.minimum(x0, holes.shape[1]), np.minimum(y0, holes.shape[0])
    count = summed[y1, x1] - summed[y0, x1] - summed[y1, x0] + summed[y0, x0]
    share = count / (span * span)
    return int((share > 0).sum()), int((share >= TILE_HIT).sum())


def check_slide(task: dict[str, Any]) -> None:
    out = task["output_dir"] / "rows" / f"{task['key']}.parquet"
    if out.exists():
        return
    rows = []
    start = time.monotonic()
    opening = np.ones((3, 3), dtype=bool)
    try:
        with OpenSlide(task["path"]) as slide:
            width, height = slide.dimensions
            grid = (-(-height // CELL), -(-width // CELL))
            present = [round(d) for d in slide.level_downsamples]
            levels = {d: present.index(d) for d in FINE + COARSE if d in present}
            share, rgb = {}, {}
            for downsample in (d for d in COARSE if d in levels):
                share[downsample], rgb[downsample] = read_whole(slide, downsample, grid)
            stained = {
                d: (share[d] < CONTENT_MAX_BLANK)
                & (saturation(rgb[d]) >= STAIN_MIN_SATURATION)
                for d in share
            }
            mask = read_tissue_mask(task, grid)
            has_mask = mask is not None
            if mask is None:
                mask = np.zeros(grid, dtype=bool)
            seed = np.logical_or.reduce([*stained.values(), mask])
            blocks = to_blocks(ndimage.binary_dilation(seed, iterations=MARGIN_CELLS))
            searched = to_cells(blocks, grid)
            for downsample in (d for d in FINE if d in levels):
                share[downsample], rgb[downsample] = read_blocks(
                    slide, downsample, grid, blocks
                )
            content = {d: share[d] < CONTENT_MAX_BLANK for d in share}
            stained = {
                d: content[d] & (saturation(rgb[d]) >= STAIN_MIN_SATURATION)
                for d in share
            }
            tissue = np.logical_or.reduce(list(stained.values()))
            missed = ndimage.binary_opening(
                tissue & ~ndimage.binary_dilation(mask, iterations=MASK_SLACK_CELLS),
                structure=opening,
            )
            _, largest_missed = largest_region(missed)
            grids = {
                "mask": mask,
                "tissue": tissue,
                "missed": missed,
                **{f"content_L{levels[d]}": content[d] for d in content},
                **{f"stained_L{levels[d]}": stained[d] for d in stained},
            }
            for level in sorted(task["levels"]):
                checked = present[level]
                others = [d for d in content if d != checked]
                holes = ndimage.binary_opening(
                    np.logical_or.reduce([content[d] for d in others])
                    & (share[checked] >= HOLE_MIN_BLANK)
                    & searched,
                    structure=opening,
                )
                tissue_holes = holes & np.logical_or.reduce(
                    [stained[d] for d in others]
                )
                grids[f"holes_L{level}"] = holes
                n_tissue_holes, largest = largest_region(tissue_holes)
                row = {
                    **task["info"],
                    "level": level,
                    "levels": ",".join(str(levels[d]) for d in sorted(levels)),
                    "has_mask": has_mask,
                    "tissue_cells": int(tissue.sum()),
                    "hole_cells": int(holes.sum()),
                    "hole_tissue_cells": int(tissue_holes.sum()),
                    "tissue_lost": float(tissue_holes.sum() / max(tissue.sum(), 1)),
                    "n_tissue_holes": n_tissue_holes,
                    "largest_tissue_hole_cells": largest,
                    "mask_cells": int(mask.sum()),
                    "hole_mask_cells": int((tissue_holes & mask).sum()),
                    "mask_missed_cells": int(missed.sum()),
                    "mask_missed_fraction": float(missed.sum() / max(tissue.sum(), 1)),
                    "largest_mask_missed_cells": largest_missed,
                    "error": "",
                } | {
                    f"seen_at_L{levels[d]}": int((tissue_holes & stained[d]).sum())
                    for d in others
                }
                tiling = task["levels"][level]
                touched, hit = tile_hits(
                    holes,
                    tiling["tiles"],
                    tiling["tile_extent"],
                    slide.level_downsamples[level],
                )
                row |= {
                    "n_tiles": len(tiling["tiles"]),
                    "tiles_touched": touched,
                    "tiles_hit": hit,
                }
                rows.append(row)
                if max(largest, largest_missed) >= PREVIEW_MIN_CELLS:
                    save_preview(
                        task["output_dir"] / "previews" / f"{task['key']}_L{level}.png",
                        rgb[27] if 27 in rgb else rgb[min(rgb)],
                        rgb[checked],
                        tissue_holes,
                        missed,
                    )
            grids_path = task["output_dir"] / "grids" / f"{task['key']}.npz"
            grids_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(grids_path, **grids)
    except Exception:
        rows = [{**task["info"], "level": -1, "error": traceback.format_exc(limit=3)}]
    out.parent.mkdir(parents=True, exist_ok=True)
    seconds = time.monotonic() - start
    pd.DataFrame(rows).assign(
        seconds=seconds, file_bytes=os.path.getsize(task["path"])
    ).to_parquet(out)
    print(
        f"{task['key']}: "
        + ", ".join(
            f"L{r['level']} tissue lost {r.get('tissue_lost', float('nan')):.1%}"
            for r in rows
        )
        + (
            " ERROR"
            if rows[0]["error"]
            else f", mask missed {rows[0]['mask_missed_fraction']:.1%}"
        ),
        flush=True,
    )


def split_dir(path: Path) -> Path:
    """The split directory of a table: `<split>/slides.parquet` or `<split>/slides/*.parquet`."""
    return path.parent if path.stem in ("slides", "tiles") else path.parent.parent


def tiling_tasks(output_dir: Path) -> list[dict[str, Any]]:
    """One task per slide, holding each tiled level's tiles."""
    tasks: dict[str, dict[str, Any]] = {}
    for dataset, uris in TILINGS.items():
        for uri in uris:
            root = output_dir / "tiling" / uri.split("/")[-2]
            if not root.exists():
                mlflow.artifacts.download_artifacts(
                    artifact_uri=uri, dst_path=str(root)
                )
            files = sorted(root.rglob("*.parquet"))
            for split in sorted({split_dir(f) for f in files}):
                tables = {
                    kind: [
                        f
                        for f in files
                        if split_dir(f) == split and kind in (f.stem, f.parent.name)
                    ]
                    for kind in ("slides", "tiles")
                }
                slides = pd.concat(map(pd.read_parquet, tables["slides"]))
                tiles = pd.concat(
                    pd.read_parquet(f, columns=["slide_id", "x", "y"])
                    for f in tables["tiles"]
                )
                by_slide = tiles.groupby("slide_id").indices
                source = f"{dataset}/{split.name.split(' - ')[0]}"
                for slide in slides.to_dict("records"):
                    name = Path(slide["path"]).stem
                    task = tasks.setdefault(
                        slide["path"],
                        {
                            "key": f"{dataset}__{source.split('/')[1]}__{name}",
                            "path": slide["path"],
                            "levels": {},
                            "info": {
                                "dataset": dataset,
                                "source": source,
                                "slide_name": name,
                                "path": slide["path"],
                                "nancy_index": slide.get("nancy_index"),
                                "clarity": slide.get("clarity"),
                            },
                            "output_dir": output_dir,
                        },
                    )
                    task["levels"][int(slide["level"])] = {
                        "tiles": tiles.iloc[by_slide.get(slide["id"], [])],
                        "tile_extent": int(slide["tile_extent_x"]),
                    }
    return list(tasks.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--processes", type=int, default=len(os.sched_getaffinity(0)))
    parser.add_argument(
        "--limit", type=int, default=None, help="first N slides, for a trial"
    )
    parser.add_argument("--only", nargs="*", default=None, help="slide names to check")
    args = parser.parse_args()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", TRACKING_URI))

    tasks = tiling_tasks(args.output_dir)
    if args.only:
        tasks = [t for t in tasks if t["info"]["slide_name"] in args.only]
    # Biggest files first, so the slowest slides don't run last on one core.
    tasks.sort(key=lambda t: -os.path.getsize(t["path"]))
    tasks = tasks[: args.limit]
    print(
        f"{len(tasks)} slides, {args.processes} processes -> {args.output_dir}",
        flush=True,
    )

    with Pool(args.processes) as pool:
        for _ in pool.imap_unordered(check_slide, tasks):
            pass

    rows = pd.concat(
        [
            pd.read_parquet(f)
            for f in sorted((args.output_dir / "rows").glob("*.parquet"))
        ],
        ignore_index=True,
    )
    rows.to_parquet(args.output_dir / "holes.parquet")
    checked = rows[rows["level"] >= 0]
    print(f"errors: {int((rows['level'] < 0).sum())}", flush=True)
    print(
        f"without a tissue mask: {int((~checked['has_mask']).sum())} rows", flush=True
    )
    for (dataset, level), group in checked.groupby(["dataset", "level"]):
        affected = group[group["largest_tissue_hole_cells"] >= PREVIEW_MIN_CELLS]
        missed = group[group["largest_mask_missed_cells"] >= PREVIEW_MIN_CELLS]
        tissue = max(group["tissue_cells"].sum(), 1)
        print(
            f"{dataset} L{level}: {len(group)} slides, {len(affected)} with a tissue "
            f"hole of >= {PREVIEW_MIN_CELLS} cells, tissue lost overall "
            f"{group['hole_tissue_cells'].sum() / tissue:.2%}; {len(missed)} where "
            f"the mask missed >= {PREVIEW_MIN_CELLS} cells, overall "
            f"{group['mask_missed_cells'].sum() / tissue:.2%}; tiles hit "
            f"{int(group['tiles_hit'].sum())} of {int(group['n_tiles'].sum())}",
            flush=True,
        )
        columns = [
            "slide_name",
            "source",
            "tissue_lost",
            "largest_tissue_hole_cells",
            "mask_missed_fraction",
            "tiles_hit",
            "n_tiles",
        ]
        columns += [c for c in group if c.startswith("seen_at_")]
        print(
            group.sort_values("tissue_lost", ascending=False)[columns]
            .head(30)
            .to_string(),
            flush=True,
        )


if __name__ == "__main__":
    main()
