"""Build per-slide NMF concept masks from patch tokens and a fitted H dictionary.

For every slide, assembles K continuous "concept intensity" masks (one per
NMF component) plus one discrete argmax mask, at patch-token-native
resolution (1 mask pixel per 14x14 source-pixel patch - Virchow2's 16x16
token grid over each 224x224 tile, see concept_mil.tex / the
explainability-pipeline-code memory) via `ratiopath.masks.mask_builders.
MaskBuilder`. Masks are *not* upsampled back to the slide's full
tiled-resolution extent - the downstream visualization tool handles that -
so each is written straight from `MaskBuilder.finalize()`'s own resolution,
with `write_big_tiff`'s `mpp_x`/`mpp_y` scaled up by `PATCH_PIXELS` to match.

Output layout is grouped by mask type first, then slide:
`output_dir/concept_{k}/{slide_id}.tiff` and `output_dir/argmax/{slide_id}.tiff`.
The whole `output_dir` (all masks + manifest.json) is logged to mlflow, not
just kept on the project mount - unlike `nmf_fit.py`'s W or the patch/cls
token tables, these are small enough (patch-grid resolution, not source
resolution) not to need the large-artifact local-mount-only treatment.

W is *not* reused from `nmf_fit.py`'s saved `w.f32.npy` - that's tied to
whatever slide subset `nmf_fit.py`'s own training/transform pass happened to
touch (the subsampled, non-overlapping "train" split). This script instead
recomputes each patch's concept weights on demand by transforming its token
through the already gauge-fixed, already scale-recovered `H` loaded from
mlflow, so it can run against *any* embeddings_xai split independently of
what `nmf_fit.py` processed - e.g. the overlapping-tile "test_preliminary"
split (`filter_tiles: false`), which is why `MeanAggregator` (blending
overlapping patches' estimates) matters here at all; the non-overlapping
"train" split wouldn't need it.

**Unverified assumption:** `patch_index -> (row, col)` in each tile's 16x16
token grid is assumed row-major / x-fastest (`patch_index % 16, patch_index
// 16`) - the standard ViT/timm patchify convention (also matches
`ratiopath.tiling.grid_tiles`'s own x-fastest doc convention) - but the
actual Virchow2 model server's token order was never verified directly
against this (out of reach from this repo - the model runs behind
`rationai.AsyncClient`, see explainability-status memory). Wrong ordering
would silently transpose/mirror concept locations *within* each tile without
any error - re-verify this first if masks ever look spatially wrong at the
tile level (slide-level placement is unaffected either way, since that comes
from each tile's own `x`/`y`, not `patch_index`).

**Every patch is its own MaskBuilder "tile"**, rather than reassembling each
real 224px tile's 16x16 token grid before writing it. `PATCH_PIXELS` (14)
evenly divides both `tile_extent` (224) and the tiling stride (112 in
`configs/preprocessing/tiling.yaml`), so every patch's true absolute pixel
coordinate (`tile_x + (patch_index % 16)*PATCH_PIXELS`, `tile_y +
(patch_index // 16)*PATCH_PIXELS`) always lands exactly on a slide-wide
14px grid, regardless of which (possibly overlapping) real tile it came
from. That lets `MaskBuilder` be configured per-patch
(`source_tile_extent=stride=PATCH_PIXELS`, `output_tile_extent=1`) instead
of per-tile, so overlapping patches from different tiles that cover the
same underlying pixels land on the same mask cell and get mean-aggregated -
identical result to reassembling 16x16 grids first, but needing no
buffering: coordinates are computed vectorized per `ray.data` batch, and
`update_batch` is called once per distinct slide present in that batch
(grouped via `pandas.DataFrame.groupby`), not once per patch or per tile.
"""

import json
import time
from pathlib import Path

import hydra
import mlflow.artifacts
import numpy as np
import pandas as pd
import pyvips
import ray
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.masks import write_big_tiff
from ratiopath.masks.mask_builders import MaskBuilder, MeanAggregator
from sklearn.decomposition import MiniBatchNMF

from explainability.nmf_fit import (
    iter_patch_batches,
    load_shift,
    resolve_percentile_stats_path,
)
from explainability.tiles import load_slides, load_tokens_dataset, resolve_token_dirs


# Virchow2 is a ViT-H/14: a 224px tile tokenizes into a 224/14 = 16x16 patch
# grid. Expressed as a constant (not hardcoded 16 below) so it stays correct
# if `config.tile_extent` ever changes.
PATCH_PIXELS = 14


def load_h(h_mlflow_uri: str) -> np.ndarray:
    """Download and load nmf_fit's gauge-fixed, scale-recovered H dictionary.

    Args:
        h_mlflow_uri: mlflow artifact URI for a specific nmf_fit run's
            `h.parquet` (already gauge-fixed and in shift-only token space -
            see `explainability/nmf_fit.py`'s recovery/gauge-fix comments).

    Returns:
        Array of shape (n_components, embed_dim).
    """
    path = Path(mlflow.artifacts.download_artifacts(h_mlflow_uri))
    return pd.read_parquet(path).sort_index().to_numpy(dtype=np.float32)


def make_transform_model(h: np.ndarray, config: DictConfig) -> MiniBatchNMF:
    """Build a MiniBatchNMF whose dictionary is the given (pre-fit) H, ready to `.transform()`.

    sklearn's `.transform()` requires the estimator to already look "fitted"
    (`check_is_fitted` looks for attributes only set during fitting, e.g.
    `n_components_`) even though the one thing that actually matters here
    (`components_`) is about to be overwritten anyway. Mirrors
    `explainability/nmf_fit.py`'s own components_-swap-before-transform
    pattern, just standing in a throwaway `fit_transform` on a couple of
    random rows for "already fit" - nothing here is actually trained, `h`
    came pre-fit from mlflow.

    Args:
        h: Dictionary matrix, shape (n_components, embed_dim).
        config: Hydra config - `nmf.init`/`beta_loss`/`random_state` (only
            affects the throwaway dummy fit below, not the loaded `h`).

    Returns:
        A MiniBatchNMF instance with `components_` set to `h`, ready for `.transform()`.
    """
    model = MiniBatchNMF(
        n_components=h.shape[0],
        init=config.nmf.init,
        beta_loss=config.nmf.beta_loss,
        random_state=config.nmf.random_state,
    )
    dummy = np.abs(
        np.random.default_rng(config.nmf.random_state).standard_normal((2, h.shape[1]))
    ).astype(np.float32)
    model.fit_transform(dummy)
    model.components_ = h
    return model


def compress(w: np.ndarray) -> np.ndarray:
    """X / (1 + x): rescale non-negative NMF weights into [0, 1)."""
    return w / (1.0 + w)


def to_uint8(x: np.ndarray) -> np.ndarray:
    """Quantize a [0, 1]-valued array to [0, 255] uint8."""
    return np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)


def patch_coords(metadata: pd.DataFrame, grid_size: int) -> np.ndarray:
    """Compute each patch's absolute (row, col) pixel coordinate in source-level pixel space.

    `patch_index`'s position within a tile's 16x16 token grid is assumed
    row-major / x-fastest (see module docstring's unverified-assumption
    note) - `patch_index % grid_size` is the column, `patch_index //
    grid_size` the row.

    Args:
        metadata: A batch's (slide_id, x, y, patch_index) metadata, as
            yielded by `explainability.nmf_fit.iter_patch_batches` with
            `with_metadata=True`.
        grid_size: Tokens per tile side (16 for a 224px tile - see
            `PATCH_PIXELS`).

    Returns:
        Array of shape (B, 2), each row (y, x) in source-level pixels -
        matching `MaskBuilder`'s (row, col) coordinate order.
    """
    patch_index = metadata["patch_index"].to_numpy()
    row = metadata["y"].to_numpy() + (patch_index // grid_size) * PATCH_PIXELS
    col = metadata["x"].to_numpy() + (patch_index % grid_size) * PATCH_PIXELS
    return np.stack([row, col], axis=1).astype(np.int64)


def to_vips_image(array: np.ndarray) -> pyvips.Image:
    """Wrap a (C, H, W) array as a pyvips.Image (H, W, C), no resize/crop.

    Same array -> pyvips conversion `MaskBuilder.resize_to_source` uses
    internally - used directly here since these masks are written at
    patch-grid-native resolution, without that method's resize-to-source
    step (see module docstring).
    """
    return pyvips.Image.new_from_array(array.transpose(1, 2, 0))


def write_slide_masks(
    slide_id: str,
    builder: MaskBuilder,
    n_components: int,
    output_dir: Path,
    mpp_x: float,
    mpp_y: float,
) -> None:
    """Finalize one slide's MaskBuilder into K quantized concept masks + one argmax mask.

    Written at patch-grid-native resolution (1 pixel per `PATCH_PIXELS`-px
    patch - `MaskBuilder.finalize()`'s own resolution, not resized up to the
    slide's full tiled extent) via `ratiopath.masks.write_big_tiff` (bigtiff
    + DEFLATE + tiled 512x512 pyramid) rather than a hand-rolled
    `pyvips.Image.write_to_file` call - same fixed settings this module
    would otherwise duplicate, plus correct physical resolution metadata and
    matches whatever else gets loaded into xOpat.

    One file per (concept, slide), grouped by mask type first:
    `output_dir/concept_{k}/{slide_id}.tiff`, `output_dir/argmax/{slide_id}.tiff`.

    Args:
        slide_id: Slide identifier - also each output file's basename.
        builder: This slide's MaskBuilder, already fed every one of its patches.
        n_components: K, the number of NMF components/concept masks.
        output_dir: Root output directory.
        mpp_x: This slide's horizontal resolution in µm/pixel (from
            `slides.parquet`) - scaled by `PATCH_PIXELS` before being handed
            to `write_big_tiff`, since each mask pixel covers `PATCH_PIXELS`
            source pixels, not one.
        mpp_y: Vertical resolution in µm/pixel, same scaling.
    """
    result = builder.finalize()
    mask = result["mask"]  # (K, H_mask, W_mask) float32, mean-aggregated raw NMF weights
    covered = result["overlap_counter"][0] > 0  # (H_mask, W_mask) - False where no patch ever wrote

    mask_mpp_x = mpp_x * PATCH_PIXELS
    mask_mpp_y = mpp_y * PATCH_PIXELS

    for k in range(n_components):
        channel = to_uint8(compress(mask[k]))
        channel[~covered] = 0  # explicit background=0, even though this is already a no-op here
        concept_dir = output_dir / f"concept_{k}"
        concept_dir.mkdir(parents=True, exist_ok=True)
        write_big_tiff(
            to_vips_image(channel[None, :, :]),
            concept_dir / f"{slide_id}.tiff",
            mask_mpp_x,
            mask_mpp_y,
        )

    # argmax is invariant to compress() - it's monotonic increasing on [0, inf) -
    # so this is computed directly on the raw (pre-compress) averaged weights.
    argmax_idx = np.argmax(mask, axis=0)  # (H_mask, W_mask); ties -> lowest index
    levels = ((255 * (argmax_idx + 1)) // n_components).astype(np.uint8)
    levels[~covered] = 0
    argmax_dir = output_dir / "argmax"
    argmax_dir.mkdir(parents=True, exist_ok=True)
    write_big_tiff(to_vips_image(levels[None, :, :]), argmax_dir / f"{slide_id}.tiff", mask_mpp_x, mask_mpp_y)


@with_cli_args(["+explainability=concept_masks"])
@hydra.main(config_path="../configs", config_name="explainability", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    h = load_h(config.h.mlflow_uri)
    n_components = h.shape[0]
    if n_components != config.n_components:
        raise ValueError(
            f"config.n_components={config.n_components} doesn't match "
            f"config.h.mlflow_uri's actual {n_components} components"
        )
    model = make_transform_model(h, config)

    stats_path = resolve_percentile_stats_path(config.shift.mlflow_uri)
    shift = load_shift(stats_path, config.shift.percentile_column)
    unscaled = np.ones_like(shift)  # h is already in shift-only space - see nmf_fit.py's transform pass

    token_dirs = resolve_token_dirs(
        config.sources, config.get("local_embeddings_xai_dir"), kind="patch", split=config.split
    )
    patches_ds = load_tokens_dataset(token_dirs)

    slides = load_slides(
        config.sources, config.get("local_embeddings_xai_dir"), split=config.split
    ).set_index("id")

    grid_size = config.tile_extent // PATCH_PIXELS
    builders: dict[str, MaskBuilder] = {}

    def get_builder(slide_id: str) -> MaskBuilder:
        if slide_id not in builders:
            row = slides.loc[slide_id]
            builders[slide_id] = MaskBuilder(
                source_extents=(int(row["extent_y"]), int(row["extent_x"])),
                # Each patch is its own MaskBuilder "tile" - source_tile_extent
                # == stride == PATCH_PIXELS, output_tile_extent=1 (one mask
                # pixel per patch, no internal grid to assemble) - see module
                # docstring for why this is equivalent to reassembling whole
                # 16x16 tiles first.
                source_tile_extent=PATCH_PIXELS,
                output_tile_extent=1,
                stride=PATCH_PIXELS,
                n_channels=n_components,
                storage="memmap",  # disk-backed - every slide's builder stays open for the full pass
                aggregation=MeanAggregator,
                dtype=np.float32,
            )
        return builders[slide_id]

    n_patches = 0
    n_updates = 0
    start = time.monotonic()
    last_log = start
    for patches, metadata in iter_patch_batches(
        patches_ds, config.batch_size, shift, unscaled, with_metadata=True
    ):
        assert metadata is not None  # guaranteed by with_metadata=True above
        w_batch = model.transform(patches)  # (B, K), non-negative
        coords = patch_coords(metadata, grid_size)

        # One update_batch call per distinct slide in this batch (typically far
        # fewer than B) rather than per patch - see module docstring.
        for slide_id, idx in metadata.groupby("slide_id").indices.items():
            get_builder(str(slide_id)).update_batch(w_batch[idx], coords[idx])
            n_updates += 1

        n_patches += w_batch.shape[0]
        now = time.monotonic()
        if now - last_log > 60:
            rate = n_patches / (now - start)
            print(
                f"concept_masks: {n_patches} patches, {len(builders)} slides so far "
                f"({rate:.0f} patches/s)",
                flush=True,
            )
            last_log = now

    print(
        f"concept_masks: processed {n_patches} patches across {len(builders)} slides "
        f"({n_updates} update_batch calls)",
        flush=True,
    )

    for slide_id, builder in builders.items():
        row = slides.loc[slide_id]
        write_slide_masks(
            slide_id, builder, n_components, output_dir, float(row["mpp_x"]), float(row["mpp_y"])
        )
        builder.cleanup()

    manifest = {
        "n_components": n_components,
        "h_mlflow_uri": config.h.mlflow_uri,
        "shift_mlflow_uri": config.shift.mlflow_uri,
        "split": config.split,
        "n_slides": len(builders),
        "n_patches": n_patches,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Log everything - manifest + every concept_{k}/ and argmax/ mask - not
    # just the manifest: at patch-grid resolution these are small enough
    # (unlike nmf_fit.py's W or the patch/cls token tables) not to need
    # project-mount-only treatment.
    logger.log_artifacts(str(output_dir))


if __name__ == "__main__":
    ctx = ray.data.DataContext.get_current()
    ctx.enable_rich_progress_bars = True
    ctx.use_ray_tqdm = False

    # num_cpus deliberately low - same oversized-parquet-row-group root cause
    # as patch_statistics.py/nmf_fit.py (see explainability-status memory).
    # Keep in sync with cpu= in scripts/explainability/concept_masks.py.
    with ray.init(num_cpus=8, runtime_env={"excludes": [".git", ".venv"]}):
        main()
