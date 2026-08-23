import shutil
from pathlib import Path
from typing import Any

import httpx
import hydra
import mlflow.artifacts
import numpy as np
import pandas as pd
import pyarrow as pa
import ray
from omegaconf import DictConfig
from rationai import AsyncClient  # type: ignore[attr-defined]
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.tiling.read_slide_tiles import read_slide_tiles
from ray.data.expressions import col


class EmbedTiles:
    """Embed one tile per call and explode its tokens into per-token rows.

    Still exactly one model call per tile - the expensive, network-bound,
    carefully concurrency-tuned part is unchanged. Only the *output shape*
    changes: instead of one row per tile holding all tokens flattened into a
    single large list (which forces every downstream consumer to decode and
    reshape the whole thing just to reach individual patch tokens), this
    yields one small row per token, tagged `kind` ("patch" or "cls") so
    `main`'s single partitioned `write_parquet` call can split them into two
    physically separate tables without a second pass over this step.
    """

    def __init__(self, model: str, concurrency: int, pool_tokens: str, cls_token_index: int) -> None:
        self.model = f"{model}/"
        self.client = AsyncClient(
            limits=httpx.Limits(
                max_connections=concurrency, max_keepalive_connections=concurrency
            ),
            timeout=500,
        )
        self.pool_tokens = pool_tokens
        self.cls_token_index = cls_token_index

    async def __call__(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        embedding = (
            await self.client.models.embed_image(
                self.model, row["tile"], pool_tokens=self.pool_tokens
            )
        ).astype(np.float32)
        base = {k: v for k, v in row.items() if k != "tile"}
        return explode_tokens(base, embedding, self.cls_token_index)


def explode_tokens(
    base: dict[str, Any], embedding: np.ndarray, cls_token_index: int
) -> list[dict[str, Any]]:
    """Turn one tile's token array into one output row per token.

    Pulled out of `EmbedTiles.__call__` as a pure function so the row-shaping
    logic (kind tagging, patch_index renumbering) is unit-testable without an
    actual model server.

    Args:
        base: Per-tile metadata to copy onto every output row (slide_id, x,
            y, ...) - anything except the tile image / raw embedding.
        embedding: This tile's tokens, shape (n_tokens, embed_dim).
        cls_token_index: Index of the CLS/summary token among `embedding`'s
            rows; every other row becomes a "patch" row.

    Returns:
        One dict per token, each with `kind` ("patch" or "cls"),
        `patch_index` (contiguous 0..n_tokens-2 for patch rows, None for the
        CLS row), and `embedding` (that single token, as a list).
    """
    rows = []
    for token_index, token in enumerate(embedding):
        is_cls = token_index == cls_token_index
        rows.append(
            {
                **base,
                "kind": "cls" if is_cls else "patch",
                # Position among the non-CLS tokens only (0..P-2), not the
                # raw token index, so patches.parquet's patch_index is
                # contiguous rather than having a CLS-shaped hole.
                "patch_index": None if is_cls else token_index - (token_index > cls_token_index),
                "embedding": token.tolist(),
            }
        )
    return rows


def subsample(
    slides: pd.DataFrame, tiles: pd.DataFrame, slides_per_index: int, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Subsample slides and tiles to a maximum number of slides per index.

    Args:
        slides: DataFrame containing slide information.
        tiles: DataFrame containing tile information.
        slides_per_index: Maximum number of slides to keep per index.
        random_state: Random seed for reproducibility.

    Returns:
        A tuple containing the subsampled slides and tiles DataFrames.
    """
    # Group slides by index and sample a maximum of slides_per_index slides per group
    sampled_slides = (
        slides.groupby("nancy_index", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), slides_per_index), random_state=random_state))
        .reset_index(drop=True)
    )

    # Filter tiles to only include those corresponding to the sampled slides
    sampled_tiles = tiles[tiles["slide_id"].isin(sampled_slides["id"])]
    sampled_tiles = sampled_tiles[
        (sampled_tiles["x"] % 224 == 0) & (sampled_tiles["y"] % 224 == 0)
    ].reset_index(drop=True)

    return sampled_slides, sampled_tiles


@with_cli_args(["+preprocessing=embeddings_xai"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    uri = config.dataset.mlflow_uris.tiling.train
    folder = Path(mlflow.artifacts.download_artifacts(uri))
    slides = pd.read_parquet(folder / "slides.parquet")
    tiles = pd.read_parquet(folder / "tiles.parquet")

    slides, tiles = subsample(
        slides, tiles, config.slides_per_index, config.random_state
    )

    slide_info = slides.set_index("id")[
        ["path", "level", "tile_extent_x", "tile_extent_y"]
    ]
    tiles_enriched = tiles.join(slide_info, on="slide_id", how="inner")

    ds = ray.data.from_arrow(
        pa.Table.from_pandas(tiles_enriched, preserve_index=False)
    ).repartition(target_num_rows_per_block=config.block_size)
    ds = ds.with_column(
        "tile",
        read_slide_tiles(  # pyright: ignore[reportCallIssue]
            col("path"),
            col("x"),
            col("y"),
            col("tile_extent_x"),
            col("tile_extent_y"),
            col("level"),
        ),
        num_cpus=1,
        memory=4 * 1024**3,
    )
    ds = ds.drop_columns(["path", "level", "tile_extent_x", "tile_extent_y"])
    ds = ds.flat_map(
        EmbedTiles,  # pyright: ignore[reportArgumentType]
        fn_constructor_args=(
            config.model,
            config.concurrency,
            config.pool_tokens,
            config.cls_token_index,
        ),
        compute=ray.data.ActorPoolStrategy(
            max_size=4,
            max_tasks_in_flight_per_actor=config.concurrency // 4,
        ),
        max_concurrency=config.concurrency,
    )

    split_dir = Path(config.output_dir) / "train"
    # Wipe the whole split dir up front, not just the tokens subdir: every
    # run rewrites all of it anyway (nothing here is incremental), and
    # `logger.log_artifacts` below uploads everything still sitting under
    # `split_dir` - a stale subdir from a previous run under a different
    # output layout (e.g. this schema's predecessor, a flat "tiles/" dir)
    # would otherwise get uploaded to mlflow right alongside the real output.
    if split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True)
    tokens_parquet_dir = split_dir / "tokens"

    slides.to_parquet(split_dir / "slides.parquet", index=False)
    # partition_cols=["kind"] splits the single pass over EmbedTiles into two
    # independently-readable tables - tokens/kind=patch/*.parquet and
    # tokens/kind=cls/*.parquet - without re-running the embedding step or
    # materializing the (~corpus-sized) dataset to derive two writes from it.
    ds.write_parquet(
        str(tokens_parquet_dir),
        partition_cols=["kind"],
        min_rows_per_file=config.rows_per_file,
    )

    logger.log_artifacts(str(split_dir), f"train - {config.dataset.institution}")


if __name__ == "__main__":
    ctx = ray.data.DataContext.get_current()
    ctx.enable_rich_progress_bars = True
    ctx.use_ray_tqdm = False

    with ray.init(runtime_env={"excludes": [".git", ".venv"]}):
        main()
