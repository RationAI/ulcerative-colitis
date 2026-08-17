import shutil
from pathlib import Path
from typing import Any

import httpx
import hydra
import mlflow.artifacts
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
    def __init__(self, model: str, concurrency: int, pool_tokens: str) -> None:
        self.model = f"{model}/"
        self.client = AsyncClient(
            limits=httpx.Limits(
                max_connections=concurrency, max_keepalive_connections=concurrency
            ),
            timeout=500,
        )
        self.pool_tokens = pool_tokens

    async def __call__(self, row: dict[str, Any]) -> dict[str, Any]:
        embedding = (
            (
                await self.client.models.embed_image(
                    self.model, row["tile"], pool_tokens=self.pool_tokens
                )
            )
            .reshape(-1)
            .tolist()
        )
        del row["tile"]
        row["embedding"] = embedding
        return row


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
    ds = ds.map(
        EmbedTiles,  # pyright: ignore[reportArgumentType]
        fn_constructor_args=(config.model, config.concurrency, config.pool_tokens),
        compute=ray.data.ActorPoolStrategy(
            max_size=4,
            max_tasks_in_flight_per_actor=config.concurrency // 4,
        ),
        max_concurrency=config.concurrency,
    )

    split_dir = Path(config.output_dir) / "train"
    split_dir.mkdir(parents=True, exist_ok=True)
    tiles_parquet_dir = split_dir / "tiles"
    if tiles_parquet_dir.exists():
        shutil.rmtree(tiles_parquet_dir)

    slides.to_parquet(split_dir / "slides.parquet", index=False)
    ds.write_parquet(str(tiles_parquet_dir), min_rows_per_file=config.rows_per_file)

    logger.log_artifacts(str(split_dir), f"train - {config.dataset.institution}")


if __name__ == "__main__":
    ctx = ray.data.DataContext.get_current()
    ctx.enable_rich_progress_bars = True
    ctx.use_ray_tqdm = False

    with ray.init(runtime_env={"excludes": [".git", ".venv"]}):
        main()
