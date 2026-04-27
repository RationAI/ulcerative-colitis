import shutil
from pathlib import Path

import hydra
import mlflow.artifacts
import pandas as pd
import ray
from omegaconf import DictConfig
from rationai import AsyncClient
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.tiling.read_slide_tiles import read_slide_tiles
from ray.data.expressions import col


class EmbedTiles:
    def __init__(self, model: str) -> None:
        self.model = model
        self.client = AsyncClient()

    async def __call__(self, row: dict) -> dict:
        embedding = (
            (await self.client.models.embed_image(self.model, row["tile"]))
            .reshape(-1)
            .tolist()
        )
        del row["tile"]
        row["embedding"] = embedding
        return row


@with_cli_args(["+preprocessing=embeddings"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    for name, split_uri in config.dataset.mlflow_uris.tiling.items():
        folder = Path(mlflow.artifacts.download_artifacts(split_uri))
        slides = pd.read_parquet(folder / "slides.parquet")
        tiles = pd.read_parquet(folder / "tiles.parquet")

        slide_info = slides.set_index("id")[
            ["path", "level", "tile_extent_x", "tile_extent_y"]
        ]
        tiles_enriched = tiles.join(slide_info, on="slide_id")

        ds = ray.data.from_pandas(tiles_enriched)
        ds = ds.with_column(
            "tile",
            read_slide_tiles(
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
            EmbedTiles,  # type: ignore[arg-type]
            fn_constructor_args=(config.model,),
            compute=ray.data.ActorPoolStrategy(
                max_tasks_in_flight_per_actor=config.concurrency
            ),
            max_concurrency=config.concurrency,
        )

        split_dir = Path(config.output_dir) / str(name)
        split_dir.mkdir(parents=True, exist_ok=True)
        tiles_parquet_dir = split_dir / "tiles.parquet"
        if tiles_parquet_dir.exists():
            shutil.rmtree(tiles_parquet_dir)

        slides.to_parquet(split_dir / "slides.parquet", index=False)
        ds.write_parquet(str(tiles_parquet_dir))

        logger.log_artifacts(str(split_dir), f"{name} - {config.dataset.institution}")


if __name__ == "__main__":
    ctx = ray.data.DataContext.get_current()
    ctx.enable_rich_progress_bars = True
    ctx.use_ray_tqdm = False

    with ray.init(runtime_env={"excludes": [".git", ".venv"]}):
        main()
