import asyncio
import shutil
from pathlib import Path

import hydra
import mlflow.artifacts
import pandas as pd
import pyarrow as pa
import rationai
import ray
from omegaconf import DictConfig
from PIL import Image
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.tiling.read_slide_tiles import read_openslide_tiles
from ratiopath.tiling.types import ReadTilesArguments


class EmbedTiles:
    def __init__(self, model: str, concurrency: int) -> None:
        self.model = model
        self.concurrency = concurrency

    async def _embed_all(self, images: list[Image.Image]) -> list[list[float]]:
        semaphore = asyncio.Semaphore(self.concurrency)

        async with rationai.AsyncClient() as client:

            async def embed_one(img: Image.Image) -> list[float]:
                async with semaphore:
                    return (
                        (await client.models.embed_image(self.model, img))
                        .reshape(-1)
                        .tolist()
                    )

            return list(await asyncio.gather(*[embed_one(img) for img in images]))

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        tile_images: list[Image.Image] = []
        batch = batch.reset_index(drop=True)

        for path, group in batch.groupby("path"):
            assert isinstance(path, str)
            kwargs: ReadTilesArguments = {
                "tile_x": pa.array(group["x"].tolist()),
                "tile_y": pa.array(group["y"].tolist()),
                "tile_extent_x": pa.array(group["tile_extent_x"].tolist()),
                "tile_extent_y": pa.array(group["tile_extent_y"].tolist()),
                "level": pa.array(group["level"].tolist()),
            }
            tiles = read_openslide_tiles(path, **kwargs)
            for i in range(len(group)):
                tile_images.append(Image.fromarray(tiles[i]))

        embeddings = asyncio.run(self._embed_all(tile_images))

        return batch.drop(
            columns=["path", "level", "tile_extent_x", "tile_extent_y"]
        ).assign(embedding=list(embeddings))  # type: ignore[call-overload]


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
        ds = ds.map_batches(
            EmbedTiles,
            fn_constructor_args=(config.model, config.concurrency),
            batch_format="pandas",
            batch_size=config.batch_size,
            concurrency=1,
        )

        split_dir = Path(config.output_dir) / str(name)
        split_dir.mkdir(parents=True, exist_ok=True)
        tiles_parquet_dir = split_dir / "tiles.parquet"
        if tiles_parquet_dir.exists():
            shutil.rmtree(tiles_parquet_dir)

        slides.to_parquet(split_dir / "slides.parquet", index=False)
        ds.write_parquet(str(tiles_parquet_dir), max_rows_per_file=100_000)

        logger.log_artifacts(str(split_dir), f"{name} - {config.dataset.institution}")


if __name__ == "__main__":
    ctx = ray.data.DataContext.get_current()
    ctx.enable_rich_progress_bars = True
    ctx.use_ray_tqdm = False

    with ray.init(runtime_env={"excludes": [".git", ".venv"]}):
        main()
