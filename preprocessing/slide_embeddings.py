import asyncio
import os
import tempfile
from pathlib import Path

import hydra
import mlflow.data.pandas_dataset
import pandas as pd
import torch
from aiohttp import (
    ClientError,
    ClientSession,
    ClientTimeout,
)
from lightning.pytorch.loggers import Logger
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig
from rationai.mlkit.autolog import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader
from tqdm import tqdm

from ulcerative_colitis.data.datasets import TileEmbeddingsPredict


async def post_request(
    session: ClientSession,
    data: bytes,
    semaphore: asyncio.Semaphore,
    config: DictConfig,
    length: int,
) -> list[float] | None:
    async with semaphore:
        url = f"{config.url}/{length}"
        try:
            async with session.post(
                url,
                data=data,
                headers={"Content-Type": "application/octet-stream"},
            ) as response:
                try:
                    response.raise_for_status()
                    result = await response.json()
                    return result["embeddings"][-1]
                except ClientError as e:
                    print(f"Request failed: {e}")
                    return None
        except TimeoutError:
            print(f"Request to {url} timed out.")
            return None


async def repeatable_post_request(
    session: ClientSession,
    data: bytes,
    semaphore: asyncio.Semaphore,
    config: DictConfig,
    length: int,
    slide_id: str,
) -> tuple[str, list[float] | None]:
    for _ in range(config.num_repeats):
        embedding = await post_request(session, data, semaphore, config, length)
        if embedding is None:
            print(f"Request failed for slide {slide_id}, retrying...")
            continue

        return slide_id, embedding
    return slide_id, None


async def slide_embeddings(
    dataset: TileEmbeddingsPredict,
    config: DictConfig,
) -> pd.DataFrame:
    semaphore = asyncio.Semaphore(config.request_limit)
    timeout = ClientTimeout(total=config.connection_parameters.request_timeout)
    async with ClientSession(timeout=timeout) as session:
        with tqdm(total=len(dataset), desc="Processing slides") as pbar:
            pending = set()
            results = []
            for x, metadata in DataLoader(dataset, batch_size=None):
                coords = torch.stack([metadata["x"], metadata["y"]], dim=-1)

                pending.add(
                    asyncio.create_task(
                        repeatable_post_request(
                            session=session,
                            semaphore=semaphore,
                            config=config.connection_parameters,
                            data=x.numpy().tobytes() + coords.numpy().tobytes(),
                            length=len(x),
                            slide_id=str(metadata["slide_id"]),
                        )
                    )
                )

                if len(pending) >= 2 * config.request_limit:
                    done, pending = await asyncio.wait(
                        pending, return_when=asyncio.FIRST_COMPLETED
                    )

                    for d in done:
                        results.append(d.result())
                        pbar.update(1)

            results.extend(await asyncio.gather(*pending))
            pbar.update(len(pending))

    return pd.DataFrame(results, columns=["slide_id", "embedding"])


def save_mlflow_dataset(
    df: pd.DataFrame, dataset_name: str, logger: MLFlowLogger
) -> None:
    with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp_dir:
        path = Path(tmp_dir) / "slide_embeddings.parquet"
        df.to_parquet(path, index=False)

        logger.experiment.log_artifact(logger.run_id, path, "slide_embeddings")

    slide_dataset = mlflow.data.pandas_dataset.from_pandas(
        df,
        name=dataset_name,
    )

    mlflow.log_input(slide_dataset, context="embeddings")


@hydra.main(config_path="../configs", config_name="slide_embeddings", version_base=None)
@autolog
def main(config: DictConfig, logger: Logger | None = None) -> None:
    assert logger is not None, "Need logger"
    assert isinstance(logger, MLFlowLogger), "Need MLFlowLogger"
    assert isinstance(logger.experiment, MlflowClient), "Need MlflowClient"
    assert logger.run_id is not None, "Need run_id"

    splits = len(config.tiling_uris)
    embeddings_uris = [config.embeddings_uri] * splits
    embeddings_folders = [Path(config.embeddings_folder)] * splits

    dataset = TileEmbeddingsPredict(
        tiling_uris=config.tiling_uris,
        embeddings_uris=embeddings_uris,
        embeddings_folders=None if config.embeddings_download else embeddings_folders,
        padding=False,
    )

    slide_embeddings_df = asyncio.run(slide_embeddings(dataset, config))
    save_mlflow_dataset(
        slide_embeddings_df, f"slide_embeddings - {config.cohort}", logger
    )


if __name__ == "__main__":
    main()
