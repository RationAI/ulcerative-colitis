import asyncio
import os
import tempfile
from pathlib import Path

import hydra
import mlflow.data.pandas_dataset
import pandas as pd
import torch
from aiohttp import ClientResponse, ClientSession, ClientTimeout
from lightning.pytorch.loggers import Logger
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig
from rationai.mlkit.autolog import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger
from requests import RequestException
from torch.utils.data import DataLoader

from ulcerative_colitis.data.datasets import TileEmbeddingsPredict


async def post_request(
    session: ClientSession,
    data: bytes,
    semaphore: asyncio.Semaphore,
    config: DictConfig,
    length: int,
) -> ClientResponse:
    timeout = ClientTimeout(total=config.request_timeout)

    print(f"Sending request to {config.url}/{length}...")
    async with (
        semaphore,
        session.post(f"{config.url}/{length}", json=data, timeout=timeout) as response,
    ):
        return response


async def repeatable_post_request(
    session: ClientSession,
    data: bytes,
    semaphore: asyncio.Semaphore,
    config: DictConfig,
    length: int,
    slide_id: str,
) -> tuple[str, list[float] | None]:
    for _ in range(config.num_repeats):
        try:
            print(f"Sending request for slide {slide_id} attempt {_ + 1}...")
            response = await post_request(session, data, semaphore, config, length)

            response.raise_for_status()
            result = await response.json()

            return slide_id, result["embeddings"][-1]

        except RequestException:
            print(f"Request failed for slide {slide_id}, retrying...")
            continue

    return slide_id, None


async def slide_embeddings(
    dataset: TileEmbeddingsPredict,
    semaphore: asyncio.Semaphore,
    config: DictConfig,
) -> pd.DataFrame:
    print(f"Using embedding server at {config.connection_parameters.url}")
    async with ClientSession() as session:
        tasks = []
        for x, metadata in DataLoader(dataset, batch_size=None):
            print(f"Processing slide {metadata['slide_id']} with {len(x)} tiles...")
            coords = torch.stack([metadata["x"], metadata["y"]], dim=-1)
            tasks.append(
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

        print(f"Sending {len(tasks)} requests to the embedding server...")
        results = await asyncio.gather(*tasks)
        print("All requests completed.")

    return pd.DataFrame(results, columns=["slide_id", "embedding"])


def save_mlflow_dataset(
    df: pd.DataFrame, dataset_name: str, logger: MLFlowLogger
) -> None:
    with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp_dir:
        path = Path(tmp_dir) / "slide_embeddings.parquet"
        df.to_parquet(path, index=False)

        logger.experiment.log_artifact(logger.run_id, path, dataset_name)

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

    semaphore = asyncio.Semaphore(config.request_limit)

    for split in ["train", "test preliminary", "test final"]:
        print(f"Processing {split} set...")
        tiling_uri = f"{config.tiling_uri}/{split} - {config.cohort}"
        embeddings_uri = f"{config.embeddings_uri}/{split} - {config.cohort}"
        embeddings_folder = Path(config.embeddings_folder)

        print(f"Loading dataset from {tiling_uri}...")
        dataset = TileEmbeddingsPredict(
            tiling_uri=tiling_uri,
            embeddings_uri=embeddings_uri,
            embeddings_folder=None if config.embeddings_download else embeddings_folder,
            padding=False,
        )

        print(f"Computing slide embeddings for {len(dataset)} slides...")
        slide_embeddings_df = asyncio.run(
            slide_embeddings(
                dataset=dataset,
                semaphore=semaphore,
                config=config,
            )
        )

        print(f"Saving slide embeddings to {split} - {config.cohort}...")
        save_mlflow_dataset(slide_embeddings_df, f"{split} - {config.cohort}", logger)


if __name__ == "__main__":
    main()
