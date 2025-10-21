import asyncio
import os
import tempfile
from pathlib import Path

import hydra
import mlflow.data.pandas_dataset
import pandas as pd
import requests
import torch
from aiohttp import (
    ClientError,
    ClientResponse,
    ClientSession,
    ClientTimeout,
    TCPConnector,
)
from lightning.pytorch.loggers import Logger
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig
from rationai.mlkit.autolog import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader
from tqdm import tqdm

from ulcerative_colitis.data.datasets import TileEmbeddingsPredict


async def _post_request(
    session: ClientSession,
    data: bytes,
    semaphore: asyncio.Semaphore,
    config: DictConfig,
    length: int,
) -> ClientResponse:
    async with semaphore:
        url = f"{config.url}/{length}"
        try:
            async with session.post(
                url,
                data=data,
                headers={"Content-Type": "application/octet-stream"},
            ) as response:
                return response
        except Exception as e:
            print(f"Error during request to {url}: {e}")
            raise


async def post_request(
    session: ClientSession,
    data: bytes,
    semaphore: asyncio.Semaphore,
    config: DictConfig,
    length: int,
) -> ClientResponse:
    timeout = ClientTimeout(
        total=config.request_timeout, sock_read=config.request_timeout
    )

    print(f"Sending request to {config.url}/{length}...")
    async with (
        semaphore,
        session.post(
            f"{config.url}/{length}",
            data=data,
            timeout=timeout,
            headers={"Content-Type": "application/octet-stream"},
        ) as response,
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
            response = await _post_request(session, data, semaphore, config, length)

            print(f"Response status for slide {slide_id}: {response.status}")
            response.raise_for_status()
            print("Response OK, parsing...")
            result = await response.json()
            print(f"Request succeeded for slide {slide_id}. ✅")

            return slide_id, result["embeddings"][-1]

        except ClientError as e:
            print(f"Request failed for slide {slide_id}, retrying... Error: {e}")
            continue

    return slide_id, None


async def _slide_embeddings(
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
            # if pending:
            #     done = await asyncio.gather(*pending)
            #     for d in done:
            #         results.append(d)
            #         pbar.update(1)

    return pd.DataFrame(results, columns=["slide_id", "embedding"])


async def slide_embeddings(
    dataset: TileEmbeddingsPredict,
    semaphore: asyncio.Semaphore,
    config: DictConfig,
) -> pd.DataFrame:
    print(f"Using embedding server at {config.connection_parameters.url}")
    connector = TCPConnector(limit_per_host=config.request_limit)
    async with ClientSession(connector=connector) as session:
        pending = set()
        results = []
        for x, metadata in DataLoader(dataset, batch_size=None):
            coords = torch.stack([metadata["x"], metadata["y"]], dim=-1)

            result = requests.post(
                f"{config.connection_parameters.url}/{len(x)}",
                data=x.numpy().tobytes() + coords.numpy().tobytes(),
                timeout=config.connection_parameters.request_timeout,
                headers={"Content-Type": "application/octet-stream"},
            )
            results.append((str(metadata["slide_id"]), result.json()["embeddings"][-1]))
            # result = await asyncio.create_task(
            #     repeatable_post_request(
            #         session=session,
            #         semaphore=semaphore,
            #         config=config.connection_parameters,
            #         data=x.numpy().tobytes() + coords.numpy().tobytes(),
            #         length=len(x),
            #         slide_id=str(metadata["slide_id"]),
            #     )
            # )
            # results.append(result)
            # pending.add(
            #     asyncio.create_task(
            #         repeatable_post_request(
            #             session=session,
            #             semaphore=semaphore,
            #             config=config.connection_parameters,
            #             data=x.numpy().tobytes() + coords.numpy().tobytes(),
            #             length=len(x),
            #             slide_id=str(metadata["slide_id"]),
            #         )
            #     )
            # )

            # if len(pending) >= config.request_limit:
            #     done, pending = await asyncio.wait(
            #         pending, return_when=asyncio.FIRST_COMPLETED
            #     )

            #     for d in done:
            #         results.append(d.result())

        print(f"Sending {len(pending)} requests to the embedding server...")
        results.extend(await asyncio.gather(*pending))
        print("All requests completed.")

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

    # semaphore = asyncio.Semaphore(config.request_limit)

    splits = len(config.tiling_uris)
    embeddings_uris = [config.embeddings_uri] * splits
    embeddings_folders = [Path(config.embeddings_folder)] * splits

    dataset = TileEmbeddingsPredict(
        tiling_uris=config.tiling_uris,
        embeddings_uris=embeddings_uris,
        embeddings_folders=None if config.embeddings_download else embeddings_folders,
        padding=False,
    )

    slide_embeddings_df = asyncio.run(_slide_embeddings(dataset, config))
    save_mlflow_dataset(
        slide_embeddings_df, f"slide_embeddings - {config.cohort}", logger
    )

    # slide_embeddings_dfs = []
    # embeddings_folder = Path(config.embeddings_folder)
    # for split in ["train", "test preliminary", "test final"]:
    #     print(f"Processing {split} set...")
    #     tiling_uri = f"{config.tiling_uri}/{split} - {config.cohort}"

    #     print(f"Loading dataset from {tiling_uri}...")
    #     dataset = TileEmbeddingsPredict(
    #         tiling_uri=tiling_uri,
    #         embeddings_uri=config.embeddings_uri,
    #         embeddings_folder=None if config.embeddings_download else embeddings_folder,
    #         padding=False,
    #     )

    #     print(f"Computing slide embeddings for {len(dataset)} slides...")
    #     slide_embeddings_dfs.append(
    #         asyncio.run(
    #             slide_embeddings(
    #                 dataset=dataset,
    #                 semaphore=semaphore,
    #                 config=config,
    #             )
    #         )
    #     )

    # print("Saving slide embeddings...")
    # save_mlflow_dataset(
    #     pd.concat(slide_embeddings_dfs), f"slide_embeddings - {config.cohort}", logger
    # )


if __name__ == "__main__":
    main()
