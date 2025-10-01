import asyncio
import tempfile
from pathlib import Path
from typing import Any

import hydra
from aiohttp import ClientConnectionError, ClientSession, ClientTimeout
from lightning.pytorch.loggers import Logger
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig
from rationai.mlkit.autolog import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger


async def put_request(
    session: ClientSession,
    data: dict[str, Any],
    semaphore: asyncio.Semaphore,
    config: DictConfig,
) -> tuple[int, str]:
    timeout = ClientTimeout(total=config.request_timeout)

    try:
        async with (
            semaphore,
            session.put(config.url, json=data, timeout=timeout) as response,
        ):
            text = await response.text()

            return response.status, text
    except TimeoutError:
        print(
            f"Failed to process {data['wsi_path']}:\n\tTimeout after {config.request_timeout} seconds\n"
        )

        return -1, "Timeout"
    except ClientConnectionError:
        return -1, "Connection error"


def artifacts(wsi_stem: str) -> list[str]:
    return [
        f"{wsi_stem}.csv",
        # f"FoldingFunction_folding_test_{wsi_stem}.tiff",
        f"Piqe_focus_score_piqe_median_{wsi_stem}.tiff",
        f"Piqe_piqe_median_activity_mask_{wsi_stem}.tiff",
        f"ResidualArtifactsAndCoverage_coverage_mask_{wsi_stem}.tiff",
        f"ResidualArtifactsAndCoverage_cov_percent_heatmap_{wsi_stem}.tiff",
    ]


def artifacts_exist(wsi_stem: str, output_path: Path) -> bool:
    return all((output_path / artifact).exists() for artifact in artifacts(wsi_stem))


def log_artifacts(
    logger: MLFlowLogger,
    wsi_stem: str,
    output_path: Path,
) -> None:
    for artifact in artifacts(wsi_stem):
        logger.experiment.log_artifact(
            logger.run_id, output_path / artifact, "qc_masks"
        )


async def repeatable_put_request(
    session: ClientSession,
    data: dict[str, Any],
    semaphore: asyncio.Semaphore,
    config: DictConfig,
    logger: MLFlowLogger,
) -> None:
    if artifacts_exist(Path(data["wsi_path"]).stem, Path(data["output_path"])):
        log_artifacts(logger, Path(data["wsi_path"]).stem, Path(data["output_path"]))

        print(f"Skipped {data['wsi_path']}:\n\tArtifacts already exist\n")
        return

    for attempt in range(1, config.num_repeats + 1):
        status, text = await put_request(session, data, semaphore, config)

        if status == -1 and text == "Timeout":
            return

        if status == -1 and text == "Connection error":
            att_count = f"attempt {attempt}/{config.num_repeats}"
            print(
                f"Connection error received for {data['wsi_path']} ({att_count}). Retrying...\n"
            )
            await asyncio.sleep(2**attempt)

            continue

        if status == 500 and text == "Internal Server Error":
            att_count = f"attempt {attempt}/{config.num_repeats}"
            print(
                f"Unexpected status 500 received for {data['wsi_path']} ({att_count}):\n\tResponse: {text}\n"
            )
            await asyncio.sleep(2**attempt)

            continue

        log_artifacts(logger, Path(data["wsi_path"]).stem, Path(data["output_path"]))
        print(
            f"Processed {data['wsi_path']}:\n\tStatus: {status} \n\tResponse: {text}\n"
        )

        return

    print(f"Failed to process {data['wsi_path']}:\n\tAll retry attempts failed\n")


async def generate_report(
    session: ClientSession,
    slides: list[Path],
    save_location: str,
    semaphore: asyncio.Semaphore,
    config: DictConfig,
) -> None:
    url = config.connection_parameters.url + "report"

    data = {
        "backgrounds": [str(slide.resolve()) for slide in slides],
        "mask_dir": config.output_path,
        "save_location": save_location,
        "compute_metrics": True,
    }

    async with semaphore, session.put(url, json=data) as response:
        result = await response.text()

        print(
            f"Report generation:\n\tStatus: {response.status} \n\tResponse: {result}\n"
        )


async def quality_control(
    report_path: str,
    slides: list[Path],
    semaphore: asyncio.Semaphore,
    config: DictConfig,
    logger: MLFlowLogger,
) -> None:
    async with ClientSession() as session:
        tasks = [
            repeatable_put_request(
                session=session,
                semaphore=semaphore,
                config=config.connection_parameters,
                logger=logger,
                data={
                    "wsi_path": slide.as_posix(),
                    "output_path": config.output_path,
                    "mask_level": config.qc_parameters.mask_level,
                    "sample_level": config.qc_parameters.sample_level,
                    "check_residual": config.qc_parameters.check_residual,
                    "check_folding": config.qc_parameters.check_folding,
                    "check_focus": config.qc_parameters.check_focus,
                    "wb_correction": config.qc_parameters.wb_correction,
                },
            )
            for slide in slides
        ]

        # Processing of the slides
        await asyncio.gather(*tasks)

        # Report generation
        await generate_report(
            session=session,
            slides=slides,
            save_location=report_path,
            semaphore=semaphore,
            config=config,
        )

        logger.experiment.log_artifact(logger.run_id, report_path)


@hydra.main(config_path="../configs", config_name="quality_control", version_base=None)
@autolog
def main(config: DictConfig, logger: Logger | None = None) -> None:
    assert logger is not None, "Need logger"
    assert isinstance(logger, MLFlowLogger), "Need MLFlowLogger"
    assert isinstance(logger.experiment, MlflowClient), "Need MlflowClient"
    assert logger.run_id is not None, "Need run_id"

    semaphore = asyncio.Semaphore(config.request_limit)

    Path(config.output_path).mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="qc_masks_report_") as tmp_dir:
        report_path = Path(tmp_dir) / "report.html"

        asyncio.run(
            quality_control(
                report_path=report_path.absolute().as_posix(),
                slides=list(Path(config.slides_folder).glob("*.tiff")),
                semaphore=semaphore,
                config=config,
                logger=logger,
            )
        )


if __name__ == "__main__":
    main()
