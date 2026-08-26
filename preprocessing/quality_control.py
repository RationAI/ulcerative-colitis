# credits: https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/lymph-nodes/-/blob/develop/preprocessing/qc.py?ref_type=heads

import asyncio
from collections.abc import Generator
from pathlib import Path
from typing import TypedDict

import hydra
import pandas as pd
import rationai
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.types import SlideCheckConfig
from tqdm.asyncio import tqdm


class QCParameters(TypedDict):
    mask_level: int
    sample_level: int
    check_residual: bool
    check_folding: bool
    check_focus: bool
    wb_correction: bool


def get_qc_masks(qc_parameters: QCParameters) -> Generator[tuple[str, str], None, None]:
    if qc_parameters["check_focus"]:
        yield ("Piqe_focus_score_piqe_median", "blur_per_tile")
        yield ("Piqe_piqe_median_activity_mask", "blur_per_pixel")

    if qc_parameters["check_residual"]:
        yield ("ResidualArtifactsAndCoverage_cov_percent_heatmap", "artifacts_per_tile")
        yield ("ResidualArtifactsAndCoverage_coverage_mask", "artifacts_per_pixel")

    if qc_parameters["check_folding"]:
        yield ("FoldingFunction_folding_test", "folds_per_pixel")


def organize_masks(output_path: Path, subdir: str, mask_prefix: str) -> None:
    prefix_dir = output_path / subdir
    prefix_dir.mkdir(parents=True, exist_ok=True)

    # Glob has to be wrapped in list, because we're modifying the directory!!!
    for file in list(output_path.glob(f"{mask_prefix}_*.tiff")):
        slide_name = file.name.replace(f"{mask_prefix}_", "")
        destination = prefix_dir / slide_name
        file.rename(destination)


def filter_dataset_by_qc_errors(
    dataset: pd.DataFrame, qc_errors_uri: str | None
) -> pd.DataFrame:
    """Keep only the rows of `dataset` whose slide failed QC.

    `qc_errors_uri` is an MLflow artifact URI pointing to a `qc_errors.log`
    file produced by `qc_main` (lines formatted as
    "Failed to process {wsi_path}: {error}"), e.g.
    "mlflow-artifacts:/86/433c941b3706450aa499f9bee4b17701/artifacts/qc_errors.log".
    If `qc_errors_uri` is None, no QC run is available to filter by and the
    whole dataset is returned unchanged.
    """
    if qc_errors_uri is None:
        return dataset

    prefix = "Failed to process "
    failed_paths: set[str] = set()
    with open(download_artifacts(qc_errors_uri)) as log_file:
        for line in log_file:
            line = line.rstrip("\n")
            if not line.startswith(prefix):
                continue
            wsi_path, _, _error = line[len(prefix) :].partition(":")
            failed_paths.add(wsi_path.strip())

    return dataset[dataset["path"].isin(failed_paths)].reset_index(drop=True)


async def qc_main(
    output_path: Path,
    slides: list[str],
    logger: MLFlowLogger,
    request_timeout: int,
    max_concurrent: int,
    qc_parameters: QCParameters,
) -> None:
    async with rationai.AsyncClient() as client:  # type: ignore[attr-defined]
        async for result in tqdm(
            client.qc.check_slides(
                slides,
                output_path,
                config=SlideCheckConfig(**qc_parameters),
                timeout=request_timeout,
                max_concurrent=max_concurrent,
            ),
            total=len(slides),
        ):
            if not result.success:
                with open(output_path / "qc_errors.log", "a") as log_file:
                    log_file.write(
                        f"Failed to process {result.wsi_path}: {result.error}\n"
                    )

        # Organize generated masks into subdirectories
        for prefix, artifact_name in get_qc_masks(qc_parameters):
            organize_masks(Path(output_path), artifact_name, prefix)

        # Merge generated csv files, appending to any pre-existing qc_metrics.csv
        metrics_path = Path(output_path, "qc_metrics.csv")
        csvs = [f for f in Path(output_path).glob("*.csv") if f != metrics_path]
        new_metrics = [pd.read_csv(f) for f in csvs]
        if metrics_path.exists():
            new_metrics.insert(0, pd.read_csv(metrics_path))
        pd.concat(new_metrics).to_csv(metrics_path, index=False)

        # Remove individual csv files
        for f in csvs:
            f.unlink()

        logger.log_artifacts(local_dir=str(output_path))


@with_cli_args(["+preprocessing=quality_control"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    dataset = pd.read_csv(download_artifacts(config.dataset.mlflow_uris.dataset))
    dataset = filter_dataset_by_qc_errors(dataset, config.get("qc_errors_uri"))

    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    asyncio.run(
        qc_main(
            output_path=output_path,
            slides=dataset["path"].to_list(),
            logger=logger,
            request_timeout=config.request_timeout,
            max_concurrent=config.max_concurrent,
            qc_parameters=config.qc_parameters,
        )
    )


if __name__ == "__main__":
    main()
