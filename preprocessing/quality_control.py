import asyncio
import tempfile
from pathlib import Path
from typing import Any

import click
import mlflow
from aiohttp import ClientConnectionError, ClientSession, ClientTimeout

from preprocessing.paths import (
    BASE_FOLDER,
    QC_MASKS_PATH,
    SLIDES_PATH_FTN,
    SLIDES_PATH_IKEM,
)


REQUEST_LIMIT = 4
REQUEST_TIMEOUT = 30 * 60  # 30 minutes
MAX_REQUEST_RETRY_ATTEMPTS = 5
BACKOFF_BASE = 2  # 2 seconds

URL = "http://rayservice-qc-serve-svc.rationai-jobs-ns.svc.cluster.local:8000/"

EXPERIMENT_NAME = "Ulcerative Colitis"


semaphore = asyncio.Semaphore(REQUEST_LIMIT)


async def put_request(
    session: ClientSession, url: str, data: dict[str, Any]
) -> tuple[int, str]:
    timeout = ClientTimeout(total=REQUEST_TIMEOUT)

    try:
        async with semaphore, session.put(url, json=data, timeout=timeout) as response:
            text = await response.text()

            return response.status, text
    except TimeoutError:
        print(
            f"Failed to process {data['wsi_path']}:\n\tTimeout after {REQUEST_TIMEOUT} seconds\n"
        )

        return -1, "Timeout"

    except ClientConnectionError:
        print(f"Failed to process {data['wsi_path']}:\n\tConnection error\n")

        return -1, "Connection error"


async def repeatable_put_request(
    session: ClientSession, url: str, data: dict[str, Any], num_repeats: int
) -> None:
    for attempt in range(1, num_repeats + 1):
        status, text = await put_request(session, url, data)

        if status == -1 and text == "Timeout":
            return

        if status == -1 and text == "Connection error":
            att_count = f"attempt {attempt}/{MAX_REQUEST_RETRY_ATTEMPTS}"
            print(f"Connection error received for {data['wsi_path']} ({att_count})\n")
            await asyncio.sleep(BACKOFF_BASE**attempt)

            continue

        if status == 500 and text == "Internal Server Error":
            att_count = f"attempt {attempt}/{MAX_REQUEST_RETRY_ATTEMPTS}"
            print(
                f"Unexpected status 500 received for {data['wsi_path']} ({att_count}):\n\tResponse: {text}\n"
            )
            await asyncio.sleep(BACKOFF_BASE**attempt)

            continue

        print(
            f"Processed {data['wsi_path']}:\n\tStatus: {status} \n\tResponse: {text}\n"
        )

        return

    print(f"Failed to process {data['wsi_path']}:\n\tAll retry attempts failed\n")


async def generate_report(
    session: ClientSession, slides: list[Path], output_dir: str, save_location: str
) -> None:
    url = URL + "report"

    data = {
        "backgrounds": [str(slide.resolve()) for slide in slides],
        "mask_dir": output_dir,
        "save_location": save_location,
        "compute_metrics": True,
    }

    async with semaphore, session.put(url, json=data) as response:
        result = await response.text()

        print(
            f"Report generation:\n\tStatus: {response.status} \n\tResponse: {result}\n"
        )


async def quality_control(
    output_path: str, report_path: str, slides: list[Path]
) -> None:
    async with ClientSession() as session:
        tasks = [
            repeatable_put_request(
                session=session,
                url=URL,
                data={
                    "wsi_path": str(slide.resolve()),
                    "output_path": output_path,
                    "mask_level": 3,
                    "sample_level": 1,
                    "check_residual": True,
                    "check_folding": True,
                    "check_focus": True,
                    "wb_correction": True,
                },
                num_repeats=MAX_REQUEST_RETRY_ATTEMPTS,
            )
            for slide in slides
        ]

        # Processing of the slides
        await asyncio.gather(*tasks)

        # Report generation
        await generate_report(
            session=session,
            slides=slides,
            output_dir=output_path,
            save_location=report_path,
        )


@click.command()
@click.option(
    "--cohort",
    type=click.Choice(["FTN", "IKEM"]),
    required=True,
    help="Cohort to process",
)
def main(cohort: str) -> None:
    mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)

    with (
        mlflow.start_run(run_name=f"🎭 Quality Control Masks: {cohort}"),
        tempfile.TemporaryDirectory(
            prefix="qc_masks_report_", dir=Path(BASE_FOLDER).as_posix()
        ) as tmp_dir,  # Create a temporary directory for the report
    ):
        report_path = Path(tmp_dir) / "report.html"
        output_dir = QC_MASKS_PATH / cohort

        output_dir.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        slides_folder = SLIDES_PATH_FTN if cohort == "FTN" else SLIDES_PATH_IKEM

        asyncio.run(
            quality_control(
                output_path=output_dir.absolute().as_posix(),
                report_path=report_path.absolute().as_posix(),
                slides=list(slides_folder.glob("*.tiff")),
            )
        )

        mlflow.log_artifact(local_path=str(report_path), artifact_path="report")
        mlflow.log_artifacts(local_dir=str(output_dir), artifact_path="qc_masks")


if __name__ == "__main__":
    main()
