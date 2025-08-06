# Credits: https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/prostate-cancer/-/blob/feature/preprocessing-masks/preprocessing/masks/quality_control.py?ref_type=heads

import asyncio
import tempfile
from pathlib import Path
from typing import Any

import aiohttp
import mlflow
from aiohttp import (
    ClientConnectorError,
    ClientSession,
    ClientTimeout,
    ServerDisconnectedError,
)

from preprocessing.paths import BASE_FOLDER, QC_MASKS_PATH, SLIDES_PATH_FTN


REQUEST_LIMIT = 2
REQUEST_TIMEOUT = 21 * 60  # 21 minutes
REPORT_REQUEST_TIMEOUT = 4 * 60  # 4 minutes
MAX_RETRIES = 3

EXPERIMENT_NAME = "Ulcerative Colitis"

URL = "http://rayservice-qc-serve-svc.rationai-jobs-ns.svc.cluster.local:8000/"

OUTPUT_DIR = QC_MASKS_PATH / "FTN"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# report.html is logged to MLflow

# Resolve to remove the symlinks
SLIDES: list[Path] = list(Path(SLIDES_PATH_FTN).resolve().rglob("*.tiff"))


semaphore = asyncio.Semaphore(REQUEST_LIMIT)


async def put_request(
    session: ClientSession, url: str, data: dict[str, Any], retry: int = MAX_RETRIES
) -> str:
    timeout = ClientTimeout(total=REQUEST_TIMEOUT, connect=60)  # Add connect timeout

    for attempt in range(retry + 1):
        try:
            async with (
                semaphore,
                session.put(url, json=data, timeout=timeout) as response,
            ):
                result = await response.text()

                print(
                    f"Processed {data['wsi_path']}:\n\tStatus: {response.status} \n\tResponse: {result}\n"
                )

                if response.status == 500 and attempt < retry:
                    print(
                        f"Retrying {data['wsi_path']} due to server error (attempt {attempt + 1}/{retry + 1})."
                    )
                    await asyncio.sleep(2**attempt)  # Exponential backoff
                    continue
                else:
                    return result

        except (ServerDisconnectedError, ClientConnectorError, TimeoutError) as e:
            slide_name = Path(data["wsi_path"]).name
            if attempt < retry:
                wait_time = 2**attempt
                print(
                    f"Connection error for {slide_name} (attempt {attempt + 1}/{retry + 1}): {type(e).__name__}. "
                    f"Retrying in {wait_time} seconds..."
                )
                await asyncio.sleep(wait_time)
                continue
            else:
                print(
                    f"Failed to process {slide_name} after {retry + 1} attempts: {type(e).__name__}"
                )
                return f"Failed: {type(e).__name__}"

        except Exception as e:
            slide_name = Path(data["wsi_path"]).name
            print(f"Unexpected error processing {slide_name}: {e}")
            return f"Error: {e}"

    return "Max retries exceeded"


async def generate_report(
    session: ClientSession, slides: list[Path], output_dir: str, save_location: str
) -> None:
    url = URL + "report"

    data = {
        "backgrounds": [str(slide) for slide in slides],
        "mask_dir": output_dir,
        "save_location": save_location,
    }

    try:
        async with (
            semaphore,
            session.put(
                url, json=data, timeout=ClientTimeout(total=REPORT_REQUEST_TIMEOUT)
            ) as response,
        ):
            result = await response.text()

            print(
                f"Report generation:\n\tStatus: {response.status} \n\tResponse: {result}\n"
            )
    except (ServerDisconnectedError, ClientConnectorError, TimeoutError) as e:
        print(f"Report generation failed: {type(e).__name__}")
    except Exception as e:
        print(f"Unexpected error during report generation: {e}")


async def main(output_path: str, report_path: str) -> None:
    # Use connector with better connection handling
    connector = aiohttp.TCPConnector(
        limit=REQUEST_LIMIT,
        limit_per_host=REQUEST_LIMIT,
        keepalive_timeout=30,
        enable_cleanup_closed=True,
    )

    async with ClientSession(connector=connector) as session:
        tasks = [
            put_request(
                session=session,
                url=URL,
                data={
                    "wsi_path": str(slide),
                    "output_path": output_path,
                    "mask_level": 3,
                    "sample_level": 1,
                    "check_residual": True,
                    "check_folding": True,
                    "check_focus": True,
                },
            )
            for slide in SLIDES
        ]

        await asyncio.gather(*tasks)

        await generate_report(
            session=session,
            slides=SLIDES,
            output_dir=output_path,
            save_location=report_path,
        )

        # Save the report to MLflow
        mlflow.log_artifact(report_path)


if __name__ == "__main__":
    mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)

    with (
        mlflow.start_run(run_name="🎭 Quality Control Masks: FTN"),
        tempfile.TemporaryDirectory(
            prefix="qc_masks_report_", dir=Path(BASE_FOLDER).as_posix()
        ) as tmp_dir,  # Create a temporary directory for the report
    ):
        report_path = Path(tmp_dir) / "report.html"

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        asyncio.run(
            main(
                output_path=OUTPUT_DIR.absolute().as_posix(),
                report_path=report_path.absolute().as_posix(),
            )
        )
