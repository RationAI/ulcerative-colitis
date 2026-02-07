import tempfile
from pathlib import Path
from typing import cast

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.data.datasets import TilesPredict


class NeutrophilDetector(torch.nn.Module):
    def __init__(self, confidence: float, weights_url: str) -> None:
        super().__init__()
        with tempfile.NamedTemporaryFile(suffix=".pt") as temp_file:
            torch.hub.download_url_to_file(weights_url, temp_file.name)
            self.model = cast(
                "torch.nn.Module",
                torch.hub.load("ultralytics/yolov5", "custom", path=temp_file.name),
            )
        self.model.conf = confidence

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def save_neutrophils(
    x0: torch.Tensor,
    y0: torch.Tensor,
    x1: torch.Tensor,
    y1: torch.Tensor,
    probability: torch.Tensor,
    neutrophils_path: Path,
) -> None:
    neutrophils_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "x0": x0.numpy(),
            "y0": y0.numpy(),
            "x1": x1.numpy(),
            "y1": y1.numpy(),
            "probability": probability.numpy(),
        }
    )

    df.to_parquet(neutrophils_path, index=False, engine="pyarrow")


@with_cli_args(["+preprocessing=neutrophils"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    dest = Path(config.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    neutrophil_detector = NeutrophilDetector(**config.model)
    neutrophil_detector = neutrophil_detector.to(device)

    with torch.no_grad():
        dataset = TilesPredict(config.dataset.uris.values(), to_tensor=False)

        for slide_dataset in tqdm(dataset.generate_datasets()):
            slide_name = str(slide_dataset.slide_metadata["name"])
            neutrophils_path = (dest / slide_name).with_suffix(".parquet")

            if neutrophils_path.exists():
                print(
                    f"Neutrophil detections for slide {slide_name} already exist, skipping..."
                )
                continue

            try:
                slide_tiles_dataloader = DataLoader(
                    slide_dataset,
                    batch_size=config.dataloader.batch_size,
                    num_workers=config.dataloader.num_workers,
                    persistent_workers=config.dataloader.persistent_workers,
                    collate_fn=lambda batch: (
                        [x for x, _ in batch],
                        [m for _, m in batch],
                    ),
                )

                x0, y0, x1, y1, probability = [], [], [], [], []

                for x, metadata in slide_tiles_dataloader:
                    result = neutrophil_detector(x)

                    result.xyxy = cast("list[torch.Tensor]", result.xyxy)
                    for m, xyxy in zip(metadata, result.xyxy, strict=True):
                        if xyxy.numel() == 0:
                            continue
                        xyxy = xyxy.cpu()
                        x0.append(xyxy[:, 0] + m["x"])
                        y0.append(xyxy[:, 1] + m["y"])
                        x1.append(xyxy[:, 2] + m["x"])
                        y1.append(xyxy[:, 3] + m["y"])
                        probability.append(xyxy[:, 4])

                if len(x0) == 0:
                    empty = torch.empty((0,), dtype=torch.float32)
                    save_neutrophils(
                        empty, empty, empty, empty, empty, neutrophils_path
                    )
                    continue

                save_neutrophils(
                    torch.concat(x0),
                    torch.concat(y0),
                    torch.concat(x1),
                    torch.concat(y1),
                    torch.concat(probability),
                    neutrophils_path,
                )

                # logger.log_artifact(
                #     local_path=str(neutrophils_path), artifact_path="neutrophils"
                # )
            except Exception as e:
                print(f"Error processing slide {slide_name}: {e}")

        logger.log_artifacts(local_dir=str(dest), artifact_path="neutrophils")


if __name__ == "__main__":
    main()
