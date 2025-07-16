from collections.abc import Iterable
from pathlib import Path

import albumentations as A
import mlflow
import timm
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from preprocessing.paths import EMBEDDINGS_PATH
from ulcerative_colitis.data.datasets import NeutrophilsPredict


URIS = [
    "mlflow-artifacts:/27/a045896edb624e9ba042b99d2b9e3d72/artifacts/Ulcerative Colitis - test preliminary",
    "mlflow-artifacts:/27/a045896edb624e9ba042b99d2b9e3d72/artifacts/Ulcerative Colitis - test final",
    "mlflow-artifacts:/27/a045896edb624e9ba042b99d2b9e3d72/artifacts/Ulcerative Colitis - train",
]

BATCH_SIZE = 2048


def load_dataset(uris: Iterable[str]) -> NeutrophilsPredict:
    transforms = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    return NeutrophilsPredict(uris, transforms=transforms)


def load_tile_encoder() -> torch.nn.Module:
    return timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)


def save_embeddings(
    slide_embeddings: torch.Tensor, partition: str, slide_name: str
) -> None:
    folder = EMBEDDINGS_PATH / partition
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(slide_embeddings, (folder / slide_name).with_suffix(".pt"))


def main() -> None:
    devide = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tile_encoder = load_tile_encoder().to(devide).eval()

    with torch.no_grad():
        for uri in URIS:
            dataset = load_dataset((uri,))
            partition = uri.split(" - ")[-1]

            for slide_dataset in tqdm(
                dataset.generate_datasets(), desc=f"{partition}: "
            ):
                slide_name = Path(slide_dataset.slide_metadata["path"]).stem
                slide_path = (EMBEDDINGS_PATH / partition / slide_name).with_suffix(
                    ".pt"
                )
                if slide_path.exists():
                    continue

                slide_dataloader = DataLoader(
                    slide_dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=8,
                    persistent_workers=True,
                )
                slide_embeddings = torch.zeros(
                    (len(slide_dataset), 1536), device=devide, dtype=torch.float32
                )
                for i, (x, _) in enumerate(slide_dataloader):
                    x = x.to(devide)
                    embeddings = tile_encoder(x)
                    start = i * BATCH_SIZE
                    end = start + embeddings.size(0)
                    slide_embeddings[start:end] = embeddings

                save_embeddings(slide_embeddings, partition, slide_name)

    mlflow.set_experiment(experiment_name="IKEM")
    with mlflow.start_run(run_name="📂 Dataset: Embeddings"):
        mlflow.log_artifacts(str(EMBEDDINGS_PATH))


if __name__ == "__main__":
    main()
