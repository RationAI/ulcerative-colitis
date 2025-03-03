from collections.abc import Iterable
from pathlib import Path

import albumentations as A
import gigapath
import gigapath.slide_encoder
import mlflow
import timm
import torch
from torch.utils.data import DataLoader

from ulcerative_colitis.data.datasets import NeutrophilsPredict


URIS = [
    "mlflow-artifacts:/27/a42de386382f48f0b61e9e7fe898208e/artifacts/Ulcerative Colitis - val",
    "mlflow-artifacts:/27/a42de386382f48f0b61e9e7fe898208e/artifacts/Ulcerative Colitis - train",
    "mlflow-artifacts:/27/a42de386382f48f0b61e9e7fe898208e/artifacts/Ulcerative Colitis - test1",
    "mlflow-artifacts:/27/a42de386382f48f0b61e9e7fe898208e/artifacts/Ulcerative Colitis - test2",
]

DESTINATION = Path(
    "/mnt/data/Projects/inflammatory_bowel_disease/ulcerative_colitis/embeddings"
)


def load_dataset(uris: Iterable[str]) -> NeutrophilsPredict:
    transforms = A.Compose(
        [
            A.CenterCrop(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    return NeutrophilsPredict(uris, transforms=transforms)


def load_tile_encoder() -> torch.nn.Module:
    return timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)


def load_slide_encoder() -> torch.nn.Module:
    # TODO
    return gigapath.slide_encoder.create_model(
        "hf_hub:prov-gigapath/prov-gigapath", "gigapath_slide_enc12l768d", 1536
    )


def save_embeddings(
    slide_embeddings: torch.Tensor, partition: str, slide_name: str
) -> None:
    torch.save(
        slide_embeddings, (DESTINATION / partition / slide_name).with_suffix(".pt")
    )


def main() -> None:
    devide = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tile_encoder = load_tile_encoder().to(devide).eval()

    for uri in URIS:
        dataset = load_dataset((uri,))
        partition = uri.split(" - ")[-1]

        slide_embeddings = torch.zeros(
            (len(dataset), 1536), device=devide, dtype=torch.float32
        )
        for slide_dataset in dataset.generate_datasets():
            slide_dataloader = DataLoader(slide_dataset, batch_size=1, shuffle=False)
            for i, (x, _) in enumerate(slide_dataloader):
                x = x.to(devide)
                embeddings = tile_encoder(x)

                slide_embeddings[i, :] = embeddings.squeeze()

            slide_name = (
                slide_dataset.slide_metadata["path"].split("/")[-1].split(".")[0]
            )
            save_embeddings(slide_embeddings, partition, slide_name)

    mlflow.set_experiment(experiment_name="IKEM")
    with mlflow.start_run(run_name="📂 Dataset: Embeddings"):
        mlflow.log_artifacts(str(DESTINATION))


if __name__ == "__main__":
    main()
