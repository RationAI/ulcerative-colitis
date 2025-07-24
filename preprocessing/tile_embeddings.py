from collections.abc import Iterable
from enum import Enum
from pathlib import Path

import albumentations as A
import click
import mlflow
import pandas as pd
import timm
import torch
from huggingface_hub import login
from timm.layers.mlp import SwiGLUPacked
from torch.utils.data import DataLoader
from tqdm import tqdm

from preprocessing.paths import EMBEDDING_REGIONS_PATH, EMBEDDINGS_PATH
from ulcerative_colitis.data.datasets import TilesPredict


URIS = [
    "mlflow-artifacts:/86/0f605c9479574c8498f64ffea5f87508/artifacts/Ulcerative Colitis - test preliminary",
    "mlflow-artifacts:/86/0f605c9479574c8498f64ffea5f87508/artifacts/Ulcerative Colitis - test final",
    "mlflow-artifacts:/86/0f605c9479574c8498f64ffea5f87508/artifacts/Ulcerative Colitis - train",
]


class FoundationModel(Enum):
    PROV_GIGAPATH = "prov-gigapath"
    UNI = "UNI"
    UNI2 = "UNI2-h"
    VIRCHOW = "Virchow"
    VIRCHOW2 = "Virchow2"


def load_dataset(uris: Iterable[str]) -> TilesPredict:
    """Load the dataset for tile embeddings.

    Assumes that the dataset has 224x224 RGB tiles.

    Args:
        uris (Iterable[str]): The URIs of the datasets to load.

    Returns:
        TilesPredict: The dataset object for tile embeddings.
    """
    transforms = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    return TilesPredict(uris, transforms=transforms)


def load_tile_encoder(model: FoundationModel) -> torch.nn.Module:
    """Load the tile encoder model for feature extraction.

    Args:
        model (FoundationModel): The foundation model to use.

    Returns:
        torch.nn.Module: The tile encoder model.
    """
    match model:
        case FoundationModel.PROV_GIGAPATH:
            return timm.create_model(
                "hf_hub:prov-gigapath/prov-gigapath",
                pretrained=True,
            )
        case FoundationModel.UNI:
            return timm.create_model(
                "hf-hub:MahmoodLab/uni",
                pretrained=True,
                init_values=1e-5,
                dynamic_img_size=True,
            )
        case FoundationModel.UNI2:
            return timm.create_model(
                "hf-hub:MahmoodLab/UNI2-h",
                pretrained=True,
                img_size=224,
                patch_size=14,
                depth=24,
                num_heads=24,
                init_values=1e-5,
                embed_dim=1536,
                mlp_ratio=2.66667 * 2,
                num_classes=0,
                no_embed_class=True,
                mlp_layer=SwiGLUPacked,
                act_layer=torch.nn.SiLU,
                reg_tokens=8,
                dynamic_img_size=True,
            )
        case FoundationModel.VIRCHOW:
            return timm.create_model(
                "hf-hub:paige-ai/Virchow",
                pretrained=True,
                mlp_layer=SwiGLUPacked,
                act_layer=torch.nn.SiLU,
            )
        case FoundationModel.VIRCHOW2:
            return timm.create_model(
                "hf_hub:paige-ai/Virchow2",
                pretrained=True,
                mlp_layer=SwiGLUPacked,
                act_layer=torch.nn.SiLU,
            )


def embeddings_dimension(model: FoundationModel) -> int:
    """Get the dimension of the embeddings for the specified model.

    Args:
        model (FoundationModel): The foundation model to use.

    Returns:
        int: The dimension of the embeddings.
    """
    match model:
        case FoundationModel.PROV_GIGAPATH | FoundationModel.UNI2:
            return 1536
        case FoundationModel.UNI:
            return 1024
        case FoundationModel.VIRCHOW | FoundationModel.VIRCHOW2:
            return 2560


def process_output(output: torch.Tensor, model: FoundationModel) -> torch.Tensor:
    """Process the output of the tile encoder model.

    Args:
        output (torch.Tensor): The raw output from the model.
        model (FoundationModel): The foundation model used.

    Returns:
        torch.Tensor: The processed embeddings.
    """
    if model == FoundationModel.VIRCHOW:
        class_token = output[:, 0]
        patch_tokens = output[:, 1:]
        return torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
    if model == FoundationModel.VIRCHOW2:
        class_token = output[:, 0]
        patch_tokens = output[:, 5:]
        return torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
    return output


def save_embeddings(slide_embeddings: torch.Tensor, slide_path: Path) -> None:
    """Save the slide embeddings to the specified path.

    Args:
        slide_embeddings (torch.Tensor): The embeddings to save.
        slide_path (Path): The path to save the embeddings to.
    """
    slide_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slide_embeddings, slide_path)


def save_embeddings_regions(
    slide_embeddings: torch.Tensor, slide_path: Path, tiles: pd.DataFrame
) -> None:
    """Save the slide embeddings and their corresponding tile regions.

    Args:
        slide_embeddings (torch.Tensor): The embeddings to save.
        slide_path (Path): The path to save the embeddings to.
        tiles (pd.DataFrame): The DataFrame containing tile metadata.
    """
    folder = EMBEDDING_REGIONS_PATH / slide_path.relative_to(EMBEDDINGS_PATH).parent
    folder.mkdir(parents=True, exist_ok=True)

    tiles = tiles.reset_index(drop=True)
    for region in tiles["region"].unique():
        region_tiles = tiles.query(f"region == {region}")
        region_embeddings = slide_embeddings[region_tiles.index.to_numpy()]

        torch.save(
            region_embeddings,
            folder / f"{slide_path.stem}_region_{region:03d}.pt",
        )


@click.command()
@click.option(
    "--token",
    type=str,
    help="Hugging Face token for accessing private models and datasets.",
    required=True,
)
@click.option(
    "--model",
    type=click.Choice([m.value for m in FoundationModel]),
    help="The foundation model to use for tile embeddings.",
    default=FoundationModel.PROV_GIGAPATH.value,
)
@click.option(
    "--batch-size",
    type=int,
    help="Batch size for processing tiles.",
    default=2048,
)
def main(token: str, model: str | FoundationModel, batch_size: int) -> None:
    login(token=token)
    model = FoundationModel(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tile_encoder = load_tile_encoder(model).to(device).eval()
    embedding_dim = embeddings_dimension(model)

    with torch.no_grad():
        for uri in URIS:
            dataset = load_dataset((uri,))
            partition = uri.split(" - ")[-1]

            for slide_dataset in tqdm(
                dataset.generate_datasets(), desc=f"{partition}: "
            ):
                slide_name = str(slide_dataset.slide_metadata["name"])
                slide_path = (
                    EMBEDDINGS_PATH / model.value / partition / slide_name
                ).with_suffix(".pt")

                if slide_path.exists():
                    continue

                slide_dataloader = DataLoader(
                    slide_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=8,
                    persistent_workers=True,
                )
                slide_embeddings = torch.zeros(
                    (len(slide_dataset), embedding_dim), dtype=torch.float32
                )
                for i, (x, _) in enumerate(slide_dataloader):
                    x = x.to(device)
                    embeddings = process_output(tile_encoder(x), model)
                    start = i * batch_size
                    end = start + embeddings.size(0)
                    slide_embeddings[start:end] = embeddings.to("cpu")

                save_embeddings(slide_embeddings, slide_path)
                save_embeddings_regions(
                    slide_embeddings,
                    slide_path,
                    slide_dataset.slide_tiles.tiles,
                )

    mlflow.set_experiment(experiment_name="Ulcerative Colitis")
    with mlflow.start_run(run_name=f"📂 Dataset: Embeddings - {model.value}"):
        mlflow.log_artifacts(str(EMBEDDINGS_PATH / model.value))

    with mlflow.start_run(run_name=f"📂 Dataset: Embedding Regions - {model.value}"):
        mlflow.log_artifacts(str(EMBEDDING_REGIONS_PATH / model.value))


if __name__ == "__main__":
    main()
