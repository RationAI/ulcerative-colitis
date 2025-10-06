from enum import Enum
from pathlib import Path

import albumentations as A
import hydra
import timm
import torch
from huggingface_hub import login
from lightning.pytorch.loggers import Logger
from mlflow import MlflowClient
from omegaconf import DictConfig
from rationai.mlkit.autolog import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger
from timm.layers.mlp import SwiGLUPacked
from torch.utils.data import DataLoader
from tqdm import tqdm

from ulcerative_colitis.data.datasets import TilesPredict


class FoundationModel(Enum):
    PROV_GIGAPATH = "prov-gigapath"
    UNI = "UNI"
    UNI2 = "UNI2-h"
    VIRCHOW = "Virchow"
    VIRCHOW2 = "Virchow2"


def load_dataset(uri: str) -> TilesPredict:
    """Load the dataset for tile embeddings.

    Assumes that the dataset has 224x224 RGB tiles.

    Args:
        uri (str): The URI of the dataset.

    Returns:
        TilesPredict: The dataset object for tile embeddings.
    """
    return TilesPredict(
        (uri,),
        transforms=A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        ),
    )


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


def save_embeddings(
    slide_tiles_embeddings: torch.Tensor,
    slide_tiles_coords: torch.Tensor,
    embeddings_path: Path,
) -> None:
    """Save the slide embeddings to the specified path.

    Args:
        slide_tiles_embeddings (torch.Tensor): The embeddings to save.
        slide_tiles_coords (torch.Tensor): The coordinates of the tiles.
        embeddings_path (Path): The path to save the embeddings to.
    """
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "embeddings": slide_tiles_embeddings,
        "coords": slide_tiles_coords,
    }
    torch.save(data, embeddings_path)


@hydra.main(
    config_path="../../configs", config_name="slide_embeddings", version_base=None
)
@autolog
def main(config: DictConfig, logger: Logger | None = None) -> None:
    assert logger is not None, "Need logger"
    assert isinstance(logger, MLFlowLogger), "Need MLFlowLogger"
    assert isinstance(logger.experiment, MlflowClient), "Need MlflowClient"
    assert logger.run_id is not None, "Need run_id"

    login(config.token)
    model = FoundationModel(config.model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tile_encoder = load_tile_encoder(model).to(device).eval()
    embedding_dim = embeddings_dimension(model)

    output_folder = Path(config.output_folder)

    with torch.no_grad():
        for partition in ("train", "test preliminary", "test final"):
            dataset = load_dataset(f"{config.tiling_uri}/{partition} - {config.cohort}")

            for slide_dataset in tqdm(
                dataset.generate_datasets(), desc=f"{partition}: "
            ):
                slide_name = str(slide_dataset.slide_metadata["name"])
                embeddings_path = (output_folder / slide_name).with_suffix(".pt")

                if config.skip_existing and embeddings_path.exists():
                    continue

                slide_tiles_dataloader = DataLoader(
                    slide_dataset,
                    batch_size=config.dataloader.batch_size,
                    shuffle=config.dataloader.shuffle,
                    num_workers=config.dataloader.num_workers,
                    persistent_workers=config.dataloader.persistent_workers,
                )
                slide_tiles_embeddings = torch.zeros(
                    (len(slide_dataset), embedding_dim), dtype=torch.float32
                )
                slide_tiles_coords = torch.zeros(
                    (len(slide_dataset), 2), dtype=torch.int32
                )

                for i, (x, metadata) in enumerate(slide_tiles_dataloader):
                    x = x.to(device)
                    embeddings = process_output(tile_encoder(x), model)
                    start = i * config.dataloader.batch_size
                    end = start + embeddings.size(0)
                    slide_tiles_embeddings[start:end] = embeddings.to("cpu")
                    slide_tiles_coords[start:end] = torch.stack(
                        [metadata["x"], metadata["y"]], dim=-1
                    )

                save_embeddings(
                    slide_tiles_embeddings, slide_tiles_coords, embeddings_path
                )

    logger.experiment.log_artifacts(
        run_id=logger.run_id, local_dir=str(output_folder), artifact_path="embeddings"
    )


if __name__ == "__main__":
    main()
