import os
from collections.abc import Iterable
from pathlib import Path
from typing import cast

import albumentations as A
import hydra
import pandas as pd
import timm
import torch
from huggingface_hub import login
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from timm.layers.mlp import SwiGLUPacked
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.data.datasets import TilesPredict


class FoundationModel(torch.nn.Module):
    def __init__(self, name: str, embed_dim: int) -> None:
        """Wrapper for a foundation model - forward and dimension differ depending on the model."""
        super().__init__()
        self.embed_dim = embed_dim


class ProvGigaPath(FoundationModel):
    def __init__(self, name: str) -> None:
        super().__init__(name, 1536)
        # For this, you need to setup HF_TOKEN=<X> env.variable.
        self.module = timm.create_model(
            "hf_hub:prov-gigapath/prov-gigapath", pretrained=True
        ).eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


class Virchow(FoundationModel):
    def __init__(self, name: str) -> None:
        super().__init__(name, 2560)

        # For this, you need to setup HF_TOKEN=<X> env.variable.
        self.module = timm.create_model(
            "hf-hub:paige-ai/Virchow",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        ).eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.module(x)

        class_token = output[:, 0]
        patch_tokens = output[:, 1:]
        return torch.cat([class_token, patch_tokens.mean(1)], dim=-1)


class Virchow2(FoundationModel):
    def __init__(self, name: str) -> None:
        super().__init__(name, 2560)

        # For this, you need to setup HF_TOKEN=<X> env.variable.
        self.module = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        ).eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.module(x)  # size: B x 261 x 1280

        class_token = output[:, 0]  # size: B x 1280
        patch_tokens = output[
            :, 5:
        ]  # size: B x 256 x 1280, tokens 1-4 are register tokens so we ignore those

        # concatenate class token and average pool of patch tokens
        return torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: B x 2560


class UNI(FoundationModel):
    def __init__(self, name: str) -> None:
        super().__init__(name, 1024)

        # For this, you need to setup HF_TOKEN=<X> env.variable.

        self.module = timm.create_model(
            "hf-hub:MahmoodLab/uni",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        ).eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


class UNI2(FoundationModel):
    def __init__(self, name: str) -> None:
        super().__init__(name, 1536)

        # For this, you need to setup HF_TOKEN=<X> env.variable.
        self.module = timm.create_model(
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
        ).eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


def load_dataset(uris: Iterable[str]) -> TilesPredict:
    """Load the dataset for tile embeddings.

    Assumes that the dataset has 224x224 RGB tiles.

    Args:
        uris (Iterable[str]): The URIs of the tiles.

    Returns:
        TilesPredict: The dataset object for tile embeddings.
    """
    return TilesPredict(
        uris,
        transforms=A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        ),
    )


def save_embeddings(
    slide_tiles_embeddings: torch.Tensor,
    slide_tiles_x: torch.Tensor,
    slide_tiles_y: torch.Tensor,
    embeddings_path: Path,
) -> None:
    """Save the slide embeddings to the specified path.

    Args:
        slide_tiles_embeddings (torch.Tensor): The embeddings to save.
        slide_tiles_x (torch.Tensor): The x-coordinates of the tiles.
        slide_tiles_y (torch.Tensor): The y-coordinates of the tiles.
        embeddings_path (Path): The path to save the embeddings to.
    """
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "x": slide_tiles_x.numpy(),
            "y": slide_tiles_y.numpy(),
            "embedding": [emb.numpy() for emb in slide_tiles_embeddings],
        }
    )

    df.to_parquet(embeddings_path, index=False, engine="pyarrow")


@with_cli_args(["+preprocessing=embeddings"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    login(token=os.environ["HF_TOKEN"])
    dest = Path(config.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tile_encoder: FoundationModel = hydra.utils.instantiate(config.tile_encoder)
    tile_encoder = tile_encoder.to(device)

    with torch.no_grad():
        dataset = load_dataset(config.dataset.tiling_uris.values())

        for slide_dataset in tqdm(dataset.generate_datasets()):
            slide_name = str(slide_dataset.slide_metadata["name"])
            embeddings_path = (dest / slide_name).with_suffix(".parquet")

            if embeddings_path.exists():
                print(f"Embeddings for slide {slide_name} already exist, skipping...")
                continue

            try:
                slide_tiles_dataloader = DataLoader(
                    slide_dataset,
                    batch_size=config.dataloader.batch_size,
                    num_workers=config.dataloader.num_workers,
                    persistent_workers=config.dataloader.persistent_workers,
                )
                slide_tiles_embeddings = torch.zeros(
                    (len(slide_dataset), tile_encoder.embed_dim), dtype=torch.float32
                )
                slide_tiles_x = torch.zeros((len(slide_dataset),), dtype=torch.int32)
                slide_tiles_y = torch.zeros((len(slide_dataset),), dtype=torch.int32)

                for i, (x, metadata) in enumerate(slide_tiles_dataloader):
                    x = x.to(device)
                    embeddings = cast("torch.Tensor", tile_encoder(x))

                    start = i * config.dataloader.batch_size
                    end = start + embeddings.size(0)

                    slide_tiles_embeddings[start:end] = embeddings.to("cpu")
                    slide_tiles_x[start:end] = metadata["x"].to("cpu")
                    slide_tiles_y[start:end] = metadata["y"].to("cpu")

                save_embeddings(
                    slide_tiles_embeddings,
                    slide_tiles_x,
                    slide_tiles_y,
                    embeddings_path,
                )

            except Exception as e:
                print(f"Error processing slide {slide_name}: {e}")

        logger.log_artifacts(str(dest), artifact_path="embeddings")


if __name__ == "__main__":
    main()
