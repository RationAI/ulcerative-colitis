from pathlib import Path

import gigapath.slide_encoder
import mlflow
import mlflow.artifacts
import pandas as pd
import torch
from pandas import DataFrame
from torch.utils.data import DataLoader, Dataset


DF_URI = "mlflow-artifacts:/27/a42de386382f48f0b61e9e7fe898208e/artifacts/Ulcerative Colitis - "
PT_URI = "mlflow-artifacts:/27/738471d6ff2c4d979bea2860cba4a399/artifacts/"
DESTINATION = Path(
    "/mnt/data/Projects/inflammatory_bowel_dissease/ulcerative_colitis/slide_embeddings"
)


class EmbeddingDataset(Dataset):
    def __init__(self, tile_df: DataFrame, slide_df: DataFrame, folder: Path) -> None:
        self.tile_df = tile_df
        self.slide_df = slide_df
        self.folder = folder

    def __len__(self) -> int:
        return len(self.slide_df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        slide_name = self._get_slide_name(index)
        embeddings = self._get_embeddings(slide_name)
        coords = self._get_coords(index)

        return embeddings, coords, slide_name

    def _get_slide_name(self, index: int) -> str:
        return str(self.slide_df.iloc[index]["path"]).split("/")[-1].split(".")[0]

    def _get_embeddings(self, slide_name: str) -> torch.Tensor:
        return torch.load((self.folder / slide_name).with_suffix(".pt"))

    def _get_coords(self, index: int) -> torch.Tensor:
        slide_id = self.slide_df.iloc[index]["id"]
        tiles = self.tile_df.query("slide_id == " + str(slide_id))

        return torch.tensor(tiles[["x", "y"]].values, dtype=torch.float32)


def load_slide_encoder() -> torch.nn.Module:
    return gigapath.slide_encoder.create_model(
        "hf_hub:prov-gigapath/prov-gigapath",
        "gigapath_slide_enc12l768d",
        1536,
        global_pool=True,
    )


def load_dataset(stage: str) -> EmbeddingDataset:
    df_folder = Path(mlflow.artifacts.download_artifacts(DF_URI + stage))
    pt_folder = Path(mlflow.artifacts.download_artifacts(PT_URI + stage))
    tile_df = pd.read_parquet(df_folder / "tiles.parquet")
    slide_df = pd.read_parquet(df_folder / "slides.parquet")

    return EmbeddingDataset(tile_df, slide_df, pt_folder)


def save_embeddings(
    slide_embeddings: torch.Tensor, partition: str, slide_name: str
) -> None:
    folder = DESTINATION / partition
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(slide_embeddings, (folder / slide_name).with_suffix(".pt"))


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    slide_encoder = load_slide_encoder().to(device).eval()

    for stage in ["train", "val", "test1", "test2"]:
        dataset = load_dataset(stage)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        for tile_embeddings, coords, name in dataloader:
            slide_embeddings = slide_encoder(
                tile_embeddings.to(device), coords.to(device)
            )
            save_embeddings(slide_embeddings, stage, name)


if __name__ == "__main__":
    main()
