import albumentations as A
import gigapath
import gigapath.slide_encoder
import timm

# from torchvision import transforms
import torch
from torch.utils.data import DataLoader

from ulcerative_colitis.data.datasets import NeutrophilsPredict


URIS = [
    # "mlflow-artifacts:/27/40d45169bc604d3782f140284e87725c/artifacts/Ulcerative Colitis - train"
    # "mlflow-artifacts:/27/40d45169bc604d3782f140284e87725c/artifacts/Ulcerative Colitis - val"
    "mlflow-artifacts:/27/40d45169bc604d3782f140284e87725c/artifacts/Ulcerative Colitis - test1"
]


def load_dataset() -> DataLoader:
    transforms = A.Compose(
        [
            A.CenterCrop(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    dataset = NeutrophilsPredict(URIS, transforms=transforms)
    return DataLoader(dataset, batch_size=1, num_workers=4)


def load_tile_encoder() -> torch.nn.Module:
    return timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)


def load_slide_encoder() -> torch.nn.Module:
    # TODO
    return gigapath.slide_encoder.create_model(
        "hf_hub:prov-gigapath/prov-gigapath", "gigapath_slide_enc12l768d", 1536
    )


def main() -> None:
    tile_encoder = load_tile_encoder()
    print(tile_encoder)
    print("\n\n\n")
    slide_encoder = load_slide_encoder()
    print(slide_encoder)


if __name__ == "__main__":
    main()
