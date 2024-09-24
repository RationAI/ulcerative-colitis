import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ulcerative_colitis.data.datasets import UlcerativeColitis


URIS = [
    "mlflow-artifacts:/27/91908b2ddeeb41009351e6c3bfb88236/artifacts/Ulcerative Colitis - train"
]


def main() -> None:
    dataset = UlcerativeColitis(URIS)
    dataloader = DataLoader(dataset, batch_size=155, num_workers=4)

    means = []
    stds = []

    for x, *_ in tqdm(dataloader):
        x = x.float()
        means.append(x.mean((0, 2, 3)))
        stds.append(x.std((0, 2, 3)))

    mean = torch.stack(means).mean(0)
    std = torch.stack(stds).mean(0)

    print(f"Mean: {mean}")
    print(f"Std: {std}")


if __name__ == "__main__":
    main()
