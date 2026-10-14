from collections import Counter
from typing import Protocol

from torch.utils.data import WeightedRandomSampler


class LabeledDataset(Protocol):
    def __len__(self) -> int: ...
    @property
    def labels(self) -> list[int]: ...


class AutoWeightedRandomSampler(WeightedRandomSampler):
    def __init__(self, dataset: LabeledDataset, replacement: bool = True) -> None:
        labels = dataset.labels
        counts = Counter(labels)
        weights = [1.0 / counts[label] for label in labels]
        super().__init__(weights, len(dataset), replacement)
