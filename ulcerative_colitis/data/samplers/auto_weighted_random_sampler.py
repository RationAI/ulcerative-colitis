from collections.abc import Sequence

import pandas as pd
from torch.utils.data import WeightedRandomSampler

from ulcerative_colitis.data.datasets import UlcerativeColitis


class AutoWeightedRandomSampler(WeightedRandomSampler):
    def __init__(self, dataset: UlcerativeColitis, replacement: bool = True) -> None:
        super().__init__(self._get_weights(dataset.tiles), len(dataset), replacement)

    def _get_weights(self, df: pd.DataFrame) -> Sequence[float]:
        value_counts = df["nancy_index"].value_counts()
        return df["nancy_index"].apply(lambda x: 1 / value_counts[x]).tolist()
