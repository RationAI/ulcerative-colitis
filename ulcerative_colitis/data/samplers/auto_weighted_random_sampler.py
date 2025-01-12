from collections.abc import Sequence

import pandas as pd
from torch.utils.data import WeightedRandomSampler

from ulcerative_colitis.data.datasets import UlcerativeColitis


class AutoWeightedRandomSampler(WeightedRandomSampler):
    def __init__(self, dataset: UlcerativeColitis, replacement: bool = True) -> None:
        merged = pd.merge(
            dataset.tiles, dataset.slides, left_on="slide_id", right_on="id"
        )
        super().__init__(self._get_weights(merged), len(dataset), replacement)

    def _get_weights(self, df: pd.DataFrame) -> Sequence[float]:
        value_counts = df["nancy_index"].value_counts()
        return df["nancy_index"].apply(lambda x: 1 / value_counts[x]).tolist()
