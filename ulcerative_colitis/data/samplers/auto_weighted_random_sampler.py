from collections.abc import Sequence

import pandas as pd
from torch.utils.data import WeightedRandomSampler

from ulcerative_colitis.data.datasets import EmbeddingsSubset


class AutoWeightedRandomSampler(WeightedRandomSampler):
    def __init__(
        self, dataset: EmbeddingsSubset, column: str, replacement: bool = True
    ) -> None:
        super().__init__(
            self._get_weights(dataset.slides[column]), len(dataset), replacement
        )

    def _get_weights(self, column: pd.Series) -> Sequence[float]:
        value_counts = column.value_counts()
        return column.apply(lambda x: 1 / value_counts[x]).tolist()
