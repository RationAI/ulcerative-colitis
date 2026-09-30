import math

from datasets import Dataset as HFDataset


def filter_tiles(tiles: HFDataset, thresholds: dict[str, float]) -> HFDataset:
    return tiles.filter(
        lambda tile: all(
            _missing(tile[col]) or tile[col] <= thr for col, thr in thresholds.items()
        )
    )


def _missing(value: float | None) -> bool:
    # tiles of slides where QC failed have no scores and are kept
    return value is None or math.isnan(value)
