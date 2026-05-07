from datasets import Dataset as HFDataset


def filter_tiles(tiles: HFDataset, thresholds: dict[str, float]) -> HFDataset:
    return tiles.filter(
        lambda tile: all(tile[col] <= thr for col, thr in thresholds.items())
    )
