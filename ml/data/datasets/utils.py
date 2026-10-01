import hashlib
import shutil
from pathlib import Path, PurePosixPath

import numpy as np
import pyarrow as pa
import torch
from datasets import Dataset as HFDataset
from filelock import FileLock
from mlflow.artifacts import download_artifacts


def filter_tiles(tiles: HFDataset, thresholds: dict[str, float]) -> HFDataset:
    # vectorized over Arrow columns; a per-row HF .filter() decodes every
    # embedding into Python lists and takes minutes per shard
    table = tiles.with_format("arrow")[:]
    keep = np.ones(len(tiles), dtype=bool)
    for col, thr in thresholds.items():
        # tiles of slides where QC failed have no scores (null/NaN) and are kept
        values = table[col].cast(pa.float64()).fill_null(np.nan).to_numpy()
        keep &= np.isnan(values) | (values <= thr)
    return tiles.select(np.flatnonzero(keep))


def embeddings_tensor(tiles: HFDataset) -> torch.Tensor:
    # go through Arrow -> NumPy instead of torch.tensor(tiles["embedding"]),
    # which builds a Python float per element and is ~200x slower
    column = tiles.with_format("arrow")["embedding"]
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    values = column.flatten().to_numpy(zero_copy_only=False)
    return torch.from_numpy(values.reshape(len(column), -1).astype(np.float32))


def column_tensor(tiles: HFDataset, column: str) -> torch.Tensor:
    # tiles[column] is a lazy Column that torch.tensor() reads element by element
    return torch.from_numpy(np.array(tiles.with_format("arrow")[column]))


def download_artifacts_cached(uri: str, cache_dir: Path) -> Path:
    # MLflow artifacts are immutable, so each URI is downloaded once to a stable
    # path; stable paths also let HF reuse its arrow cache (keyed by file path)
    target = cache_dir / hashlib.sha256(uri.encode()).hexdigest()[:16]
    target.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(f"{target}.lock"):  # concurrent jobs wait for the first one
        if not (target / ".complete").exists():
            shutil.rmtree(target, ignore_errors=True)  # interrupted download
            download_artifacts(artifact_uri=uri, dst_path=str(target))
            (target / ".complete").touch()
    return target / PurePosixPath(uri).name
