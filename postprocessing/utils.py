import json
from collections.abc import Mapping
from pathlib import Path

import mlflow
import mlflow.artifacts
import pandas as pd


TASKS = ["neutrophils", "nancy_low", "nancy_high"]


def load_label_map(dataset_uri: str) -> dict[str, int]:
    dataset = pd.read_csv(mlflow.artifacts.download_artifacts(dataset_uri), index_col=0)
    return {str(k): int(v) for k, v in dataset["nancy"].items()}


def load_fold_predictions(uri: str, label_map: dict[str, int]) -> pd.DataFrame:
    artifact_path = Path(mlflow.artifacts.download_artifacts(uri))
    if artifact_path.is_dir():
        (artifact_path,) = artifact_path.glob("*.json")
    with open(artifact_path) as f:
        d = json.load(f)
    df = pd.DataFrame(d["data"], columns=d["columns"])
    df["nancy"] = df["slide"].map(label_map)
    df = df[df["nancy"].notna()].copy()
    df["nancy"] = df["nancy"].astype(int)
    return df.set_index("slide")


def load_folds(
    uris: Mapping[str, Mapping[str, str]],
    folds: list[str],
    label_map: dict[str, int],
) -> list[dict[str, pd.DataFrame]]:
    folds_data = []
    for fold in folds:
        fold_dfs = {
            task: load_fold_predictions(uris[task][fold], label_map) for task in TASKS
        }
        common = fold_dfs[TASKS[0]].index
        for df in fold_dfs.values():
            common = common.intersection(df.index)
        folds_data.append({k: v.loc[common] for k, v in fold_dfs.items()})
        print(f"Fold {fold}: {len(common)} slides")
    return folds_data
