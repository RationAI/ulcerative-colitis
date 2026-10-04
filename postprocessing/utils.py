import json
from collections.abc import Mapping
from pathlib import Path

import mlflow
import mlflow.artifacts
import pandas as pd
from omegaconf import DictConfig


TASKS = ["neutrophils", "nancy_low", "nancy_high"]

# NHI 0-1 (negative) vs 2-4 (positive)
BINARY_THRESHOLD = 2


def load_dataset(dataset_uri: str) -> pd.DataFrame:
    return pd.read_csv(mlflow.artifacts.download_artifacts(dataset_uri), index_col=0)


def load_label_map(dataset_uri: str) -> dict[str, int]:
    dataset = load_dataset(dataset_uri)
    return {str(k): int(v) for k, v in dataset["nancy"].items()}


def load_case_map(dataset_uri: str) -> dict[str, str]:
    dataset = load_dataset(dataset_uri)
    return {str(k): str(v) for k, v in dataset["case_id"].items()}


def load_task_predictions(
    uri: str, label_map: dict[str, int] | None = None
) -> pd.DataFrame:
    artifact_path = Path(mlflow.artifacts.download_artifacts(uri))
    if artifact_path.is_dir():
        (artifact_path,) = artifact_path.glob("*.json")
    with open(artifact_path) as f:
        d = json.load(f)
    df = pd.DataFrame(d["data"], columns=d["columns"])
    if label_map is None:
        return df.set_index("slide")
    df["nancy"] = df["slide"].map(label_map)
    df = df[df["nancy"].notna()].copy()
    df["nancy"] = df["nancy"].astype(int)
    return df.set_index("slide")


def load_predictions(
    uris: Mapping[str, str],
    label_map: dict[str, int] | None = None,
) -> dict[str, pd.DataFrame]:
    task_dfs = {task: load_task_predictions(uris[task], label_map) for task in TASKS}
    common = task_dfs[TASKS[0]].index
    for df in task_dfs.values():
        common = common.intersection(df.index)
    return {k: v.loc[common] for k, v in task_dfs.items()}


def load_maps(config: DictConfig) -> tuple[dict[str, int], dict[str, str]]:
    # whole test set pools several institutions in `datasets`
    datasets = config.get("datasets") or {"dataset": config.dataset}
    label_map: dict[str, int] = {}
    case_map: dict[str, str] = {}
    for dataset in datasets.values():
        uri = dataset.mlflow_uris.dataset
        label_map |= load_label_map(uri)
        # case ids are numbered per institution and collide across institutions
        case_map |= {
            slide: f"{dataset.institution}/{case}"
            for slide, case in load_case_map(uri).items()
        }
    return label_map, case_map
