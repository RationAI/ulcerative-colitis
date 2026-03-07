from math import isclose
from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import pandas as pd
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.model_selection import train_test_split
from sklearn.model_selection import StratifiedGroupKFold


def split_dataset(
    dataset: pd.DataFrame, splits: DictConfig, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert isclose(sum(splits.values()), 1.0), "Splits must sum to 1.0"

    if isclose(splits["train"], 0.0):
        train = pd.DataFrame(columns=dataset.columns)
        test = dataset
    else:
        train, test = train_test_split(
            dataset,
            train_size=splits["train"],
            random_state=random_state,
            stratify=dataset["nancy"],
            groups=dataset["case_id"],
        )

    if isclose(splits["test_preliminary"], 0.0):
        test_preliminary = pd.DataFrame(columns=dataset.columns)
        test_final = test
    else:
        preliminary_size = splits["test_preliminary"] / (1.0 - splits["train"])
        test_preliminary, test_final = train_test_split(
            test,
            train_size=preliminary_size,
            random_state=random_state,
            stratify=test["nancy"],
            groups=test["case_id"],
        )

    return train, test_preliminary, test_final


def add_folds(train: pd.DataFrame, n_folds: int, random_state: int) -> pd.DataFrame:
    if train.empty:
        return train

    splitter = StratifiedGroupKFold(
        n_splits=n_folds, shuffle=True, random_state=random_state
    )
    train["fold"] = -1
    for fold, (_, val_idx) in enumerate(
        splitter.split(train, y=train["nancy"], groups=train["case_id"])
    ):
        train.loc[train.iloc[val_idx].index, "fold"] = fold
    return train


@with_cli_args(["+preprocessing=split_dataset"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    dataset = pd.read_csv(download_artifacts(config.dataset.uri))

    train, test_preliminary, test_final = split_dataset(
        dataset, config.splits, config.random_state
    )
    train = add_folds(train, config.n_folds, config.random_state)

    with TemporaryDirectory() as tmpdir:
        for name, df in (
            ("train", train),
            ("test_preliminary", test_preliminary),
            ("test_final", test_final),
        ):
            if df.empty:
                continue

            output_path = Path(tmpdir) / f"{name}.csv"
            df.to_csv(output_path, index=False)
            logger.log_artifact(str(output_path))


if __name__ == "__main__":
    main()
