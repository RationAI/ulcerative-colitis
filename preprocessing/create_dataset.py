import re
import tempfile
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


def get_labels(folder_path: Path, labels: list[str]) -> pd.DataFrame:
    dfs = []
    for labels_file in labels:
        labels_path = folder_path / labels_file

        if labels_path.suffix == ".csv":
            # One labels file is in CSV format
            df = pd.read_csv(labels_path, index_col=0)
        else:
            df = pd.read_excel(labels_path, index_col=0)

        df.columns = df.columns.str.lower()
        dfs.append(df)

    labels_df = pd.concat(dfs)
    # id is in format [case_id]/YY (e.g., 01234/24 or 1234/24)
    labels_df.index = labels_df.index.str.lstrip("0")
    labels_df.index = labels_df.index.str.strip()
    labels_df.index = labels_df.index.str.replace("/", "_", regex=False)

    return labels_df


def get_slides(folder_path: Path, pattern: re.Pattern) -> pd.DataFrame:
    slides = []
    for slide_path in folder_path.iterdir():
        if not pattern.fullmatch(slide_path.name):
            continue
        case_id = "_".join(slide_path.stem.split("_")[:2])

        slides.append(
            {"slide_id": slide_path.stem, "case_id": case_id, "path": str(slide_path)}
        )

    slides_df = pd.DataFrame(slides).set_index("slide_id")
    return slides_df


def create_dataset(
    folder: str, labels: list[str], institution: str, pattern: re.Pattern
) -> tuple[pd.DataFrame, list[str], list[str]]:
    folder_path = Path(folder)
    labels_df = get_labels(folder_path, labels)
    slides_df = get_slides(folder_path, pattern)

    # IKEM has only case-level labels (FTN has one slide per case)
    on = "case_id" if institution == "ikem" else "slide_id"
    dataset_df = slides_df.join(labels_df, on=on, how="outer")
    dataset_df.index.name = "slide_id"

    if institution == "ikem":
        # IKEM has 'Lokalita' and 'Diagnóza' columns
        # Slides inside ileum are not used
        dataset_df = dataset_df[dataset_df["lokalita"] != "ileum"]
        # Columns 'Lokalita' and 'Diagnóza' are no longer needed

    missing_labels = dataset_df[dataset_df["nancy"].isna()].index.to_list()
    missing_slides = dataset_df[dataset_df["path"].isna()][on].to_list()

    dataset_df = dataset_df[["case_id", "path", "nancy"]]
    dataset_df = dataset_df.dropna()
    dataset_df["nancy"] = dataset_df["nancy"].astype(int)

    return dataset_df, missing_slides, missing_labels


@with_cli_args(["+preprocessing=create_dataset"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    dataset, missing_slides, missing_labels = create_dataset(
        config.dataset.folder,
        config.dataset.labels,
        config.dataset.institution,
        re.compile(config.dataset.regex_pattern),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        output_path = tmpdir_path / "dataset.csv"
        dataset.to_csv(output_path, index=True)
        logger.log_artifact(str(output_path))

        def _log_missing_items(items: list[str], filename: str) -> None:
            if not items:
                return
            file_path = tmpdir_path / filename
            file_path.write_text("\n".join(items) + "\n")
            logger.log_artifact(str(file_path))

        _log_missing_items(missing_slides, "missing_slides.txt")
        _log_missing_items(missing_labels, "missing_labels.txt")


if __name__ == "__main__":
    main()
