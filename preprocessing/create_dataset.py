import re
import tempfile
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


REGEX = {
    # [0-9]{1,5} - case ID (1 to 5 digits) (in year scope)
    # _ - underscore separator
    # 2[1-4] - year (2021 to 2024)
    # _ - underscore separator
    # HE - stain type
    # (?:_0[1-6])? - optional underscore and slide number (01 to 06)
    # .czi - file extension
    "ikem": re.compile(r"^[0-9]{1,5}_2[1-4]_HE(?:_0[1-6])?\.czi"),
    # [0-9]{1,6} - case ID (1 to 6 digits) (in year scope)
    # _ - underscore separator
    # 2[0-5] - year (2020 to 2025)
    # .czi - file extension
    "ftn": re.compile(r"[0-9]{1,6}_2[0-5]\.czi"),
    # [0-9]{1,5} - case ID (1 to 5 digits) (in year scope)
    # _ - underscore separator
    # 25 - year 2025
    # _ - underscore separator
    # [A-F] - block identifier (A to F)
    # _ - underscore separator
    # HE - stain type
    # (0[1-9]|1[0-2]) - slide number (01 to 12)
    # .czi - file extension
    "knl_patos": re.compile(r"[0-9]{1,5}_25_[A-F]_HE(0[1-9]|1[0-2])\.czi"),
}


def get_labels(folder_path: Path, labels: list[str]) -> pd.DataFrame:
    dfs = []
    for labels_file in labels:
        labels_path = Path(folder_path) / labels_file

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


def get_slides(folder_path: Path, institution: str) -> pd.DataFrame:
    slides = []
    for slide_path in folder_path.iterdir():
        if not REGEX[institution].fullmatch(slide_path.name):
            continue
        case_id = "_".join(slide_path.stem.split("_")[:2])

        slides.append(
            {"slide_id": slide_path.stem, "case_id": case_id, "path": str(slide_path)}
        )

    slides_df = pd.DataFrame(slides).set_index("slide_id")
    return slides_df


def create_dataset(
    folder: str, labels: list[str], institution: str
) -> tuple[pd.DataFrame, list[str], list[str]]:
    folder_path = Path(folder)
    labels_df = get_labels(folder_path, labels)
    slides_df = get_slides(folder_path, institution)

    # IKEM has only case-level labels (FTN has one slide per case)
    on = "case_id" if institution == "ikem" else "slide_id"
    dataset_df = slides_df.join(labels_df, on=on, how="outer")

    if institution == "ikem":
        # IKEM has 'Lokalita' and 'Diagnóza' columns
        # Slides inside ileum are not used
        dataset_df = dataset_df[dataset_df["lokalita"] != "ileum"]
        # Columns 'Lokalita' and 'Diagnóza' are no longer needed

    missing_labels = dataset_df[dataset_df["nancy"].isna()].index.to_list()
    missing_slides = dataset_df[dataset_df["path"].isna()][on].to_list()

    dataset_df = dataset_df[["case_id", "path", "nancy"]]
    dataset_df = dataset_df.dropna()

    return dataset_df, missing_slides, missing_labels


@with_cli_args(["+preprocessing=create_dataset"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    dataset, missing_slides, missing_labels = create_dataset(
        config.dataset.folder, config.dataset.labels, config.dataset.institution
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        output_path = tmpdir_path / "dataset.csv"
        dataset.to_csv(output_path, index=True)
        logger.log_artifact(str(output_path))

        if missing_slides:
            missing_slides_path = tmpdir_path / "missing_slides.txt"
            with open(missing_slides_path, "w") as f:
                for slide in missing_slides:
                    f.write(f"{slide}\n")
            logger.log_artifact(str(missing_slides_path))

        if missing_labels:
            missing_labels_path = tmpdir_path / "missing_labels.txt"
            with open(missing_labels_path, "w") as f:
                for label in missing_labels:
                    f.write(f"{label}\n")
            logger.log_artifact(str(missing_labels_path))


if __name__ == "__main__":
    main()
