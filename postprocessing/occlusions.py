from pathlib import Path

import mlflow
import mlflow.artifacts
import pyvips
from rationai.masks import write_big_tiff


INPUT_DIR = Path("in").absolute()
OUTPUT_DIR = Path("out")


def change_background_color(mask_path: Path, color: int = 128) -> None:
    # open the mask image
    mask = pyvips.Image.new_from_file(str(mask_path), access="sequential")

    # change every pixel with value 0 to the new color
    mask = (mask == 0).ifthenelse(color, mask)

    # save the modified image
    dest_path = OUTPUT_DIR / mask_path.relative_to(INPUT_DIR)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    write_big_tiff(mask, dest_path, 0.34, 0.34)


def main() -> None:
    masks = Path(
        mlflow.artifacts.download_artifacts(
            "mlflow-artifacts:/27/32559541cc864cdebe0e3e58ffbcb220/artifacts/masks",
            dst_path="in",
        )
    )

    for folder in masks.iterdir():
        for masks in folder.iterdir():
            if masks.suffix == ".tiff":
                change_background_color(masks, color=128)
                print(f"Changed background color of {masks} to 128")

    with mlflow.start_run(
        run_name="🗣️ Explaining: Ulcerative Colitis - Neutrophils MIL"
    ):
        mlflow.log_artifact(str(OUTPUT_DIR / "masks"))


if __name__ == "__main__":
    mlflow.set_experiment("IKEM")
    main()
