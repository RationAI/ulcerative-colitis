from pathlib import Path
from typing import cast

import mlflow
import pandas as pd
from lightning import LightningModule, Trainer
from rationai.masks.mask_builders import ScalarMaskBuilder
from rationai.mlkit.lightning.callbacks import MultiloaderLifecycle

from ulcerative_colitis.data import DataModule
from ulcerative_colitis.typing import Output, PredictInput


class MaskBuilderCallback(MultiloaderLifecycle):
    mask_builders: list[ScalarMaskBuilder]
    mask_builder_argmax: ScalarMaskBuilder

    def on_predict_dataloader_start(
        self, trainer: Trainer, pl_module: LightningModule, dataloader_idx: int
    ) -> None:
        if not hasattr(trainer, "datamodule"):
            raise ValueError("Trainer must have a datamodule to use this callback")

        datamodule = cast(DataModule, trainer.datamodule)
        slide = cast(pd.DataFrame, datamodule.predict.slides).iloc[dataloader_idx]

        # TODO: fix mmp -> mpp
        self.mask_builders = [
            ScalarMaskBuilder(
                save_dir=Path("masks"),
                filename=Path(slide.path).stem,
                extent_x=slide.extent_x,
                extent_y=slide.extent_y,
                mmp_x=slide.mpp_x,
                mmp_y=slide.mpp_y,
                extent_tile=slide.tile_extent_x,
                stride=slide.stride_x,
            )
            for _ in range(5)
        ]

        self.mask_builder_argmax = ScalarMaskBuilder(
            save_dir=Path("masks"),
            filename=Path(slide.path).stem,
            extent_x=slide.extent_x,
            extent_y=slide.extent_y,
            mmp_x=slide.mpp_x,
            mmp_y=slide.mpp_y,
            extent_tile=slide.tile_extent_x,
            stride=slide.stride_x,
        )

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Output,
        batch: PredictInput,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        metadata = batch[1]
        # outputs is a tensor of shape (batch_size, 1)
        # TODO: fix move ImageBuilder to right device

        for i, mask_builder in enumerate(self.mask_builders):
            mask_builder.update(outputs[:, i].cpu(), metadata["x"], metadata["y"])
        self.mask_builder_argmax.update(
            (outputs.argmax(dim=1).cpu() + 1) / 5, metadata["x"], metadata["y"]
        )

    def on_predict_dataloader_end(
        self, trainer: Trainer, pl_module: LightningModule, dataloader_idx: int
    ) -> None:
        for i, mask_builder in enumerate(self.mask_builders):
            mlflow.log_artifact(str(mask_builder.save()), artifact_path=f"nancy_{i}")

        mlflow.log_artifact(
            str(self.mask_builder_argmax.save()), artifact_path="nancy_argmax"
        )
