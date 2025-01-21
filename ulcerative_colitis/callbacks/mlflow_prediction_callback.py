from pathlib import Path
from typing import cast

import pandas as pd
from lightning import LightningModule, Trainer
from rationai.mlkit.lightning.callbacks import MultiloaderLifecycle
from rationai.mlkit.lightning.loggers.mlflow import MLFlowLogger
from rationai.mlkit.metrics import MeanAggregator

from ulcerative_colitis.data import DataModule
from ulcerative_colitis.typing import Output, PredictInput


class MLFlowPredictionCallback(MultiloaderLifecycle):
    aggregator: MeanAggregator

    def on_predict_dataloader_start(
        self, trainer: Trainer, pl_module: LightningModule, dataloader_idx: int
    ) -> None:
        self.aggregator = MeanAggregator()

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Output,
        batch: PredictInput,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        for output in outputs.cpu().squeeze(1):
            self.aggregator.update(output, output)

    def on_predict_dataloader_end(
        self, trainer: Trainer, pl_module: LightningModule, dataloader_idx: int
    ) -> None:
        if not isinstance(trainer.logger, MLFlowLogger):
            return

        if not hasattr(trainer, "datamodule"):
            raise ValueError("Trainer must have a datamodule to use this callback")

        datamodule = cast(DataModule, trainer.datamodule)
        slide = cast(pd.DataFrame, datamodule.predict.slides).iloc[dataloader_idx]

        table = {
            "slide": Path(slide.path).stem,
            "pred_mean": self.aggregator.compute()[0].cpu().item(),
        }
        trainer.logger.log_table(table, artifact_file="predictions.json")
