import lightning.pytorch as pl
from rationai.mlkit import Trainer
from rationai.mlkit.lightning.loggers.mlflow import MLFlowLogger

from ulcerative_colitis.typing import Input, Output


class MLFlowPredictionCallback(pl.Callback):
    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: pl.LightningModule,
        outputs: Output,
        batch: Input,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not isinstance(trainer.logger, MLFlowLogger):
            return

        _, target, metadata = batch

        table = {
            "slide": metadata["slide"],
            "x": metadata["x"].cpu(),
            "y": metadata["y"].cpu(),
            "prediction": outputs.cpu(),
            "target": target.cpu().flatten(),
        }
        print(table)
        trainer.logger.log_table(table, artifact_file="predictions.parquet")
