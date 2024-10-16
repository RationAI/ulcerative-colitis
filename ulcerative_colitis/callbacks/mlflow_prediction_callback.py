import lightning.pytorch as pl
from rationai.mlkit import Trainer
from rationai.mlkit.lightning.loggers.mlflow import MLFlowLogger

from ulcerative_colitis.typing import Output, PredictInput


class MLFlowPredictionCallback(pl.Callback):
    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: pl.LightningModule,
        outputs: Output,
        batch: PredictInput,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not isinstance(trainer.logger, MLFlowLogger):
            return

        _, metadata = batch

        table = {
            "slide": metadata["slide"],
            "x": metadata["x"].cpu(),
            "y": metadata["y"].cpu(),
        }
        for i in range(outputs.shape[1]):
            table[f"pred_{i}"] = outputs[:, i].cpu()
        trainer.logger.log_table(table, artifact_file="predictions.parquet")
