import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from rationai.mlkit import Trainer
from rationai.mlkit.lightning.loggers.mlflow import MLFlowLogger

from ulcerative_colitis.typing import Input, Output


class MLFlowPredictionCallback(Callback):
    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: pl.LightningModule,
        outputs: Output,
        batch: Input,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if not isinstance(trainer.logger, MLFlowLogger):
            return

        _, _, metadata = batch

        table = {
            "slide": metadata["slide"],
            "x": metadata["x"],
            "y": metadata["y"],
            "prediction": outputs,
        }
        trainer.logger.log_table(table, artifact_file="")
