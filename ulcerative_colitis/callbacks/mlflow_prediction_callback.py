from lightning import Callback, LightningModule, Trainer
from rationai.mlkit.lightning.loggers.mlflow import MLFlowLogger

from ulcerative_colitis.typing import MILPredictInput, Output


class MLFlowPredictionCallback(Callback):
    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Output,
        batch: MILPredictInput,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        assert isinstance(trainer.logger, MLFlowLogger)

        metadatas = batch[1]
        for output, metadata in zip(outputs.cpu(), metadatas, strict=True):
            trainer.logger.log_table(
                {
                    "slide": metadata["slide"],
                    "prediction": output.item(),
                },
                artifact_file="predictions.json",
            )
