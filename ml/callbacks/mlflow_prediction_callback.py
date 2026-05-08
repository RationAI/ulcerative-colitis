from lightning import Callback, LightningModule, Trainer
from rationai.mlkit.lightning.loggers.mlflow import MLFlowLogger

from ml.typing import Output


def _slide_names(metadata) -> list[str]:
    if isinstance(metadata, dict):
        return metadata["slide_name"]
    return [m["slide_name"] for m in metadata]


class MLFlowPredictionCallback(Callback):
    def _on_batch_end(
        self,
        trainer: Trainer,
        outputs: Output,
        batch: tuple,
    ) -> None:
        assert isinstance(trainer.logger, MLFlowLogger)
        trainer.logger.log_table(
            {
                "slide": _slide_names(batch[-1]),
                "prediction": outputs.tolist(),
            },
            artifact_file="predictions.json",
        )

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Output,
        batch: tuple,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._on_batch_end(trainer, outputs, batch)
        assert isinstance(trainer.logger, MLFlowLogger)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Output,
        batch: tuple,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._on_batch_end(trainer, outputs, batch)
