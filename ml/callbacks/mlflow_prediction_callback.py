from typing import cast

from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from rationai.mlkit.lightning.loggers.mlflow import MLFlowLogger

from ml.typing import BagsInput, BagsPredictInput, Output


class MLFlowPredictionCallback(Callback):
    def _on_batch_end(
        self, trainer: Trainer, outputs: Output, batch: BagsInput | BagsPredictInput
    ) -> None:
        assert isinstance(trainer.logger, MLFlowLogger)
        trainer.logger.log_table(
            {
                "slide": [m["slide_name"] for m in batch[-1]],
                "prediction": outputs.tolist(),
            },
            artifact_file="predictions.json",
        )

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: BagsInput,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._on_batch_end(trainer, cast("Output", outputs), batch)

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Output,
        batch: BagsPredictInput,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._on_batch_end(trainer, outputs, batch)
