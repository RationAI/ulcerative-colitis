from typing import cast

from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from rationai.mlkit.lightning.loggers.mlflow import MLFlowLogger

from ml.typing import BagsInput, Output


class MLFlowPredictionCallback(Callback):
    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: BagsInput,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        assert isinstance(trainer.logger, MLFlowLogger)
        trainer.logger.log_table(
            {
                "slide": [m["slide_name"] for m in batch[2]],
                "prediction": cast("Output", outputs).tolist(),
            },
            artifact_file="predictions.json",
        )
