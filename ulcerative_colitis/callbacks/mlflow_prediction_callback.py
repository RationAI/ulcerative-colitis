from lightning import Callback, LightningModule, Trainer
from rationai.mlkit.lightning.loggers.mlflow import MLFlowLogger

from ulcerative_colitis.typing import (
    Output,
    TileEmbeddingsInput,
    TileEmbeddingsPredictInput,
)


class MLFlowPredictionCallback(Callback):
    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Output,
        batch: TileEmbeddingsPredictInput,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        assert isinstance(trainer.logger, MLFlowLogger)
        trainer.logger.log_table(
            {
                "slide": [m["slide_name"] for m in batch[1]],
                "prediction": outputs.tolist(),
            },
            artifact_file="predictions.json",
        )

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Output,
        batch: TileEmbeddingsInput,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        assert isinstance(trainer.logger, MLFlowLogger)
        trainer.logger.log_table(
            {
                "slide": [m["slide_name"] for m in batch[2]],
                "prediction": outputs.tolist(),
            },
            artifact_file="predictions.json",
        )
