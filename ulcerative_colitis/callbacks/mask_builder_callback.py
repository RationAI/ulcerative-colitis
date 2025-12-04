from pathlib import Path
from typing import TypedDict, cast

import mlflow
import pandas as pd
import torch
from lightning import Callback, Trainer
from rationai.masks.mask_builders import ScalarMaskBuilder

from ulcerative_colitis.data import DataModule
from ulcerative_colitis.modeling.normalization import sigmoid_normalization
from ulcerative_colitis.typing import Output, TileEmbeddingsPredictInput
from ulcerative_colitis.ulcerative_colitis_attention_mil import (
    UlcerativeColitisModelAttentionMIL,
)


class MaskBuilders(TypedDict):
    attention_rescaled: ScalarMaskBuilder
    classification_binary: ScalarMaskBuilder
    classification_2: ScalarMaskBuilder
    classification_3: ScalarMaskBuilder
    classification_4: ScalarMaskBuilder


class MaskBuilderCallback(Callback):
    def get_mask_builders(self, slide_name: str, trainer: Trainer) -> MaskBuilders:
        datamodule = cast("DataModule", trainer.datamodule)
        slides = cast("pd.DataFrame", datamodule.predict.slides)
        slides["name"] = slides["path"].apply(lambda x: Path(x).stem)

        _slide = slides[slides["name"] == slide_name]
        assert len(_slide) == 1
        slide = _slide.iloc[0]

        kwargs = {
            "filename": Path(slide.path).stem,
            "extent_x": slide.extent_x,
            "extent_y": slide.extent_y,
            "mpp_x": slide.mpp_x,
            "mpp_y": slide.mpp_y,
            "extent_tile": slide.tile_extent_x,
            "stride": slide.stride_x,
        }

        return {
            "attention_rescaled": ScalarMaskBuilder(
                save_dir=Path("masks/attention_rescaled"), **kwargs
            ),
            "classification_binary": ScalarMaskBuilder(
                save_dir=Path("masks/classification_binary"), **kwargs
            ),
            "classification_2": ScalarMaskBuilder(
                save_dir=Path("masks/classification_2"), **kwargs
            ),
            "classification_3": ScalarMaskBuilder(
                save_dir=Path("masks/classification_3"), **kwargs
            ),
            "classification_4": ScalarMaskBuilder(
                save_dir=Path("masks/classification_4"), **kwargs
            ),
        }

    def save_mask_builders(self, mask_builders: MaskBuilders) -> None:
        for mask_builder in mask_builders.values():
            assert isinstance(mask_builder, ScalarMaskBuilder)
            mlflow.log_artifact(
                str(mask_builder.save()), artifact_path=str(mask_builder.save_dir)
            )

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: UlcerativeColitisModelAttentionMIL,
        outputs: Output,
        batch: TileEmbeddingsPredictInput,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        for bag, metadata in zip(*batch, strict=True):
            mask_builders = self.get_mask_builders(metadata["slide_name"], trainer)

            bag = bag[: len(metadata["x"])]

            bag = pl_module.encoder(bag)
            attention_weights = sigmoid_normalization(pl_module.attention(bag)).cpu()

            mask_builders["attention_rescaled"].update(
                min_max_normalization(attention_weights),
                metadata["x"],
                metadata["y"],
            )

            classification = pl_module.classifier(bag).cpu()

            if outputs.shape[-1] == 3:
                classification = torch.softmax(classification, dim=-1)
                mask_builders["classification_2"].update(
                    classification[:, 0],
                    metadata["x"],
                    metadata["y"],
                )
                mask_builders["classification_3"].update(
                    classification[:, 1],
                    metadata["x"],
                    metadata["y"],
                )
                mask_builders["classification_4"].update(
                    classification[:, 2],
                    metadata["x"],
                    metadata["y"],
                )
            else:
                classification = classification.sigmoid()
                mask_builders["classification_binary"].update(
                    classification,
                    metadata["x"],
                    metadata["y"],
                )

            self.save_mask_builders(mask_builders)


def min_max_normalization(tensor: torch.Tensor) -> torch.Tensor:
    weights_max = tensor.max()
    weights_min = tensor.min()
    return (tensor - weights_min) / (weights_max - weights_min)
