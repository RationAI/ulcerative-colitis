from pathlib import Path
from typing import TypedDict, cast

import mlflow
import pandas as pd
import torch
from lightning import Callback, Trainer
from rationai.masks.mask_builders import ScalarMaskBuilder

from ulcerative_colitis.data import DataModule
from ulcerative_colitis.typing import MILPredictInput, Output
from ulcerative_colitis.ulcerative_colitis_attention_mil import (
    UlcerativeColitisModelAttentionMIL,
)


class MaskBuilders(TypedDict):
    attention: ScalarMaskBuilder
    attention_rescaled: ScalarMaskBuilder
    classification: ScalarMaskBuilder
    classification_attention: ScalarMaskBuilder
    classification_attention_rescaled: ScalarMaskBuilder


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
            "attention": ScalarMaskBuilder(save_dir=Path("masks/attention"), **kwargs),
            "attention_rescaled": ScalarMaskBuilder(
                save_dir=Path("masks/attention_rescaled"), **kwargs
            ),
            "classification": ScalarMaskBuilder(
                save_dir=Path("masks/classifications"), **kwargs
            ),
            "classification_attention": ScalarMaskBuilder(
                save_dir=Path("masks/classifications_attention"), **kwargs
            ),
            "classification_attention_rescaled": ScalarMaskBuilder(
                save_dir=Path("masks/classifications_attention_rescaled"), **kwargs
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
        batch: MILPredictInput,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        for bag, metadata in zip(*batch, strict=True):
            mask_builders = self.get_mask_builders(metadata["slide"], trainer)

            bag = pl_module.encoder(bag)
            attention_weights = torch.softmax(pl_module.attention(bag), dim=0).cpu()
            classification = torch.sigmoid(pl_module.classifier(bag)).cpu()

            weights_max = attention_weights.max()

            mask_builders["attention"].update(
                attention_weights, metadata["x"], metadata["y"]
            )
            mask_builders["attention_rescaled"].update(
                attention_weights / weights_max, metadata["x"], metadata["y"]
            )
            mask_builders["classification"].update(
                classification, metadata["x"], metadata["y"]
            )
            mask_builders["classification_attention"].update(
                attention_weights * classification, metadata["x"], metadata["y"]
            )
            mask_builders["classification_attention_rescaled"].update(
                attention_weights * classification / weights_max,
                metadata["x"],
                metadata["y"],
            )

            self.save_mask_builders(mask_builders)
