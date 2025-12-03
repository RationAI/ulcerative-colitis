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
    attention: ScalarMaskBuilder
    attention_rescaled: ScalarMaskBuilder
    # attention_percentile: ScalarMaskBuilder
    attention_cumulative: ScalarMaskBuilder
    attention_cumulative_log5: ScalarMaskBuilder
    # classification: ScalarMaskBuilder
    # classification_attention_cumulative_log5: ScalarMaskBuilder
    # classification_attention: ScalarMaskBuilder
    # classification_attention_rescaled: ScalarMaskBuilder
    # classification_attention_percentile: ScalarMaskBuilder
    # classification_attention_cumulative: ScalarMaskBuilder


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
            # "attention_percentile": ScalarMaskBuilder(
            #     save_dir=Path("masks/attention_percentile"), **kwargs
            # ),
            "attention_cumulative": ScalarMaskBuilder(
                save_dir=Path("masks/attention_cumulative"), **kwargs
            ),
            "attention_cumulative_log5": ScalarMaskBuilder(
                save_dir=Path("masks/attention_cumulative_log5"), **kwargs
            ),
            # "classification": ScalarMaskBuilder(
            #     save_dir=Path("masks/classifications"), **kwargs
            # ),
            # "classification_attention": ScalarMaskBuilder(
            #     save_dir=Path("masks/classifications_attention"), **kwargs
            # ),
            # "classification_attention_rescaled": ScalarMaskBuilder(
            #     save_dir=Path("masks/classifications_attention_rescaled"), **kwargs
            # ),
            # "classification_attention_percentile": ScalarMaskBuilder(
            #     save_dir=Path("masks/classifications_attention_percentile"), **kwargs
            # ),
            # "classification_attention_cumulative_log5": ScalarMaskBuilder(
            #     save_dir=Path("masks/classifications_attention_cumulative"), **kwargs
            # ),
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
            # attention_weights = torch.softmax(pl_modattention(bag), dim=0).cpu()
            attention_weights = sigmoid_normalization(pl_module.attention(bag))
            mlflow.log_dict(
                {"attention_weights": attention_weights.tolist()},
                f"attention_{metadata['slide_name']}.json",
            )
            # classification = torch.sigmoid(pl_module.classifier(bag)).cpu()

            weights_max = attention_weights.max()
            weights_min = attention_weights.min()
            _, attention_cumulative = values_to_percentiles(attention_weights)
            attention_cumulative_log5 = log2_1p_rec(attention_cumulative, 5)

            mask_builders["attention"].update(
                attention_weights.to("cpu"), metadata["x"], metadata["y"]
            )
            mask_builders["attention_rescaled"].update(
                ((attention_weights - weights_min) / (weights_max - weights_min)).to(
                    "cpu"
                ),
                metadata["x"],
                metadata["y"],
            )
            # mask_builders["attention_percentile"].update(
            #     attention_percentiles, metadata["x"], metadata["y"]
            # )
            mask_builders["attention_cumulative"].update(
                attention_cumulative.to("cpu"), metadata["x"], metadata["y"]
            )
            mask_builders["attention_cumulative_log5"].update(
                attention_cumulative_log5.to("cpu"), metadata["x"], metadata["y"]
            )
            # mask_builders["classification"].update(
            #     classification, metadata["x"], metadata["y"]
            # )
            # mask_builders["classification_attention"].update(
            #     attention_weights * classification, metadata["x"], metadata["y"]
            # )
            # mask_builders["classification_attention_rescaled"].update(
            #     attention_weights * classification / weights_max,
            #     metadata["x"],
            #     metadata["y"],
            # )
            # mask_builders["classification_attention_percentile"].update(
            #     attention_percentiles * classification,
            #     metadata["x"],
            #     metadata["y"],
            # )
            # mask_builders["classification_attention_cumulative_log5"].update(
            #     attention_cumulative_log5 * classification, metadata["x"], metadata["y"]
            # )

            self.save_mask_builders(mask_builders)


def values_to_percentiles(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values = values.flatten()
    sorted_indices = values.argsort()
    ranks = torch.empty_like(sorted_indices, dtype=torch.float)
    ranks[sorted_indices] = torch.linspace(0, 1, len(values), device=values.device)

    cumulative_values = torch.cumsum(values[sorted_indices], dim=0)
    original_order_cumulative = torch.empty_like(cumulative_values)
    original_order_cumulative[sorted_indices] = cumulative_values

    return ranks.unsqueeze(1), original_order_cumulative.unsqueeze(1)


def log2_1p(x: torch.Tensor) -> torch.Tensor:
    return torch.log1p(x) / torch.log(torch.tensor(2.0, device=x.device))


def log2_1p_rec(x: torch.Tensor, depth: int = 1) -> torch.Tensor:
    if depth == 0:
        return x
    return log2_1p_rec(log2_1p(x), depth - 1)
