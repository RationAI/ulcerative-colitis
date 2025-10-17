from itertools import islice
from pathlib import Path
from typing import TypedDict, cast

import albumentations as A
import mlflow
import numpy as np
import pandas as pd
import pyvips
import timm
import torch
from lightning import Callback, Trainer
from numpy.typing import NDArray
from rationai.masks import write_big_tiff
from rationai.masks.mask_builders import ScalarMaskBuilder
from rationai.mlkit.data.datasets import OpenSlideTilesDataset
from tqdm import tqdm

from ulcerative_colitis.data import DataModule
from ulcerative_colitis.typing import Output, TileEmbeddingsPredictInput
from ulcerative_colitis.ulcerative_colitis_attention_mil import (
    UlcerativeColitisModelAttentionMIL,
)


class MaskBuilders(TypedDict):
    occlusion_attention_abs: ScalarMaskBuilder
    occlusion_attention_rel: ScalarMaskBuilder
    occlusion_attention_rescaled: ScalarMaskBuilder
    occlusion_classification_abs: ScalarMaskBuilder
    occlusion_classification_rel: ScalarMaskBuilder
    occlusion_classification_rescaled: ScalarMaskBuilder


class OcclusionCallback(Callback):
    def __init__(
        self,
        sliding_window: int,
        stride: int,
        color: int,
        batch_size: int = 1,
    ) -> None:
        self.tile_encoder = self.load_tile_encoder()
        self.transforms = self.load_transforms()
        self.to_tensor = A.ToTensorV2()

        self.sliding_window = sliding_window
        self.stride = stride
        self.color = color

        self.batch_size = batch_size

    def load_tile_encoder(self) -> torch.nn.Module:
        return timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)

    def load_transforms(self) -> A.Compose:
        return A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

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
            "extent_tile": self.sliding_window,
            "stride": self.stride,
        }

        return {
            "occlusion_attention_abs": ScalarMaskBuilder(
                save_dir=Path("masks/occlusion_attention_abs"), **kwargs
            ),
            "occlusion_attention_rel": ScalarMaskBuilder(
                save_dir=Path("masks/occlusion_attention_rel"), **kwargs
            ),
            "occlusion_attention_rescaled": ScalarMaskBuilder(
                save_dir=Path("masks/occlusion_attention_rescaled"), **kwargs
            ),
            "occlusion_classification_abs": ScalarMaskBuilder(
                save_dir=Path("masks/occlusion_classification_abs"), **kwargs
            ),
            "occlusion_classification_rel": ScalarMaskBuilder(
                save_dir=Path("masks/occlusion_classification_rel"), **kwargs
            ),
            "occlusion_classification_rescaled": ScalarMaskBuilder(
                save_dir=Path("masks/occlusion_classification_rescaled"), **kwargs
            ),
        }

    def change_background_color(
        self, mask_path: Path, mpp_x: float, mpp_y: float, color: int = 128
    ) -> None:
        mask = pyvips.Image.new_from_file(str(mask_path), access="sequential")
        mask = (mask == 0).ifthenelse(color, mask)
        write_big_tiff(mask, mask_path, mpp_x, mpp_y)

    def save_mask_builders(self, mask_builders: MaskBuilders) -> None:
        for mask_builder in mask_builders.values():
            assert isinstance(mask_builder, ScalarMaskBuilder)
            path = mask_builder.save()
            # self.change_background_color(
            #     path, mask_builder.mpp_x, mask_builder.mpp_y, self.color
            # )
            mlflow.log_artifact(str(path), artifact_path=str(mask_builder.save_dir))

    def rescale(self, probabilities: torch.Tensor) -> torch.Tensor:
        probabilities -= 0.5
        probabilities *= 0.45 / probabilities.abs().max()
        probabilities += 0.5
        return probabilities

    def batch(
        self, images: list[NDArray[np.uint8]], xs: torch.Tensor, ys: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        occlusions = []
        ys_new, xs_new = [], []

        for image, x, y in zip(images, xs, ys, strict=True):
            for x_delta in range(
                0, image.shape[1] - self.sliding_window + 1, self.stride
            ):
                for y_delta in range(
                    0, image.shape[0] - self.sliding_window + 1, self.stride
                ):
                    occlusion = image.copy()
                    occlusion[
                        y_delta : y_delta + self.sliding_window,
                        x_delta : x_delta + self.sliding_window,
                        :,
                    ] = self.color

                    occlusion = self.transforms(image=occlusion)["image"]
                    occlusion = self.to_tensor(image=occlusion)["image"]

                    occlusions.append(occlusion)
                    ys_new.append(y + y_delta)
                    xs_new.append(x + x_delta)

        return (
            torch.stack(occlusions).to(xs.device),
            torch.tensor(xs_new, device=xs.device),
            torch.tensor(ys_new, device=ys.device),
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
        self.tile_encoder.to(device=pl_module.device)
        self.tile_encoder.eval()
        for bag, metadata in zip(*batch, strict=True):
            slide_tiles = OpenSlideTilesDataset(
                slide_path=metadata["slide_path"],
                level=metadata["level"],
                tile_extent_x=metadata["tile_extent_x"],
                tile_extent_y=metadata["tile_extent_y"],
                tiles=metadata["tiles"],
            )

            mask_builders = self.get_mask_builders(metadata["slide"], trainer)

            raw_attention = cast("torch.Tensor", pl_module.attention(bag)).cpu()
            classifications_logits = cast(
                "torch.Tensor", pl_module.classifier(bag)
            ).cpu()

            # pick top 2% of the attention weights
            top_k = int(0.02 * len(raw_attention))
            top_k_indices = torch.topk(
                torch.flatten(raw_attention), top_k
            ).indices.tolist()

            attention_diffs = []
            classification_diffs = []
            xss = []
            yss = []
            bag_iter = iter(top_k_indices)
            progress_bar_iter = iter(
                tqdm(
                    bag_iter,
                    desc=f"Occlusion_{metadata['slide']}",
                    total=len(top_k_indices),
                )
            )
            while indices := list(islice(progress_bar_iter, self.batch_size)):
                images = [slide_tiles[i] for i in indices]
                occlusions, xs, ys = self.batch(
                    images, metadata["x"][indices], metadata["y"][indices]
                )

                embeddings = self.tile_encoder(occlusions)
                occlusion_raw_attention = cast(
                    "torch.Tensor", pl_module.attention(embeddings)
                ).cpu()
                occlusion_classification_logits = cast(
                    "torch.Tensor", pl_module.classifier(embeddings)
                ).cpu()

                _indices = torch.repeat_interleave(
                    torch.tensor(indices), len(xs) // len(indices)
                )
                original_raw_attention = raw_attention[_indices]
                original_classification_logits = classifications_logits[_indices]

                attention_diff_raw = occlusion_raw_attention - original_raw_attention
                classification_diff_logits = (
                    occlusion_classification_logits - original_classification_logits
                )
                mask_builders["occlusion_attention_abs"].update(
                    attention_diff_raw.sigmoid(), xs, ys
                )
                mask_builders["occlusion_attention_rel"].update(
                    (attention_diff_raw / original_raw_attention).sigmoid(), xs, ys
                )
                mask_builders["occlusion_classification_abs"].update(
                    classification_diff_logits.sigmoid(), xs, ys
                )
                mask_builders["occlusion_classification_rel"].update(
                    (
                        classification_diff_logits / original_classification_logits
                    ).sigmoid(),
                    xs,
                    ys,
                )

                attention_diffs.append(attention_diff_raw.sigmoid())
                classification_diffs.append(classification_diff_logits.sigmoid())
                xss.append(xs)
                yss.append(ys)

            attention_diffs = torch.cat(attention_diffs)
            classification_diffs = torch.cat(classification_diffs)
            xss = torch.cat(xss)
            yss = torch.cat(yss)

            attention_diffs = self.rescale(attention_diffs)
            classification_diffs = self.rescale(classification_diffs)
            mask_builders["occlusion_attention_rescaled"].update(
                attention_diffs, xss, yss
            )
            mask_builders["occlusion_classification_rescaled"].update(
                classification_diffs, xss, yss
            )

            self.save_mask_builders(mask_builders)
