from itertools import islice
from pathlib import Path
from typing import TypedDict, cast

import albumentations as A
import mlflow
import numpy as np
import pandas as pd
import timm
import torch
from lightning import Callback, Trainer
from numpy.typing import NDArray
from rationai.masks.mask_builders import ScalarMaskBuilder
from rationai.mlkit.data.datasets import OpenSlideTilesDataset
from tqdm import tqdm

from ulcerative_colitis.data import DataModule
from ulcerative_colitis.typing import MILPredictInput, Output
from ulcerative_colitis.ulcerative_colitis_attention_mil import (
    UlcerativeColitisModelAttentionMIL,
)


class MaskBuilders(TypedDict):
    occlusion_attention_abs: ScalarMaskBuilder
    occlusion_attention_rel: ScalarMaskBuilder
    occlusion_classification_abs: ScalarMaskBuilder
    occlusion_classification_rel: ScalarMaskBuilder


class OcclusionCallback(Callback):
    def __init__(
        self,
        sliding_window: int,
        stride: int,
        color: int = 255,
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
            "occlusion_classification_abs": ScalarMaskBuilder(
                save_dir=Path("masks/occlusion_classification_abs"), **kwargs
            ),
            "occlusion_classification_rel": ScalarMaskBuilder(
                save_dir=Path("masks/occlusion_classification_rel"), **kwargs
            ),
        }

    def save_mask_builders(self, mask_builders: MaskBuilders) -> None:
        for mask_builder in mask_builders.values():
            assert isinstance(mask_builder, ScalarMaskBuilder)
            mlflow.log_artifact(
                str(mask_builder.save()), artifact_path=str(mask_builder.save_dir)
            )

    # def batched(
    #     self, image: NDArray[np.uint8]
    # ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    #     y_iter = iter(range(0, image.shape[0] - self.sliding_window, self.stride))
    #     x_iter = iter(range(0, image.shape[1] - self.sliding_window, self.stride))
    #     prod_iter = iter(product(y_iter, x_iter))

    #     while positions := list(islice(prod_iter, self.batch_size)):
    #         occlusions = []
    #         ys, xs = [], []
    #         for y, x in positions:
    #             occlusion = image.copy()
    #             occlusion[
    #                 y : y + self.sliding_window,
    #                 x : x + self.sliding_window,
    #                 :,
    #             ] = self.color

    #             occlusion = self.transforms(image=occlusion)["image"]
    #             occlusion = self.to_tensor(image=occlusion)["image"]

    #             occlusions.append(occlusion)
    #             ys.append(y)
    #             xs.append(x)

    #         yield torch.stack(occlusions), torch.tensor(xs), torch.tensor(ys)

    def batch(
        self, images: list[NDArray[np.uint8]], xs: torch.Tensor, ys: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        occlusions = []
        ys_new, xs_new = [], []

        for image, x, y in zip(images, xs, ys, strict=True):
            for x_delta in range(0, image.shape[1] - self.sliding_window, self.stride):
                for y_delta in range(
                    0, image.shape[0] - self.sliding_window, self.stride
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
        batch: MILPredictInput,
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

            raw_attention = pl_module.attention(bag)
            raw_attention_exp = torch.exp(raw_attention)
            softmax_denominator = torch.sum(raw_attention_exp, dim=0, keepdim=True)
            attention_weights = torch.softmax(raw_attention, dim=0).cpu()
            classifications = torch.sigmoid(pl_module.classifier(bag)).cpu()

            bag_iter = iter(range(len(bag)))
            progress_bar_iter = iter(
                tqdm(bag_iter, desc=f"Occlusion_{metadata['slide']}", total=len(bag))
            )
            while indices := list(islice(progress_bar_iter, self.batch_size)):
                images = [slide_tiles[i] for i in indices]
                occlusions, xs, ys = self.batch(
                    images, metadata["x"][indices], metadata["y"][indices]
                )

                embeddings = self.tile_encoder(occlusions)
                occlusion_attention = pl_module.attention(embeddings)
                occlusion_attention_exp = torch.exp(occlusion_attention)
                occlusion_attention_weights = occlusion_attention_exp / (
                    softmax_denominator
                    + occlusion_attention_exp
                    - raw_attention_exp[indices]
                )

                occlusion_classification = torch.sigmoid(
                    pl_module.classifier(embeddings)
                )
                original_attention_weight = attention_weights[indices]
                original_classification = classifications[indices]

                attention_diff = (
                    occlusion_attention_weights.cpu() - original_attention_weight
                )
                classification_diff = (
                    occlusion_classification.cpu() - original_classification
                )
                mask_builders["occlusion_attention_abs"].update(
                    attention_diff.sigmoid(), xs, ys
                )
                mask_builders["occlusion_attention_rel"].update(
                    (attention_diff / original_attention_weight).sigmoid(), xs, ys
                )
                mask_builders["occlusion_classification_abs"].update(
                    classification_diff.sigmoid(), xs, ys
                )
                mask_builders["occlusion_classification_rel"].update(
                    (classification_diff / original_classification).sigmoid(), xs, ys
                )

            # for i in tqdm(range(len(bag)), desc=f"Occlusion_{metadata['slide']}"):
            #     image = slide_tiles[i]

            #     for occlusions, xs, ys in self.batched(image):
            #         occlusions = occlusions.to(device=pl_module.device)
            #         xs = xs.to(device=pl_module.device)
            #         ys = ys.to(device=pl_module.device)

            #         embeddings = self.tile_encoder(occlusions)

            #         occlusion_attention = pl_module.attention(embeddings)
            #         occlusion_attention_exp = torch.exp(occlusion_attention)

            #         occlusion_attention_weights = occlusion_attention_exp / (
            #             softmax_denominator
            #             + occlusion_attention_exp
            #             - raw_attention_exp[i]
            #         )

            #         occlusion_classification = torch.sigmoid(
            #             pl_module.classifier(embeddings)
            #         )

            #         original_attention_weight = attention_weights[i]
            #         original_classification = classifications[i]

            #         attention_diff = (
            #             occlusion_attention_weights.cpu() - original_attention_weight
            #         )

            #         classification_diff = (
            #             occlusion_classification.cpu() - original_classification
            #         )

            #         mask_builders["occlusion_attention_abs"].update(
            #             attention_diff.sigmoid(),
            #             metadata["x"][i] + xs,
            #             metadata["y"][i] + ys,
            #         )

            #         mask_builders["occlusion_attention_rel"].update(
            #             (attention_diff / original_attention_weight).sigmoid(),
            #             metadata["x"][i] + xs,
            #             metadata["y"][i] + ys,
            #         )

            #         mask_builders["occlusion_classification_abs"].update(
            #             classification_diff.sigmoid(),
            #             metadata["x"][i] + xs,
            #             metadata["y"][i] + ys,
            #         )

            #         mask_builders["occlusion_classification_rel"].update(
            #             (classification_diff / original_classification).sigmoid(),
            #             metadata["x"][i] + xs,
            #             metadata["y"][i] + ys,
            #         )

            self.save_mask_builders(mask_builders)
