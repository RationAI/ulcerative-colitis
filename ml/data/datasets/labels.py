from enum import Enum
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset as HFDataset


class LabelMode(Enum):
    NEUTROPHILS = "neutrophils"
    NANCY_HIGH = "nancy_high"
    NANCY_LOW = "nancy_low"
    ALL = "all"


def process_slides(
    slides: HFDataset,
    mode: LabelMode | None = None,
    val_fold: int | None = None,
    is_val: bool = False,
) -> HFDataset:
    match mode:
        case LabelMode.NEUTROPHILS:
            slides = slides.map(lambda x: {"neutrophils": x["nancy_index"] >= 2})
        case LabelMode.NANCY_HIGH:
            # new labels: 0,1 -> 0; 2,3,4 -> 1,2,3
            slides = slides.map(lambda x: {"nancy_index": max(0, x["nancy_index"] - 1)})
        case LabelMode.NANCY_LOW:
            # new labels: 0,1 -> 0,1; 2,3,4 -> 2
            slides = slides.map(lambda x: {"nancy_index": min(x["nancy_index"], 2)})
        case LabelMode.ALL:
            # Keep all labels
            pass

    if val_fold is not None:
        if is_val:
            slides = slides.filter(lambda x: x["fold"] == val_fold)
        else:
            slides = slides.filter(lambda x: x["fold"] != val_fold)

    slides = slides.map(lambda x: {"name": Path(x["path"]).stem})
    return slides


def get_label(slide_metadata: dict[str, Any], mode: LabelMode) -> torch.Tensor:
    match mode:
        case LabelMode.NEUTROPHILS:
            return torch.tensor(slide_metadata["neutrophils"]).float()
        case LabelMode.NANCY_HIGH | LabelMode.NANCY_LOW | LabelMode.ALL:
            return torch.tensor(slide_metadata["nancy_index"]).long()


def get_target_column(mode: LabelMode) -> str:
    return "neutrophils" if mode == LabelMode.NEUTROPHILS else "nancy_index"
