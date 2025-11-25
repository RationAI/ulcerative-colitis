from enum import Enum
from pathlib import Path

import pandas as pd
import torch


class LabelMode(Enum):
    NEUTROPHILS = "neutrophils"
    NANCY_HIGH = "nancy_high"
    NANCY_LOW = "nancy_low"
    ULCERATION = "ulceration"
    NANCY_HIGH_ALL = "nancy_high_all"
    NANCY_LOW_ALL = "nancy_low_all"
    ULCERATION_ALL = "ulceration_all"


def process_slides(slides: pd.DataFrame, mode: LabelMode | None = None) -> pd.DataFrame:
    match mode:
        case LabelMode.NEUTROPHILS:
            slides["neutrophils"] = slides["nancy_index"] >= 2
        case LabelMode.NANCY_LOW:
            slides = slides[slides["nancy_index"] < 2].copy()
        case LabelMode.NANCY_HIGH:
            slides = slides[slides["nancy_index"] >= 2].copy()
            slides["nancy_index"] -= 2
        case LabelMode.ULCERATION:
            slides = slides[slides["nancy_index"] >= 2].copy()
            slides["ulceration"] = slides["nancy_index"] == 4
        case LabelMode.NANCY_HIGH_ALL:
            slides["nancy_index"] = slides["nancy_index"].apply(lambda x: max(0, x - 2))
        case LabelMode.NANCY_LOW_ALL:
            slides["nancy_index"] = slides["nancy_index"].apply(lambda x: min(x, 1))
        case LabelMode.ULCERATION_ALL:
            slides["ulceration"] = slides["nancy_index"] == 4

    slides["name"] = slides["path"].apply(lambda x: Path(x).stem)
    return slides


def get_label(slide_metadata: pd.Series, mode: LabelMode) -> torch.Tensor:
    match mode:
        case LabelMode.NEUTROPHILS:
            return torch.tensor(slide_metadata["neutrophils"].item()).float()
        case (
            LabelMode.NANCY_LOW
            | LabelMode.NANCY_HIGH
            | LabelMode.NANCY_HIGH_ALL
            | LabelMode.NANCY_LOW_ALL
        ):
            return torch.tensor(slide_metadata["nancy_index"].item()).float()
        case LabelMode.ULCERATION | LabelMode.ULCERATION_ALL:
            return torch.tensor(slide_metadata["ulceration"].item()).float()
