from enum import Enum
from pathlib import Path

import pandas as pd
import torch


class LabelMode(Enum):
    NEUTROPHILS = "neutrophils"
    NANCY_HIGH = "nancy_high"
    NANCY_LOW = "nancy_low"


def process_slides(slides: pd.DataFrame, mode: LabelMode | None = None) -> pd.DataFrame:
    match mode:
        case LabelMode.NEUTROPHILS:
            slides["neutrophils"] = slides["nancy_index"] >= 2
        case LabelMode.NANCY_LOW:
            slides = slides[slides["nancy_index"] < 2]
        case LabelMode.NANCY_HIGH:
            slides = slides[slides["nancy_index"] >= 2]
            slides["ulceration"] = slides["nancy_index"] == 4
            slides["nancy_index"] -= 2

    slides["name"] = slides["path"].apply(lambda x: Path(x).stem)
    return slides


def get_label(slide_metadata: pd.Series, mode: LabelMode) -> torch.Tensor:
    match mode:
        case LabelMode.NEUTROPHILS:
            return torch.tensor(slide_metadata["neutrophils"].item()).float()
        case LabelMode.NANCY_LOW:
            return torch.tensor(slide_metadata["nancy_index"].item()).float()
        case LabelMode.NANCY_HIGH:
            return torch.tensor(slide_metadata["ulceration"].item()).float()
