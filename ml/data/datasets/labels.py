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
        case LabelMode.NANCY_HIGH:
            # new labels: 0,1 -> 0; 2,3,4 -> 1,2,3
            slides["nancy_index"] = slides["nancy_index"].apply(lambda x: max(0, x - 1))
        case LabelMode.NANCY_LOW:
            # new labels: 0,1 -> 0,1; 2,3,4 -> 2
            slides["nancy_index"] = slides["nancy_index"].apply(lambda x: min(x, 2))

    slides["name"] = slides["path"].apply(lambda x: Path(x).stem)
    return slides


def get_label(slide_metadata: pd.Series, mode: LabelMode) -> torch.Tensor:
    match mode:
        case LabelMode.NEUTROPHILS:
            return torch.tensor(slide_metadata["neutrophils"].item()).float()
        case LabelMode.NANCY_HIGH | LabelMode.NANCY_LOW:
            return torch.tensor(slide_metadata["nancy_index"].item()).long()


def get_target_column(mode: LabelMode) -> str:
    return "neutrophils" if mode == LabelMode.NEUTROPHILS else "nancy_index"
