import hydra
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

from postprocessing.markov_chain_confidence import (
    Confidence,
    absorption_distribution,
    compute_confidence,
)
from postprocessing.utils import load_predictions


@with_cli_args(["+postprocessing=markov_chain_confidence_predict"])
@hydra.main(config_path="../configs", config_name="postprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    confidence_mode = Confidence(config.confidence)
    uris = {
        task: config.predictions.mlflow_uris[task]
        for task in ["neutrophils", "nancy_low", "nancy_high"]
    }
    data = load_predictions(uris)

    pi = absorption_distribution(
        data["neutrophils"], data["nancy_low"], data["nancy_high"]
    )
    confidence = compute_confidence(pi, confidence_mode)

    rows = [
        {
            "slide": slide,
            "pi_0": pi_row[0],
            "pi_1": pi_row[1],
            "pi_2": pi_row[2],
            "pi_3": pi_row[3],
            "pi_4": pi_row[4],
            "confidence": conf,
        }
        for slide, pi_row, conf in zip(
            data["neutrophils"].index, pi, confidence, strict=True
        )
    ]

    logger.log_table(pd.DataFrame(rows), artifact_file="confidence.json")


if __name__ == "__main__":
    main()
