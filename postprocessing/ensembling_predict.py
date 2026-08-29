from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

from postprocessing.utils import load_predictions

PredictionData = dict[str, pd.DataFrame]


def run_ensembling_predict(data: PredictionData) -> pd.DataFrame:
    neut_df = data["neutrophils"]
    nlow_df = data["nancy_low"]
    nhigh_df = data["nancy_high"]

    neut_prob = np.array(neut_df["prediction"])
    low_probs = np.array(nlow_df["prediction"].tolist())
    high_probs = np.array(nhigh_df["prediction"].tolist())

    low_branch = low_probs[:, :2].argmax(axis=1)
    high_branch = high_probs[:, 1:].argmax(axis=1) + 2

    # Ensembling: soft majority vote across three tasks
    route_high_ens = (
        neut_prob + low_probs[:, 2] + high_probs[:, 1:].sum(axis=1)
    ) >= 1.5
    ens_pred = np.where(route_high_ens, high_branch, low_branch)

    # Hierarchical: neutrophils task alone routes
    hier_pred = np.where(neut_prob >= 0.5, high_branch, low_branch)

    return pd.DataFrame(
        {
            "slide": neut_df.index,
            "pred_ensembling": ens_pred.astype(int),
            "pred_hierarchical": hier_pred.astype(int),
        }
    )


@with_cli_args(["+postprocessing=ensembling_predict"])
@hydra.main(config_path="../configs", config_name="postprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    uris = {
        task: config.predictions.mlflow_uris[task]
        for task in ["neutrophils", "nancy_low", "nancy_high"]
    }
    data = load_predictions(uris)
    results = run_ensembling_predict(data)

    with TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "results.csv"
        results.to_csv(output_path, index=False)
        logger.log_artifact(str(output_path))


if __name__ == "__main__":
    main()
