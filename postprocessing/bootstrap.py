from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

from postprocessing.ensembling import compute_metrics, run_ensembling
from postprocessing.utils import load_case_map, load_label_map, load_predictions


METHODS = ["ensembling", "hierarchical"]


def load_maps(datasets: DictConfig) -> tuple[dict[str, int], dict[str, str]]:
    label_map: dict[str, int] = {}
    case_map: dict[str, str] = {}
    for dataset in datasets.values():
        uri = dataset.mlflow_uris.dataset
        label_map |= load_label_map(uri)
        # case ids are numbered per institution and collide across institutions
        case_map |= {
            slide: f"{dataset.institution}/{case}"
            for slide, case in load_case_map(uri).items()
        }
    return label_map, case_map


def case_bootstrap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cases: np.ndarray,
    n_resamples: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    # resample whole cases with replacement, slides of one case are correlated
    case_ids, case_index = np.unique(cases, return_inverse=True)
    slides_by_case = [np.flatnonzero(case_index == i) for i in range(len(case_ids))]

    samples = []
    for _ in range(n_resamples):
        drawn = rng.integers(len(case_ids), size=len(case_ids))
        idx = np.concatenate([slides_by_case[i] for i in drawn])
        samples.append(compute_metrics(y_true[idx], y_pred[idx]))
    samples_df = pd.DataFrame(samples)

    alpha = (1.0 - confidence_level) / 2
    return pd.DataFrame(
        {
            "estimate": compute_metrics(y_true, y_pred),
            "ci_lower": samples_df.quantile(alpha),
            "ci_upper": samples_df.quantile(1.0 - alpha),
        }
    )


@with_cli_args(["+postprocessing=bootstrap"])
@hydra.main(config_path="../configs", config_name="postprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    # whole test set pools several institutions in `datasets`
    datasets = config.get("datasets") or {"dataset": config.dataset}
    label_map, case_map = load_maps(datasets)
    data = load_predictions(config.predictions.mlflow_uris, label_map)
    _, _, results = run_ensembling(data)

    y_true = results["nancy"].to_numpy()
    cases = results["slide"].map(case_map).to_numpy()

    tables = []
    for method in METHODS:
        table = case_bootstrap(
            y_true,
            results[f"pred_{method}"].to_numpy(),
            cases,
            config.n_resamples,
            config.confidence_level,
            np.random.default_rng(config.seed),
        )
        logger.log_metrics(
            {
                f"{method}/{metric}{suffix}": float(row[column])
                for metric, row in table.iterrows()
                for column, suffix in [
                    ("estimate", ""),
                    ("ci_lower", "_ci_lower"),
                    ("ci_upper", "_ci_upper"),
                ]
            }
        )
        tables.append(table.rename_axis("metric").reset_index().assign(method=method))

    with TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "confidence_intervals.csv"
        pd.concat(tables)[
            ["method", "metric", "estimate", "ci_lower", "ci_upper"]
        ].to_csv(output_path, index=False)
        logger.log_artifact(str(output_path))


if __name__ == "__main__":
    main()
