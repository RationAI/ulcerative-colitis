import json
import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import ray
from omegaconf import DictConfig, OmegaConf
from pytdigest import TDigest
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

from explainability.tiles import load_embeddings_dataset, resolve_embedding_split_dirs


def compute_percentiles(
    dataset: ray.data.Dataset, percentiles: list[float], batch_size: int
) -> pd.DataFrame:
    """Estimate per-dimension percentiles by streaming t-digests over every tile embedding.

    Reads every tile embedding exactly once, with no `.random_sample()` step:
    sampling doesn't actually save the expensive part here, since Ray has no
    pushdown for it and has to decode every row to filter it anyway - so it
    only ever saved *downstream* storage/compute, which this streaming
    approach no longer has (no memmap, no scratch file, no O(n log n)
    transpose). One `TDigest` per embedding dimension is updated batch by
    batch; digest updates are the dominant cost (the old ~39.5M-row patch
    corpus took ~100 min single-threaded on the driver; the ~1.3M-row tile
    embedding corpus at twice the width is far cheaper) and could be
    parallelized across a ray actor pool if that ever becomes the bottleneck.

    Args:
        dataset: Pooled `ray.data.Dataset` of tile embeddings (from
            `explainability.tiles.load_embeddings_dataset`).
        percentiles: Quantile levels in [0, 1] to estimate for each dimension.
        batch_size: Number of tiles to read per batch.

    Returns:
        A DataFrame indexed by dimension, one column per requested percentile
        plus an exact `mean` column.
    """
    digests: list[TDigest] | None = None
    total: np.ndarray | None = None
    n_rows = 0
    start = time.monotonic()
    last_log = start
    for batch in dataset.select_columns(["embedding"]).iter_batches(
        batch_size=batch_size, batch_format="numpy"
    ):
        rows = np.stack(batch["embedding"]).astype(np.float64, copy=False)
        if digests is None:
            digests = [TDigest() for _ in range(rows.shape[1])]
            total = np.zeros(rows.shape[1])
        assert total is not None
        total += rows.sum(axis=0)
        for dim, digest in enumerate(digests):
            digest.update(rows[:, dim])

        n_rows += rows.shape[0]
        now = time.monotonic()
        # Digest updates dominate wall-clock here (see docstring), so a
        # simple elapsed-time-based log is the only progress signal - there's
        # no file growing on disk to watch the way the earlier memmap-based
        # version had (see explainability-status memory). Uses print(), not
        # `logging` - configs/hydra/default.yaml sets job_logging: disabled,
        # which leaves no handler attached anywhere (verified directly:
        # log.warning() is silently dropped, not just filtered by level), so
        # anything through `logging` never appears in job output at all.
        # flush=True since stdout is fully-buffered (not line-buffered) once
        # it's piped/redirected rather than a TTY - without it this could sit
        # in Python's internal buffer for a long time before actually
        # reaching whatever log the job's output is captured into.
        if now - last_log > 60:
            rate = n_rows / (now - start)
            print(f"compute_percentiles: {n_rows} embeddings processed ({rate:.0f} rows/s)", flush=True)
            last_log = now

    if digests is None:
        raise ValueError("Dataset is empty - no embeddings to compute percentiles over.")

    stats = np.array([[digest.inverse_cdf(p) for p in percentiles] for digest in digests])
    columns = [f"p{p:g}" for p in percentiles]
    frame = pd.DataFrame(stats, columns=columns).rename_axis("dimension")
    # Exact (running sum, not a digest estimate) - the centring offset for
    # semi-NMF (explainability.nmf_fit.select_shift).
    assert total is not None
    frame["mean"] = total / n_rows
    return frame


@with_cli_args(["+explainability=token_statistics"])
@hydra.main(config_path="../configs", config_name="explainability", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "percentile_stats.parquet"

    if stats_path.exists() and not config.overwrite:
        stats = pd.read_parquet(stats_path)
    else:
        split_dirs = resolve_embedding_split_dirs(
            config.sources, config.get("local_embeddings_dir"), split=config.split
        )
        dataset = load_embeddings_dataset(split_dirs)
        percentiles = OmegaConf.to_object(config.percentiles)
        stats = compute_percentiles(dataset, percentiles, batch_size=config.batch_size)
        stats.to_parquet(stats_path)

    manifest = {
        "percentile_stats": {"path": str(stats_path), "n_dims": len(stats)},
        "percentiles": OmegaConf.to_object(config.percentiles),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    logger.log_artifact(str(stats_path))
    logger.log_artifact(str(manifest_path))


if __name__ == "__main__":
    ctx = ray.data.DataContext.get_current()
    ctx.enable_rich_progress_bars = True
    ctx.use_ray_tqdm = False

    # num_cpus is set deliberately *low*. Confirmed root cause on the old
    # embeddings_xai patch tables (see explainability-status memory): parquet
    # files with huge single row groups make even Ray's automatic per-file
    # metadata sampling materialize close to the whole file, and on a single
    # local Ray instance num_cpus is what bounds how many of those load
    # concurrently. Not re-benchmarked against the per-tile embeddings files -
    # kept as the conservative default. Keep in sync with cpu= in
    # scripts/explainability/token_statistics.py.
    with ray.init(num_cpus=8, runtime_env={"excludes": [".git", ".venv"]}):
        main()
