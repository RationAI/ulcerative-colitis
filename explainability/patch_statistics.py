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

from explainability.tiles import load_tokens_dataset, resolve_token_dirs


def compute_percentiles(
    dataset: ray.data.Dataset, percentiles: list[float], batch_size: int
) -> pd.DataFrame:
    """Estimate per-dimension percentiles by streaming t-digests over every patch.

    Reads every patch token exactly once, with no `.random_sample()` step:
    sampling doesn't actually save the expensive part here, since Ray has no
    pushdown for it and has to decode every row to filter it anyway - so it
    only ever saved *downstream* storage/compute, which this streaming
    approach no longer has (no memmap, no scratch file, no O(n log n)
    transpose). One `TDigest` per embedding dimension is updated batch by
    batch; digest updates are the dominant cost here (~100 min for the full
    ftn+ikem corpus, single-threaded on the driver, measured locally) and
    could be parallelized across a ray actor pool if that ever becomes the
    bottleneck - not done here since it currently overlaps reasonably with
    the ray-side read/decode happening concurrently in separate processes.

    Args:
        dataset: Pooled `ray.data.Dataset` of patch tokens (e.g. from
            `load_tokens_dataset` with `kind="patch"`).
        percentiles: Quantile levels in [0, 1] to estimate for each dimension.
        batch_size: Number of patches to read per batch.

    Returns:
        A DataFrame indexed by dimension, one column per requested percentile.
    """
    digests: list[TDigest] | None = None
    n_patches = 0
    start = time.monotonic()
    last_log = start
    for batch in dataset.select_columns(["embedding"]).iter_batches(
        batch_size=batch_size, batch_format="numpy"
    ):
        tokens = np.stack(batch["embedding"]).astype(np.float64, copy=False)
        if digests is None:
            digests = [TDigest() for _ in range(tokens.shape[1])]
        for dim, digest in enumerate(digests):
            digest.update(tokens[:, dim])

        n_patches += tokens.shape[0]
        now = time.monotonic()
        # Digest updates dominate wall-clock here (see docstring), so a
        # simple elapsed-time-based log is the only progress signal - there's
        # no file growing on disk to watch the way the earlier memmap-based
        # version had (see explainability-status memory). Uses print(), not
        # `logging` - configs/hydra/default.yaml sets job_logging: disabled,
        # which leaves no handler attached anywhere (verified directly:
        # log.warning() is silently dropped, not just filtered by level), so
        # anything through `logging` never appears in job output at all.
        if now - last_log > 60:
            rate = n_patches / (now - start)
            print(f"compute_percentiles: {n_patches} patches processed ({rate:.0f} patches/s)")
            last_log = now

    if digests is None:
        raise ValueError("Dataset is empty - no patches to compute percentiles over.")

    stats = np.array([[digest.inverse_cdf(p) for p in percentiles] for digest in digests])
    columns = [f"p{p:g}" for p in percentiles]
    return pd.DataFrame(stats, columns=columns).rename_axis("dimension")


@with_cli_args(["+explainability=patch_statistics"])
@hydra.main(config_path="../configs", config_name="explainability", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "percentile_stats.parquet"

    if stats_path.exists() and not config.overwrite:
        stats = pd.read_parquet(stats_path)
    else:
        token_dirs = resolve_token_dirs(
            config.sources, config.get("local_embeddings_xai_dir"), kind="patch"
        )
        dataset = load_tokens_dataset(token_dirs)
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

    # num_cpus is set deliberately *low*. Confirmed root cause (reproduced
    # directly against the real files, see explainability-status memory):
    # every patch token parquet file is a single ~1.6GB row group (Parquet
    # compresses each column chunk as one continuous stream per row group),
    # so even Ray's own automatic per-file metadata sampling
    # (`_fetch_parquet_file_info`, runs before any read task) has to
    # materialize close to the whole file - measured at 2-5GB per file just
    # to sample 1024 rows. This is independent of sampling/memmaps (still
    # true after dropping both - confirmed: removing num_cpus here
    # reintroduced the exact same stall), so it isn't going away on its own.
    # Ray's mitigation (SPREAD scheduling across cluster nodes) does nothing
    # for a single local Ray instance (one pod, no cluster - every job log
    # shows "Started a local Ray instance"), so num_cpus is what actually
    # bounds how many ~2-5GB files get processed *concurrently on this one
    # node*. Keep in sync with cpu= in scripts/explainability/patch_statistics.py.
    with ray.init(num_cpus=8, runtime_env={"excludes": [".git", ".venv"]}):
        main()
