import json
import re
from collections.abc import Iterator
from pathlib import Path

import hydra
import mlflow.artifacts
import numpy as np
import pandas as pd
import ray
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from sklearn.decomposition import MiniBatchNMF

from explainability.tiles import (
    load_tokens_dataset,
    resolve_grade_token_dir,
    resolve_token_dirs,
)


_MLFLOW_RUN_ID_RE = re.compile(r"^mlflow-artifacts:/[^/]+/(?P<run_id>[0-9a-f]{32})/artifacts/")


def check_shift_kind(mlflow_uri: str, expected_kind: str) -> None:
    """Guard against `shift.mlflow_uri` pointing at a token_statistics run of the wrong kind.

    Confirmed real failure (2026-09-18): a kind=cls nmf_fit run was submitted
    with only `kind`/`grade`/`n_components`/`nmf.scale_power` overridden on
    the CLI, leaving `shift.mlflow_uri` at nmf_fit.yaml's kind=patch default.
    The fit completed without error - wrong shift/scale silently produces a
    valid-looking, non-crashing, wrong result, not an exception - but applying
    patch-distribution shift constants to cls tokens spuriously clipped up to
    ~78% of one dimension's real values to zero (that dimension's patch-based
    shift sat far above cls's own true minimum for it; 22.5% of dimensions
    lost >1% of their mass this way, vs. the ~0.01% the p0.0001 shift column
    is designed for). Cheap enough to check outright rather than document as
    a footgun: every token_statistics run logs its own `kind` param, so a
    mismatch against this run's `config.kind` is detected directly.

    Args:
        mlflow_uri: `config.shift[config.kind].mlflow_uri` - expected to be a
            standard "mlflow-artifacts:/<experiment_id>/<run_id>/artifacts/..."
            URI (true for every token_statistics.py run so far).
        expected_kind: This nmf_fit run's own `config.kind`.

    Raises:
        ValueError: If the token_statistics run's logged `kind` param exists
            and disagrees with `expected_kind`.
    """
    match = _MLFLOW_RUN_ID_RE.match(mlflow_uri)
    if match is None:
        return  # Not a standard mlflow-artifacts run URI - can't check, skip rather than guess.
    run_kind = mlflow.get_run(match.group("run_id")).data.params.get("kind")
    if run_kind is not None and run_kind != expected_kind:
        raise ValueError(
            f"shift.mlflow_uri ({mlflow_uri}) is a token_statistics run with kind={run_kind!r}, "
            f"but this nmf_fit run has kind={expected_kind!r} - update shift.mlflow_uri to a "
            f"token_statistics run of the matching kind."
        )


def resolve_percentile_stats_path(mlflow_uri: str) -> Path:
    """Download token_statistics' percentile_stats.parquet from mlflow.

    Unlike the patch/cls token parquet (huge - see `explainability.tiles.
    resolve_token_dirs`'s local-mount-preferred fast path) or W/H (also
    memmap-sized), this is a small per-dimension summary - a few hundred KB
    at most - so there's no large-artifact exception here: it's always
    fetched from mlflow, the one source of truth, rather than assuming a
    particular pod happens to have a local copy sitting around.

    Args:
        mlflow_uri: mlflow artifact URI for a specific token_statistics
            run's percentile_stats.parquet, e.g.
            "mlflow-artifacts:/86/<run_id>/artifacts/percentile_stats.parquet".

    Returns:
        Local filesystem path to the downloaded file (mlflow's own cache).
    """
    return Path(mlflow.artifacts.download_artifacts(mlflow_uri))


def load_shift(percentile_stats_path: Path, percentile_column: str) -> np.ndarray:
    """Load the per-dimension shift constant picked from token_statistics' output.

    Args:
        percentile_stats_path: Path to the percentile_stats.parquet produced by
            `explainability.token_statistics`.
        percentile_column: Which percentile column to use as the shift, e.g.
            "p0.0001" for the 1e-4 quantile.

    Returns:
        A 1D array of shape (embed_dim,), one shift value per dimension.
    """
    stats = pd.read_parquet(percentile_stats_path).sort_index()
    return stats[percentile_column].to_numpy(dtype=np.float32)


def load_scale(
    percentile_stats_path: Path, low_column: str = "p0.25", high_column: str = "p0.75"
) -> np.ndarray:
    """Load the per-dimension IQR scale (p0.75 - p0.25) from token_statistics' output.

    A handful of embedding dimensions carry far larger magnitude than the
    rest (observed: dimension 1 spans roughly -53..41 vs. a typical ~-4..4 -
    see explainability-status memory), and NMF's (unweighted, Frobenius) loss
    would otherwise let those few dimensions dominate what the dictionary
    fits. IQR is used rather than std since it's robust to exactly the
    outliers being downweighted (std would itself be inflated by them), and
    it comes for free from the same per-dimension percentiles already being
    computed.

    Args:
        percentile_stats_path: Path to the percentile_stats.parquet produced by
            `explainability.token_statistics` (must include the `low_column`
            and `high_column` percentiles).
        low_column: Percentile column for the IQR's lower bound.
        high_column: Percentile column for the IQR's upper bound.

    Returns:
        A 1D array of shape (embed_dim,), one scale value per dimension.
        Dimensions with a zero (or negative, shouldn't happen) IQR fall back
        to a scale of 1 rather than dividing by zero.
    """
    stats = pd.read_parquet(percentile_stats_path).sort_index()
    iqr = (stats[high_column] - stats[low_column]).to_numpy(dtype=np.float32)
    return np.where(iqr <= 0, 1.0, iqr)


def iter_token_batches(
    tokens_ds: ray.data.Dataset,
    batch_size: int,
    shift: np.ndarray,
    scale: np.ndarray,
    with_metadata: bool = False,
    shuffle_seed: int | None = None,
    shuffle_buffer_size: int | None = None,
) -> Iterator[tuple[np.ndarray, pd.DataFrame | None]]:
    """Yield shifted, scaled, non-negative token batches with optional provenance.

    `tokens_ds` already holds one row per token (see
    `preprocessing/embeddings_xai.py`, which explodes and Hive-partitions by
    `kind` at write time), so this reads and transforms tokens directly - no
    per-tile reshaping or CLS stripping needed here any more. Works for
    either `kind="patch"` (many rows per tile, `patch_index` 0..P-2) or
    `kind="cls"` (exactly one row per tile, `patch_index` always null) -
    the transform itself doesn't care which.

    Args:
        tokens_ds: `ray.data.Dataset` of tokens (e.g. from
            `explainability.tiles.load_tokens_dataset`), in whatever order
            the caller wants read (see `shuffle_seed`).
        batch_size: Number of tokens to read per yielded batch.
        shift: Per-dimension shift constant `c`, shape (embed_dim,).
        scale: Per-dimension scale constant `d` (the IQR, see `load_scale`),
            shape (embed_dim,) - matches concept_mil.tex's non-negativity
            transform t~ = (t + c) / d (here `shift` plays the role of `-c`).
        with_metadata: If True, also yield a DataFrame of (slide_id, x, y,
            patch_index) rows aligned with the yielded token batch, so each
            row of W can be traced back to the token it came from.
        shuffle_seed: If given, tokens are read in a locally-shuffled order
            (a cheap, per-worker approximate shuffle - see `Dataset.iter_batches`'s
            `local_shuffle_buffer_size`, no cross-node data movement) - used
            for the per-epoch NMF training passes. Leave as None (read order
            preserved) for the final transform pass, since its output rows
            must line up 1:1 with the yielded metadata.
        shuffle_buffer_size: Row buffer size for the local shuffle; required
            together with `shuffle_seed`, ignored otherwise.

    Yields:
        Tuples of (tokens, metadata), where tokens has shape
        (n_rows_in_batch, embed_dim) and metadata is None unless
        `with_metadata` is set.
    """
    columns = ["slide_id", "x", "y", "patch_index", "embedding"] if with_metadata else ["embedding"]

    for batch in tokens_ds.select_columns(columns).iter_batches(
        batch_size=batch_size,
        batch_format="numpy",
        local_shuffle_seed=shuffle_seed,
        local_shuffle_buffer_size=shuffle_buffer_size,
    ):
        raw = np.stack(batch["embedding"]).astype(np.float32, copy=False)
        tokens = np.maximum((raw - shift) / scale, 0.0)

        metadata = None
        if with_metadata:
            metadata = pd.DataFrame(
                {
                    "slide_id": batch["slide_id"],
                    "x": batch["x"],
                    "y": batch["y"],
                    "patch_index": batch["patch_index"],
                }
            )
        yield tokens, metadata


def gauge_fix_dictionary(h: np.ndarray) -> np.ndarray:
    """Fix the WH scale ambiguity: rescale H to unit rows.

    For any positive diagonal S, W @ H == (W @ S^-1) @ (S @ H), so component
    magnitudes carry no meaning until this is fixed. No corresponding W
    correction is needed from the caller: `main` re-transforms every token
    against this gauge-fixed H directly (by pointing `model.components_` at
    it before the transform pass) rather than computing W against the
    pre-fix H and rescaling it after the fact.

    Args:
        h: Dictionary/components matrix, shape (n_components, n_features).

    Returns:
        The gauge-fixed H (unit L2-norm rows).
    """
    norms = np.linalg.norm(h, axis=1)
    norms = np.where(norms == 0, 1.0, norms)
    return h / norms[:, None]


@with_cli_args(["+explainability=nmf_fit"])
@hydra.main(config_path="../configs", config_name="explainability", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shift_config = config.shift[config.kind]
    check_shift_kind(shift_config.mlflow_uri, config.kind)
    stats_path = resolve_percentile_stats_path(shift_config.mlflow_uri)
    shift = load_shift(stats_path, shift_config.percentile_column)
    # scale_power=1 -> IQR (original), 0.5 -> sqrt(IQR) (gentler), 0 -> all
    # ones (no scaling) - IQR is always >0 (load_scale's own zero/negative
    # fallback), so **0 is exactly 1 for every dimension, not just close to
    # it. See explainability-status memory for why this knob exists: raw
    # |Theta_m| looked anti-correlated with IQR, but |Theta_m|*IQR (each
    # dimension's actual contribution to the logit) is *positively*
    # correlated - i.e. plain IQR scaling risks suppressing exactly the
    # dimensions the trained classifiers rely on most. (theta_z_check.py
    # found a similar, weaker effect for z_i/Theta_z - this knob applies the
    # same way regardless of config.kind.)
    scale = load_scale(stats_path) ** config.nmf.scale_power

    # grade="all" fits the whole pooled corpus (every grade, both
    # institutions) via the flat pre-grade_split loader - concept_mil.tex's
    # own described design (a single dictionary fit on a sample "balanced
    # across centre and grade"), as opposed to every grade-specific run so
    # far, which reads one of explainability.grade_split's per-grade
    # partitions instead. See configs/explainability/nmf_fit.yaml's `grade`
    # comment for why this option exists.
    if config.grade == "all":
        token_dirs = resolve_token_dirs(
            config.sources, config.get("local_embeddings_xai_dir"), kind=config.kind
        )
    else:
        token_dirs = [
            resolve_grade_token_dir(
                config.get("local_grade_split_dir"),
                config.grade_split.mlflow_uri,
                kind=config.kind,
                grade=config.grade,
            )
        ]
    tokens_ds = load_tokens_dataset(token_dirs)
    # A plain, unfiltered read_parquet count is metadata-only (row counts come
    # from the parquet footers, no column data decoded) - unlike
    # token_statistics.py's sampled count, nothing here forces a full read.
    n_tokens = tokens_ds.count()

    model = MiniBatchNMF(
        n_components=config.n_components,
        init=config.nmf.init,
        beta_loss=config.nmf.beta_loss,
        random_state=config.nmf.random_state,
    )

    # Fit: several shuffled passes over the full corpus, updating H only.
    # W is deliberately not collected here - the H seen by an early batch in
    # a later epoch is already better than the H an early epoch started
    # with, so W from mid-training batches would reflect a moving target
    # rather than the final dictionary.
    for epoch in range(config.nmf.epochs):
        for tokens, _ in iter_token_batches(
            tokens_ds,
            config.nmf.batch_size,
            shift,
            scale,
            shuffle_seed=config.nmf.random_state + epoch,
            shuffle_buffer_size=config.nmf.shuffle_buffer_size,
        ):
            model.partial_fit(tokens)

    # Recover H_k = H~_k * d (concept_mil.tex eq 2.24): the dictionary was
    # fit on scaled tokens, so its raw coefficients are per unit of
    # (dimension j / scale[j]), not per unit of dimension j directly - this
    # multiplies that back out into the original (shifted-only) token space.
    # Must happen *before* gauge-fixing: gauge-fixing normalizes row norms,
    # and this recovery changes those norms (scaling each column by a
    # different amount).
    h = model.components_ * scale[None, :]
    if config.nmf.gauge_fix:
        h = gauge_fix_dictionary(h)

    # Point the model at the final (recovered, possibly gauge-fixed) H and
    # transform *shift-only* tokens (scale=1 - h is no longer in the
    # scaled-fit space, so the input mustn't be either) against it: W then
    # comes out of transform() already correct, with no separate rescale
    # needed the way leaving model.components_ unchanged would have required.
    model.components_ = h
    unscaled = np.ones_like(scale)

    # Transform: one clean pass with the now-final H to get every token's W.
    # For kind="cls" (exactly one token per tile), W's rows are already the
    # tile-level phi_ik concept_mil.tex needs directly - no per-tile
    # averaging over patches required the way kind="patch" needs downstream.
    w_path = output_dir / "w.f32.npy"
    w = np.lib.format.open_memmap(
        w_path, mode="w+", dtype=np.float32, shape=(n_tokens, config.n_components)
    )
    metadata_chunks = []
    offset = 0
    for tokens, metadata in iter_token_batches(
        tokens_ds,
        config.nmf.batch_size,
        shift,
        unscaled,
        with_metadata=True,
    ):
        w_batch = model.transform(tokens)
        w[offset : offset + w_batch.shape[0]] = w_batch
        metadata_chunks.append(metadata)
        offset += w_batch.shape[0]
    w.flush()
    assert offset == n_tokens

    pd.concat(metadata_chunks, ignore_index=True).to_parquet(
        output_dir / "w_metadata.parquet", index=False
    )

    h_df = pd.DataFrame(h).rename_axis("component")
    h_df.to_parquet(output_dir / "h.parquet")

    manifest = {
        "w": {"path": str(w_path), "shape": list(w.shape), "dtype": str(w.dtype)},
        "kind": config.kind,
        "grade": config.grade,
        "n_components": config.n_components,
        "percentile_column": shift_config.percentile_column,
        "scale_columns": "p0.75 - p0.25 (IQR)",
        "gauge_fixed": config.nmf.gauge_fix,
        "epochs": config.nmf.epochs,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # W and its per-token metadata stay on the project mount (too large for
    # mlflow, same treatment as patch_sample.f32.npy in token_statistics.py);
    # only H and the manifest are small enough to log directly.
    logger.log_artifact(str(output_dir / "h.parquet"))
    logger.log_artifact(str(manifest_path))


if __name__ == "__main__":
    ctx = ray.data.DataContext.get_current()
    ctx.enable_rich_progress_bars = True
    ctx.use_ray_tqdm = False

    # num_cpus set deliberately *low* - same fix, same root cause as
    # explainability/token_statistics.py (see that file's comment and
    # explainability-status memory): the original (pre-grade_split) patch
    # token parquet files were each a single ~1.6GB row group, so even Ray's
    # own automatic per-file metadata sampling had to materialize close to
    # the whole file - measured at 2-5GB per file. With no cap, Ray schedules
    # up to num_cpus of those concurrently on this single local Ray instance,
    # which is what OOM-killed this job. Now reading grade_split.py's output
    # instead (resolve_grade_token_dir) - those files are far smaller
    # (~16MB avg, observed directly), so this specific OOM mode likely no
    # longer applies, but num_cpus=8 is left as-is (conservative, not yet
    # re-benchmarked against the new input) rather than dropped without being
    # asked. Keep in sync with cpu= in scripts/explainability/nmf_fit.py.
    with ray.init(num_cpus=8, runtime_env={"excludes": [".git", ".venv"]}):
        main()
