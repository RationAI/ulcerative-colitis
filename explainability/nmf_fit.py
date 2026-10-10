import json
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

from explainability.factorization import (
    METHODS,
    kmeans_init,
    nnls_batch,
    reseed_dead_components,
)
from explainability.tiles import load_embeddings_dataset, resolve_embedding_split_dirs


def resolve_percentile_stats_path(mlflow_uri: str) -> Path:
    """Download token_statistics' percentile_stats.parquet from mlflow.

    Unlike the embeddings parquet (huge - see `explainability.tiles.
    resolve_embedding_split_dirs`'s local-mount-preferred fast path) or W
    (memmap-sized), this is a small per-dimension summary - a few hundred KB
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


def iter_embedding_batches(
    dataset: ray.data.Dataset,
    batch_size: int,
    shift: np.ndarray,
    scale: np.ndarray,
    metadata_columns: tuple[str, ...] = (),
    shuffle_seed: int | None = None,
    shuffle_buffer_size: int | None = None,
    clip: bool = True,
) -> Iterator[tuple[np.ndarray, pd.DataFrame | None]]:
    """Yield shifted, scaled (and by default non-negative) embedding batches with optional provenance.

    Args:
        dataset: `ray.data.Dataset` with an `embedding` column, in whatever
            order the caller wants read (see `shuffle_seed`).
        batch_size: Number of rows to read per yielded batch.
        shift: Per-dimension shift constant `c`, shape (embed_dim,).
        scale: Per-dimension scale constant `d` (the IQR, see `load_scale`),
            shape (embed_dim,) - matches concept_mil.tex's non-negativity
            transform t~ = (t + c) / d (here `shift` plays the role of `-c`).
        metadata_columns: Columns (e.g. `("slide_id", "x", "y")`) to also
            yield as a DataFrame aligned with each batch, so each row of W
            can be traced back to where it came from. Empty -> no metadata.
        shuffle_seed: If given, rows are read in a locally-shuffled order
            (a cheap, per-worker approximate shuffle - see `Dataset.iter_batches`'s
            `local_shuffle_buffer_size`, no cross-node data movement) - used
            for the per-epoch NMF training passes. Leave as None (read order
            preserved) for the final transform pass, since its output rows
            must line up 1:1 with the yielded metadata.
        shuffle_buffer_size: Row buffer size for the local shuffle; required
            together with `shuffle_seed`, ignored otherwise.
        clip: Clip negatives to zero (NMF's non-negativity transform). Off
            for semi-NMF, which factors the signed embeddings directly.

    Yields:
        Tuples of (rows, metadata), where rows has shape
        (n_rows_in_batch, embed_dim) and metadata is None unless
        `metadata_columns` is non-empty.
    """
    for batch in dataset.select_columns([*metadata_columns, "embedding"]).iter_batches(
        batch_size=batch_size,
        batch_format="numpy",
        local_shuffle_seed=shuffle_seed,
        local_shuffle_buffer_size=shuffle_buffer_size,
    ):
        raw = np.stack(batch["embedding"]).astype(np.float32, copy=False)
        rows = (raw - shift) / scale
        if clip:
            rows = np.maximum(rows, 0.0)

        metadata = None
        if metadata_columns:
            metadata = pd.DataFrame({column: batch[column] for column in metadata_columns})
        yield rows, metadata


def load_mean(percentile_stats_path: Path) -> np.ndarray:
    """Load the exact per-dimension mean embedding from token_statistics' output."""
    stats = pd.read_parquet(percentile_stats_path).sort_index()
    if "mean" not in stats:
        raise ValueError(
            f"{percentile_stats_path} has no `mean` column - re-run token_statistics.py "
            "(runs before 2026-10-10 predate it) and point shift.mlflow_uri at the new run."
        )
    return stats["mean"].to_numpy(dtype=np.float32)


def select_shift(method: str, center: bool, stats_path: Path, percentile_column: str) -> np.ndarray:
    """The offset mu subtracted before factorizing (and added back by the decoder).

    nmf: the non-negativity shift (a low percentile per dimension) - centring
    would leave about half of every dimension negative, all clipped to zero
    by NMF's non-negativity transform, so `center` must be false. semi_nmf:
    the corpus mean when `center` (otherwise several components get spent
    reproducing the shared mean every tile carries), else zero.
    """
    if method == "nmf":
        if center:
            raise ValueError("center=true is incompatible with method=nmf - set center=false")
        return load_shift(stats_path, percentile_column)
    if center:
        return load_mean(stats_path)
    return np.zeros_like(load_scale(stats_path))


def gauge_fix_dictionary(h: np.ndarray) -> np.ndarray:
    """Fix the WH scale ambiguity: rescale H to unit rows.

    For any positive diagonal S, W @ H == (W @ S^-1) @ (S @ H), so component
    magnitudes carry no meaning until this is fixed. No corresponding W
    correction is needed from the caller: `main` re-transforms every row
    against this gauge-fixed H directly rather than computing W against the
    pre-fix H and rescaling it after the fact.

    Args:
        h: Dictionary/components matrix, shape (n_components, n_features).

    Returns:
        The gauge-fixed H (unit L2-norm rows).
    """
    norms = np.linalg.norm(h, axis=1)
    norms = np.where(norms == 0, 1.0, norms)
    return h / norms[:, None]


def fit_nmf_online(
    dataset: ray.data.Dataset, config: DictConfig, shift: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """MiniBatchNMF over shuffled passes of the shifted, scaled, clipped embeddings.

    Updates H only - W from mid-training batches would reflect a moving
    target rather than the final dictionary, so W comes from a separate
    transform pass against the final H.

    Returns:
        H in the scaled fit space, shape `(n_components, embed_dim)`.
    """
    model = MiniBatchNMF(
        n_components=config.n_components,
        init=config.nmf.init,
        beta_loss=config.nmf.beta_loss,
        random_state=config.nmf.random_state,
    )
    for epoch in range(config.nmf.epochs):
        for rows, _ in iter_embedding_batches(
            dataset,
            config.nmf.batch_size,
            shift,
            scale,
            shuffle_seed=config.nmf.random_state + epoch,
            shuffle_buffer_size=config.nmf.shuffle_buffer_size,
        ):
            model.partial_fit(rows)
    return model.components_


def fit_semi_nmf_online(
    dataset: ray.data.Dataset, config: DictConfig, shift: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """Online semi-NMF (W >= 0, H signed) over shuffled passes of the shifted, scaled, signed embeddings.

    Online dictionary learning (Mairal et al. 2010) with semi-NMF's
    unconstrained H-step: per batch, W_b is the exact NNLS solution against
    the current H, and H is re-solved from exponentially-forgotten
    sufficient statistics `A = sum W_b^T W_b`, `B = sum W_b^T X_b` -
    `H = (A + ridge I)^-1 B`. `forget` < 1 down-weights batches seen with an
    early, worse H. H is initialized from k-means centroids of the first
    batch (Ding et al.'s recommended semi-NMF init).

    Returns:
        H in the scaled fit space, shape `(n_components, embed_dim)`.
    """
    rng = np.random.default_rng(config.nmf.random_state)
    k = config.n_components
    h = a = b = None
    for epoch in range(config.nmf.epochs):
        for rows, _ in iter_embedding_batches(
            dataset,
            config.nmf.batch_size,
            shift,
            scale,
            shuffle_seed=config.nmf.random_state + epoch,
            shuffle_buffer_size=config.nmf.shuffle_buffer_size,
            clip=False,
        ):
            if h is None:
                h = kmeans_init(rows, k, config.nmf.random_state)
                a = np.zeros((k, k), dtype=np.float64)
                b = np.zeros((k, rows.shape[1]), dtype=np.float64)
            assert a is not None and b is not None
            w = nnls_batch(rows, h, config.nmf.nnls_iter)
            a = config.semi_nmf.forget * a + w.T @ w
            b = config.semi_nmf.forget * b + w.T @ rows
            h = np.linalg.solve(a + config.semi_nmf.ridge * np.eye(k), b).astype(np.float32)
            h = reseed_dead_components(h, np.diag(a), rows, rng)
    if h is None:
        raise ValueError("Dataset is empty - nothing to fit.")
    return h


@with_cli_args(["+explainability=nmf_fit"])
@hydra.main(config_path="../configs", config_name="explainability", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    if config.method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {config.method!r}")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_path = resolve_percentile_stats_path(config.shift.mlflow_uri)
    # scale_power=1 -> IQR (original), 0.5 -> sqrt(IQR) (gentler), 0 -> all
    # ones (no scaling) - IQR is always >0 (load_scale's own zero/negative
    # fallback), so **0 is exactly 1 for every dimension, not just close to
    # it. Why this knob exists: earlier weight checks (since removed) found
    # |Theta|*IQR (each dimension's actual contribution to the logit)
    # *positively* correlated with IQR for both halves of h_i - i.e. plain
    # IQR scaling risks suppressing exactly the dimensions the trained
    # classifiers rely on most.
    scale = load_scale(stats_path) ** config.nmf.scale_power

    split_dirs = resolve_embedding_split_dirs(
        config.sources, config.get("local_embeddings_dir"), split=config.split
    )
    dataset = load_embeddings_dataset(split_dirs)
    # Saved as shift.npy, so h_i ~= shift + phi_i @ H holds for every variant.
    shift = select_shift(config.method, config.center, stats_path, config.shift.percentile_column)
    # A plain, unfiltered read_parquet count is metadata-only (row counts come
    # from the parquet footers, no column data decoded).
    n_rows = dataset.count()

    if config.method == "nmf":
        h_fit = fit_nmf_online(dataset, config, shift, scale)
    else:
        h_fit = fit_semi_nmf_online(dataset, config, shift, scale)

    # Recover H_k = H~_k * d: the dictionary was fit on scaled embeddings, so
    # its raw coefficients are per unit of (dimension j / scale[j]) - this
    # multiplies that back out into the original (shifted-only) embedding
    # space. Must happen *before* gauge-fixing: gauge-fixing normalizes row
    # norms, and this recovery changes those norms.
    h = (h_fit * scale[None, :]).astype(np.float32)
    if config.nmf.gauge_fix:
        h = gauge_fix_dictionary(h)

    # Transform: one clean pass of *shift-only* (unscaled - H is no longer in
    # the scaled fit space) embeddings against the final H. One row per tile,
    # so W's rows are the tile-level concept weights phi_i directly, with
    # h_i ~= shift + phi_i @ H.
    unscaled = np.ones_like(scale)
    w_path = output_dir / "w.f32.npy"
    w = np.lib.format.open_memmap(
        w_path, mode="w+", dtype=np.float32, shape=(n_rows, config.n_components)
    )
    metadata_chunks = []
    offset = 0
    for rows, metadata in iter_embedding_batches(
        dataset,
        config.nmf.batch_size,
        shift,
        unscaled,
        metadata_columns=("slide_id", "x", "y"),
        clip=config.method == "nmf",
    ):
        w_batch = nnls_batch(rows, h, config.nmf.nnls_iter)
        w[offset : offset + w_batch.shape[0]] = w_batch
        metadata_chunks.append(metadata)
        offset += w_batch.shape[0]
    w.flush()
    assert offset == n_rows

    pd.concat(metadata_chunks, ignore_index=True).to_parquet(
        output_dir / "w_metadata.parquet", index=False
    )

    h_df = pd.DataFrame(h).rename_axis("component")
    h_df.to_parquet(output_dir / "h.parquet")
    # Saved next to W/H so downstream checks can rebuild the embedding
    # reconstruction h_i ~= shift + phi_i @ H without re-resolving the
    # token_statistics run this fit used.
    np.save(output_dir / "shift.npy", shift)

    manifest = {
        "w": {"path": str(w_path), "shape": list(w.shape), "dtype": str(w.dtype)},
        "method": config.method,
        "center": config.center,
        "split": config.split,
        "n_components": config.n_components,
        "shift_mlflow_uri": config.shift.mlflow_uri,
        "percentile_column": config.shift.percentile_column if config.method == "nmf" else None,
        "scale_columns": "p0.75 - p0.25 (IQR)",
        "gauge_fixed": config.nmf.gauge_fix,
        "epochs": config.nmf.epochs,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # W and its per-tile metadata stay on the project mount (too large for
    # mlflow); only H and the manifest are small enough to log directly.
    logger.log_artifact(str(output_dir / "h.parquet"))
    logger.log_artifact(str(manifest_path))


if __name__ == "__main__":
    ctx = ray.data.DataContext.get_current()
    ctx.enable_rich_progress_bars = True
    ctx.use_ray_tqdm = False

    # num_cpus set deliberately *low* - same root cause as
    # explainability/token_statistics.py (see that file's comment): confirmed
    # OOM on the old embeddings_xai patch tables without it, not
    # re-benchmarked against the per-tile embeddings - kept as the
    # conservative default. Keep in sync with cpu= in
    # scripts/explainability/nmf_fit.py.
    with ray.init(num_cpus=8, runtime_env={"excludes": [".git", ".venv"]}):
        main()
