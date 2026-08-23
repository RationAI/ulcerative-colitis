import logging
from pathlib import Path

import ray.data
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig


log = logging.getLogger(__name__)


def resolve_token_dirs(
    sources: DictConfig, local_embeddings_xai_dir: str | None, kind: str
) -> list[str]:
    """Locate each institution's `kind` token directory ("patch" or "cls").

    `preprocessing/embeddings_xai.py` writes one row per *token* (not per
    tile), Hive-partitioned by `kind` into `tokens/kind=patch/` and
    `tokens/kind=cls/` under each institution's `train` dir - so which one is
    wanted has to be picked here, before any read happens.

    Prefers the pre-existing local copy under `local_embeddings_xai_dir`
    (matching `output_dir` in `configs/preprocessing/embeddings_xai.yaml`,
    i.e. `<local_embeddings_xai_dir>/<institution>/train`) since it's the
    same data `download_artifacts` would otherwise fetch, just without the
    slow mlflow round-trip. Falls back to `download_artifacts` when no local
    copy is found (e.g. on a fresh kube job that isn't on the project mount).

    Args:
        sources: Mapping of institution name to its embeddings_xai dataset
            config (as produced by `configs/dataset/embeddings_xai/*.yaml`).
        local_embeddings_xai_dir: Root directory to look for a local copy
            under, or None to always go through `download_artifacts`.
        kind: Which token partition to resolve - "patch" or "cls".

    Returns:
        Directories, one per institution, holding that institution's `kind`
        token parquet files - ready to hand straight to `ray.data.read_parquet`.
    """
    token_dirs = []
    for institution, source in sources.items():
        token_dir = None
        if local_embeddings_xai_dir is not None:
            candidate = (
                Path(local_embeddings_xai_dir)
                / source.institution
                / "train"
                / "tokens"
                / f"kind={kind}"
            )
            if candidate.is_dir():
                log.info("Using local %s tokens for %s: %s", kind, institution, candidate)
                token_dir = candidate

        if token_dir is None:
            folder = Path(download_artifacts(source.mlflow_uris.embeddings_xai.train))
            token_dir = folder / "tokens" / f"kind={kind}"
            if not token_dir.is_dir():
                raise FileNotFoundError(
                    f"No {kind} tokens directory found for {institution} under {folder}"
                )

        token_dirs.append(str(token_dir))
    return token_dirs


def load_tokens_dataset(
    token_dirs: list[str], read_task_memory: float | None = None
) -> ray.data.Dataset:
    """Load and pool token parquet files (across institutions) into one dataset.

    Uses `ray.data.read_parquet` rather than HF `datasets`: it reads straight
    through PyArrow's native parquet reader - the same path `embeddings_xai.py`
    used to write these files in the first place - with no separate row-by-row
    Arrow-conversion pass. That conversion is what makes `datasets.load_dataset`
    slow (tens of examples/s), and `ray.data` scales across the job's whole
    cluster rather than one node's cores.

    Args:
        token_dirs: Per-institution directories of token parquet files, as
            returned by `resolve_token_dirs`. A single `read_parquet` call
            pools all of them into one dataset.
        read_task_memory: Heap memory (bytes) to reserve per parallel read
            task, forwarded to `ray.data.read_parquet`'s `memory` arg. Each
            written file here is a *single* parquet row group (~1.6GB
            compressed / ~2.1GB uncompressed - verified directly against the
            written files, see explainability-status memory) rather than
            several smaller ones, so a read task can't decode less than
            roughly a whole file's worth of data at once. Left unset by
            default (Ray's own per-task scheduling applies, i.e. up to
            `num_cpus` files decoding concurrently); pass a value close to
            that per-file size to make Ray throttle concurrent read tasks to
            what actually fits in memory, instead of launching as many
            multi-GB decodes at once as there are CPUs.

    Returns:
        A single pooled, lazy `ray.data.Dataset` over all tokens.
    """
    # write_parquet(partition_cols=["kind"]) still writes "kind" as a real
    # column inside each file (verified empirically - it isn't Hive-optimized
    # away the way some writers do), even though it's already encoded in
    # which of these directories was resolved. Drop it here, once, so no
    # downstream consumer carries around a redundant constant column.
    return ray.data.read_parquet(token_dirs, memory=read_task_memory).drop_columns(["kind"])
