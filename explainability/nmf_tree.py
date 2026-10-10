"""Recursive (binary-split tree) concept dictionary over the per-tile embeddings.

A flat Frobenius fit is dominated by whatever occupies the most mass, so a
rare but decisive tile phenotype (e.g. crypt abscesses, which separate Nancy 1
from Nancy >= 2) never earns its own component at moderate K. Growing a
binary tree fixes this: once a split isolates, say, inflamed mucosa, the rare
pattern is a large fraction *within that node*, and the next split can
resolve it.

**Construction** (on an in-memory sample of whole slides, in the fit space -
scaled, and for `nmf` also shifted/clipped - same transform as nmf_fit.py):

- Each split is a rank-2 factorization with the chosen `method` (`nmf` or
  `semi_nmf`, explainability.factorization.factorize); rows go hard to the
  child whose weight is larger, and each child's atom is its unit-norm H row.
- Best-first: the leaf with the highest priority is tried next,
      priority(n) = mass(n) * ||varsigma(a_n)|| * err(n)
  `mass` = fraction of sample rows in the node, `err` = relative residual of
  the node's rows against its own atom, and `varsigma(a)` the classifier
  response to the atom stacked over heads - 8 entries: the neutrophils
  logit, plus the nancy_low (3) and nancy_high (4) logits each centred
  (softmax is invariant to a shared offset, so only the centred part means
  anything). Each factor guards a different failure: mass alone refines
  noise, relevance alone chases negligible nodes, error alone refines what
  no head reads.
- A split is accepted only if it passes both tests, so depth is adaptive:
    stability  re-fit the split separately on two slide-disjoint halves of
               the node's rows; the cosine of the best-matched atom pair must
               be >= `stability_threshold` (else it's sampling noise).
               Halves are by *slide*, never by tile - tiles of one slide are
               too correlated to count as independent evidence.
    tau_diff   ||varsigma(a_L) - varsigma(a_R)|| >= `tau_diff` (else no head
               distinguishes the children).
  A rejected node is final. Growth stops at `max_leaves`.
- `supervised: false` drops varsigma from the priority and disables the
  tau_diff test - the unsupervised baseline tree, to check that steering the
  tree with the classifier isn't circular.

**Inference**: the leaf atoms (recovered to the original embedding space,
unit-norm) are stacked into one dictionary H, and every tile's weights are
the NNLS solution against all leaves jointly - so H is a single global
linear decoder exactly like a flat fit (internal nodes carry no attribution)
and the outputs (`w.f32.npy`, `w_metadata.parquet`, `h.parquet`,
`shift.npy`) plug straight into concept_completeness.py and concept_gallery.py. `tree.parquet`
logs every node, including every rejected candidate split's stability and
varsigma difference, to calibrate the thresholds.
"""

import heapq
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import ray
import ray.data
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

from explainability.factorization import METHODS, factorize, nnls_batch
from explainability.model import ModelWeights, load_full_model
from explainability.nmf_fit import (
    iter_embedding_batches,
    load_scale,
    resolve_percentile_stats_path,
    select_shift,
)
from explainability.tiles import (
    load_embedding_slides,
    load_embeddings_dataset,
    resolve_embedding_split_dirs,
)


@dataclass
class Node:
    node_id: int
    parent: int | None
    depth: int
    rows: np.ndarray  # indices into the in-memory sample
    atom_fit: np.ndarray | None = None  # unit atom in the fit space
    atom: np.ndarray | None = None  # unit atom in the original embedding space
    mass: float = 1.0
    err: float = float("nan")
    varsigma: np.ndarray | None = None
    priority: float = float("inf")
    status: str = "leaf"  # leaf (unexpanded) | split | final (split rejected)
    reason: str = ""
    stability: float = float("nan")
    varsigma_diff: float = float("nan")


def unit_rows(a: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.where(norms == 0, 1.0, norms)


def head_response(atom: np.ndarray, models: dict[str, ModelWeights]) -> np.ndarray:
    """varsigma(a): each head's logit response to the atom, multiclass heads centred - 8 entries."""
    parts = []
    for model in models.values():
        logits = model.cls_w @ atom
        if len(logits) > 1:
            logits = logits - logits.mean()
        parts.append(logits)
    return np.concatenate(parts)


def relative_error(x: np.ndarray, sq_norms: np.ndarray, atom_fit: np.ndarray) -> float:
    """1 - ||projection onto the (unit) atom||^2 / ||X||^2, with non-negative weights."""
    proj = np.maximum(x @ atom_fit, 0.0)
    total = float(sq_norms.sum())
    return 1.0 - float(np.sum(proj * proj)) / total if total > 0 else 0.0


def split_rows(x: np.ndarray, config: DictConfig) -> tuple[np.ndarray, np.ndarray]:
    """Rank-2 factorization -> (hard child labels per row, the two unit atoms in fit space)."""
    w, h = factorize(
        x,
        2,
        config.method,
        config.tree.random_state,
        config.factorization.n_iter,
        config.factorization.nnls_iter,
    )
    return np.argmax(w, axis=1), unit_rows(h)


def matched_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine agreement of two atom pairs under their best matching (min over the matched pair)."""
    c = unit_rows(a) @ unit_rows(b).T
    return float(max(min(c[0, 0], c[1, 1]), min(c[0, 1], c[1, 0])))


def load_sample(
    dataset: ray.data.Dataset,
    sampled_slides: set[str],
    shift: np.ndarray,
    scale: np.ndarray,
    config: DictConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Read every tile of the sampled slides into memory, in the fit space.

    Returns:
        `(x, slide_ids)` - `x` float32 `(n, d)`, `slide_ids` one per row.
    """
    chunks, slide_chunks = [], []
    for rows, metadata in iter_embedding_batches(
        dataset,
        config.batch_size,
        shift,
        scale,
        metadata_columns=("slide_id",),
        clip=config.method == "nmf",
    ):
        assert metadata is not None
        keep = metadata["slide_id"].isin(sampled_slides).to_numpy()
        if keep.any():
            chunks.append(rows[keep])
            slide_chunks.append(metadata["slide_id"].to_numpy()[keep])
    return np.concatenate(chunks), np.concatenate(slide_chunks)


def build_tree(
    x: np.ndarray,
    half: np.ndarray,
    scale: np.ndarray,
    models: dict[str, ModelWeights],
    config: DictConfig,
) -> list[Node]:
    """Grow the tree best-first (see module docstring). Returns every node, root first."""
    tree = config.tree
    sq_norms = np.einsum("ij,ij->i", x, x)
    nodes = [Node(node_id=0, parent=None, depth=0, rows=np.arange(len(x)))]
    heap: list[tuple[float, int]] = [(-float("inf"), 0)]
    n_leaves = 1

    while heap and n_leaves < tree.max_leaves:
        _, node_id = heapq.heappop(heap)
        node = nodes[node_id]
        print(
            f"Trying node {node_id} (depth {node.depth}, {len(node.rows)} rows, "
            f"{n_leaves} leaves so far)",
            flush=True,
        )
        if len(node.rows) < 2 * tree.min_node_rows:
            node.status, node.reason = "final", "too_small"
            continue

        x_node = x[node.rows]
        labels, atoms_fit = split_rows(x_node, config)
        child_rows = [node.rows[labels == j] for j in range(2)]
        if min(len(r) for r in child_rows) < tree.min_node_rows:
            node.status, node.reason = "final", "small_child"
            continue

        half_atoms = []
        for side in range(2):
            side_rows = half[node.rows] == side
            if side_rows.sum() < 2 * tree.min_node_rows:
                break
            half_atoms.append(split_rows(x_node[side_rows], config)[1])
        del x_node
        if len(half_atoms) < 2:
            node.status, node.reason = "final", "half_too_small"
            continue
        node.stability = matched_cosine(half_atoms[0], half_atoms[1])

        atoms = unit_rows(atoms_fit * scale[None, :])
        varsigmas = [head_response(atom, models) for atom in atoms]
        node.varsigma_diff = float(np.linalg.norm(varsigmas[0] - varsigmas[1]))

        if node.stability < tree.stability_threshold:
            node.status, node.reason = "final", "unstable"
            continue
        if tree.supervised and node.varsigma_diff < tree.tau_diff:
            node.status, node.reason = "final", "indistinguishable"
            continue

        node.status = "split"
        n_leaves += 1
        for j in range(2):
            child = Node(
                node_id=len(nodes),
                parent=node_id,
                depth=node.depth + 1,
                rows=child_rows[j],
                atom_fit=atoms_fit[j],
                atom=atoms[j],
                mass=len(child_rows[j]) / len(x),
                varsigma=varsigmas[j],
            )
            child.err = relative_error(x[child.rows], sq_norms[child.rows], atoms_fit[j])
            relevance = float(np.linalg.norm(varsigmas[j])) if tree.supervised else 1.0
            child.priority = child.mass * relevance * child.err
            nodes.append(child)
            heapq.heappush(heap, (-child.priority, child.node_id))
        print(
            f"  split accepted: stability={node.stability:.3f}, "
            f"varsigma_diff={node.varsigma_diff:.4f}, children "
            f"{len(child_rows[0])}/{len(child_rows[1])} rows",
            flush=True,
        )

    if nodes[0].status != "split":
        raise ValueError(f"Root split rejected ({nodes[0].reason}) - no dictionary to build.")
    return nodes


def leaves(nodes: list[Node]) -> list[Node]:
    return [node for node in nodes if node.status != "split" and node.parent is not None]


def nodes_table(nodes: list[Node], leaf_index: dict[int, int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "node_id": node.node_id,
            "parent": node.parent,
            "depth": node.depth,
            "n_rows": len(node.rows),
            "mass": node.mass,
            "err": node.err,
            "priority": node.priority,
            "varsigma": None if node.varsigma is None else node.varsigma.tolist(),
            "status": node.status,
            "reason": node.reason,
            "stability": node.stability,
            "varsigma_diff": node.varsigma_diff,
            "component": leaf_index.get(node.node_id),
        }
        for node in nodes
    ]
    return pd.DataFrame(rows)


@with_cli_args(["+explainability=nmf_tree"])
@hydra.main(config_path="../configs", config_name="explainability", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    if config.method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {config.method!r}")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.tree.random_state)

    stats_path = resolve_percentile_stats_path(config.shift.mlflow_uri)
    scale = load_scale(stats_path) ** config.scale_power

    models = {
        head: load_full_model(config.checkpoints[head].checkpoint, config.embed_dim)
        for head in ("neutrophils", "nancy_low", "nancy_high")
    }

    split_dirs = resolve_embedding_split_dirs(
        config.sources, config.get("local_embeddings_dir"), split=config.split
    )
    dataset = load_embeddings_dataset(split_dirs)
    # Corpus mean (centred semi-NMF), NMF's non-negativity shift, or zero -
    # the same offset nmf_fit.py would use, saved as shift.npy.
    shift = select_shift(config.method, config.center, stats_path, config.shift.percentile_column)

    slide_ids = load_embedding_slides(split_dirs)["id"].to_numpy()
    n_sampled = max(2, round(config.tree.sample_slide_fraction * len(slide_ids)))
    sampled = rng.choice(slide_ids, n_sampled, replace=False)
    # Slide-disjoint halves for the stability test.
    slide_half = dict(zip(sampled, rng.permutation(len(sampled)) % 2, strict=True))

    x, row_slides = load_sample(dataset, set(sampled), shift, scale, config)
    half = np.array([slide_half[s] for s in row_slides])
    print(f"Loaded {len(x)} tiles from {n_sampled} sampled slides for tree construction.", flush=True)

    nodes = build_tree(x, half, scale, models, config)
    del x
    leaf_nodes = leaves(nodes)
    leaf_index = {node.node_id: k for k, node in enumerate(leaf_nodes)}
    h = np.stack([node.atom for node in leaf_nodes if node.atom is not None]).astype(np.float32)
    n_components = len(h)
    print(f"Tree has {n_components} leaves.", flush=True)

    # Joint NNLS of every tile (shift-only, unscaled) against all leaves.
    n_rows = dataset.count()
    w_path = output_dir / "w.f32.npy"
    w = np.lib.format.open_memmap(w_path, mode="w+", dtype=np.float32, shape=(n_rows, n_components))
    metadata_chunks = []
    offset = 0
    for rows, metadata in iter_embedding_batches(
        dataset,
        config.batch_size,
        shift,
        np.ones_like(scale),
        metadata_columns=("slide_id", "x", "y"),
        clip=config.method == "nmf",
    ):
        w_batch = nnls_batch(rows, h, config.factorization.nnls_iter)
        w[offset : offset + len(w_batch)] = w_batch
        metadata_chunks.append(metadata)
        offset += len(w_batch)
    w.flush()
    assert offset == n_rows

    pd.concat(metadata_chunks, ignore_index=True).to_parquet(
        output_dir / "w_metadata.parquet", index=False
    )
    pd.DataFrame(h).rename_axis("component").to_parquet(output_dir / "h.parquet")
    np.save(output_dir / "shift.npy", shift)
    tree_path = output_dir / "tree.parquet"
    nodes_table(nodes, leaf_index).to_parquet(tree_path, index=False)

    reasons = pd.Series([node.reason for node in nodes if node.status == "final"]).value_counts()
    manifest = {
        "w": {"path": str(w_path), "shape": list(w.shape), "dtype": str(w.dtype)},
        "method": config.method,
        "center": config.center,
        "split": config.split,
        "n_components": n_components,
        "n_nodes": len(nodes),
        "max_depth": max(node.depth for node in leaf_nodes),
        "rejected_splits": reasons.to_dict(),
        "n_sample_rows": len(half),
        "shift_mlflow_uri": config.shift.mlflow_uri,
        "tree": OmegaConf.to_container(config.tree),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    logger.log_artifact(str(output_dir / "h.parquet"))
    logger.log_artifact(str(tree_path))
    logger.log_artifact(str(manifest_path))
    logger.log_metrics({"n_components": n_components, "n_nodes": len(nodes)})


if __name__ == "__main__":
    ctx = ray.data.DataContext.get_current()
    ctx.enable_rich_progress_bars = True
    ctx.use_ray_tqdm = False

    # num_cpus=8: same conservative cap as nmf_fit.py (full embeddings corpus
    # read, twice). Keep in sync with cpu= in scripts/explainability/nmf_tree.py.
    with ray.init(num_cpus=8, runtime_env={"excludes": [".git", ".venv"]}):
        main()
