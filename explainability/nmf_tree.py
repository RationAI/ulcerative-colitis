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
- Best-first: the leaf with the highest priority is tried next. `varsigma`
  is the classifier response stacked over heads - 8 entries: the
  neutrophils logit, plus the nancy_low (3) and nancy_high (4) logits each
  centred (softmax is invariant to a shared offset, so only the centred part
  means anything). Two priorities (`tree.priority`):
    variance  priority(n) = (1/N) sum_{i in n} ||varsigma_i - mean_n varsigma||^2
              with varsigma_i each sample tile's own response (N = sample
              size) - the model-relevant variation sitting inside the node.
              Tiles the model reads in opposite directions add up here
              instead of cancelling, and the sum (not mean) keeps mass in.
              Unsupervised: the same with each tile's fit-space embedding.
    atom      priority(n) = mass(n) * ||varsigma(a_n)|| * err(n) (the
              original): mass = fraction of sample rows, err = relative
              residual of the node's rows against its own atom. Its relevance
              looks at the node's averaged atom, so children the model reads
              in opposite directions cancel and the most productive splits
              rank low (19 of 30 splits in the first centred tree had
              opposing children). Unsupervised: relevance = 1.
- A split is accepted only if it passes both tests, so the tree decides its
  own size - no K to choose:
    stability  re-fit the split separately on two slide-disjoint halves of
               the node's rows; the cosine of the best-matched atom pair must
               be >= `stability_threshold` (else it's sampling noise).
               Halves are by *slide*, never by tile - tiles of one slide are
               too correlated to count as independent evidence.
    min_gain   the split must explain at least `min_gain` of the
               model-relevant variation, measured *held out*: the split fit
               on one half assigns the other half's tiles to children (NNLS,
               argmax) and the gain there is the between-children sum of
               squares of varsigma_i over the half's total sum of squares;
               averaged over both directions. Mass is built in - a tiny node
               can't gain much - and held-out evaluation keeps deep splits
               from memorising single slides.
  A rejected node is final. Growth stops when every leaf is rejected
  (converged); `max_leaves` is only a safety cap.
- `supervised: false` is the unsupervised baseline tree, to check that
  steering it with the classifier isn't circular: priority and gain use each
  tile's fit-space embedding instead of its head responses.

**Inference**: the leaf atoms (recovered to the original embedding space,
unit-norm) are stacked into one dictionary H, and every tile's weights are
the NNLS solution against all leaves jointly - so H is a single global
linear decoder exactly like a flat fit (internal nodes carry no attribution)
and the outputs (`w.f32.npy`, `w_metadata.parquet`, `h.parquet`,
`shift.npy`) plug straight into concept_completeness.py and concept_gallery.py. `tree.parquet`
logs every node, including every rejected candidate split's stability, gain
and varsigma difference, to calibrate the thresholds; `node_atoms.parquet`
keeps every node's atom (internal ones too) for the pathologist's tree view.
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
    gain: float = float("nan")
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


def tile_responses(
    x: np.ndarray, scale: np.ndarray, models: dict[str, ModelWeights], chunk_size: int = 65536
) -> np.ndarray:
    """varsigma_i for every fit-space row: head responses of x_i * scale, shape `(n, 8)`.

    The fit-space offset (mu) is a constant shift of every varsigma_i, so it
    cancels in any within-node spread and is left out.
    """
    parts = []
    for start in range(0, len(x), chunk_size):
        z = x[start : start + chunk_size] * scale[None, :]
        heads = []
        for model in models.values():
            logits = z @ model.cls_w.T
            if logits.shape[1] > 1:
                logits = logits - logits.mean(axis=1, keepdims=True)
            heads.append(logits)
        parts.append(np.concatenate(heads, axis=1))
    return np.concatenate(parts)


def node_spread(values: np.ndarray, n_total: int) -> float:
    """(1/N) sum_i ||values_i - mean(values)||^2 over the node's rows."""
    return float(np.sum((values - values.mean(axis=0)) ** 2)) / n_total


def embedding_spread(
    x: np.ndarray, rows: np.ndarray, sq_norms: np.ndarray, n_total: int, chunk_size: int = 65536
) -> float:
    """node_spread of the fit-space rows themselves, via sum ||x||^2 - ||sum x||^2 / n.

    Chunked so no full copy of a large node's (n, 2560) block is ever made -
    the unsupervised counterpart of the response spread.
    """
    total = np.zeros(x.shape[1])
    for start in range(0, len(rows), chunk_size):
        total += x[rows[start : start + chunk_size]].sum(axis=0, dtype=np.float64)
    return (float(sq_norms[rows].sum()) - float(total @ total) / len(rows)) / n_total


def between_ss(values: np.ndarray, labels: np.ndarray) -> float:
    """Between-children sum of squares: sum_c n_c ||mean_c - mean||^2."""
    mean = values.mean(axis=0)
    total = 0.0
    for c in np.unique(labels):
        group = values[labels == c]
        total += len(group) * float(np.sum((group.mean(axis=0) - mean) ** 2))
    return total


def assign_rows(x: np.ndarray, atoms_fit: np.ndarray, config: DictConfig) -> np.ndarray:
    """Child labels for rows a split wasn't fit on: NNLS against its two atoms, then argmax."""
    w = nnls_batch(
        x, atoms_fit, config.factorization.nnls_iter, tol=config.factorization.get("tol")
    )
    return np.argmax(w, axis=1)


def relative_error(x: np.ndarray, sq_norms: np.ndarray, atom_fit: np.ndarray) -> float:
    """1 - ||projection onto the (unit) atom||^2 / ||X||^2, with non-negative weights."""
    proj = np.maximum(x @ atom_fit, 0.0)
    total = float(sq_norms.sum())
    return 1.0 - float(np.sum(proj * proj)) / total if total > 0 else 0.0


def split_rows(x: np.ndarray, config: DictConfig) -> tuple[np.ndarray, np.ndarray]:
    """Rank-2 factorization -> (hard child labels per row, the two unit atoms in fit space)."""
    w, h, iterations, final_change = factorize(
        x,
        2,
        config.method,
        config.tree.random_state,
        config.factorization.n_iter,
        config.factorization.nnls_iter,
        tol=config.factorization.get("tol"),
    )
    print(
        f"  rank-2 fit on {len(x)} rows: {iterations} iterations, "
        f"final relative H change {final_change:.2e}",
        flush=True,
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
    if tree.priority not in ("variance", "atom"):
        raise ValueError(f"tree.priority must be 'variance' or 'atom', got {tree.priority!r}")
    sq_norms = np.einsum("ij,ij->i", x, x)
    # Per-tile head responses drive the variance priority and the gain
    # (supervised); the unsupervised tree uses the fit-space rows themselves.
    responses = tile_responses(x, scale, models) if tree.supervised else None
    half_total = [
        (
            node_spread(responses[half == side], 1)
            if responses is not None
            else embedding_spread(x, np.flatnonzero(half == side), sq_norms, 1)
        )
        for side in range(2)
    ]
    nodes = [Node(node_id=0, parent=None, depth=0, rows=np.arange(len(x)))]
    heap: list[tuple[float, int]] = [(-float("inf"), 0)]
    n_leaves = 1

    while heap:
        if n_leaves >= tree.max_leaves:
            print(f"Stopped at the max_leaves safety cap ({tree.max_leaves}).", flush=True)
            break
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
        half_masks = [half[node.rows] == side for side in range(2)]
        for side_rows in half_masks:
            if side_rows.sum() < 2 * tree.min_node_rows:
                break
            half_atoms.append(split_rows(x_node[side_rows], config)[1])
        if len(half_atoms) < 2:
            node.status, node.reason = "final", "half_too_small"
            continue
        node.stability = matched_cosine(half_atoms[0], half_atoms[1])

        # Held-out gain: the split fit on one half, scored on the other.
        gains = []
        for fit_side, eval_side in ((0, 1), (1, 0)):
            eval_x = x_node[half_masks[eval_side]]
            held_out = assign_rows(eval_x, half_atoms[fit_side], config)
            values = (
                responses[node.rows[half_masks[eval_side]]] if responses is not None else eval_x
            )
            gains.append(between_ss(values, held_out) / half_total[eval_side])
        node.gain = float(np.mean(gains))
        del x_node

        atoms = unit_rows(atoms_fit * scale[None, :])
        varsigmas = [head_response(atom, models) for atom in atoms]
        node.varsigma_diff = float(np.linalg.norm(varsigmas[0] - varsigmas[1]))

        if node.stability < tree.stability_threshold:
            node.status, node.reason = "final", "unstable"
            continue
        if node.gain < tree.min_gain:
            node.status, node.reason = "final", "low_gain"
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
            if tree.priority == "variance":
                child.priority = (
                    node_spread(responses[child.rows], len(x))
                    if responses is not None
                    else embedding_spread(x, child.rows, sq_norms, len(x))
                )
            else:
                relevance = float(np.linalg.norm(varsigmas[j])) if tree.supervised else 1.0
                child.priority = child.mass * relevance * child.err
            nodes.append(child)
            heapq.heappush(heap, (-child.priority, child.node_id))
        print(
            f"  split accepted: stability={node.stability:.3f}, gain={node.gain:.4f}, "
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
            "gain": node.gain,
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
    # Every node's atom (embedding space, unit norm) - internal nodes are
    # scaffolding for attribution but context for the pathologist's labels.
    with_atom = [node for node in nodes if node.atom is not None]
    pd.DataFrame(
        np.stack([node.atom for node in with_atom if node.atom is not None]),
        index=pd.Index([node.node_id for node in with_atom], name="node_id"),
    ).to_parquet(output_dir / "node_atoms.parquet")

    reasons = pd.Series([node.reason for node in nodes if node.status == "final"]).value_counts()
    manifest = {
        "w": {"path": str(w_path), "shape": list(w.shape), "dtype": str(w.dtype)},
        "method": config.method,
        "center": config.center,
        "split": config.split,
        "n_components": n_components,
        "n_nodes": len(nodes),
        "stopped_by": "max_leaves" if n_components >= config.tree.max_leaves else "converged",
        "max_depth": max(node.depth for node in leaf_nodes),
        "rejected_splits": reasons.to_dict(),
        "n_sample_rows": len(half),
        "shift_mlflow_uri": config.shift.mlflow_uri,
        "tree": OmegaConf.to_container(config.tree),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    logger.log_artifact(str(output_dir / "h.parquet"))
    # Small (one value per dimension) - logged so the dictionary is complete in
    # mlflow: concept_completeness.py / concept_gallery.py read only from there.
    logger.log_artifact(str(output_dir / "shift.npy"))
    logger.log_artifact(str(tree_path))
    logger.log_artifact(str(output_dir / "node_atoms.parquet"))
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
