"""The concept autoencoder A = (E, D) of a fitted dictionary, and slide-level pooling.

Notation follows "A Unifying Framework of Concept-based Explainable AI with
Completeness Guarantees" (arXiv:2609.34750), adapted to the attention-MIL model:

    z_i = h(x_i)                 tile embedding (frozen Virchow2), in Z = R^2560
    v_i = E(z_i)                 concept code in V = R^K_+: NNLS of (z_i - mu)
                                 against the dictionary rows d_k (clipped at 0
                                 first for method=nmf, matching nmf_fit.py)
    D(v) = mu + sum_k v_k d_k    decoder; mu is nmf_fit.py's saved shift
    g_s(z) = Theta z + b         tile-logit head (centred for multiclass heads)
    g_u(z) = q^T tanh(U z + b1) + b2   attention head
    omega = softmax_i(g_u(z_i))  attention weights within a slide
    L(Z) = sum_i omega_i g_s(z_i)      slide logit; f = sigmoid / softmax of L
    f_A                          the same model with every z_i replaced by D(E(z_i))
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import ray.data
from scipy.special import softmax

from explainability.factorization import nnls_batch


KEYS = ["slide_id", "x", "y"]


@dataclass
class ConceptAutoencoder:
    dictionary: np.ndarray  # (K, D), rows d_k
    mu: np.ndarray  # (D,)
    clip: bool  # NMF's non-negativity clip before encoding
    nnls_iter: int

    def encode(self, z: np.ndarray) -> np.ndarray:
        """V = E(z), shape `(n, K)`."""
        x = z - self.mu
        if self.clip:
            x = np.maximum(x, 0.0)
        return nnls_batch(x, self.dictionary, self.nnls_iter)

    def decode(self, v: np.ndarray) -> np.ndarray:
        """D(v) = mu + v @ dictionary, shape `(n, D)`."""
        return self.mu + v @ self.dictionary


def load_autoencoder(w_dir: Path, nnls_iter: int) -> ConceptAutoencoder:
    """The (E, D) pair of one nmf_fit.py / nmf_tree.py run, from its output dir."""
    h = pd.read_parquet(w_dir / "h.parquet").sort_index(axis=0).sort_index(axis=1)
    method = json.loads((w_dir / "manifest.json").read_text())["method"]
    return ConceptAutoencoder(
        dictionary=h.to_numpy(dtype=np.float32),
        mu=np.load(w_dir / "shift.npy").astype(np.float32),
        clip=method == "nmf",
        nnls_iter=nnls_iter,
    )


def iter_tiles(
    dataset: ray.data.Dataset, batch_size: int
) -> Iterator[tuple[pd.DataFrame, np.ndarray]]:
    """Yield `(keys, z)` batches - (slide_id, x, y) and float32 embeddings."""
    n_rows = 0
    for batch in dataset.select_columns([*KEYS, "embedding"]).iter_batches(
        batch_size=batch_size, batch_format="numpy"
    ):
        z = np.stack(batch["embedding"]).astype(np.float32, copy=False)
        n_rows += len(z)
        print(f"iter_tiles: {n_rows} tiles", flush=True)
        yield pd.DataFrame({key: batch[key] for key in KEYS}), z


class SlideIndex:
    """Groups tile rows by slide once, for repeated per-slide pooling."""

    def __init__(self, slide_ids: pd.Series) -> None:
        self.codes, self.ids = pd.factorize(slide_ids)
        self.order = np.argsort(self.codes, kind="stable")
        self.bounds = np.searchsorted(self.codes[self.order], np.arange(len(self.ids) + 1))

    def __len__(self) -> int:
        return len(self.ids)

    def rows(self, slide: int) -> np.ndarray:
        return self.order[self.bounds[slide] : self.bounds[slide + 1]]

    def attention_pool(self, values: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Per slide, `softmax(scores) @ values`, shape `(n_slides, values.shape[1])`."""
        pooled = np.empty((len(self), values.shape[1]))
        for slide in range(len(self)):
            rows = self.rows(slide)
            pooled[slide] = softmax(scores[rows]) @ values[rows]
        return pooled

    def mean(self, values: np.ndarray) -> np.ndarray:
        """Per slide, the uniform mean of `values` over its tiles."""
        pooled = np.empty((len(self), values.shape[1]))
        for slide in range(len(self)):
            pooled[slide] = values[self.rows(slide)].mean(axis=0)
        return pooled


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def r2_score(true: np.ndarray, pred: np.ndarray) -> float:
    """1 - MSE / Var(true); NaN when `true` has zero variance."""
    var = float(np.var(true))
    return 1.0 - float(np.mean((true - pred) ** 2)) / var if var > 0 else float("nan")


def ols_predict(v: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Best linear head on the concepts: OLS of `target` on `[v, 1]`, in-sample predictions."""
    design = np.hstack([v, np.ones((len(v), 1), dtype=v.dtype)])
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    return design @ beta
