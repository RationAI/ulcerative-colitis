"""In-memory matrix-factorization building blocks shared by nmf_fit.py and nmf_tree.py.

Both methods factor rows as `X ~= W @ H` with `W >= 0`:

    nmf       H >= 0 too - needs non-negative X (shifted/clipped embeddings).
    semi_nmf  H signed (Ding, Li & Jordan 2010) - works on the raw, signed
              embeddings directly, no shift needed.

Every "W given H" step - semi-NMF's own W-update, and the final transform of
every tile against a fixed dictionary, for either method - is the same
non-negative least squares problem, solved here by `nnls_batch`, vectorized
across rows.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF


METHODS = ("nmf", "semi_nmf")


def nnls_batch(
    x: np.ndarray, h: np.ndarray, n_iter: int, w0: np.ndarray | None = None
) -> np.ndarray:
    """Solve `min_W ||X - W H||_F^2  s.t. W >= 0` for every row of X at once.

    Accelerated projected gradient (FISTA) with step 1/L, L the largest
    eigenvalue of `H H^T` - each iteration is only `(n, K) @ (K, K)`, so
    hundreds of iterations are cheap at small K. Starts from `w0` if given,
    else from the clipped unconstrained least-squares solution.

    Args:
        x: Rows to fit, shape `(n, d)`.
        h: Fixed dictionary, shape `(k, d)` - any sign.
        n_iter: Number of FISTA iterations.
        w0: Optional warm start, shape `(n, k)`.

    Returns:
        W, shape `(n, k)`, non-negative.
    """
    gram = h @ h.T
    xh = x @ h.T
    lipschitz = float(np.linalg.eigvalsh(gram).max())
    if lipschitz <= 0:
        return np.zeros_like(xh)
    if w0 is None:
        w0 = np.maximum(np.linalg.lstsq(gram, xh.T, rcond=None)[0].T, 0.0)
    w = w0.astype(xh.dtype, copy=True)
    y = w.copy()
    t = 1.0
    for _ in range(n_iter):
        w_next = np.maximum(y - (y @ gram - xh) / lipschitz, 0.0)
        t_next = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = w_next + ((t - 1.0) / t_next) * (w_next - w)
        w, t = w_next, t_next
    return w


def kmeans_init(x: np.ndarray, k: int, random_state: int, max_rows: int = 20_000) -> np.ndarray:
    """Initial dictionary from k-means centroids on (at most `max_rows` of) `x`.

    Ding et al.'s recommended semi-NMF initialization - semi-NMF is a relaxed
    soft k-means, so k-means centroids are already a good fixed point to
    start from.
    """
    rng = np.random.default_rng(random_state)
    sample = x if len(x) <= max_rows else x[rng.choice(len(x), max_rows, replace=False)]
    kmeans = KMeans(n_clusters=k, n_init=1, max_iter=100, random_state=random_state)
    return kmeans.fit(sample).cluster_centers_.astype(x.dtype)


def update_h(w: np.ndarray, x: np.ndarray, ridge: float) -> np.ndarray:
    """Semi-NMF's H-step: unconstrained least squares `H = (W^T W + ridge I)^-1 W^T X`."""
    gram = w.T @ w + ridge * np.eye(w.shape[1], dtype=w.dtype)
    return np.linalg.solve(gram, w.T @ x)


def reseed_dead_components(
    h: np.ndarray, usage: np.ndarray, x: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Replace dictionary rows no sample uses (`usage == 0`) with random rows of `x`."""
    dead = np.flatnonzero(usage <= 0)
    if len(dead):
        h = h.copy()
        h[dead] = x[rng.choice(len(x), len(dead), replace=False)]
    return h


def semi_nmf(
    x: np.ndarray, k: int, n_iter: int, nnls_iter: int, random_state: int, ridge: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """In-memory semi-NMF `X ~= W H`, `W >= 0`, H signed - alternating NNLS / least squares.

    Returns:
        `(W, H)` with shapes `(n, k)` and `(k, d)`.
    """
    rng = np.random.default_rng(random_state)
    h = kmeans_init(x, k, random_state)
    w = nnls_batch(x, h, nnls_iter)
    for _ in range(n_iter):
        h = reseed_dead_components(update_h(w, x, ridge), w.sum(axis=0), x, rng)
        w = nnls_batch(x, h, nnls_iter, w0=w)
    return w, h


def factorize(
    x: np.ndarray,
    k: int,
    method: str,
    random_state: int,
    n_iter: int,
    nnls_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    """In-memory rank-k factorization of `x` with the chosen `method`.

    `nmf` requires `x >= 0` (callers pass shifted/clipped embeddings).

    Returns:
        `(W, H)` with shapes `(n, k)` and `(k, d)`.
    """
    if method == "nmf":
        model = NMF(n_components=k, init="nndsvda", max_iter=n_iter, random_state=random_state)
        w = model.fit_transform(x)
        return w.astype(x.dtype), model.components_.astype(x.dtype)
    if method == "semi_nmf":
        return semi_nmf(x, k, n_iter, nnls_iter, random_state)
    raise ValueError(f"Unknown method {method!r}, expected one of {METHODS}")
