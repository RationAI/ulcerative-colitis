"""Step 2 of concept_mil.tex's order-of-work table: R^2 check of eq:tilelogit.

`sigma_table.py` (step 1) answers "*can* the concept basis express what
Theta_m reads?" (readout coverage, weights only, no data). This script
answers the companion question from the same Q4 table (Sec. 4): "*do* they,
on real tiles?" -

    R^2 of  varphi @ sigma   vs.   m @ Theta_m^T        (eq:tilelogit, tile level)

`varphi_ik = (1/P) sum_p w_ipk` is the tile-level concept weight (mean of a
tile's per-patch NMF activations); `varphi @ sigma` is exactly the m-driven
term of the tile logit approximation (eq:tilelogit) when routed through the
K-concept bottleneck. `m @ Theta_m^T` is the same term computed directly,
without ever going through H at all - the "ground truth" this script checks
the approximation against.

**Relation to sigma_table.py's coverage**: readout coverage is a strict upper
bound on what this R^2 can be (see the session note this script's own commit
message points back to) - `varphi @ sigma == m_parallel @ Theta_m^T` always
(m_parallel = m's projection onto span(H)), so R^2 here can only fall short of
that ceiling if a tile's own m doesn't vary along Theta_m's in-span component
the way varphi happens to, never exceed it. Coverage alone (step 1) is enough
to rule an H out early (free, no data); this step (real tiles, "minutes" per
concept_mil.tex's own order-of-work table) is what actually confirms it once
coverage looks promising enough to bother.

**Single streaming pass, driver-side aggregation, not `ray.data.groupby`**:
`iter_patch_batches` (from `explainability.nmf_fit`) is reused exactly as
`nmf_fit.py`'s own fit/transform passes use it - read a batch, transform it
against the fixed H under test (`model.transform`, the same call
`nmf_fit.py`'s own final pass and `concept_masks.py` use), then reduce that
batch's rows into a per-tile running sum, vectorized locally (`np.add.at`
keyed by `pd.factorize`). A tile's `m` and `varphi` need the same per-tile
grouping and are both derived from the same batch, so this does both in one
pass rather than two. Kept off `ray.data.groupby(...).aggregate(...)`
entirely: the number of distinct tiles in even the largest grade partition
(tens of thousands) comfortably fits an in-driver dict, so there's no need
for `ray.data`'s own distributed hash-shuffle aggregation here - which would
otherwise reintroduce `grade_split.mean_pool_patches`'s pyarrow
mixed-accumulator pitfall for no benefit.

**Uses raw (uncentered) sigma**, unlike `sigma_table.py`'s printed table:
eq:tilelogit's actual logit computation is `... + sum_k varphi_ik sigma_kc +
b_c`, and eq. 2.9/2.10 are both stated in terms of raw sigma - row-centering
(remark below eq 3.10) is only for interpreting sigma's *signs* elsewhere,
not part of the computation this script is checking.

**Session discipline** (see explainability-status memory): written and
validated only against small synthetic `ray.data` datasets plus
`ruff`/`mypy strict` - never run against the real corpus in this session,
even though this step is the cheapest of the three that touch real patch
data (concept_mil.tex's own order-of-work table puts it at "minutes" for
~5k tiles, now plausibly true for real given it reads grade_split.py's much
smaller per-file output rather than the original giant-row-group tables).
"""

import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import ray
import ray.data
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from sklearn.decomposition import MiniBatchNMF

from explainability.nmf_fit import (
    iter_patch_batches,
    load_shift,
    resolve_percentile_stats_path,
)
from explainability.sigma_table import load_h
from explainability.theta_m_check import load_theta_m
from explainability.tiles import load_tokens_dataset, resolve_grade_token_dir


def build_transform_model(
    h: np.ndarray, n_components: int, init: str, beta_loss: str, random_state: int, embed_dim: int
) -> MiniBatchNMF:
    """A `MiniBatchNMF` whose `components_` is a fixed, already-fitted `H` - ready for `.transform()`.

    `sklearn`'s `transform()` refuses to run before `fit`/`partial_fit` has
    set the estimator's internal fitted state (`n_components_` etc.), even
    though `transform()` itself only actually needs `components_` - so this
    does one throwaway `partial_fit` on a tiny random dummy batch purely to
    satisfy that check, then immediately overwrites `components_` with the
    real `h` before any real data is ever transformed. Confirmed directly
    (synthetic low-rank data, exact recovery) that the dummy batch's own
    content doesn't affect the resulting transform - only `components_` (and
    `beta_loss`/`init`, which govern the update rule itself and must match
    whatever `h` was actually fit with) does. Same pattern
    `explainability/nmf_fit.py`'s own final pass uses (`model.components_ =
    h` before `model.transform(...)`), just without the preceding real
    `partial_fit` training loop nmf_fit.py has and this script doesn't need.

    Args:
        h: The dictionary to check, shape (n_components, embed_dim) - already
            fully recovered (scale multiplied back in) and gauge-fixed, i.e.
            exactly `nmf_fit.py`'s own `h.parquet`.
        n_components: Must equal `h.shape[0]` (checked by the caller).
        init: Must match the `nmf_fit.py` run `h` came from - governs
            `transform()`'s own update rule despite `components_` being
            overwritten (see this function's docstring above).
        beta_loss: Same caveat as `init`.
        random_state: Same caveat as `init`, though weaker still - only seeds
            the throwaway dummy batch, with no effect on the actual result.
        embed_dim: Width of one patch token's `embedding`.

    Returns:
        A `MiniBatchNMF` ready for `.transform()` against `h`.
    """
    model = MiniBatchNMF(
        n_components=n_components, init=init, beta_loss=beta_loss, random_state=random_state
    )
    dummy = np.abs(
        np.random.default_rng(random_state).standard_normal((2, embed_dim))
    ).astype(np.float32)
    model.partial_fit(dummy)
    model.components_ = h
    return model


def mean_pool_tile_features(
    patches_ds: ray.data.Dataset,
    model: MiniBatchNMF,
    shift: np.ndarray,
    embed_dim: int,
    n_components: int,
    batch_size: int,
) -> pd.DataFrame:
    """One streaming pass: per-tile mean raw (shift-only) embedding `m` and mean NMF activation `varphi`.

    `varphi_ik = (1/P) sum_p w_ipk` (concept_mil.tex): the tile-level concept
    weight is the *mean* of a tile's per-patch NMF activations `w`, computed
    here by transforming each patch against `model` (already pointed at the
    fixed `H` under test - see `build_transform_model`) and mean-pooling the
    result by tile, in exactly the same pass and the same per-tile grouping
    as `m` (the mean-pooled *raw* shift-only patch embedding - i.e. what
    `grade_split.mean_pool_patches` computes, recomputed here rather than
    reused so both quantities come from one streaming pass instead of two).

    Uses `unscaled = ones(embed_dim)`: `h` is already in shift-only space
    (`nmf_fit.py`'s recovery step multiplies scale back into H before writing
    `h.parquet`), so both `m` and the patches fed to `model.transform` must
    be shift-only too, matching `nmf_fit.py`'s own final transform pass.

    Args:
        patches_ds: Patch token dataset for one grade partition (see
            `explainability.tiles.resolve_grade_token_dir`).
        model: `MiniBatchNMF` with `components_` fixed to the `H` under test
            (see `build_transform_model`).
        shift: Per-dimension shift constant `c`, shape (embed_dim,).
        embed_dim: Width of one patch token's `embedding`.
        n_components: `model`'s `n_components` (`h.shape[0]`).
        batch_size: Patches read per batch (see `iter_patch_batches`).

    Returns:
        One row per tile: `slide_id`, `x`, `y`, `n_patches`, `m` (list of
        `embed_dim` floats), `varphi` (list of `n_components` floats).
    """
    unscaled = np.ones(embed_dim, dtype=np.float32)
    sum_m: dict[tuple[str, int, int], np.ndarray] = {}
    sum_w: dict[tuple[str, int, int], np.ndarray] = {}
    counts: dict[tuple[str, int, int], int] = {}

    for patches, metadata in iter_patch_batches(
        patches_ds, batch_size, shift, unscaled, with_metadata=True
    ):
        w = model.transform(patches)
        assert metadata is not None

        keys = pd.Series(
            list(zip(metadata["slide_id"], metadata["x"].tolist(), metadata["y"].tolist(), strict=True)),
            dtype=object,
        )
        codes, uniques = pd.factorize(keys)
        n_groups = len(uniques)

        batch_sum_m = np.zeros((n_groups, embed_dim))
        batch_sum_w = np.zeros((n_groups, n_components))
        np.add.at(batch_sum_m, codes, patches)
        np.add.at(batch_sum_w, codes, w)
        batch_count = np.bincount(codes, minlength=n_groups)

        for i, key in enumerate(uniques):
            if key not in counts:
                sum_m[key] = np.zeros(embed_dim, dtype=np.float64)
                sum_w[key] = np.zeros(n_components, dtype=np.float64)
                counts[key] = 0
            sum_m[key] += batch_sum_m[i]
            sum_w[key] += batch_sum_w[i]
            counts[key] += int(batch_count[i])

    rows = []
    for key, count in counts.items():
        slide_id, x, y = key
        rows.append(
            {
                "slide_id": slide_id,
                "x": x,
                "y": y,
                "n_patches": count,
                "m": (sum_m[key] / count).tolist(),
                "varphi": (sum_w[key] / count).tolist(),
            }
        )
    return pd.DataFrame(rows)


def compute_r2(tile_df: pd.DataFrame, sigma: np.ndarray, theta_m: np.ndarray) -> pd.DataFrame:
    """R^2 (and Pearson r) of the concept-routed vs. direct m-driven logit contribution, per class.

    predicted_ic = varphi_i @ sigma[:,c]        (goes through the K-concept bottleneck)
    true_ic      = m_i @ Theta_m[c,:]^T         (direct, full embed_dim - "ground truth")

    Args:
        tile_df: Output of `mean_pool_tile_features` (`m`, `varphi` columns).
        sigma: `H @ Theta_m^T` (eq. 2.9), shape (n_components, num_classes).
        theta_m: Shape (num_classes, embed_dim).

    Returns:
        One row per class: `class`, `r2`, `pearson_r`, `n_tiles`. `r2`/
        `pearson_r` are NaN for a class whose true values are constant across
        every tile (zero variance - R^2/Pearson r are undefined there, not
        just numerically unstable).
    """
    m = np.stack(tile_df["m"].to_numpy())
    phi = np.stack(tile_df["varphi"].to_numpy())

    true = m @ theta_m.T
    pred = phi @ sigma

    rows = []
    for c in range(theta_m.shape[0]):
        true_c, pred_c = true[:, c], pred[:, c]
        ss_tot = float(np.sum((true_c - true_c.mean()) ** 2))
        if ss_tot > 0:
            ss_res = float(np.sum((true_c - pred_c) ** 2))
            r2 = 1.0 - ss_res / ss_tot
            pearson_r = float(np.corrcoef(true_c, pred_c)[0, 1])
        else:
            r2 = float("nan")
            pearson_r = float("nan")
        rows.append({"class": c, "r2": r2, "pearson_r": pearson_r, "n_tiles": len(tile_df)})
    return pd.DataFrame(rows)


@with_cli_args(["+explainability=tile_r2_check"])
@hydra.main(config_path="../configs", config_name="explainability", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_path = resolve_percentile_stats_path(config.shift.mlflow_uri)
    shift = load_shift(stats_path, config.shift.percentile_column)

    h = load_h(config.h.mlflow_uri)
    if h.shape[0] != config.n_components:
        raise ValueError(
            f"h.parquet at {config.h.mlflow_uri} has {h.shape[0]} rows, "
            f"expected n_components={config.n_components}"
        )

    theta_ms = {
        name: load_theta_m(checkpoint_cfg.checkpoint, config.embed_dim)
        for name, checkpoint_cfg in config.checkpoints.items()
    }

    token_dir = resolve_grade_token_dir(
        config.get("local_grade_split_dir"), config.grade_split.mlflow_uri, kind="patch", grade=config.grade
    )
    patches_ds = load_tokens_dataset([token_dir])

    model = build_transform_model(
        h, config.n_components, config.nmf.init, config.nmf.beta_loss, config.nmf.random_state,
        config.embed_dim,
    )

    print(
        f"Streaming grade={config.grade} patch tokens once: mean-pooling raw "
        "embeddings (m) and NMF activations (varphi) by tile...",
        flush=True,
    )
    tile_df = mean_pool_tile_features(
        patches_ds, model, shift, config.embed_dim, config.n_components, config.nmf.batch_size
    )
    print(f"Pooled {len(tile_df)} tiles.", flush=True)

    r2_frames = []
    for head, theta_m in theta_ms.items():
        sigma = h @ theta_m.T  # eq. 2.9, raw (uncentered) - see module docstring
        r2_df = compute_r2(tile_df, sigma, theta_m)
        r2_df.insert(0, "head", head)
        r2_frames.append(r2_df)
        print(f"=== {head} ===", flush=True)
        print(r2_df.to_string(index=False), flush=True)

    r2_all = pd.concat(r2_frames, ignore_index=True)
    print("\n=== pooled R^2 across heads/classes ===", flush=True)
    print(f"mean R^2 = {r2_all['r2'].mean():.4f}  min R^2 = {r2_all['r2'].min():.4f}", flush=True)

    tile_path = output_dir / "tile_features.parquet"
    r2_path = output_dir / "r2.parquet"
    tile_df.to_parquet(tile_path, index=False)
    r2_all.to_parquet(r2_path, index=False)

    manifest = {
        "grade": config.grade,
        "n_components": config.n_components,
        "n_tiles": len(tile_df),
        "r2_by_head_class": r2_all.to_dict(orient="records"),
        "mean_r2": float(r2_all["r2"].mean()),
        "min_r2": float(r2_all["r2"].min()),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # tile_features.parquet stays project-mount-only (m is embed_dim=1280-wide
    # per tile - same large-artifact treatment as nmf_fit.py's w.f32.npy /
    # w_metadata.parquet, not small like r2.parquet/manifest.json).
    logger.log_artifact(str(r2_path))
    logger.log_artifact(str(manifest_path))
    for _, row in r2_all.iterrows():
        tag = f"{row['head']}_class{row['class']}"
        logger.log_metrics({f"r2/{tag}": row["r2"], f"pearson_r/{tag}": row["pearson_r"]})


if __name__ == "__main__":
    ctx = ray.data.DataContext.get_current()
    ctx.enable_rich_progress_bars = True
    ctx.use_ray_tqdm = False

    # num_cpus set the same conservative way as explainability/nmf_fit.py's
    # own grade-split read (see that file's ray.init() comment): the OOM root
    # cause this originally guarded against (giant single-row-group parquet
    # files) applies to the *old* embeddings_xai token tables, not
    # grade_split.py's much smaller per-file output this script reads - kept
    # anyway pending an actual re-benchmark. Keep in sync with cpu= in
    # scripts/explainability/tile_r2_check.py.
    with ray.init(num_cpus=8, runtime_env={"excludes": [".git", ".venv"]}):
        main()
