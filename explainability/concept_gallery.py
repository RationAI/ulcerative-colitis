"""Concept gallery: example tiles per concept plus each concept's coefficient for every output class.

Modelled on Table 1 of "Explaining Digital Pathology Models via Clustering
Activations" (arXiv:2511.14558), which regresses the model's slide-level
prediction on per-slide concept features and reports one coefficient per
concept. Our MIL model makes that regression principled: the slide logit is
`L_c = sum_i a_i s_ic` with attention `a = softmax(u)`, and the tile logit is
linear in the concept weights (`s_i ~= phi_i @ sigma + const`), so

    L_c ~= Phi_s @ sigma_c + const,     Phi_s = sum_i a_i phi_i

- the attention-pooled concept weights of slide s. Per output class this
script reports:

    beta   OLS coefficient (with intercept) of the real slide logit on Phi_s,
           fit across slides - the paper's coefficient, using the real
           model's own attention for pooling. `r2` of each regression says
           how much of that output's slide-level variation the concepts
           explain.
    sigma  closed-form per-concept tile-logit contribution, `H @ Theta^T` -
           what beta would be if the reconstruction were exact.

For the multiclass heads both are centred across the head's classes (softmax
ignores a shared offset, so only differences between classes carry meaning);
the single neutrophils logit is left as is. Columns, per
`explainability.model.nancy_to_target`: neutrophils; nancy_low NHI 0 / 1 /
>=2; nancy_high NHI <2 / 2 / 3 / 4.

Example tiles: the `n_tiles` highest-phi_k tiles of each concept, at most one
per slide (so one slide can't fill a whole row), read from the WSIs with the
same `read_slide_tiles` call preprocessing/embeddings.py embedded them with.

Outputs: `gallery.html` (single file, images embedded), `coefficients.parquet`
(long format), `slide_regression.parquet` (r2 per output), `tiles.parquet`
(which tiles were shown) and one PNG per shown tile under `tiles/`.
"""

import base64
import html
import io
import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import pyarrow as pa
import ray
import ray.data
from omegaconf import DictConfig
from PIL import Image
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.tiling.read_slide_tiles import read_slide_tiles
from ray.data.expressions import col
from scipy.special import softmax

from explainability.concept_surrogate_check import (
    align,
    compute_readouts,
    load_dictionary,
    r2_score,
)
from explainability.model import ModelWeights, load_full_model
from explainability.tiles import (
    load_embedding_slides,
    load_embeddings_dataset,
    resolve_embedding_split_dirs,
)


HEADS = ("neutrophils", "nancy_low", "nancy_high")
CLASS_LABELS = {
    "neutrophils": ["neutrophils"],
    "nancy_low": ["NHI 0", "NHI 1", "NHI ≥2"],
    "nancy_high": ["NHI <2", "NHI 2", "NHI 3", "NHI 4"],
}


def centre_classes(x: np.ndarray) -> np.ndarray:
    """Subtract the mean across classes (last axis) for multiclass heads; single-logit heads unchanged."""
    return x - x.mean(axis=-1, keepdims=True) if x.shape[-1] > 1 else x


def attention_pool(
    values: np.ndarray, u: np.ndarray, slide_codes: np.ndarray, n_slides: int
) -> np.ndarray:
    """Per slide, `softmax(u) @ values` over its tiles, shape `(n_slides, values.shape[1])`."""
    order = np.argsort(slide_codes, kind="stable")
    bounds = np.searchsorted(slide_codes[order], np.arange(n_slides + 1))
    pooled = np.empty((n_slides, values.shape[1]))
    for slide in range(n_slides):
        rows = order[bounds[slide] : bounds[slide + 1]]
        pooled[slide] = softmax(u[rows]) @ values[rows]
    return pooled


def slide_regression(phi_pooled: np.ndarray, logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS of each slide logit column on `[Phi, 1]` -> (coefficients `(K, C)`, r2 per column)."""
    design = np.hstack([phi_pooled, np.ones((len(phi_pooled), 1))])
    beta, *_ = np.linalg.lstsq(design, logits, rcond=None)
    pred = design @ beta
    r2 = np.array([r2_score(logits[:, c], pred[:, c]) for c in range(logits.shape[1])])
    return beta[:-1], r2


def select_tiles(
    phi: np.ndarray, keys: pd.DataFrame, n_tiles: int
) -> pd.DataFrame:
    """Top-`n_tiles` tiles per concept by phi_k, at most one per slide."""
    frames = []
    share = phi / np.maximum(phi.sum(axis=1, keepdims=True), 1e-12)
    for k in range(phi.shape[1]):
        top = np.argsort(-phi[:, k])
        frame = keys.iloc[top].assign(phi=phi[top, k], share=share[top, k])
        frame = frame.drop_duplicates("slide_id").head(n_tiles)
        frames.append(frame.assign(concept=k, rank=np.arange(len(frame))))
    return pd.concat(frames, ignore_index=True)


def read_tiles(selected: pd.DataFrame, slides: pd.DataFrame) -> list[np.ndarray]:
    """Read the selected tiles' RGB pixels from their WSIs, in `selected`'s row order."""
    info = slides.set_index("id")[["path", "level", "tile_extent_x", "tile_extent_y"]]
    enriched = selected.join(info, on="slide_id").assign(_order=np.arange(len(selected)))
    # from_arrow, not from_pandas: same as preprocessing/embeddings.py - a
    # pandas block hands the string `path` column to the UDF as list<string>.
    table = pa.Table.from_pandas(
        enriched[["_order", "path", "x", "y", "tile_extent_x", "tile_extent_y", "level"]],
        preserve_index=False,
    )
    ds = ray.data.from_arrow(table).with_column(
        "tile",
        read_slide_tiles(  # pyright: ignore[reportCallIssue]
            col("path"),
            col("x"),
            col("y"),
            col("tile_extent_x"),
            col("tile_extent_y"),
            col("level"),
        ),
    )
    result = ds.select_columns(["_order", "tile"]).to_pandas().sort_values("_order")
    return [np.asarray(tile)[..., :3].astype(np.uint8) for tile in result["tile"]]


def png_bytes(tile: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(tile).save(buffer, format="PNG")
    return buffer.getvalue()


def cell_colour(value: float, scale: float) -> str:
    """Red for positive, blue for negative, intensity relative to the column's max |value|."""
    alpha = 0.0 if scale <= 0 or not np.isfinite(value) else min(abs(value) / scale, 1.0) * 0.6
    rgb = "214, 39, 40" if value > 0 else "31, 119, 180"
    return f"rgba({rgb}, {alpha:.3f})"


def render_html(
    coefficients: pd.DataFrame,
    regression: pd.DataFrame,
    selected: pd.DataFrame,
    images: list[bytes],
    title: str,
) -> str:
    columns = [(head, c, label) for head in HEADS for c, label in enumerate(CLASS_LABELS[head])]
    beta_scale = {
        (head, c): coefficients.query("head == @head and `class` == @c")["beta"].abs().max()
        for head, c, _ in columns
    }
    r2 = regression.set_index(["head", "class"])["r2"]

    head_row = "".join(
        f'<th colspan="{len(CLASS_LABELS[h])}">{h}</th>' for h in HEADS
    )
    class_row = "".join(
        f"<th>{html.escape(label)}<br><small>r<sup>2</sup>={r2[(head, c)]:.2f}</small></th>"
        for head, c, label in columns
    )
    body = []
    for k in sorted(coefficients["concept"].unique()):
        concept_tiles = selected.index[selected["concept"] == k]
        imgs = "".join(
            f'<img src="data:image/png;base64,{base64.b64encode(images[i]).decode()}" '
            f'title="{html.escape(str(selected.at[i, "slide_id"]))} '
            f'({selected.at[i, "x"]}, {selected.at[i, "y"]}) '
            f'phi={selected.at[i, "phi"]:.3g} share={selected.at[i, "share"]:.2f}">'
            for i in concept_tiles
        )
        cells = []
        for head, c, _ in columns:
            row = coefficients.query("concept == @k and head == @head and `class` == @c").iloc[0]
            cells.append(
                f'<td style="background:{cell_colour(row["beta"], beta_scale[(head, c)])}">'
                f'<b>{row["beta"]:+.3f}</b><br><small>&sigma; {row["sigma"]:+.3f}</small></td>'
            )
        body.append(f'<tr><th>{k}</th><td class="tiles">{imgs}</td>{"".join(cells)}</tr>')

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 16px; background: #fff; color: #111; }}
table {{ border-collapse: collapse; }}
th, td {{ border: 1px solid #ccc; padding: 4px 6px; text-align: center; vertical-align: middle; }}
td.tiles {{ text-align: left; white-space: nowrap; }}
td.tiles img {{ width: 96px; height: 96px; margin: 1px; }}
small {{ color: #555; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p>Per concept: the highest-&phi; tiles (one per slide; hover for slide, position, &phi; and its
share of the tile's total &phi;). Per output class: <b>&beta;</b>, the OLS coefficient of the real slide
logit on attention-pooled concept weights (r<sup>2</sup> per column in the header), and <small>&sigma;</small>,
the closed-form tile-logit contribution H&nbsp;&Theta;<sup>T</sup>. Multiclass heads are centred across classes.
Red = pushes towards the class, blue = away.</p>
<table>
<tr><th rowspan="2">concept</th><th rowspan="2">example tiles</th>{head_row}</tr>
<tr>{class_row}</tr>
{"".join(body)}
</table></body></html>
"""


@with_cli_args(["+explainability=concept_gallery"])
@hydra.main(config_path="../configs", config_name="explainability", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    output_dir = Path(config.output_dir)
    (output_dir / "tiles").mkdir(parents=True, exist_ok=True)

    models: dict[str, ModelWeights] = {
        head: load_full_model(config.checkpoints[head].checkpoint, config.embed_dim)
        for head in HEADS
    }
    split_dirs = resolve_embedding_split_dirs(
        config.sources, config.get("local_embeddings_dir"), split=config.split
    )
    readout_keys, s_real, u_real = compute_readouts(
        load_embeddings_dataset(split_dirs), models, config.batch_size
    )
    w_keys, w, h, _shift = load_dictionary(Path(config.w_dir))
    r_idx, w_idx = align(readout_keys, w_keys)
    phi = w[w_idx]
    keys = readout_keys.iloc[r_idx].reset_index(drop=True)
    slide_codes, slide_ids = pd.factorize(keys["slide_id"])
    print(f"Aligned {len(keys)} tiles / {len(slide_ids)} slides, K={h.shape[0]}", flush=True)

    coefficient_rows = []
    regression_rows = []
    for head, model in models.items():
        u = u_real[head][r_idx]
        slide_logits = centre_classes(
            attention_pool(s_real[head][r_idx], u, slide_codes, len(slide_ids))
        )
        phi_pooled = attention_pool(phi, u, slide_codes, len(slide_ids))
        beta, r2 = slide_regression(phi_pooled, slide_logits)
        sigma = centre_classes(h @ model.cls_w.T)
        for c in range(model.cls_w.shape[0]):
            regression_rows.append({"head": head, "class": c, "r2": r2[c]})
            for k in range(h.shape[0]):
                coefficient_rows.append(
                    {"concept": k, "head": head, "class": c, "beta": beta[k, c], "sigma": sigma[k, c]}
                )
    coefficients = pd.DataFrame(coefficient_rows)
    regression = pd.DataFrame(regression_rows)
    print(regression.to_string(index=False), flush=True)

    selected = select_tiles(phi, keys, config.n_tiles)
    tiles = read_tiles(selected, load_embedding_slides(split_dirs))
    images = [png_bytes(tile) for tile in tiles]
    for i, image in enumerate(images):
        (output_dir / "tiles" / f"concept{selected.at[i, 'concept']}_rank{selected.at[i, 'rank']}.png").write_bytes(image)

    title = f"Concept gallery - {Path(config.w_dir).name} ({config.split})"
    gallery_path = output_dir / "gallery.html"
    gallery_path.write_text(render_html(coefficients, regression, selected, images, title))

    coefficients.to_parquet(output_dir / "coefficients.parquet", index=False)
    regression.to_parquet(output_dir / "slide_regression.parquet", index=False)
    selected.to_parquet(output_dir / "tiles.parquet", index=False)
    manifest = {
        "w_dir": config.w_dir,
        "split": config.split,
        "n_components": int(h.shape[0]),
        "n_tiles_per_concept": config.n_tiles,
        "n_slides": len(slide_ids),
        "slide_regression_r2": regression.to_dict(orient="records"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    for name in ("gallery.html", "coefficients.parquet", "slide_regression.parquet", "manifest.json"):
        logger.log_artifact(str(output_dir / name))
    for row in regression.to_dict(orient="records"):
        logger.log_metrics({f"slide_regression_r2/{row['head']}_class{row['class']}": row["r2"]})


if __name__ == "__main__":
    ctx = ray.data.DataContext.get_current()
    ctx.enable_rich_progress_bars = True
    ctx.use_ray_tqdm = False

    # num_cpus=8: same conservative cap as concept_surrogate_check.py (full
    # embeddings corpus read). Keep in sync with cpu= in
    # scripts/explainability/concept_gallery.py.
    with ray.init(num_cpus=8, runtime_env={"excludes": [".git", ".venv"]}):
        main()
