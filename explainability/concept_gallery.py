"""Q2 and Q3 - which concepts matter for which class, and which attract each model's attention.

Notation as in explainability/concepts.py (after arXiv:2609.34750). Example
tiles per concept: the `n_tiles` highest-v_k tiles, at most one per slide, read
from the WSIs exactly as preprocessing/embeddings.py embedded them.

**Q2 - concept importance per output class.** The tile-logit head through the
decoder, g_s o D, is affine, so the paper's insertion, occlusion and
gradient-x-input attributions coincide and are exact:

    a_kc = <d_k, Theta_c>                 per unit of concept k, effect on logit c
    phi_kc(x_i) = v_ik a_kc               tile attribution
    phi_kc(s) = vbar_sk a_kc              slide attribution, vbar_s = sum_i omega_i v_i
                                          with the model's real attention omega

and the slide logit decomposes additively, L_c = b'_c + sum_k phi_kc(s) + residual
(b' = Theta mu + b; the residual is the reconstruction error passed through
the head - `ate` is its RMSE, the paper's attribution error). Per output:

    separation_gt    E[phi_kc(s) | y = c] - E[phi_kc(s) | y != c] over slides,
                     y ground truth: how much of the logit gap between class-c
                     slides and the rest flows through concept k. The per-output
                     `logit_gap_gt` is the full gap these parts (plus residual)
                     add up to.
    separation_pred  the same with the model's own predicted label.
    importance       E|phi_kc(s)| - overall magnitude, whatever the class.

**Q3 - which concepts attract attention.** g_u o D is nonlinear (tanh), so
attributions differ by method; this uses occlusion,

    phi^u_k(x_i) = g_u(D(v_i)) - g_u(D(v_i - v_ik e_k))

(`attention_occlusion` = its mean over tiles), with the paper's additivity
error `add` = RMSE(g_u(D(v)), g_u(mu) + sum_k phi^u_k) saying how far attention
is from additive in the concepts. And, method-free,

    enrichment_k = sum_s vbar_sk / sum_s vtilde_sk

attention-pooled over uniformly pooled code (vtilde_s = mean_i v_i): > 1 means
the model looks preferentially at tiles carrying concept k. Also per
ground-truth class of each head.

Multiclass logits are centred across the head's classes. In-sample, like
concept_completeness.py.

The report opens with a Q1 panel (RE / FE / MCE / agreement / AUC) read from
`<w_dir>/concept_completeness/manifest.json` if concept_completeness.py has
been run on this dictionary - run it first; nothing is recomputed here.
"""

import base64
import html
import io
import json
from pathlib import Path
from typing import Any

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

from explainability.concepts import (
    SlideIndex,
    iter_tiles,
    load_autoencoder,
    r2_score,
    rmse,
)
from explainability.model import (
    ModelWeights,
    attention_from_preactivation,
    attention_preactivation,
    attention_scores,
    centred_classifier,
    class_mask,
    load_full_model,
    logits_to_prob,
    nancy_to_target,
    predicted_labels,
    tile_logits,
)
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


def attention_occlusion(
    v: np.ndarray, z_hat: np.ndarray, dictionary: np.ndarray, model: ModelWeights
) -> tuple[np.ndarray, np.ndarray]:
    """`(g_u(D(v)), phi^u)`: decoded attention score and its per-concept occlusion, `(n,)` / `(n, K)`.

    Removing concept k shifts the decoded embedding by -v_k d_k, i.e. the
    attention pre-activation by -v_k U d_k - precomputed once per concept.
    """
    pre = attention_preactivation(z_hat, model)
    u_hat = attention_from_preactivation(pre, model)
    u_d = dictionary @ model.attn_w1.T  # (K, hidden)
    occlusion = np.empty_like(v)
    for k in range(v.shape[1]):
        occlusion[:, k] = u_hat - attention_from_preactivation(pre - v[:, k, None] * u_d[k], model)
    return u_hat, occlusion


def conditional_difference(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """E[values | mask] - E[values | ~mask] over the first axis; NaN if either side is empty."""
    if mask.all() or not mask.any():
        return np.full(values.shape[1:], np.nan)
    return values[mask].mean(axis=0) - values[~mask].mean(axis=0)


def select_tiles(v: np.ndarray, keys: pd.DataFrame, n_tiles: int) -> pd.DataFrame:
    """Top-`n_tiles` tiles per concept by v_k, at most one per slide."""
    frames = []
    share = v / np.maximum(v.sum(axis=1, keepdims=True), 1e-12)
    for k in range(v.shape[1]):
        top = np.argsort(-v[:, k])
        frame = keys.iloc[top].assign(v=v[top, k], share=share[top, k])
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
    """Red for positive, blue for negative, intensity relative to `scale` (the column's max |value|)."""
    alpha = 0.0 if scale <= 0 or not np.isfinite(value) else min(abs(value) / scale, 1.0) * 0.6
    rgb = "214, 39, 40" if value > 0 else "31, 119, 180"
    return f"rgba({rgb}, {alpha:.3f})"


def render_completeness(manifest: dict[str, Any] | None) -> str:
    """Q1 panel from concept_completeness.py's manifest.json, or a note if it hasn't run."""
    if manifest is None:
        return (
            "<h2>Q1 - encoder / decoder quality</h2><p><i>concept_completeness.py has not been "
            "run for this dictionary - run it first to show RE / FE / MCE here.</i></p>"
        )

    def fmt(value: Any, spec: str = ".3f") -> str:
        return "-" if value is None or not np.isfinite(value) else format(value, spec)

    tile = {(r["head"], r["target"]): r for r in manifest["tile_fidelity"]}
    agreement = {(r["head"], r["surrogate"]): r["label_agreement"] for r in manifest["slide_agreement"]}
    output_rows = []
    for r in manifest["slide_fidelity"]:
        head, c = r["head"], r["class"]
        t = tile[(head, f"s_class{c}")]
        output_rows.append(
            f"<tr><th>{head}</th><th>{html.escape(CLASS_LABELS[head][c])}</th>"
            f"<td>{fmt(t['fe'])} <small>({fmt(t['r2_fe'], '.2f')})</small></td>"
            f"<td>{fmt(t['rho_fe'], '.2f')}</td>"
            f"<td>{fmt(t['r2_mce_ols'], '.2f')}</td>"
            f"<td>{fmt(r['logit_fe'])} <small>({fmt(r['logit_r2_fe'], '.2f')})</small></td>"
            f"<td>{fmt(r['logit_r2_mce_ols'], '.2f')}</td>"
            f"<td>{fmt(r['auc_real_vs_groundtruth'])}</td>"
            f"<td>{fmt(r['auc_A_vs_groundtruth'])}</td><td>{fmt(r['auc_eta_vs_groundtruth'])}</td>"
            f"<td>{fmt(r['auc_A_vs_predicted'])}</td><td>{fmt(r['prob_corr_A'], '.2f')}</td></tr>"
        )
    model_rows = []
    for head in HEADS:
        t = tile[(head, "u")]
        model_rows.append(
            f"<tr><th>{head}</th>"
            f"<td>{fmt(t['fe'])} <small>({fmt(t['r2_fe'], '.2f')})</small></td>"
            f"<td>{fmt(t['rho_fe'], '.2f')}</td><td>{fmt(t['r2_mce_ols'], '.2f')}</td>"
            f"<td>{fmt(agreement.get((head, 'A')), '.3f')}</td>"
            f"<td>{fmt(agreement.get((head, 'eta')), '.3f')}</td></tr>"
        )
    return f"""<h2>Q1 - encoder / decoder quality</h2>
<p>RE = {fmt(manifest['re'])} (relative {fmt(manifest['re_relative'])}) &middot;
K = {manifest['n_components']} &middot; {manifest['n_tiles']} tiles &middot;
{manifest['n_slides']} slides &middot; split {html.escape(str(manifest['split']))}.
FE = RMSE of the real model vs f<sub>A</sub> (the model on D(E(z))), r<sup>2</sup> = 1 - FE<sup>2</sup>/Var
in brackets; &rho; = FE / (Lipschitz &times; RE); MCE r<sup>2</sup>: best linear head on the
concepts (OLS, an upper estimate of the error). In-sample.</p>
<table>
<tr><th rowspan="2">head</th><th rowspan="2">output</th><th colspan="3">tile logit</th>
<th colspan="2">slide logit</th><th colspan="3">AUC vs ground truth</th>
<th colspan="2">f<sub>A</sub> vs model prediction</th></tr>
<tr><th>FE<sub>s</sub></th><th>&rho;</th><th>MCE r<sup>2</sup></th><th>FE<sub>L</sub></th>
<th>MCE r<sup>2</sup></th><th>real</th><th>f<sub>A</sub></th><th>OLS head</th>
<th>AUC</th><th>prob corr</th></tr>
{"".join(output_rows)}
</table>
<p></p>
<table>
<tr><th rowspan="2">model</th><th colspan="3">attention score</th>
<th colspan="2">slide label agreement with the model</th></tr>
<tr><th>FE<sub>u</sub></th><th>&rho;</th><th>MCE r<sup>2</sup></th><th>f<sub>A</sub></th>
<th>OLS head</th></tr>
{"".join(model_rows)}
</table>
<h2>Q2 / Q3 - concepts per class and attention</h2>
"""


def render_html(
    concept_class: pd.DataFrame,
    concept_attention: pd.DataFrame,
    slide_additivity: pd.DataFrame,
    attention_additivity: pd.DataFrame,
    selected: pd.DataFrame,
    images: list[bytes],
    title: str,
    completeness: dict[str, Any] | None = None,
) -> str:
    outputs = [(head, c, label) for head in HEADS for c, label in enumerate(CLASS_LABELS[head])]
    cc = concept_class.set_index(["concept", "head", "class"])
    sep_scale = concept_class.groupby(["head", "class"])["separation_gt"].apply(
        lambda x: float(np.nanmax(np.abs(x)))
    )
    att = concept_attention[concept_attention["class"] == -1].set_index(["concept", "head"])
    log_enrich = np.log(att["enrichment"].clip(lower=1e-6))
    enrich_scale = log_enrich.abs().groupby(level="head").max()
    additivity = slide_additivity.set_index(["head", "class"])
    attention_add = attention_additivity.set_index("head")

    group_row = (
        "".join(f'<th colspan="{len(CLASS_LABELS[h])}">{h}: separation</th>' for h in HEADS)
        + f'<th colspan="{len(HEADS)}">attention</th>'
    )
    output_row = "".join(
        f"<th>{html.escape(label)}<br><small>gap {additivity.at[(head, c), 'logit_gap_gt']:+.2f}"
        f"<br>ATE r<sup>2</sup> {additivity.at[(head, c), 'r2_concepts']:.2f}</small></th>"
        for head, c, label in outputs
    ) + "".join(
        f"<th>{head}<br><small>ADD {attention_add.at[head, 'add']:.3f}</small></th>" for head in HEADS
    )

    body = []
    for k in sorted(concept_class["concept"].unique()):
        imgs = "".join(
            f'<img src="data:image/png;base64,{base64.b64encode(images[i]).decode()}" '
            f'title="{html.escape(str(selected.at[i, "slide_id"]))} '
            f'({selected.at[i, "x"]}, {selected.at[i, "y"]}) '
            f'v={selected.at[i, "v"]:.3g} share={selected.at[i, "share"]:.2f}">'
            for i in selected.index[selected["concept"] == k]
        )
        cells = []
        for head, c, _ in outputs:
            row = cc.loc[(k, head, c)]
            cells.append(
                f'<td style="background:{cell_colour(row["separation_gt"], sep_scale[(head, c)])}">'
                f'<b>{row["separation_gt"]:+.3f}</b><br>'
                f'<small>pred {row["separation_pred"]:+.3f}<br>a {row["a"]:+.3f}</small></td>'
            )
        for head in HEADS:
            row = att.loc[(k, head)]
            cells.append(
                f'<td style="background:{cell_colour(log_enrich[(k, head)], enrich_scale[head])}">'
                f'<b>&times;{row["enrichment"]:.2f}</b><br>'
                f'<small>occl {row["attention_occlusion"]:+.3f}</small></td>'
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
{render_completeness(completeness)}
<p><b>Example tiles</b>: highest concept code v<sub>k</sub>, one per slide (hover: slide,
position, v<sub>k</sub>, share of the tile's total code).</p>
<p><b>Separation</b> (per output): how much of the logit gap between slides of that class and
the rest flows through the concept, using the exact slide attribution
&phi;<sub>kc</sub>(s) = v&#772;<sub>sk</sub> a<sub>kc</sub> with the model's real attention -
bold: ground-truth classes, <small>pred</small>: the model's own predicted classes,
<small>a</small>: per-unit effect &lang;d<sub>k</sub>, &Theta;<sub>c</sub>&rang;. Header: the full
gap and the r<sup>2</sup> of the concept decomposition (1 - ATE<sup>2</sup>/Var). Red = towards
the class, blue = away. Multiclass heads centred across classes.</p>
<p><b>Attention</b> (per model): enrichment &times; = attention-pooled / uniformly pooled concept
code (red &gt; 1: the model looks at tiles with this concept), <small>occl</small>: mean occlusion
attribution to the attention score. Header: additivity error of the occlusion attributions.</p>
<table>
<tr><th rowspan="2">concept</th><th rowspan="2">example tiles</th>{group_row}</tr>
<tr>{output_row}</tr>
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
    autoencoder = load_autoencoder(Path(config.w_dir), config.nnls_iter)
    split_dirs = resolve_embedding_split_dirs(
        config.sources, config.get("local_embeddings_dir"), split=config.split
    )

    key_chunks, v_chunks = [], []
    readouts: dict[str, dict[str, list[np.ndarray]]] = {
        head: {"s": [], "u": [], "u_hat": [], "occlusion": []} for head in HEADS
    }
    for keys, z in iter_tiles(load_embeddings_dataset(split_dirs), config.batch_size):
        v = autoencoder.encode(z)
        z_hat = autoencoder.decode(v)
        for head, model in models.items():
            u_hat, occlusion = attention_occlusion(v, z_hat, autoencoder.dictionary, model)
            readouts[head]["s"].append(tile_logits(z, model).astype(np.float32))
            readouts[head]["u"].append(attention_scores(z, model).astype(np.float32))
            readouts[head]["u_hat"].append(u_hat.astype(np.float32))
            readouts[head]["occlusion"].append(occlusion.astype(np.float32))
        key_chunks.append(keys)
        v_chunks.append(v)
    keys = pd.concat(key_chunks, ignore_index=True)
    v = np.concatenate(v_chunks)
    n_components = v.shape[1]

    slides = SlideIndex(keys["slide_id"])
    nancy_index = (
        load_embedding_slides(split_dirs).set_index("id").loc[slides.ids, "nancy_index"].to_numpy()
    )
    if pd.isna(nancy_index).any():
        raise ValueError(f"{int(pd.isna(nancy_index).sum())} slides lack nancy_index")
    v_uniform = slides.mean(v)

    class_rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []
    additivity_rows: list[dict[str, Any]] = []
    attention_additivity_rows: list[dict[str, Any]] = []
    for head, model in models.items():
        r = {name: np.concatenate(chunks) for name, chunks in readouts[head].items()}
        theta, b = centred_classifier(model)
        num_classes = theta.shape[0]
        a = autoencoder.dictionary @ theta.T  # (K, C)
        offset = theta @ autoencoder.mu + b  # b'
        v_bar = slides.attention_pool(v, r["u"])  # (S, K), real attention
        logits = slides.attention_pool(r["s"], r["u"])  # (S, C)
        phi = v_bar[:, :, None] * a[None]  # (S, K, C)
        concept_part = offset + phi.sum(axis=1)
        y_gt = nancy_to_target(nancy_index, head)
        y_pred = predicted_labels(logits_to_prob(logits, num_classes))

        for c in range(num_classes):
            gt = class_mask(y_gt, c, num_classes)
            pred = class_mask(y_pred, c, num_classes)
            gap = conditional_difference(logits[:, c, None], gt)[0]
            additivity_rows.append(
                {"head": head, "class": c, "ate": rmse(logits[:, c], concept_part[:, c]),
                 "r2_concepts": r2_score(logits[:, c], concept_part[:, c]), "logit_gap_gt": gap}
            )
            sep_gt = conditional_difference(phi[:, :, c], gt)
            sep_pred = conditional_difference(phi[:, :, c], pred)
            importance = np.abs(phi[:, :, c]).mean(axis=0)
            for k in range(n_components):
                class_rows.append(
                    {"concept": k, "head": head, "class": c, "a": a[k, c],
                     "separation_gt": sep_gt[k], "separation_pred": sep_pred[k],
                     "importance": importance[k]}
                )

        occlusion_mean = r["occlusion"].mean(axis=0)
        base = attention_scores(autoencoder.mu[None], model)[0]  # g_u(D(0))
        additive = base + r["occlusion"].sum(axis=1)
        attention_additivity_rows.append(
            {"head": head, "add": rmse(r["u_hat"], additive), "ate": rmse(r["u"], additive),
             "fe": rmse(r["u"], r["u_hat"])}
        )
        groups = [(-1, np.ones(len(slides), dtype=bool))] + [
            (c, class_mask(y_gt, c, num_classes)) for c in range(num_classes)
        ]
        for c, mask in groups:
            enrichment = v_bar[mask].sum(axis=0) / np.maximum(v_uniform[mask].sum(axis=0), 1e-12)
            for k in range(n_components):
                attention_rows.append(
                    {"concept": k, "head": head, "class": c, "n_slides": int(mask.sum()),
                     "enrichment": enrichment[k], "attention_occlusion": occlusion_mean[k]}
                )
        print(f"Done {head}.", flush=True)

    concept_class = pd.DataFrame(class_rows)
    concept_attention = pd.DataFrame(attention_rows)
    slide_additivity = pd.DataFrame(additivity_rows)
    attention_additivity = pd.DataFrame(attention_additivity_rows)
    print(slide_additivity.to_string(index=False), flush=True)
    print(attention_additivity.to_string(index=False), flush=True)

    selected = select_tiles(v, keys, config.n_tiles)
    tiles = read_tiles(selected, load_embedding_slides(split_dirs))
    images = [png_bytes(tile) for tile in tiles]
    for i, image in enumerate(images):
        name = f"concept{selected.at[i, 'concept']}_rank{selected.at[i, 'rank']}.png"
        (output_dir / "tiles" / name).write_bytes(image)

    title = f"Concept gallery - {Path(config.w_dir).name} ({config.split})"
    # Q1 numbers come from concept_completeness.py's own run on this
    # dictionary, if there is one - nothing is recomputed here.
    completeness_path = Path(config.w_dir) / "concept_completeness" / "manifest.json"
    completeness = (
        json.loads(completeness_path.read_text()) if completeness_path.exists() else None
    )
    if completeness is not None and completeness["split"] != config.split:
        print(f"WARNING: {completeness_path} is for split {completeness['split']}", flush=True)
    (output_dir / "gallery.html").write_text(
        render_html(concept_class, concept_attention, slide_additivity, attention_additivity,
                    selected, images, title, completeness)
    )
    concept_class.to_parquet(output_dir / "concept_class.parquet", index=False)
    concept_attention.to_parquet(output_dir / "concept_attention.parquet", index=False)
    slide_additivity.to_parquet(output_dir / "slide_additivity.parquet", index=False)
    attention_additivity.to_parquet(output_dir / "attention_additivity.parquet", index=False)
    selected.to_parquet(output_dir / "tiles.parquet", index=False)
    manifest = {
        "w_dir": config.w_dir,
        "split": config.split,
        "n_components": n_components,
        "n_tiles_per_concept": config.n_tiles,
        "n_slides": len(slides),
        "slide_additivity": slide_additivity.to_dict(orient="records"),
        "attention_additivity": attention_additivity.to_dict(orient="records"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    for name in ("gallery.html", "concept_class.parquet", "concept_attention.parquet",
                 "slide_additivity.parquet", "attention_additivity.parquet", "manifest.json"):
        logger.log_artifact(str(output_dir / name))


if __name__ == "__main__":
    ctx = ray.data.DataContext.get_current()
    ctx.enable_rich_progress_bars = True
    ctx.use_ray_tqdm = False

    # num_cpus=8: same conservative cap as concept_completeness.py (full
    # embeddings corpus read). Keep in sync with cpu= in
    # scripts/explainability/concept_gallery.py.
    with ray.init(num_cpus=8, runtime_env={"excludes": [".git", ".venv"]}):
        main()
