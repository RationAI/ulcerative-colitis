"""Pathologist's view of a concept dictionary: example tiles and one number per Nancy grade.

**Effect on the final grade.** Each concept is pushed both ways on every tile of
a slide, with the rest of the code left alone, and the whole concept model
(attention, heads and the `ensemble_probs` soft vote) is recomputed:

    d_i = step_sds * sd_k * sum_l v_il      sd_k: SD over slides of concept k's mean share
    add       P_g(v_ik + d_i) - P_g(v_ik)
    subtract  P_g(v_ik) - P_g(max(v_ik - d_i, 0))
    effect    (add + subtract) / 2, mean over the slides the real model gave grade g

Pushing in both directions keeps the sign, and it doesn't assume a slide can
always take more or less of a concept: subtraction stops at zero. Where the two
arms' means disagree in sign the cell is flagged, as is usual near an effect of
zero. `same_direction` is the share of those slides moving as the mean does.
Values are percentage points of P_g. Saturated grades move little however
important the concept is (nearly every NHI 4 slide sits at P_4 close to 1).

**Presence.** The concept's mean share over slides of grade g, relative to all
slides. Model-free: it says where the concept occurs, not what it does.

The page is one HTML file with the tiles embedded. It opens with the numbers
hidden, so concepts can be named from the tiles alone, and with concepts whose
largest effect is under `minor_points` folded away at the end.
"""

import html
from collections.abc import Callable

import numpy as np
import pandas as pd

from explainability.concepts import SlideIndex
from explainability.model import (
    ModelWeights,
    attention_from_preactivation,
    centred_classifier,
    ensemble_probs,
    logits_to_prob,
)
from explainability.tree_view import TILE_CSS, code_share, tile_link


N_GRADES = 5


def pushed_grade_probs(
    delta: np.ndarray,
    atom: np.ndarray,
    decoded: dict[str, tuple[np.ndarray, np.ndarray]],
    models: dict[str, ModelWeights],
    slides: SlideIndex,
) -> np.ndarray:
    """Slide grade scores with each tile moved by `delta` along `atom`, shape `(n_slides, 5)`.

    `decoded` maps a head to its tile logits and attention pre-activations on
    the unpushed D(v). Logits and pre-activations are affine in the code, so
    the push is added to them instead of decoding every tile again.
    """
    probs = {}
    for head, model in models.items():
        s_hat, pre_hat = decoded[head]
        theta, _ = centred_classifier(model)
        s = s_hat + delta[:, None] * (atom @ theta.T)
        u = attention_from_preactivation(
            pre_hat + delta[:, None] * (atom @ model.attn_w1.T), model
        )
        probs[head] = logits_to_prob(slides.attention_pool(s, u), model.cls_w.shape[0])
    return ensemble_probs(probs)


def concept_effects(
    v: np.ndarray,
    atoms: np.ndarray,
    decoded: dict[str, tuple[np.ndarray, np.ndarray]],
    models: dict[str, ModelWeights],
    slides: SlideIndex,
    predicted: np.ndarray,
    step_sds: float,
) -> pd.DataFrame:
    """One row per concept and grade the model predicted, see the module docstring."""
    total = v.sum(axis=1)
    slide_share = slides.mean(code_share(v))
    spread = slide_share.std(axis=0, ddof=1)
    base = pushed_grade_probs(np.zeros_like(total), atoms[0], decoded, models, slides)

    rows = []
    for k in range(v.shape[1]):
        step = step_sds * spread[k] * total
        added = pushed_grade_probs(step, atoms[k], decoded, models, slides) - base
        clipped = np.minimum(step, v[:, k])
        removed = base - pushed_grade_probs(-clipped, atoms[k], decoded, models, slides)
        for g in range(N_GRADES):
            mask = predicted == g
            if not mask.any():
                continue
            central = (added[mask, g] + removed[mask, g]) / 2
            rows.append(
                {
                    "concept": k,
                    "grade": g,
                    "n_slides": int(mask.sum()),
                    "add": 100 * added[mask, g].mean(),
                    "subtract": 100 * removed[mask, g].mean(),
                    "effect": 100 * central.mean(),
                    "same_direction": float(
                        (np.sign(central) == np.sign(central.mean())).mean()
                    ),
                    "presence": slide_share[mask, k].mean()
                    / max(slide_share[:, k].mean(), 1e-12),
                    "clipped_tiles": float((v[:, k] < step).mean()),
                }
            )
    return pd.DataFrame(rows)


def cell_colour(value: float, scale: float) -> str:
    """Red for positive, blue for negative, intensity relative to `scale` (the column's max |value|)."""
    alpha = (
        0.0
        if scale <= 0 or not np.isfinite(value)
        else min(abs(value) / scale, 1.0) * 0.6
    )
    rgb = "214, 39, 40" if value > 0 else "31, 119, 180"
    return f"rgba({rgb}, {alpha:.3f})"


def concept_rows(
    concepts: list[int],
    effects: pd.DataFrame,
    selected: pd.DataFrame,
    images: list[str],
) -> str:
    """Two table rows per concept: effects, then presence, beside the concept's tiles."""
    table = effects.set_index(["concept", "grade"])
    effect_scale = float(effects["effect"].abs().max())
    presence_scale = float(np.log(effects["presence"].clip(lower=1e-6)).abs().max())

    def cells(k: int, render: Callable[[pd.Series], str]) -> str:
        out = []
        for g in range(N_GRADES):
            out.append(
                render(table.loc[(k, g)])
                if (k, g) in table.index
                else '<td class="num">-</td>'
            )
        return "".join(out)

    def effect_cell(row: pd.Series) -> str:
        unreliable = np.sign(row["add"]) != np.sign(row["subtract"])
        mark = "<sup>&dagger;</sup>" if unreliable else ""
        return (
            f'<td class="num" style="background:{cell_colour(row["effect"], effect_scale)}">'
            f"<b>{row['effect']:+.1f}</b>{mark}<br><small>{row['same_direction']:.0%}</small></td>"
        )

    def presence_cell(row: pd.Series) -> str:
        colour = cell_colour(float(np.log(max(row["presence"], 1e-6))), presence_scale)
        return f'<td class="num" style="background:{colour}">&times;{row["presence"]:.2f}</td>'

    rows = []
    for k in concepts:
        tiles = "".join(
            tile_link(f"data:image/jpeg;base64,{images[i]}", selected.loc[i])
            for i in selected.index[selected["concept"] == k]
        )
        rows.append(
            f'<tr><th rowspan="2">{k}</th><td class="tiles" rowspan="2">{tiles}</td>'
            f'<th class="num label">effect</th>{cells(k, effect_cell)}</tr>'
            f'<tr><th class="num label">presence</th>{cells(k, presence_cell)}</tr>'
        )
    return "".join(rows)


SCRIPT = """
function toggleNumbers(button) {
  const hidden = document.body.classList.toggle("blind");
  button.textContent = hidden ? "Show numbers" : "Hide numbers";
}
"""


def render_pathologist_html(
    effects: pd.DataFrame,
    selected: pd.DataFrame,
    images: list[str],
    title: str,
    minor_points: float,
    step_sds: float,
) -> str:
    """The page: `selected` is a `with_slide_links` table, `images` its tiles as base64 JPEG."""
    largest = effects.groupby("concept")["effect"].apply(lambda x: float(x.abs().max()))
    active = [int(k) for k in largest.index[largest >= minor_points]]
    minor = [int(k) for k in largest.index[largest < minor_points]]
    n_slides = effects.groupby("grade")["n_slides"].first()
    grades = "".join(
        f'<th class="num">NHI {g}<br><small>{n_slides.get(g, 0)} slides</small></th>'
        for g in range(N_GRADES)
    )
    head = (
        f'<tr><th>concept</th><th>example tiles</th><th class="num"></th>{grades}</tr>'
    )
    minor_table = (
        f"<details><summary>Minor concepts ({len(minor)}): each changes the predicted grade "
        f"by under {minor_points:g} points</summary><table>{head}"
        f"{concept_rows(minor, effects, selected, images)}</table></details>"
        if minor
        else ""
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 16px; background: #fff; color: #111; }}
table {{ border-collapse: collapse; }}
th, td {{ border: 1px solid #ccc; padding: 4px 6px; text-align: center; vertical-align: middle; }}
td.tiles {{ text-align: left; min-width: 660px; }}
th.label {{ font-weight: normal; font-size: 12px; color: #555; }}
{TILE_CSS.format(size=160)}
small {{ color: #555; }}
button {{ margin: 8px 0; padding: 6px 12px; }}
details {{ margin-top: 16px; }} summary {{ cursor: pointer; margin-bottom: 8px; }}
body.blind .num {{ display: none; }}
</style><script>{SCRIPT}</script></head><body class="blind">
<h1>{html.escape(title)}</h1>
<p>Each row is one concept the model uses, shown by the tiles that carry it most (one per slide;
click a tile to open it in xOpat). The numbers start hidden so that each concept can be named from
its tiles alone.</p>
<button onclick="toggleNumbers(this)">Show numbers</button>
<div class="num">
<p><b>Effect</b>: how the probability of a grade changes, in percentage points, among slides the
model gave that grade, when the concept is made slightly stronger or weaker on every tile
({step_sds:g} of its spread between slides). Red: the concept supports the grade, blue: it counts
against it. Below it, the share of those slides that move this way. <sup>&dagger;</sup>Adding and
removing the concept disagree on the direction, so treat the number as zero.</p>
<p><b>Presence</b>: how much of the concept slides of that grade contain, relative to all slides
(&times;2.00 is twice as much). It says where the concept occurs, not what the model does with it.</p>
<p>Numbers describe the model, not the ground truth, and they are small for grades the model is
already very sure about, above all NHI 4.</p>
</div>
<table>{head}{concept_rows(active, effects, selected, images)}</table>
{minor_table}
</body></html>
"""
