"""Pathologist's view of a whole nmf_tree.py tree: every node with example tiles, for labelling.

Internal nodes are scaffolding for attribution (only leaves are concepts), but
naming top-down - coarse groups first, then their subtypes - is how a
pathologist reads tissue, and marking a split as "don't split" (children not
meaningfully different) is direct expert feedback on the tree's stopping rule.

A node's example tiles: the highest summed weight over its descendant leaves,
`sum_{l below n} v_il` (for a leaf, just its own v_il), at most one per slide -
so the view needs only tree.parquet plus the tiles' concept codes, and works
for trees fit before node atoms were saved.

The page is a single HTML file (tiles embedded as JPEG): nested, collapsible
nodes, a label and comment field per node and a "don't split" box per
internal node. Entries are kept in the browser's local storage while
editing and exported with "Download labels (CSV)".
"""

import base64
import html
import io

import numpy as np
import pandas as pd
from PIL import Image


def descendant_components(tree: pd.DataFrame) -> dict[int, list[int]]:
    """node_id -> dictionary components (leaf indices) of every leaf at or below it."""
    children: dict[int, list[int]] = {}
    for node_id, parent in zip(tree["node_id"], tree["parent"], strict=True):
        if pd.notna(parent):
            children.setdefault(int(parent), []).append(int(node_id))
    component = {
        int(n): int(c) for n, c in zip(tree["node_id"], tree["component"], strict=True) if pd.notna(c)
    }

    def collect(node_id: int) -> list[int]:
        if node_id in component:
            return [component[node_id]]
        return [c for child in children.get(node_id, []) for c in collect(child)]

    return {int(node_id): collect(int(node_id)) for node_id in tree["node_id"]}


def select_node_tiles(
    v: np.ndarray, keys: pd.DataFrame, tree: pd.DataFrame, n_tiles: int
) -> pd.DataFrame:
    """Top-`n_tiles` tiles per non-root node by summed descendant-leaf code, one per slide."""
    frames = []
    for node_id, components in descendant_components(tree).items():
        if node_id == 0 or not components:
            continue
        score = v[:, components].sum(axis=1)
        top = np.argsort(-score)
        frame = keys.iloc[top].assign(score=score[top]).drop_duplicates("slide_id").head(n_tiles)
        frames.append(frame.assign(node_id=node_id, rank=np.arange(len(frame))))
    return pd.concat(frames, ignore_index=True)


def jpeg_base64(tile: np.ndarray, quality: int = 90) -> str:
    buffer = io.BytesIO()
    Image.fromarray(tile).save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode()


SCRIPT = """
const KEY = "tree-labels:" + document.title;
function rows() {
  return [...document.querySelectorAll(".node")].map(n => ({
    node_id: n.dataset.node,
    label: n.querySelector(".label").value,
    dont_split: n.querySelector(".nosplit") ? n.querySelector(".nosplit").checked : "",
    comment: n.querySelector(".comment").value,
  }));
}
function save() { try { localStorage.setItem(KEY, JSON.stringify(rows())); } catch (e) {} }
function restore() {
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(KEY) || "null"); } catch (e) {}
  if (!saved) return;
  for (const r of saved) {
    const n = document.querySelector('.node[data-node="' + r.node_id + '"]');
    if (!n) continue;
    n.querySelector(".label").value = r.label;
    n.querySelector(".comment").value = r.comment;
    if (n.querySelector(".nosplit")) n.querySelector(".nosplit").checked = r.dont_split === true;
  }
}
function download() {
  const q = s => '"' + String(s).replace(/"/g, '""') + '"';
  const lines = ["node_id,label,dont_split,comment"].concat(
    rows().map(r => [r.node_id, q(r.label), r.dont_split, q(r.comment)].join(",")));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lines.join("\\n")], {type: "text/csv"}));
  a.download = "tree_labels.csv";
  a.click();
}
document.addEventListener("input", save);
document.addEventListener("DOMContentLoaded", restore);
"""


def render_tree_html(
    tree: pd.DataFrame, node_tiles: pd.DataFrame, images: list[str], title: str
) -> str:
    """Nested, collapsible tree; `images[i]` is the base64 JPEG of `node_tiles` row i."""
    info = tree.set_index("node_id")
    children: dict[int, list[int]] = {}
    for node_id, parent in zip(tree["node_id"], tree["parent"], strict=True):
        if pd.notna(parent):
            children.setdefault(int(parent), []).append(int(node_id))
    tiles_of = node_tiles.groupby("node_id").indices

    def render(node_id: int) -> str:
        row = info.loc[node_id]
        is_leaf = node_id not in children
        kind = f"leaf &middot; concept {int(row['component'])}" if is_leaf else "internal"
        gain = row.get("gain", float("nan"))
        details = f"{row['n_rows']} sample tiles ({row['mass']:.1%})"
        if pd.notna(gain) and not is_leaf:
            details += f" &middot; split explains {gain:.1%} held out"
        imgs = "".join(
            f'<img src="data:image/jpeg;base64,{images[i]}" '
            f'title="{html.escape(str(node_tiles.at[i, "slide_id"]))} '
            f'({node_tiles.at[i, "x"]}, {node_tiles.at[i, "y"]})">'
            for i in tiles_of.get(node_id, [])
        )
        nosplit = (
            "" if is_leaf else '<label><input type="checkbox" class="nosplit"> don\'t split</label>'
        )
        inner = "".join(render(child) for child in children.get(node_id, []))
        return (
            f'<details open class="node" data-node="{node_id}"><summary><b>node {node_id}</b> '
            f"&middot; depth {int(row['depth'])} &middot; {kind} <small>{details}</small></summary>"
            f'<div class="tiles">{imgs}</div>'
            f'<div class="form"><input class="label" placeholder="label"> {nosplit} '
            f'<input class="comment" placeholder="comment"></div>{inner}</details>'
        )

    body = "".join(render(child) for child in children.get(0, []))
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 16px; background: #fff; color: #111; }}
details {{ border-left: 2px solid #ccc; margin: 6px 0 6px 12px; padding-left: 10px; }}
summary {{ cursor: pointer; padding: 2px 0; }}
.tiles img {{ width: 112px; height: 112px; margin: 1px; }}
.form input.label {{ width: 260px; }} .form input.comment {{ width: 420px; }}
small {{ color: #555; }}
button {{ margin: 8px 0; padding: 6px 12px; }}
</style><script>{SCRIPT}</script></head><body>
<h1>{html.escape(title)}</h1>
<p>Each node shows the tiles that most strongly carry it (one per slide; hover for the slide and
position). Internal nodes group everything below them. Please give each node a label, top-down,
and tick <i>don't split</i> where the two children are not meaningfully different. Entries are
kept in this browser while you work; use the button to export them.</p>
<button onclick="download()">Download labels (CSV)</button>
{body}
<button onclick="download()">Download labels (CSV)</button>
</body></html>
"""
