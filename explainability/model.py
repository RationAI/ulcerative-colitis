"""The trained MIL heads' weights and output conventions, read straight from their checkpoints."""

from dataclasses import dataclass

import mlflow.artifacts
import numpy as np
import torch
from scipy.special import expit, softmax


@dataclass
class ModelWeights:
    """One head's full MIL model, pulled straight from its checkpoint's `state_dict`.

    `attn_w1`/`attn_b1` is `attention.0` (`U`), `attn_w2`/`attn_b2` is
    `attention.2` (`q`) - confirmed present in the same checkpoint as
    `classifier.weight`/`.bias` by directly inspecting a downloaded
    checkpoint's `state_dict` keys, so no external model class is needed to
    reconstruct `u_i = q^T tanh(U h_i + b1) + b2` and `s_i = Theta h_i + b`.
    """

    attn_w1: np.ndarray  # (hidden, 2*embed_dim)
    attn_b1: np.ndarray  # (hidden,)
    attn_w2: np.ndarray  # (1, hidden)
    attn_b2: np.ndarray  # (1,)
    cls_w: np.ndarray  # (num_classes, 2*embed_dim)
    cls_b: np.ndarray  # (num_classes,)


def load_full_model(checkpoint_uri: str, embed_dim: int) -> ModelWeights:
    """Download a MIL checkpoint and pull out both the attention module and the classifier."""
    checkpoint_path = mlflow.artifacts.download_artifacts(checkpoint_uri)
    # weights_only=False: lightning checkpoints bundle optimizer/callback
    # state alongside the tensors, so a strict weights-only unpickle can't
    # load them - trusted source (our own mlflow-logged checkpoints).
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["state_dict"]
    weight = state_dict["classifier.weight"].numpy()
    if weight.shape[1] != 2 * embed_dim:
        raise ValueError(
            f"classifier.weight has {weight.shape[1]} input columns, expected {2 * embed_dim}"
        )
    return ModelWeights(
        attn_w1=state_dict["attention.0.weight"].numpy(),
        attn_b1=state_dict["attention.0.bias"].numpy(),
        attn_w2=state_dict["attention.2.weight"].numpy(),
        attn_b2=state_dict["attention.2.bias"].numpy(),
        cls_w=weight,
        cls_b=state_dict["classifier.bias"].numpy(),
    )


def nancy_to_target(nancy_index: np.ndarray, head: str) -> np.ndarray:
    """Map ground-truth `nancy_index` (0-4) to one head's class index.

    Class orders follow the slide-grading routing rule (`pred_ensembling` on
    the `thesis` branch):
        neutrophils (binary):  nancy_index >= 2
        nancy_low   (3-class): {0->0, 1->1, >=2->2}
        nancy_high  (4-class): {<2->0, 2->1, 3->2, 4->3}
    **Unverified assumption**: neutrophils has no separate ground-truth
    column in this repo - its positive class is inferred as `nancy_index >= 2`
    (neutrophils first appear at Nancy grade 2 per the clinical criteria).
    Revisit this first if neutrophils AUCs look implausible.
    """
    if head == "neutrophils":
        return (nancy_index >= 2).astype(np.int64)
    if head == "nancy_low":
        return np.minimum(nancy_index, 2)
    if head == "nancy_high":
        return np.where(nancy_index < 2, 0, nancy_index - 1)
    raise ValueError(f"Unknown head: {head}")


def logits_to_prob(logits: np.ndarray, num_classes: int) -> np.ndarray:
    """Sigmoid for a single-logit (binary) head, softmax otherwise."""
    return expit(logits) if num_classes == 1 else softmax(logits, axis=-1)


def centred_classifier(model: ModelWeights) -> tuple[np.ndarray, np.ndarray]:
    """The tile-logit head `g_s` as `(Theta, b)`, centred across classes for multiclass heads.

    Softmax ignores a shared offset, so for multiclass heads only logits
    relative to the head's mean carry meaning - centring Theta's rows (and b)
    makes every logit, attribution and error computed from them invariant to
    that offset. The single neutrophils logit is left as is.
    """
    theta, b = model.cls_w, model.cls_b
    if len(b) > 1:
        theta, b = theta - theta.mean(axis=0), b - b.mean()
    return theta, b


def tile_logits(z: np.ndarray, model: ModelWeights) -> np.ndarray:
    """g_s(z) = Theta z + b, centred across classes for multiclass heads; shape `(n, num_classes)`."""
    theta, b = centred_classifier(model)
    return z @ theta.T + b


def attention_preactivation(z: np.ndarray, model: ModelWeights) -> np.ndarray:
    """U z + b1, shape `(n, hidden)` - split out so occlusion can perturb it cheaply."""
    return z @ model.attn_w1.T + model.attn_b1


def attention_from_preactivation(pre: np.ndarray, model: ModelWeights) -> np.ndarray:
    """q^T tanh(pre) + b2, shape `pre.shape[:-1]`."""
    return (np.tanh(pre) @ model.attn_w2.T + model.attn_b2)[..., 0]


def attention_scores(z: np.ndarray, model: ModelWeights) -> np.ndarray:
    """g_u(z) = q^T tanh(U z + b1) + b2, shape `(n,)`."""
    return attention_from_preactivation(attention_preactivation(z, model), model)


def predicted_labels(probs: np.ndarray) -> np.ndarray:
    """Hard labels from slide probabilities: threshold 0.5 for one output, argmax otherwise."""
    return (probs[:, 0] >= 0.5).astype(np.int64) if probs.shape[1] == 1 else probs.argmax(axis=1)


def class_mask(labels: np.ndarray, c: int, num_classes: int) -> np.ndarray:
    """Slides 'of output c': label == c, or the positive label for a single-output head."""
    return labels == (1 if num_classes == 1 else c)


def output_metric(
    z: np.ndarray, models: dict[str, ModelWeights], attention_weight: float, ridge: float
) -> np.ndarray:
    """T with ||T e||^2 the heads' first-order tile-output error for an embedding error e.

    The heads read a tile only through their outputs - every (centred) tile
    logit and the attention score of each head. Per output o, an error e in
    the embedding moves it by about grad o(z) . e, so with
    M = sum_o E_z[grad o grad o^T] / Var(o) (expectations and variances over
    the sample tiles `z`, raw embeddings), e^T M e is the expected squared
    output error with every output in units of its own spread - a first-order
    sum of (1 - R^2) over the outputs. Logits are linear, so their gradients
    are Theta's rows; attention scores are not, so their gradients
    U^T (q * tanh'(U z + b1)) are averaged over the tiles. Attention terms are
    multiplied by `attention_weight`.

    M is rescaled to mean eigenvalue 1 and `ridge` * I is added, so directions
    no head reads still cost a little and the atoms stay tied to how tiles
    actually look (and T stays invertible). Returns T = Lambda^1/2 V^T, so
    T^T T = M; a dictionary fit on T (z - mu) minimises this metric.
    """
    z = z.astype(np.float64)
    dim = z.shape[1]
    metric = np.zeros((dim, dim))
    for model in models.values():
        theta, b = centred_classifier(model)
        logits = z @ theta.T + b
        for c in range(len(theta)):
            metric += np.outer(theta[c], theta[c]) / logits[:, c].var()
        pre = attention_preactivation(z, model)
        grads = (model.attn_w2[0] * (1 - np.tanh(pre) ** 2)) @ model.attn_w1
        scores = attention_from_preactivation(pre, model)
        metric += attention_weight * (grads.T @ grads) / len(z) / scores.var()
    metric *= dim / np.trace(metric)
    metric += ridge * np.eye(dim)
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    return (eigenvectors * np.sqrt(eigenvalues)).T.astype(np.float32)


def ensemble_route(probs: dict[str, np.ndarray]) -> np.ndarray:
    """Soft vote that the slide is Nancy >= 2, in [0, 1]: the mean of the three heads' votes.

    The heads' slide probabilities are `logits_to_prob` outputs keyed by head.
    Thresholding it at 0.5 is the `pred_ensembling` routing rule
    (`postprocessing/ensembling.py` on the `thesis` branch).
    """
    votes = probs["neutrophils"][:, 0] + probs["nancy_low"][:, 2] + probs["nancy_high"][:, 1:].sum(axis=1)
    return votes / 3


def ensemble_labels(probs: dict[str, np.ndarray]) -> np.ndarray:
    """Final Nancy grade 0-4 of the ensembling method, exactly as `pred_ensembling`."""
    low_branch = probs["nancy_low"][:, :2].argmax(axis=1)
    high_branch = probs["nancy_high"][:, 1:].argmax(axis=1) + 2
    return np.where(ensemble_route(probs) >= 0.5, high_branch, low_branch)


def ensemble_probs(probs: dict[str, np.ndarray]) -> np.ndarray:
    """Per-grade scores of the ensembling method, shape `(n_slides, 5)`, rows summing to 1.

    `pred_ensembling` only yields hard labels, so the scores follow its
    routing: the route probability q is split over Nancy 2-4 in nancy_high's
    proportions, 1 - q over Nancy 0-1 in nancy_low's. Their argmax can
    differ from `ensemble_labels` near the routing threshold.
    """
    q = ensemble_route(probs)[:, None]
    low = probs["nancy_low"][:, :2]
    high = probs["nancy_high"][:, 1:]
    return np.hstack(
        [(1 - q) * low / low.sum(axis=1, keepdims=True), q * high / high.sum(axis=1, keepdims=True)]
    )
