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
