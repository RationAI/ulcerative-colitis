"""Slide-grading routing rule, ported from the `thesis` branch's `postprocessing/ensembling_predict.py`.

`postprocessing/` (thesis) doesn't exist on `feature/explainability` - it lives
on a branch that diverged from this one's history before `explainability/`
existed at all, so it can't be imported. Only the pure routing arithmetic is
ported here (not `thesis`'s mlflow-table loading, which assumed one prediction
row per *slide* from a bag-level MIL forward pass); `explainability/
grade_split.py` calls this per *tile* instead, on logits computed directly
from `c_psi` (concept_mil.tex's tile-level classifier) - see that module's
docstring for why.

Of `thesis`'s two hard-label routing rules (`pred_ensembling` and
`pred_hierarchical`), only `route_grade` (== `pred_ensembling`, a soft
majority vote across all three tasks) is ported - user's explicit choice
over `pred_hierarchical` (neutrophils-only routing) and over taking
`argmax` of `markov_chain_confidence_predict.py`'s absorbing-Markov-chain
soft distribution.
"""

import numpy as np


def route_grade(neut_prob: np.ndarray, low_probs: np.ndarray, high_probs: np.ndarray) -> np.ndarray:
    """Combine the three task heads' probabilities into one 0-4 Nancy grade per row.

    Verbatim port of `run_ensembling_predict`'s `ens_pred` logic (`thesis`
    branch, `postprocessing/ensembling_predict.py`) - a soft majority vote
    across all three tasks decides whether a row routes to the nancy_low or
    nancy_high branch, rather than trusting the neutrophils head alone
    (`pred_hierarchical`, not used here).

    Args:
        neut_prob: Neutrophils head's P(high), shape (n,).
        low_probs: nancy_low head's softmax output, shape (n, 3) - classes
            are [NHI-0, NHI-1, ->nancy_high].
        high_probs: nancy_high head's softmax output, shape (n, 4) - classes
            are [->nancy_low, NHI-2, NHI-3, NHI-4].

    Returns:
        Integer array, shape (n,), values in {0, 1, 2, 3, 4}.
    """
    low_branch = low_probs[:, :2].argmax(axis=1)
    high_branch = high_probs[:, 1:].argmax(axis=1) + 2

    # Soft majority vote: each task casts one "vote" (as a probability) for
    # routing to the high branch - >=1.5 out of 3 possible votes means at
    # least a majority leaned high, even if no single task crossed 0.5 alone.
    route_high = (neut_prob + low_probs[:, 2] + high_probs[:, 1:].sum(axis=1)) >= 1.5
    return np.where(route_high, high_branch, low_branch).astype(np.int64)
