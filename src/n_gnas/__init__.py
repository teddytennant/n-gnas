"""N-GNAS core: node-score gate, residual op mix, contrastive loss.

Equations follow arXiv 2610.09297. See the README for what is not implemented.
"""

from n_gnas.gate import masked_features, soft_gate
from n_gnas.loss import binary_cross_entropy, contrastive_loss, total_loss
from n_gnas.mix import gcn_op, identity_op, residual_mix
from n_gnas.scores import (
    feature_projection,
    l2_normalize,
    mlp_score,
    normalized_scores,
    shortest_path_matrix,
    walk_matrix,
)
from n_gnas.search import darts_softmax, discretize, predict

__all__ = [
    "binary_cross_entropy",
    "contrastive_loss",
    "darts_softmax",
    "discretize",
    "feature_projection",
    "gcn_op",
    "identity_op",
    "l2_normalize",
    "masked_features",
    "mlp_score",
    "normalized_scores",
    "predict",
    "residual_mix",
    "shortest_path_matrix",
    "soft_gate",
    "total_loss",
    "walk_matrix",
]
