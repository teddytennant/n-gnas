"""DARTS softmax, discretize, and a small supervised forward.

discretize is argmax. Ties take the smallest index along the reduced axis.
"""

import jax
import jax.numpy as jnp

from n_gnas.gate import masked_features, soft_gate
from n_gnas.mix import residual_mix
from n_gnas.scores import normalized_scores


def darts_softmax(alpha, axis=-1):
    """Softmax over the operator axis."""
    return jax.nn.softmax(jnp.asarray(alpha), axis=axis)


def discretize(alpha, axis=-1):
    """argmax along axis. Ties break to the smallest index.

    jnp.argmax returns the first maximum, which is the smallest index.
    """
    return jnp.argmax(jnp.asarray(alpha), axis=axis)


def predict(params, X, A):
    """Node-gate, residual mix, and a linear sigmoid head.

    params keys: W_s1, b_s1, W1, b1, W2, b2, alpha_S, alpha_G, W_gcn,
    W_head, b_head.

    Returns p of shape (N,) and Z = X_out of shape (N, F).
    The gate uses raw alpha_S. The mix uses softmax(alpha_G).
    """
    X = jnp.asarray(X)
    S = normalized_scores(
        A,
        X,
        params["W_s1"],
        params["b_s1"],
        params["W1"],
        params["b1"],
        params["W2"],
        params["b2"],
    )
    n = X.shape[0]
    xhat = soft_gate(S, params["alpha_S"], jnp.ones((n,), dtype=X.dtype))
    X_G = masked_features(xhat, X)
    X_out = residual_mix(X_G, params["alpha_G"], A, params["W_gcn"])
    logits = jnp.reshape(X_out @ params["W_head"] + params["b_head"], (n,))
    return jax.nn.sigmoid(logits), X_out
