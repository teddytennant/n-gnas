"""Soft node gate.

Default, locked by tests: Xhat = sigmoid(S @ alpha) * xhat_prev.
alpha is raw and does not need to sum to 1.
Set softmax=True to mix with softmax(alpha) instead.
"""

import jax
import jax.numpy as jnp


def soft_gate(S, alpha, xhat_prev, softmax=False):
    """One-layer soft gate.

    S is (N, n_scores), alpha is (n_scores,), xhat_prev is (N,) or broadcasts.
    Raw form: Xhat = sigmoid(S @ alpha) * xhat_prev.
    """
    weights = jax.nn.softmax(alpha) if softmax else alpha
    mixed = jnp.asarray(S) @ jnp.asarray(weights)
    return jax.nn.sigmoid(mixed) * jnp.asarray(xhat_prev)


def masked_features(xhat, X):
    """X_G = xhat[:, None] * X, shape (N, F)."""
    xhat = jnp.reshape(jnp.asarray(xhat), (-1,))
    return xhat[:, None] * jnp.asarray(X)
