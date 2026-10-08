"""Residual operator mix.

X_out = X_G + sum_i softmax(alpha)_i * f_i(X_G).

Identity and a one-layer GCN are stand-ins. GCN, GAT, GraphSAGE, GIN, and
GatedGCN from the paper's table are not implemented.
"""

import jax
import jax.numpy as jnp


def identity_op(X, A, W):
    """f(X) = X. A and W are unused."""
    return X


def gcn_adjacency(A):
    """D_hat^{-1/2} A_hat D_hat^{-1/2} with A_hat = A + I.

    D_hat is the degree of A_hat (row sum).
    """
    A = jnp.asarray(A)
    n = A.shape[0]
    A_hat = A + jnp.eye(n, dtype=A.dtype)
    degree = jnp.sum(A_hat, axis=1)
    inv_sqrt = jnp.where(degree > 0, jax.lax.rsqrt(degree), jnp.zeros_like(degree))
    return inv_sqrt[:, None] * A_hat * inv_sqrt[None, :]


def gcn_op(X, A, W):
    """One-layer GCN stand-in: D_hat^{-1/2} A_hat D_hat^{-1/2} X W.

    W is (F, F), supplied by the caller.
    """
    return gcn_adjacency(A) @ jnp.asarray(X) @ jnp.asarray(W)


def residual_mix(X_G, alpha, A, W, ops=None):
    """Residual mix over operator stand-ins.

    Default ops are identity, then the one-layer GCN stand-in.
    """
    if ops is None:
        ops = (identity_op, gcn_op)
    X_G = jnp.asarray(X_G)
    weights = jax.nn.softmax(jnp.asarray(alpha))
    mixed = jnp.zeros_like(X_G)
    for i, op in enumerate(ops):
        mixed = mixed + weights[i] * op(X_G, A, W)
    # Locked formula: X_out = X_G + sum_i softmax(alpha)_i * f_i(X_G).
    # identity is f(X) = X, so a one-hot weight on identity returns X_G + X_G.
    return X_G + mixed
