"""Contrastive loss and binary cross-entropy as written in arXiv 2610.09297.

The contrastive double sum includes the i = j diagonal.
L_ce is binary, not multi-class, even though the paper's datasets are multi-class.
L = L_ce + L_cl.
"""

import jax.numpy as jnp


def contrastive_terms(Z, y, tau):
    """pos, neg, and L_cl.

    cos_ij = (z_i · z_j) / (|z_i| |z_j|)
    sim_ij = exp(cos_ij / tau)
    pos = sum_i sum_j 1_{y_i = y_j} sim_ij
    neg = sum_i sum_j 1_{y_i != y_j} sim_ij
    L_cl = -pos / (pos + neg)

    The i = j terms are included. A node always matches its own label, so pos
    contains the diagonal even when every pair of distinct nodes is negative.
    """
    Z = jnp.asarray(Z)
    y = jnp.asarray(y)
    norms = jnp.sqrt(jnp.sum(jnp.square(Z), axis=-1))
    cos = (Z @ Z.T) / (norms[:, None] * norms[None, :])
    sim = jnp.exp(cos / tau)
    same = y[:, None] == y[None, :]
    pos = jnp.sum(jnp.where(same, sim, 0.0))
    neg = jnp.sum(jnp.where(same, 0.0, sim))
    loss = -pos / (pos + neg)
    return pos, neg, loss


def contrastive_loss(Z, y, tau):
    """L_cl, diagonal included."""
    return contrastive_terms(Z, y, tau)[2]


def binary_cross_entropy(y, p):
    """L_ce = -mean_i [y_i log p_i + (1 - y_i) log(1 - p_i)].

    y is in {0, 1} and p is in (0, 1). This is the binary formula as written,
    not a multi-class cross-entropy.
    """
    y = jnp.asarray(y)
    p = jnp.asarray(p)
    y = y.astype(p.dtype)
    terms = y * jnp.log(p) + (1.0 - y) * jnp.log(1.0 - p)
    return -jnp.mean(terms)


def total_loss(y, p, Z, tau):
    """L = L_ce + L_cl."""
    return binary_cross_entropy(y, p) + contrastive_loss(Z, y, tau)
