"""Hand double sums for contrastive loss, plus binary CE."""

import math

import jax.numpy as jnp

from n_gnas.loss import binary_cross_entropy, contrastive_terms, total_loss


def test_contrastive_includes_diagonal():
    # Unit orthogonal rows, tau = 1.
    # cos = [[1, 0], [0, 1]], sim = [[e, 1], [1, e]]
    Z = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    tau = 1.0
    e = math.exp(1.0)

    # same label: every pair is positive, including the diagonal
    pos_same = e + 1.0 + 1.0 + e
    neg_same = 0.0
    pos, neg, loss = contrastive_terms(Z, jnp.array([1, 1]), tau)
    assert jnp.allclose(pos, pos_same)
    assert jnp.allclose(neg, neg_same)
    assert jnp.allclose(loss, -pos_same / (pos_same + neg_same))
    assert jnp.allclose(loss, -1.0)

    # different labels: diagonal stays in pos, off-diagonal is neg
    pos_diff = e + e
    neg_diff = 1.0 + 1.0
    pos, neg, loss = contrastive_terms(Z, jnp.array([0, 1]), tau)
    assert jnp.allclose(pos, pos_diff)
    assert jnp.allclose(neg, neg_diff)
    assert pos_diff > 0.0
    assert jnp.allclose(loss, -pos_diff / (pos_diff + neg_diff))


def test_binary_ce_and_total_are_the_sum():
    y = jnp.array([1.0, 0.0])
    p = jnp.array([0.5, 0.25])
    # term 0: log(0.5)
    # term 1: log(0.75)
    expected_ce = -(math.log(0.5) + math.log(0.75)) / 2.0
    assert jnp.allclose(binary_cross_entropy(y, p), expected_ce)

    Z = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    e = math.exp(1.0)
    # different labels, tau = 1, diagonal included
    # pos = 2e, neg = 2, L_cl = -e / (e + 1)
    expected_cl = -e / (e + 1.0)
    expected = expected_ce + expected_cl
    assert jnp.allclose(total_loss(y, p, Z, 1.0), expected)
