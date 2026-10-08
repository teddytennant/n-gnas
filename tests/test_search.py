"""DARTS discretize, residual mix, and a short synthetic update."""

import jax
import jax.numpy as jnp
import optax

from n_gnas.loss import total_loss
from n_gnas.mix import residual_mix
from n_gnas.search import darts_softmax, discretize, predict


def test_discretize_breaks_ties_by_smallest_index():
    alpha = jnp.array(
        [
            [1.0, 4.0, 4.0],
            [2.0, 2.0, 0.0],
            [0.5, -1.0, 3.0],
        ]
    )
    # row 0: tie at 1 and 2 -> 1
    # row 1: tie at 0 and 1 -> 0
    # row 2: 2
    assert jnp.array_equal(discretize(alpha), jnp.array([1, 0, 2]))
    assert int(discretize(jnp.array([5.0, 5.0, 5.0]))) == 0


def test_softmax_rows_sum_to_one():
    alpha = jnp.array(
        [
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
        ]
    )
    weights = darts_softmax(alpha)
    assert jnp.allclose(jnp.sum(weights, axis=-1), jnp.ones(2))
    assert jnp.allclose(jnp.sum(darts_softmax(jnp.array([1.0, -2.0, 0.5]))), 1.0)


def test_identity_one_hot_returns_twice_features():
    X_G = jnp.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [0.5, -1.0],
        ]
    )
    A = jnp.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    W = jnp.eye(2)
    # softmax([100, -100]) is exactly [1, 0] in float32
    alpha = jnp.array([100.0, -100.0])
    out = residual_mix(X_G, alpha, A, W)
    assert jnp.array_equal(out, X_G + X_G)


def test_gcn_one_hot_matches_hand_normalized_adjacency():
    # 4-node complete graph, no self-loops in A. A_hat is all ones.
    # degree of A_hat is 4, so D_hat^{-1/2} entries are 1/2.
    # A_norm_ij = (1/2) * 1 * (1/2) = 1/4 for every entry.
    A = jnp.ones((4, 4)) - jnp.eye(4)
    X_G = jnp.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.0, 2.0],
        ]
    )
    W = jnp.array(
        [
            [2.0, 0.0],
            [0.0, 4.0],
        ]
    )
    # row sum of X_G is [2, 4]
    # A_norm @ X_G has every row equal to (1/4) * [2, 4] = [0.5, 1]
    # times W: [0.5 * 2, 1 * 4] = [1, 4]
    f = jnp.array(
        [
            [1.0, 4.0],
            [1.0, 4.0],
            [1.0, 4.0],
            [1.0, 4.0],
        ]
    )
    alpha = jnp.array([-100.0, 100.0])
    out = residual_mix(X_G, alpha, A, W)
    assert jnp.array_equal(out, X_G + f)


def test_three_optax_steps_keep_loss_finite():
    A = jnp.array(
        [
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
        ]
    )
    X = jnp.array(
        [
            [1.0, 0.5],
            [0.5, 1.0],
            [1.0, 1.0],
            [0.25, 0.5],
        ]
    )
    y = jnp.array([0.0, 1.0, 0.0, 1.0])
    params = {
        "W_s1": jnp.array([[0.5], [0.25]]),
        "b_s1": jnp.array(0.25),
        "W1": jnp.array([[0.5, 0.0], [0.0, 0.5]]),
        "b1": jnp.zeros(2),
        "W2": jnp.array([0.5, 0.5]),
        "b2": jnp.array(0.0),
        "alpha_S": jnp.zeros((6,)),
        "alpha_G": jnp.zeros((2,)),
        "W_gcn": jnp.eye(2),
        "W_head": jnp.array([[0.25], [-0.25]]),
        "b_head": jnp.array(0.0),
    }

    def loss_fn(p):
        prob, Z = predict(p, X, A)
        return total_loss(y, prob, Z, 1.0)

    tx = optax.adam(1e-2)
    state = tx.init(params)
    for _ in range(3):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        assert bool(jnp.isfinite(loss))
        updates, state = tx.update(grads, state, params)
        params = optax.apply_updates(params, updates)
    assert bool(jnp.isfinite(loss_fn(params)))
