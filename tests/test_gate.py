"""Hand checks for the raw soft gate and masked features."""

import math

import jax.numpy as jnp

from n_gnas.gate import masked_features, soft_gate


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def test_raw_gate_matches_hand_sigmoid():
    S = jnp.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 2.0],
        ]
    )
    # does not sum to 1
    alpha = jnp.array([0.5, -1.0])
    # mixed = [0.5, -1.0, 1.0 - 2.0] = [0.5, -1.0, -1.0]
    xhat_prev = jnp.array([2.0, 4.0, 0.5])
    expected = jnp.array(
        [
            _sigmoid(0.5) * 2.0,
            _sigmoid(-1.0) * 4.0,
            _sigmoid(-1.0) * 0.5,
        ]
    )
    got = soft_gate(S, alpha, xhat_prev)
    assert jnp.allclose(got, expected)
    # raw alpha is the default, so an explicit False matches
    assert jnp.allclose(soft_gate(S, alpha, xhat_prev, softmax=False), expected)


def test_softmax_flag_differs_from_raw():
    S = jnp.array(
        [
            [1.0, 3.0],
            [2.0, 0.0],
        ]
    )
    alpha = jnp.array([0.0, 0.0])
    prev = jnp.array([1.0, 2.0])
    # raw mix is 0, sigmoid(0) = 0.5
    raw = soft_gate(S, alpha, prev, softmax=False)
    assert jnp.allclose(raw, jnp.array([0.5, 1.0]))
    # softmax([0, 0]) = [0.5, 0.5]
    # mixed = [2.0, 1.0]
    expected = jnp.array([_sigmoid(2.0) * 1.0, _sigmoid(1.0) * 2.0])
    got = soft_gate(S, alpha, prev, softmax=True)
    assert jnp.allclose(got, expected)
    assert not jnp.allclose(got, raw)


def test_masked_features_scale_rows():
    xhat = jnp.array([0.5, 2.0, 0.25])
    X = jnp.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )
    expected = jnp.array(
        [
            [0.5, 1.0],
            [6.0, 8.0],
            [1.25, 1.5],
        ]
    )
    assert jnp.array_equal(masked_features(xhat, X), expected)
