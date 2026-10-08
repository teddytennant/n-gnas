"""Hand checks for stage-I scores on a 3-node path."""

import jax.numpy as jnp

from n_gnas.scores import (
    degree_vector,
    feature_projection,
    l2_normalize,
    mlp_score,
    normalized_scores,
    score_s3,
    score_s4,
    score_s5,
    score_s6,
    shortest_path_matrix,
    walk_matrix,
)


def path_graph():
    # 0 - 1 - 2, no self-loops.
    return jnp.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )


def test_walk_matrix_column_sums_and_degree():
    A = path_graph()
    # d = [1, 2, 1], M = D @ A.
    # row 0: 1 * [0, 1, 0]
    # row 1: 2 * [1, 0, 1]
    # row 2: 1 * [0, 1, 0]
    expected_M = jnp.array(
        [
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 2.0],
            [0.0, 1.0, 0.0],
        ]
    )
    M = walk_matrix(A)
    assert jnp.array_equal(M, expected_M)
    # column sums: [2, 2, 2]
    assert jnp.array_equal(score_s3(A), jnp.array([2.0, 2.0, 2.0]))
    assert jnp.array_equal(jnp.sum(M, axis=0), jnp.array([2.0, 2.0, 2.0]))
    # degree score is the column sum of A, which matches d on this symmetric graph
    assert jnp.array_equal(score_s5(A), jnp.array([1.0, 2.0, 1.0]))
    assert jnp.array_equal(degree_vector(A), jnp.array([1.0, 2.0, 1.0]))
    # not the usual random walk D^{-1} A
    usual = jnp.array([1.0, 0.5, 1.0])[:, None] * A
    assert not jnp.allclose(M, usual)


def test_floyd_column_sums_on_path():
    A = path_graph()
    # hops:
    # 0 1 2
    # 1 0 1
    # 2 1 0
    # column sums [3, 2, 3]
    expected_F = jnp.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ]
    )
    F = shortest_path_matrix(A)
    assert jnp.array_equal(F, expected_F)
    assert jnp.array_equal(score_s4(A), jnp.array([3.0, 2.0, 3.0]))


def test_disconnected_pairs_use_sentinel_n():
    # edge only between 0 and 1. node 2 is isolated. sentinel is n = 3.
    A = jnp.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    expected_F = jnp.array(
        [
            [0.0, 1.0, 3.0],
            [1.0, 0.0, 3.0],
            [3.0, 3.0, 0.0],
        ]
    )
    F = shortest_path_matrix(A)
    assert jnp.array_equal(F, expected_F)
    # column sums: [4, 4, 6]
    assert jnp.array_equal(score_s4(A), jnp.array([4.0, 4.0, 6.0]))


def test_projection_and_mlp_hand_forward():
    X = jnp.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    W_s1 = jnp.array([[0.5], [0.25]])
    b_s1 = jnp.array(0.25)
    # X @ W = [0.5, 0.25, 0.75], plus 0.25 -> [0.75, 0.5, 1.0]
    s1 = feature_projection(X, W_s1, b_s1)
    assert jnp.array_equal(s1, jnp.array([0.75, 0.5, 1.0]))

    W1 = jnp.array([[1.0, -1.0], [0.5, 0.5]])
    b1 = jnp.array([0.0, 0.5])
    W2 = jnp.array([1.0, -0.5])
    b2 = jnp.array(0.25)
    # x0 = [1, 0] -> relu([1, 1]) -> 1 - 0.5 + 0.25 = 0.75
    # x1 = [0, 1] -> relu([0, 1]) -> 0 - 0.5 + 0.25 = -0.25
    # x2 = [1, 1] -> relu([0, 1.5]) -> 0 - 0.75 + 0.25 = -0.5
    s2 = mlp_score(X, W1, b1, W2, b2)
    assert jnp.array_equal(s2, jnp.array([0.75, -0.25, -0.5]))
    # same left-multiply when W2 is stored as (1, H)
    s2_row = mlp_score(X, W1, b1, W2[None, :], b2)
    assert jnp.array_equal(s2_row, s2)

    A = path_graph()
    # A @ [0.75, 0.5, 1.0] = [0.5, 1.75, 0.5]
    assert jnp.array_equal(score_s6(A, s1), jnp.array([0.5, 1.75, 0.5]))


def test_l2_norm_and_six_scores():
    assert jnp.array_equal(l2_normalize(jnp.zeros(4)), jnp.zeros(4))
    # [3, 4] has length 5
    assert jnp.allclose(l2_normalize(jnp.array([3.0, 4.0])), jnp.array([0.6, 0.8]))

    A = path_graph()
    X = jnp.ones((3, 2))
    S = normalized_scores(
        A,
        X,
        jnp.array([[0.5], [0.25]]),
        jnp.array(0.0),
        jnp.eye(2),
        jnp.zeros(2),
        jnp.ones(2),
        jnp.array(0.0),
    )
    # s1..s6 only. S7 is not implemented.
    assert S.shape == (3, 6)
    norms = jnp.sqrt(jnp.sum(jnp.square(S), axis=0))
    assert jnp.allclose(norms, jnp.ones(6))
