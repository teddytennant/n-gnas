"""Stage-I node scores for N-GNAS (arXiv 2610.09297).

M is D @ A, degree times adjacency, not D^{-1} A.
Norm is an L2 normalization. The paper writes Norm without a formula.
S7 (ASAP) is not implemented. The paper cites another method and does not
give a self-contained formula for it.
Disconnected pairs in the hop-count matrix use sentinel n.
"""

import jax
import jax.numpy as jnp


def degree_vector(A):
    """d_i = sum_j A_ij."""
    return jnp.sum(jnp.asarray(A), axis=1)


def degree_matrix(A):
    """D = diag(d), with d_i = sum_j A_ij."""
    return jnp.diag(degree_vector(A))


def walk_matrix(A):
    """M = D @ A.

    The paper calls this a random-walk matrix and writes M = D @ A.
    That is degree times adjacency, not the usual random walk D^{-1} A.
    """
    A = jnp.asarray(A)
    return degree_matrix(A) @ A


def feature_projection(X, W, b):
    """X_S1 = X @ W + b, shape (N,).

    W is (F, 1) or (F,). b broadcasts against the projection.
    """
    projected = jnp.asarray(X) @ jnp.asarray(W) + jnp.asarray(b)
    if projected.ndim == 2 and projected.shape[-1] == 1:
        projected = jnp.squeeze(projected, axis=-1)
    return projected


def mlp_score(X, W1, b1, W2, b2):
    """X_S2 = W2 @ relu(W1 @ x + b1) + b2 for each node feature x.

    X is (N, F). W1 is (H, F) and b1 is (H,), so W1 @ x is a left-multiply
    on one node. W2 is (H,) or (1, H). b2 is a scalar. Output is (N,).
    Batched, hidden = relu(X @ W1.T + b1), then W2 is applied on that axis.
    """
    hidden = jax.nn.relu(jnp.asarray(X) @ jnp.asarray(W1).T + jnp.asarray(b1))
    W2 = jnp.asarray(W2)
    b2 = jnp.asarray(b2)
    if W2.ndim == 1:
        return hidden @ W2 + b2
    out = hidden @ W2.T + b2
    if out.ndim == 2 and out.shape[-1] == 1:
        out = jnp.squeeze(out, axis=-1)
    return out


def shortest_path_matrix(A, disconnected=None):
    """Hop-count shortest paths on the unweighted graph.

    A missing edge starts as a large sentinel (n * n) during Floyd-Warshall.
    An edge has length 1, so entries are hop counts. The diagonal is 0 even
    if the caller added self-loops. After relaxation, pairs that are still
    disconnected are replaced by a finite sentinel so column sums are defined.
    That sentinel is n, the node count, unless `disconnected` is passed.
    A non-zero entry is an edge. Direction follows A.
    """
    A = jnp.asarray(A)
    n = A.shape[0]
    if disconnected is None:
        disconnected = n
    dtype = jnp.promote_types(A.dtype, jnp.float32)
    big = jnp.asarray(n * n, dtype=dtype)
    zero = jnp.asarray(0, dtype=dtype)
    one = jnp.asarray(1, dtype=dtype)
    diag = jnp.eye(n, dtype=bool)
    dist = jnp.where(diag, zero, jnp.where(A != 0, one, big))

    def relax(dist, k):
        via = dist[:, k][:, None] + dist[k, :][None, :]
        return jnp.minimum(dist, via)

    for k in range(n):
        dist = relax(dist, k)
    return jnp.where(dist >= big, jnp.asarray(disconnected, dtype=dtype), dist)


def score_s3(A):
    """X_S3_j = sum_i M_ij, column sums of M = D @ A."""
    return jnp.sum(walk_matrix(A), axis=0)


def score_s4(A, disconnected=None):
    """X_S4_j = sum_i F_ij, column sums of the hop-count matrix."""
    return jnp.sum(shortest_path_matrix(A, disconnected=disconnected), axis=0)


def score_s5(A):
    """X_S5_j = sum_i A_ij.

    Column sum of A. On a symmetric graph this is the degree.
    """
    return jnp.sum(jnp.asarray(A), axis=0)


def score_s6(A, x_s1):
    """X_S6 = A @ X_S1, neighbor aggregation of the projected features."""
    return jnp.asarray(A) @ jnp.asarray(x_s1)


def l2_normalize(v):
    """Divide by sqrt(sum of squares). A zero vector stays zero.

    The paper writes Norm without a formula. This core uses L2.
    """
    v = jnp.asarray(v)
    norm = jnp.sqrt(jnp.sum(jnp.square(v)))
    safe = jnp.where(norm == 0, jnp.ones_like(norm), norm)
    return jnp.where(norm == 0, jnp.zeros_like(v), v / safe)


def node_scores(A, X, W_s1, b_s1, W1, b1, W2, b2):
    """Raw scores s1 through s6. S7 (ASAP) is not computed."""
    s1 = feature_projection(X, W_s1, b_s1)
    s2 = mlp_score(X, W1, b1, W2, b2)
    s3 = score_s3(A)
    s4 = score_s4(A)
    s5 = score_s5(A)
    s6 = score_s6(A, s1)
    return s1, s2, s3, s4, s5, s6


def normalized_scores(A, X, W_s1, b_s1, W1, b1, W2, b2):
    """Stack L2-normalized s1..s6 into S with shape (N, 6)."""
    raw = node_scores(A, X, W_s1, b_s1, W1, b1, W2, b2)
    return jnp.stack([l2_normalize(s) for s in raw], axis=1)
