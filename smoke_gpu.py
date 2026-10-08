"""20 Adam steps of the node-level search loss on a 4-node graph. 1x H200 smoke."""

import jax
import jax.numpy as jnp
import optax

from n_gnas.loss import total_loss
from n_gnas.search import predict


def main():
    print("jax", jax.__version__, jax.devices())
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
    initial = None
    loss = None
    for step in range(20):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        if not bool(jnp.isfinite(loss)):
            raise SystemExit(f"non-finite loss at step {step}: {loss}")
        if initial is None:
            initial = float(loss)
        print(f"step {step} loss {float(loss):.6f}")
        updates, state = tx.update(grads, state, params)
        params = optax.apply_updates(params, updates)
    print(f"initial {initial:.6f} final {float(loss):.6f}")
    print("n-gnas gpu smoke done")


if __name__ == "__main__":
    main()
