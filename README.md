# n-gnas

JAX core of N-GNAS, node-level graph neural architecture search, arXiv 2610.09297. Authors: Lintao Yanga, Sirui Lia, Yaqing Wang, Pietro Liò, Xu Shen, Baisong Liu, Chengbin Peng.

The core builds stage-I node scores, gates node features with a soft mix of those scores, mixes two operator stand-ins with a residual DARTS weight, and adds binary cross-entropy to a contrastive loss. Search exposes softmax over operators and argmax discretize.

## Install

```bash
pip install -e .
```

## Run

```bash
JAX_PLATFORMS=cpu python -m pytest -q
```

H200 smoke is `train.sbatch` plus `smoke_gpu.py`: 20 Adam steps of the search loss on a 4-node graph. Job 759285 on compute-gpu-01, jax 0.11.1 CudaDevice(id=0), loss 0.207747 to 0.175477, exit 0.

## Mismatches

No CiteSeer, Cora, or other graph dataset is loaded, and there are no accuracy numbers.

S7 (ASAP) is not implemented. The paper cites another method and does not give a self-contained formula in the sections this core covers.

The matrix they call a random-walk matrix is M = D @ A, degree times adjacency, as written. It is not the usual random walk D^{-1} A.

Norm has no formula in the paper. Each score vector is divided by the square root of the sum of squares. A zero vector stays zero.

Their experimental datasets are multi-class. This core implements the binary cross-entropy formula they wrote.

The contrastive double sum includes the i = j diagonal.

GCN, GAT, GraphSAGE, GIN, and GatedGCN from their table are not implemented. Identity and a one-layer GCN are stand-ins so the mix equation can be tested. The mix is X_out = X_G + sum_i softmax(alpha_G)_i * f_i(X_G). Identity is f(X) = X, so a one-hot weight on identity returns X_G + X_G.

The MLP score is the written left-multiply on each node feature vector. With X stored as (N, F), hidden = relu(X @ W1.T + b1), then W2 is applied on that hidden vector.

Disconnected hop counts use sentinel n, the node count. Missing edges start as the larger sentinel n*n during Floyd-Warshall so a real path can replace them. The diagonal is 0.
