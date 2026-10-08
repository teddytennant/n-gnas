# n-gnas

JAX core of N-GNAS, node-level graph neural architecture search, arXiv 2610.09297. Authors: Lintao Yanga, Sirui Lia, Yaqing Wang, Pietro Lio, Xu Shen, Baisong Liu, Chengbin Peng.

The core builds stage-I node scores, gates node features with a soft mix of those scores, mixes two operator stand-ins with a residual DARTS weight, and adds binary cross-entropy to a contrastive loss. Search exposes softmax over operators and argmax discretize.

## Tests

Synthetic graphs only. CPU is enough. From this directory:

    export LD_LIBRARY_PATH="/nix/store/larys5yihddj2diyab60hpdri8n11kn7-ld-library-path/share/nix-ld/lib:/nix/store/ab3753m6i7isgvzphlar0a8xb84gl96i-gcc-15.2.0-lib/lib:/nix/store/2kdz3m7ic8w226pcvkz1dlg169v91p6a-zlib-1.3.2/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    /home/nixos/arxiv-impl-work/.venv/bin/python -m pytest -q

pytest uses `pythonpath = ["src"]` from pyproject.toml. Do not install this package into the shared venv.

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
