#!/usr/bin/env bash
#
# Smoke run for N-GNAS (arXiv 2610.09297). Copied from templates/sbatch.sh
# with the two placeholders filled in.
#
# __VENV_SETUP__   : reuse the shared CUDA jax venv
# __RUN_COMMAND__  : 20 Adam steps of the node-level search loss, 1x H200

#SBATCH --job-name=arxiv-impl-n-gnas
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -uo pipefail
cd "$(dirname "$0")"
ROOT="${SLURM_SUBMIT_DIR:-/work/ttennant1/arxiv-impl/20261008T102003Z}"
exec > "$ROOT/n-gnas.out" 2> "$ROOT/n-gnas.err"

PY="$HOME/arxiv-jax/venv/bin/python"
export JAX_PLATFORMS=cuda
unset JAX_PLATFORM_NAME || true
export PYTHONPATH="$ROOT/n-gnas/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT/n-gnas"

"$PY" -c "import jax; print('jax', jax.__version__, jax.devices())"
"$PY" smoke_gpu.py
rc=$?

echo "exit $rc"
exit $rc
