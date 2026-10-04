#!/bin/bash
# Short ICF Slurm smoke test for TabPFN wiring. This intentionally uses tiny samples so you
# can verify the environment, CUDA bridge, TABPFN_TOKEN, and result paths before launching the
# full TabPFN job.
#
# Submit from the repository root on ICF:
#
#   sbatch scripts/cluster/icf_tabpfn_smoke.sbatch

#SBATCH --job-name=nesy-tabpfn-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slogs/%x-%j.out
#SBATCH --error=slogs/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_DIR/results_smoke}"
TABPFN_DEVICE="${TABPFN_DEVICE:-cuda}"

cd "$REPO_DIR"
mkdir -p slogs "$RESULTS_ROOT"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
if [[ -z "${TABPFN_TOKEN:-}" && -n "${tabpfn_token:-}" ]]; then
  export TABPFN_TOKEN="$tabpfn_token"
fi

module add cuda || true

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required on the cluster login/compute environment." >&2
  exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
  uv sync --extra salt --extra heloc
fi

source .venv/bin/activate
if command -v add_pytorch >/dev/null 2>&1; then
  add_pytorch || true
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
PY

python scripts/run_salt_audit.py \
  --model tabpfn \
  --results-root "$RESULTS_ROOT" \
  --tabpfn-device "$TABPFN_DEVICE" \
  --train-sample-size 256 \
  --test-sample-size 64 \
  --seed 0
