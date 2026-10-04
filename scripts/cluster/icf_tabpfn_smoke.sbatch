#!/bin/bash
# Short ICF Slurm smoke test for TabPFN wiring. This intentionally uses tiny samples so you
# can verify the environment, CUDA bridge, TABPFN_TOKEN, and result paths before launching the
# full TabPFN job.
#
# Submit from the repository root on ICF:
#
#   sbatch scripts/cluster/icf_tabpfn_smoke.sbatch

#SBATCH --job-name=nesy-tabpfn-smoke
#SBATCH --partition=ICF-Research
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slogs/%x-%j.out
#SBATCH --error=slogs/%x-%j.err

set -eo pipefail

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
if [[ -z "${TABPFN_TOKEN:-}" ]]; then
  echo "ERROR: TABPFN_TOKEN is not set (checked .env, tabpfn_token, and the environment)." >&2
  echo "Set it before submitting, e.g.: export TABPFN_TOKEN=\"<token>\"" >&2
  exit 1
fi

module add cuda || true

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "--- nvidia-smi ---"
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
  echo "------------------"
fi

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

python - <<PY
import sys, torch
cuda_ok = torch.cuda.is_available()
print("torch", torch.__version__, "cuda_available", cuda_ok)
if "$TABPFN_DEVICE" == "cuda" and not cuda_ok:
    print("ERROR: TABPFN_DEVICE=cuda but torch.cuda.is_available() is False; "
          "aborting before spending the GPU allocation.", file=sys.stderr)
    sys.exit(1)
PY

python scripts/run_salt_audit.py \
  --model tabpfn \
  --results-root "$RESULTS_ROOT" \
  --tabpfn-device "$TABPFN_DEVICE" \
  --train-sample-size 256 \
  --test-sample-size 64 \
  --seed 0
