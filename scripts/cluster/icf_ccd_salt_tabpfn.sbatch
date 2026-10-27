#!/bin/bash
# ICF Slurm job: Constraint-Conditioned Decoding (CCD) on rel-salt with TabPFN.
#   - predict: TabPFN per-class probabilities for item-shippoint and item-plant (8,000-row
#     context, full test split), the stage that needs a GPU;
#   - decode: CCD vs raw decisions and repair baselines (CPU, a few minutes).
#
# Submit from the repository root on ICF:
#
#   sbatch scripts/cluster/icf_ccd_salt_tabpfn.sbatch

#SBATCH --job-name=nesy-ccd-salt-tabpfn
#SBATCH --partition=ICF-Research
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
# 2 of the 8 tasks from the full-test run (17 min for all 8 on scotia06), so ~10 min;
# 1h leaves a wide margin for slower GPUs and the decode stage.
#SBATCH --output=slogs/%x-%j.out
#SBATCH --error=slogs/%x-%j.err

set -eo pipefail

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_DIR/results_ccd}"
TABPFN_DEVICE="${TABPFN_DEVICE:-cuda}"
SALT_TRAIN_SAMPLE_SIZE="${SALT_TRAIN_SAMPLE_SIZE:-8000}"
SALT_PREDICT_BATCH_SIZE="${SALT_PREDICT_BATCH_SIZE:-1024}"
SEED="${SEED:-0}"

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
  exit 1
fi

module add cuda || true

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "--- nvidia-smi ---"
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
  echo "------------------"
fi

if ! command -v uv >/dev/null 2>&1; then
  export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; installing to --user site with pip."
  python3 -m pip install --user uv
  export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is still unavailable after attempted install." >&2
  exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
  uv sync --extra salt --extra heloc
fi

source .venv/bin/activate
if command -v add_pytorch >/dev/null 2>&1; then
  add_pytorch || true
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "job=$SLURM_JOB_ID host=$(hostname) repo=$REPO_DIR results=$RESULTS_ROOT"
python - <<PY
import sys, torch
cuda_ok = torch.cuda.is_available()
print("torch", torch.__version__, "cuda_available", cuda_ok)
if "$TABPFN_DEVICE" == "cuda" and not cuda_ok:
    print("ERROR: TABPFN_DEVICE=cuda but torch.cuda.is_available() is False.", file=sys.stderr)
    sys.exit(1)
PY


python scripts/run_ccd_salt.py predict \
  --model tabpfn \
  --results-root "$RESULTS_ROOT" \
  --tabpfn-device "$TABPFN_DEVICE" \
  --train-sample-size "$SALT_TRAIN_SAMPLE_SIZE" \
  --predict-batch-size "$SALT_PREDICT_BATCH_SIZE" \
  --seed "$SEED"

python scripts/run_ccd_salt.py decode --model tabpfn --results-root "$RESULTS_ROOT" --seed "$SEED"
