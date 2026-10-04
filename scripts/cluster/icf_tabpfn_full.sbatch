#!/bin/bash
# ICF Slurm job for the in-repo TabPFN audits.
#
# Submit from the repository root on ICF:
#
#   sbatch scripts/cluster/icf_tabpfn_full.sbatch
#
# The job runs the TabPFN parts first:
#   1. rel-salt TabPFN audit
#   2. HELOC TabPFN monotonicity audit
#
# rel-f1's TabPFN-Rel predictions are produced by the separate relarena_predict project;
# this job only runs nesy_tabfm's in-process TabPFN experiments.

#SBATCH --job-name=nesy-tabpfn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=slogs/%x-%j.out
#SBATCH --error=slogs/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_DIR/results}"
TABPFN_DEVICE="${TABPFN_DEVICE:-cuda}"
SALT_TRAIN_SAMPLE_SIZE="${SALT_TRAIN_SAMPLE_SIZE:-8000}"
SALT_TEST_SAMPLE_SIZE="${SALT_TEST_SAMPLE_SIZE:-1000}"
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

module add cuda || true

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required on the cluster login/compute environment." >&2
  echo "Install or load uv, then resubmit this job." >&2
  exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
  uv sync --extra salt --extra heloc
fi

source .venv/bin/activate

# ICF advertises add_pytorch as the bridge from a project venv to the site PyTorch/CUDA venv.
# If the command is unavailable, continue; the project's own locked environment may already
# provide torch.
if command -v add_pytorch >/dev/null 2>&1; then
  add_pytorch || true
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "job=$SLURM_JOB_ID host=$(hostname) repo=$REPO_DIR results=$RESULTS_ROOT"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
PY

python scripts/run_salt_audit.py \
  --model tabpfn \
  --results-root "$RESULTS_ROOT" \
  --tabpfn-device "$TABPFN_DEVICE" \
  --train-sample-size "$SALT_TRAIN_SAMPLE_SIZE" \
  --test-sample-size "$SALT_TEST_SAMPLE_SIZE" \
  --seed "$SEED"

python scripts/run_heloc_audit.py \
  --model tabpfn \
  --results-root "$RESULTS_ROOT" \
  --tabpfn-device "$TABPFN_DEVICE"

python scripts/run_tier1.py --results-root "$RESULTS_ROOT"
