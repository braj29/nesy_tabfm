#!/bin/bash
# ICF Slurm job for the in-repo nesy_tabfm experiment, with TabPFN run first.
#
# Submit from the repository root on ICF:
#
#   sbatch scripts/cluster/icf_full_experiment.sbatch
#
# Order:
#   1. rel-salt TabPFN
#   2. HELOC TabPFN
#   3. rel-f1 TabPFN-Rel audit, if precomputed predictions are present
#   4. rel-salt LightGBM reference
#   5. HELOC LightGBM reference
#   6. rel-f1 LightGBM audit, if precomputed predictions are present
#   7. Tier-1 summary

#SBATCH --job-name=nesy-full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slogs/%x-%j.out
#SBATCH --error=slogs/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_DIR/results}"
TABPFN_DEVICE="${TABPFN_DEVICE:-cuda}"
SALT_TRAIN_SAMPLE_SIZE="${SALT_TRAIN_SAMPLE_SIZE:-8000}"
SALT_TEST_SAMPLE_SIZE="${SALT_TEST_SAMPLE_SIZE:-1000}"
SEED="${SEED:-0}"

REL_F1_TABPFN_PREDS="${REL_F1_TABPFN_PREDS:-$REPO_DIR/relarena_predict/results/tabpfn-rel-local}"
REL_F1_LIGHTGBM_PREDS="${REL_F1_LIGHTGBM_PREDS:-$REPO_DIR/relarena_predict/results/lightgbm}"

cd "$REPO_DIR"
mkdir -p slogs "$RESULTS_ROOT"

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

if [[ -d "$REL_F1_TABPFN_PREDS" ]]; then
  python scripts/run_audit.py \
    --model tabpfn-rel-local \
    --predictions "$REL_F1_TABPFN_PREDS" \
    --results-root "$RESULTS_ROOT"
else
  echo "Skipping rel-f1 TabPFN-Rel audit; predictions not found: $REL_F1_TABPFN_PREDS"
fi

python scripts/run_salt_audit.py --model lightgbm --results-root "$RESULTS_ROOT"
python scripts/run_heloc_audit.py --model lightgbm --results-root "$RESULTS_ROOT"

if [[ -d "$REL_F1_LIGHTGBM_PREDS" ]]; then
  python scripts/run_audit.py \
    --model lightgbm \
    --predictions "$REL_F1_LIGHTGBM_PREDS" \
    --results-root "$RESULTS_ROOT"
else
  echo "Skipping rel-f1 LightGBM audit; predictions not found: $REL_F1_LIGHTGBM_PREDS"
fi

python scripts/run_tier1.py --results-root "$RESULTS_ROOT"
