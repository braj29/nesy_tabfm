# ICF TabPFN Runs

These files are meant to be run from an ICF Slurm window such as `icf-1`, not on the
`nuremberg` shell and not directly on the ICF login node.

## 1. Get the repo onto ICF

From `hastings`, put the repository somewhere under your workspace, for example:

```bash
cd ~/workspace
git clone <your-repo-url> nesy_tabfm
cd nesy_tabfm
```

If you copied it another way, just `cd` to that copy before submitting jobs.

## 2. Set the TabPFN token

TabPFN may need a license/API token. Put it in the environment before `sbatch`:

```bash
export TABPFN_TOKEN="<your-token>"
```

For a persistent setup, add that export to a private shell startup file on the cluster.

## 3. Submit a smoke test first

```bash
mkdir -p slogs
sbatch scripts/cluster/icf_tabpfn_smoke.sbatch
squeue -u "$USER"
```

Watch logs:

```bash
tail -f slogs/nesy-tabpfn-smoke-<jobid>.out
tail -f slogs/nesy-tabpfn-smoke-<jobid>.err
```

The smoke test writes to `results_smoke/` and uses tiny rel-salt samples. It is only for
checking the environment.

## 4. Submit the full TabPFN-only job

```bash
sbatch scripts/cluster/icf_tabpfn_full.sbatch
squeue -u "$USER"
```

The full job runs:

```bash
python scripts/run_salt_audit.py --model tabpfn --tabpfn-device cuda
python scripts/run_heloc_audit.py --model tabpfn --tabpfn-device cuda
python scripts/run_tier1.py
```

Outputs go to `results/salt/tabpfn/`, `results/heloc/tabpfn/`, and the tier-1 summary.

## 5. Submit the full experiment with TabPFN first

Once the smoke test is clean, this is the broadest in-repo job:

```bash
sbatch scripts/cluster/icf_full_experiment.sbatch
```

It runs TabPFN first, then LightGBM reference baselines, then `run_tier1.py`. If rel-f1
prediction directories already exist under `relarena_predict/results/`, it audits them too;
otherwise it prints a skip message for rel-f1 and keeps going.

## Useful overrides

Use environment variables before `sbatch` to change the job without editing the script:

```bash
export REPO_DIR="$PWD"
export RESULTS_ROOT="$PWD/results_cluster"
export TABPFN_DEVICE="cuda"
export SALT_TRAIN_SAMPLE_SIZE=8000
export SALT_TEST_SAMPLE_SIZE=1000
export SEED=0
sbatch scripts/cluster/icf_full_experiment.sbatch
```

To force CPU for debugging:

```bash
TABPFN_DEVICE=cpu sbatch scripts/cluster/icf_tabpfn_smoke.sbatch
```

## Do not run the experiment directly on the login node

Use `sbatch`. Commands like `python scripts/run_salt_audit.py --model tabpfn` should run
inside a Slurm allocation, not at the `[hastings]...` prompt.
