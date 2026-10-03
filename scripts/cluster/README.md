# Cluster jobs

## ICF TabPFN audits

Use `icf_tabpfn_smoke.sbatch`, `icf_tabpfn_full.sbatch`, and
`icf_full_experiment.sbatch` for the ICF runs. Start with the smoke test, then submit either
the TabPFN-only job or the broader full experiment:

```bash
sbatch scripts/cluster/icf_tabpfn_smoke.sbatch
sbatch scripts/cluster/icf_tabpfn_full.sbatch
sbatch scripts/cluster/icf_full_experiment.sbatch
```

See `scripts/cluster/icf_tabpfn.md` for the full ICF workflow, environment variables, and log
paths. These jobs run through Slurm only; do not run the TabPFN audit directly on the ICF
login/head node.

## Training RelGNN on rel-f1 (cluster)

RelGNN training runs as a **separate process in a separate environment** from this repo --
not through `nesy_tabfm` code. This repo's audit only needs the resulting per-instance
prediction parquet (see `nesy_tabfm.models.precomputed.PrecomputedAdapter`); it never needs
`torch`/`torch_geometric` installed itself.

## 1. Environment: pin `relbench==1.1.0`, not the latest

`snap-stanford/RelGNN`'s scripts import the pre-rewrite relbench API
(`from relbench.datasets import get_dataset`, `from relbench.tasks import get_task`). relbench
was rewritten to a manifest-driven loader (`relbench.load_dataset("org/repo/subdir")`) at the
2.x/3.x releases -- the old `relbench.datasets`/`relbench.tasks` modules don't exist there.
**Following RelGNN's README literally (`pip install relbench[full]`) today pulls the latest
relbench and breaks immediately** with `ModuleNotFoundError: No module named 'relbench.datasets'`.
Verified: `relbench==1.1.0` still has the old API; nothing later than `2.0.0` does.

```bash
git clone https://github.com/snap-stanford/RelGNN.git
cd RelGNN
conda create -n relgnn python=3.10 -y && conda activate relgnn
pip install "relbench[full]==1.1.0"        # pin, do not take latest
pip install torch --index-url <cuda-wheel-url-for-your-cluster>   # match cluster CUDA version
pip install pyg-lib torch-geometric torch-frame sentence-transformers tqdm
```

## 2. Train + dump predictions for our 3 tasks

Run once per task (script/args per RelGNN's own `examples/relgnn_task_node.py --help`):

```bash
python examples/relgnn_task_node.py --dataset rel-f1 --task driver-position
python examples/relgnn_task_node.py --dataset rel-f1 --task driver-dnf
python examples/relgnn_task_node.py --dataset rel-f1 --task driver-top3
```

As shipped, the script only prints a test metric at the end:

```python
test_pred = test(loader_dict["test"])
test_metrics = task.evaluate(test_pred)
print(f"Test metric: {test_metrics[tune_metric]}")
```

Patch that tail (add `import pandas as pd` near the top of the file) to also dump a parquet
matching the schema every adapter in this repo returns
(`[entity_col, time_col, "y_true", "y_pred"]`, hard 0/1 for classification):

```python
test_pred = test(loader_dict["test"])
test_metrics = task.evaluate(test_pred)
print(f"Test metric: {test_metrics[tune_metric]}")

test_table = task.get_table("test", mask_input_cols=False).df
y_pred = (test_pred >= 0.5).astype(int) if task.task_type == TaskType.BINARY_CLASSIFICATION else test_pred
pd.DataFrame({
    task.entity_col: test_table[task.entity_col].to_numpy(),
    task.time_col: test_table[task.time_col].to_numpy(),
    "y_true": test_table[task.target_col].to_numpy(),
    "y_pred": y_pred,
}).to_parquet(f"{args.task}_predictions.parquet")
```

`relgnn_train.sbatch.example` is a starting point for a Slurm job doing the above; edit the
partition/account/module lines for your actual cluster.

## 3. Run the audit locally

Copy the 3 resulting `<task-name>_predictions.parquet` files into one local directory, then:

```bash
uv run python scripts/run_audit.py --model relgnn --predictions /path/to/that/directory
```
