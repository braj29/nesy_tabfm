# relarena_predict

Runs [RelArena-α](https://github.com/PriorLabs/relarena)'s CPU baselines (`lightgbm`,
`tabpfn-rel-local`) on rel-f1's 3 official tasks and dumps per-instance predictions for
`nesy_tabfm`'s constraint audit (`../scripts/run_audit.py`).

**This is a separate `uv` project from the parent `nesy_tabfm` repo, on purpose.** relarena
pins `relbench==2.1.2` (the pre-rewrite `relbench.datasets`/`relbench.tasks` API); the parent
repo depends on `relbench>=3.0` (the new manifest-driven `relbench.load_dataset(...)` API).
These cannot be installed in the same venv — hence two separate `pyproject.toml`s, run
separately, connected only by the parquet files this script writes.

## Setup

```bash
cd relarena_predict
uv sync
```

## Run

```bash
OMP_NUM_THREADS=1 uv run python predict.py                       # both models, all 3 tasks
OMP_NUM_THREADS=1 uv run python predict.py --models lightgbm     # skip tabpfn-rel-local
```

**`OMP_NUM_THREADS=1` is required on at least this machine (macOS), not optional.**
`lightgbm`'s `fit_lgb` builds a native `lgb.Dataset` with a high-cardinality categorical
column (`driverRef`, 771 categories) and, multi-threaded, that `.construct()` call
segfaults (exit code 139) -- silently: no Python exception, so `predict.py`'s own
`except Exception` around each `(model, task)` pair never catches it and the process just
dies. Confirmed via bisection this is specific to real rel-f1 driver-name data (structurally
identical synthetic data of the same cardinality does not crash) and to LightGBM's
multi-threaded categorical-histogram construction specifically (`num_threads=1` inside the
`lgb.Dataset` params, or the `OMP_NUM_THREADS=1` env var equivalent, both avoid it every
time). Root cause not fully isolated beyond that (looks like a native race condition in this
`lightgbm==4.6.0` build on this platform, not a bug in this repo's code), so the env var is
the practical fix, not a real one.

`lightgbm` needs no external account. `tabpfn-rel-local` uses the same `tabpfn` package (v3
backend) as nesy_tabfm's earlier TabPFN attempt, and will hit the same one-time license wall
until `TABPFN_TOKEN` is set (register/accept the license at https://ux.priorlabs.ai, copy an
API key from the account page, `export TABPFN_TOKEN="<key>"`). The script keeps going and
still writes `lightgbm`'s results if `tabpfn-rel-local` fails.

Output: `results/<model-name>/<task-name>_predictions.parquet`, one row per test instance,
columns `[driverId, date, y_true, y_pred]`.

## Feed into the audit

From the parent repo's own venv (`relbench>=3.0`, not this one):

```bash
cd ..
uv run python scripts/run_audit.py --model lightgbm --predictions relarena_predict/results/lightgbm
uv run python scripts/run_audit.py --model tabpfn-rel-local --predictions relarena_predict/results/tabpfn-rel-local
```
