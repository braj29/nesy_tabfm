#!/usr/bin/env python
"""Run RelArena-alpha's CPU baselines on rel-f1's 3 official tasks and dump per-instance
predictions in the schema nesy_tabfm's PrecomputedAdapter expects:
``[entity_col, time_col, "y_true", "y_pred"]`` parquet files, one per task, under
``results/<model_name>/``.

Deliberately thin: relarena's own `run_experiment` already does the tuning / selection /
final-fit correctly (nested temporal validation -- see relarena's docs/temporal-validation.md);
this script only translates its output into the schema the audit side consumes.

    uv run python predict.py                       # both models, all 3 tasks
    uv run python predict.py --models lightgbm      # skip tabpfn-rel-local (needs TABPFN_TOKEN)
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import relarena.models  # noqa: F401 -- import side effect: registers built-in models
from relarena.registry import registry
from relarena.runner import run_experiment
from relbench.base import TaskType
from relbench.tasks import get_task  # relarena's pinned relbench==2.1.2's own (pre-rewrite) API

def log(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S} | {msg}", flush=True)


MODELS = ["lightgbm", "tabpfn-rel-local"]
TASKS = ["driver-position", "driver-dnf", "driver-top3"]

REPO_ROOT = Path(__file__).resolve().parent


def predict_one(model_name: str, task_name: str, out_dir: Path, n_trials: int) -> None:
    model_cls = registry.get(model_name)
    log(f"{model_name}/{task_name}: run_experiment (n_trials={n_trials}) -- tuning + refit, can take minutes")
    t0 = time.time()
    summary = run_experiment(model_cls, "rel-f1", task_name, n_trials=n_trials)
    log(f"{model_name}/{task_name}: run_experiment finished in {time.time() - t0:.0f}s")
    trial = summary.tuned or summary.default
    if trial is None or trial.test_pred is None:
        raise RuntimeError(
            f"{model_name} on rel-f1/{task_name}: no successful trial with cached "
            f"predictions (see the exception logged during the run above)"
        )

    task = get_task("rel-f1", task_name, download=True)
    test_df = task.get_table("test", mask_input_cols=False).df

    y_pred = trial.test_pred
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        # relarena keeps test_pred as a continuous score (it scores roc_auc on this) --
        # our adapter contract (models/base.py) is a hard 0/1 prediction, which is what the
        # constraint specs threshold on directly (e.g. counting predicted top3_pred == 1).
        y_pred = (y_pred >= 0.5).astype(int)

    out = pd.DataFrame(
        {
            task.entity_col: test_df[task.entity_col].to_numpy(),
            task.time_col: test_df[task.time_col].to_numpy(),
            "y_true": test_df[task.target_col].to_numpy(),
            "y_pred": y_pred,
        }
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task_name}_predictions.parquet"
    out.to_parquet(out_path)
    log(f"{model_name}/{task_name}: wrote {out_path} ({len(out)} rows, test_score={trial.test_score})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=MODELS, choices=MODELS)
    parser.add_argument("--tasks", nargs="*", default=TASKS, choices=TASKS)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="HPO trials per (model, task); 0 = just the zero-tuning default config, "
        "for a fast first look instead of relarena's full default search",
    )
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    n_total = len(args.models) * len(args.tasks)
    n_done = 0
    for model_name in args.models:
        log(f"=== {model_name} ===")
        for task_name in args.tasks:
            log(f"[{n_done}/{n_total} done] starting {model_name}/{task_name}")
            try:
                predict_one(model_name, task_name, out_root / model_name, args.n_trials)
            except Exception as e:  # noqa: BLE001 -- keep going across (model, task) pairs
                log(f"{model_name}/{task_name}: FAILED: {e}")
            n_done += 1


if __name__ == "__main__":
    main()
