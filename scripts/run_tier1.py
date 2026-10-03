#!/usr/bin/env python
"""Tier 1 (DB integrity) experiments: run every Tier 1 ground-truth constraint on rel-f1 and
rel-salt, then compile ``results/tier1_db_integrity.md`` from those plus the prediction-level /
HELOC CSVs already written by the model audits (run those first; anything missing is listed as
"not run" in the report rather than silently skipped):

    uv run python scripts/run_tier1.py
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from nesy_tabfm.constraints.check import run_constraints, summarize
from nesy_tabfm.data.relbench_loader import as_frames, get_full_db, load_rel_f1, load_salt
from nesy_tabfm.progress import get_logger, setup_run_logging, timed
from nesy_tabfm.specs import rel_f1, salt
from nesy_tabfm.specs.tiers import tier1_only, tier_of

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_db_level(out_dir: Path) -> dict[str, pd.DataFrame]:
    logger = get_logger()
    summaries: dict[str, pd.DataFrame] = {}

    with timed("load rel-f1"):
        db = get_full_db(load_rel_f1())
    tables = as_frames(db)
    tables.update(rel_f1.derive_db_level_tables(db))
    results = run_constraints(tier1_only(rel_f1.db_level_constraints()), tables, progress=True, desc="rel-f1 Tier 1")
    summaries["rel-f1"] = summarize(results)

    with timed("load rel-salt"):
        db = get_full_db(load_salt())
    tables = as_frames(db)
    results = run_constraints(tier1_only(salt.db_level_constraints()), tables, progress=True, desc="rel-salt Tier 1")
    summaries["rel-salt"] = summarize(results)

    for name, df in summaries.items():
        path = out_dir / f"db_level_{name}.csv"
        df.to_csv(path, index=False)
        logger.info("wrote %s (%d constraints)", path, len(df))
    return summaries


def _read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")


def _table(df: pd.DataFrame) -> str:
    cols = ["name", "kind", "n_checked", "n_violations", "violation_rate"]
    lines = ["| constraint | type | checked | violations | rate |", "|---|---|---:|---:|---:|"]
    for r in df[cols].itertuples(index=False):
        lines.append(f"| `{r.name}` | {r.kind} | {r.n_checked:,} | {r.n_violations:,} | {r.violation_rate:.4%} |")
    return "\n".join(lines)


NOTES = """## 5. Read this before citing any number

- **Baseline noise is real.** Three ground-truth constraints are not exact and the model-level
  numbers should be read against them: `results_unique_per_race_driver` 0.33% (historical F1
  shared drives, 1950-1978), `season_points_consistency` 2.49% (rel-f1 has no sprint-results
  table), and the SAP functional dependencies 0.0006-0.28% (real business-config data, not
  schema-declared keys). A model rate near these is not evidence of a model problem.
- **TabPFN on rel-salt is a feasibility trial, not a full-power baseline.** It uses an 8,000-row
  training subsample and a 1,000-row test subsample (LightGBM uses the full ~1.6M-row train and
  ~400K/89K-row test sets), `ignore_pretraining_limits=True`, and a smaller feature set (five
  near-unique ID columns dropped). Its 1,000-row FD check covers 1,000 rows against LightGBM's
  402,855, so the two rates are not directly comparable in statistical power. `sales-group`
  (502 classes) hits TabPFN's hard 160-class ceiling and is excluded, so the
  `salesgroup_pred_determines_salesoffice_pred` check has no TabPFN row.
- **The LightGBM `shippingpoint_pred_determines_plant_pred` rate (27.29%) tracks a weak model, not
  a subtle constraint failure.** LightGBM's `item-shippoint` accuracy is 2.97%, below the 48.2%
  majority-class baseline; raising `n_estimators` 200->800 changed nothing. The cause is not
  established (unseen-customer difficulty is a hypothesis, untested).
- **rel-f1 `tabpfn-rel-local` predictions are now a clean, single run** (`predict.py --models
  tabpfn-rel-local`, started 2026-09-25 09:12, all 3 tasks written by 2026-09-27 08:12 -- the
  ~2-day spread is the machine sleeping between checks, not 2 days of compute: total CPU time
  was under an hour). Superseded the earlier mixed-provenance run (driver-dnf/driver-top3 from
  Sept 13, driver-position from a run started outside this session); the audit was re-run
  against it and the numbers were unchanged (36.7% / 11 of 30 windows for
  `top3_predictions_per_window`), so the earlier mixed files weren't actually masking a
  different result here -- but treat that as this instance, not a general guarantee.
  With only 30 test windows, 36.7% vs LightGBM's 13.3% is 11 vs 4 windows.
- **Prediction-level dates in section 2 are the date the audit last ran**, not when the model
  was trained (for rel-f1, see the previous point).
- **Timings vary a lot with machine load.** On this MacBook Air the same rel-salt TabPFN task
  took 100-960s and LightGBM `sales-payterms` 821s (vs ~105s in an earlier run) while the machine
  was under load (load average 13, swap in use, another app at 100% CPU). Per-task times in
  `results/salt/*/run.log` are real but not a benchmark.
- **HELOC TabPFN takes ~8 minutes**, not the ~2 minutes I first estimated.
"""


def compile_report(results_root: Path, db_level: dict[str, pd.DataFrame]) -> Path:
    out = ["# Tier 1 (DB integrity) results", "", f"Generated {datetime.now():%Y-%m-%d %H:%M}. Tier 1 = generic relational/statistical",
           "invariants; the domain-sourced Tier 2 constraints are excluded here (see `specs/tiers.py`).", ""]

    out += ["## 1. Ground-truth (DB-level) checks", "",
            "Run fresh by `scripts/run_tier1.py` against the full real databases. These validate that each",
            "constraint *spec* actually holds in real data before it is used to audit any model.", ""]
    for name, df in db_level.items():
        out += [f"### {name}", "", _table(df), ""]

    out += ["## 2. Prediction-level checks (Tier 1 only)", "",
            "Read from the CSVs the model audits wrote; the file date is shown so stale results are visible.", ""]
    pred_sources = [
        ("rel-f1", "lightgbm", results_root / "rel_f1/lightgbm/prediction_violation_rates.csv"),
        ("rel-f1", "tabpfn-rel-local", results_root / "rel_f1/tabpfn-rel-local/prediction_violation_rates.csv"),
        ("rel-salt", "lightgbm", results_root / "salt/lightgbm/prediction_violation_rates.csv"),
        ("rel-salt", "tabpfn", results_root / "salt/tabpfn/prediction_violation_rates.csv"),
    ]
    for dataset, model, path in pred_sources:
        df = _read(path)
        out.append(f"### {dataset} / {model}")
        out.append("")
        if df is None:
            out += ["_not run (no results file)_", ""]
            continue
        df = df[[tier_of(n) == 1 for n in df["name"]]]
        out += [f"_source: `{path.relative_to(results_root.parent)}`, written {_mtime(path)}_", "", _table(df), ""]

    out += ["## 3. Monotonicity (HELOC)", "",
            "A violation = predicted creditworthiness moved the *wrong way* when one feature was nudged in its",
            "'should help' direction, all else fixed.", ""]
    for model in ("lightgbm", "tabpfn"):
        path = results_root / f"heloc/{model}/monotonicity_violation_rates.csv"
        df = _read(path)
        out += [f"### HELOC / {model}", ""]
        out += ["_not run_", ""] if df is None else [f"_written {_mtime(path)}_", "", _table(df), ""]

    out += ["## 4. Context: rel-salt model accuracy behind the prediction-level numbers", "",
            "A violation rate is only interpretable next to how good the predictions are.", ""]
    for model in ("lightgbm", "tabpfn"):
        path = results_root / f"salt/{model}/accuracy_by_task.csv"
        df = _read(path)
        out += [f"### {model}", ""]
        if df is None:
            out += ["_not run_", ""]
            continue
        out += ["| task | target | n | accuracy |", "|---|---|---:|---:|"]
        out += [f"| {r.task} | {r.target_col} | {r.n:,} | {r.accuracy:.2%} |" for r in df.itertuples(index=False)]
        out.append("")

    out.append(NOTES)
    path = results_root / "tier1_db_integrity.md"
    path.write_text("\n".join(out))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(REPO_ROOT / "results"))
    args = parser.parse_args()
    results_root = Path(args.results_root)
    out_dir = results_root / "tier1"
    setup_run_logging(out_dir)
    db_level = run_db_level(out_dir)
    path = compile_report(results_root, db_level)
    get_logger().info("wrote %s", path)


if __name__ == "__main__":
    main()
