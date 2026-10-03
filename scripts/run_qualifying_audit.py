#!/usr/bin/env python
"""CLI entrypoint for the rel-f1 qualifying-position permutation audit: within a race, no
two drivers' predicted (rounded) qualifying positions may collide.

    uv run python scripts/run_qualifying_audit.py --model lightgbm
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nesy_tabfm.constraints.check import summarize
from nesy_tabfm.data.relbench_loader import get_full_db, get_model_visible_db, load_rel_f1
from nesy_tabfm.specs.rel_f1_autocomplete import position_no_duplicates_constraint

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_adapter(name: str):
    if name == "lightgbm":
        from nesy_tabfm.models.qualifying_lgbm import QualifyingLGBMAdapter

        return QualifyingLGBMAdapter()
    raise SystemExit(f"unknown model {name!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["lightgbm"])
    parser.add_argument("--results-root", default=str(REPO_ROOT / "results"))
    args = parser.parse_args()

    dataset = load_rel_f1()
    db_visible = get_model_visible_db(dataset)
    db_full = get_full_db(dataset)
    task = dataset.load_task("qualifying-position")

    adapter = build_adapter(args.model)
    preds = adapter.fit_predict(task, db_visible, db_full)
    preds["position_pred_rounded"] = preds["y_pred"].round()

    result = position_no_duplicates_constraint("qualifying_position_no_duplicates_per_race").check(
        {"predictions": preds}
    )
    print(f"n_checked (race, position) groups: {result.n_checked}")
    print(f"n_violations: {result.n_violations} ({result.rate:.2%})")

    out_dir = Path(args.results_root) / "rel_f1" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out_dir / "qualifying-position_predictions.parquet")
    out_path = out_dir / "qualifying_position_permutation_violation_rate.csv"
    summarize([result]).to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
