#!/usr/bin/env python
"""CLI entrypoint for the HELOC monotonicity audit: does a model's predicted
creditworthiness ever move the wrong way when a feature is nudged in its "should help"
direction, holding everything else fixed?

    uv run python scripts/run_heloc_audit.py --model lightgbm
    uv run python scripts/run_heloc_audit.py --model tabpfn   # needs TABPFN_TOKEN
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from nesy_tabfm.constraints.check import summarize
from nesy_tabfm.data.heloc_loader import load_heloc
from nesy_tabfm.progress import get_logger, setup_run_logging, timed, track
from nesy_tabfm.specs.heloc import FEATURES, monotonicity_constraint, perturb

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_adapter(args: argparse.Namespace):
    name = args.model
    if name == "lightgbm":
        from nesy_tabfm.models.heloc_lgbm import HelocLGBMAdapter

        return HelocLGBMAdapter()
    if name == "tabpfn":
        from nesy_tabfm.models.heloc_tabpfn import HelocTabPFNAdapter

        tabpfn_kwargs = {}
        if args.tabpfn_device:
            tabpfn_kwargs["device"] = args.tabpfn_device
        return HelocTabPFNAdapter(**tabpfn_kwargs)
    raise SystemExit(f"unknown model {name!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["lightgbm", "tabpfn"])
    parser.add_argument("--results-root", default=str(REPO_ROOT / "results"))
    parser.add_argument(
        "--tabpfn-device",
        default=None,
        help="TabPFN device override for --model tabpfn, e.g. 'cpu' or 'cuda'.",
    )
    args = parser.parse_args()

    out_dir = Path(args.results_root) / "heloc" / args.model
    setup_run_logging(out_dir)
    logger = get_logger()
    logger.info("HELOC monotonicity audit, model=%s", args.model)

    with timed("load HELOC"):
        x, y = load_heloc()
    x_train, x_test, y_train, _ = train_test_split(
        x, y, test_size=0.2, random_state=0, stratify=y
    )
    x_test = x_test.reset_index(drop=True)
    x_test["_row_id"] = range(len(x_test))

    adapter = build_adapter(args)
    logger.info("train=%d rows, test=%d rows", len(x_train), len(x_test))
    with timed(f"fit {args.model}"):
        adapter.fit(x_train, y_train)

    results = []
    for feature in track(FEATURES, desc=f"HELOC features [{args.model}]"):
        x_test_features = x_test.drop(columns=["_row_id"])
        x_pert, valid_mask = perturb(x_test_features, feature)

        y_pred_orig = adapter.predict_proba_positive(x_test_features.loc[valid_mask])
        y_pred_pert = adapter.predict_proba_positive(x_pert.loc[valid_mask])
        row_ids = x_test.loc[valid_mask, "_row_id"].to_numpy()

        original = pd.DataFrame({"_row_id": row_ids, "y_pred": y_pred_orig})
        perturbed = pd.DataFrame({"_row_id": row_ids, "y_pred": y_pred_pert})

        result = monotonicity_constraint(feature).check(
            {"original": original, "perturbed": perturbed}
        )
        results.append(result)
        logger.info(
            "%s: %d/%d violations (%.2f%%) -- %s",
            feature.name, result.n_violations, result.n_checked, 100 * result.rate, feature.note,
        )

    out_path = out_dir / "monotonicity_violation_rates.csv"
    summarize(results).to_csv(out_path, index=False)
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
