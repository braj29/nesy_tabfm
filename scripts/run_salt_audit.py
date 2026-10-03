#!/usr/bin/env python
"""CLI entrypoint for an RQ1 constraint-violation audit run against rel-salt.

Unlike rel-f1, the model here runs in-process (no separate environment/predictions
directory needed) -- SaltLGBMAdapter is plain LightGBM against this repo's own relbench>=3.0:

    uv run python scripts/run_salt_audit.py --model lightgbm
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nesy_tabfm.audit.salt_runner import run_audit

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_adapter(args: argparse.Namespace):
    if args.model == "lightgbm":
        from nesy_tabfm.models.salt_lgbm import SaltLGBMAdapter

        return SaltLGBMAdapter()
    if args.model == "tabpfn":
        from nesy_tabfm.models.salt_tabpfn import SaltTabPFNAdapter

        tabpfn_kwargs = {}
        if args.tabpfn_device:
            tabpfn_kwargs["device"] = args.tabpfn_device
        return SaltTabPFNAdapter(
            train_sample_size=None if args.train_sample_size <= 0 else args.train_sample_size,
            test_sample_size=None if args.test_sample_size <= 0 else args.test_sample_size,
            seed=args.seed,
            **tabpfn_kwargs,
        )
    raise SystemExit(f"unknown model {args.model!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["lightgbm", "tabpfn"])
    parser.add_argument("--results-root", default=str(REPO_ROOT / "results"))
    parser.add_argument(
        "--tabpfn-device",
        default=None,
        help="TabPFN device override for --model tabpfn, e.g. 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--train-sample-size",
        type=int,
        default=8000,
        help="TabPFN context rows for rel-salt; use 0 for the full train split.",
    )
    parser.add_argument(
        "--test-sample-size",
        type=int,
        default=1000,
        help="TabPFN test rows for rel-salt; use 0 for the full test split.",
    )
    parser.add_argument("--seed", type=int, default=0, help="TabPFN sampling seed.")
    args = parser.parse_args()

    adapter = build_adapter(args)
    findings_path = run_audit(adapter, args.model, args.results_root)
    print(f"wrote {findings_path}")


if __name__ == "__main__":
    main()
