#!/usr/bin/env python
"""CLI entrypoint for an RQ1 constraint-violation audit run against rel-f1.

Every model's predictions are produced outside this repo, in whichever environment that
model actually needs (RelGNN on a GPU cluster, RelArena-alpha baselines locally via
relarena_predict/ -- see their READMEs), then read here as a directory of
"<task-name>_predictions.parquet" files:

    uv run python scripts/run_audit.py --model lightgbm --predictions relarena_predict/results/lightgbm
    uv run python scripts/run_audit.py --model tabpfn-rel-local --predictions relarena_predict/results/tabpfn-rel-local
    uv run python scripts/run_audit.py --model relgnn --predictions <cluster-output-dir>

--model is just a label: it names the output directory, results/rel_f1/<model>/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nesy_tabfm.audit.runner import run_audit
from nesy_tabfm.models.precomputed import PrecomputedAdapter

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="label for results/rel_f1/<model>/")
    parser.add_argument(
        "--predictions", required=True, help="directory of precomputed prediction parquets"
    )
    parser.add_argument("--results-root", default=str(REPO_ROOT / "results"))
    args = parser.parse_args()

    adapter = PrecomputedAdapter(args.predictions)
    findings_path = run_audit(adapter, args.model, args.results_root)
    print(f"wrote {findings_path}")


if __name__ == "__main__":
    main()
