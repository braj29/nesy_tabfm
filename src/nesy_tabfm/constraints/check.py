"""Run a set of constraints against one snapshot of tables and summarize the results."""

from __future__ import annotations

import time

import pandas as pd

from nesy_tabfm.constraints.dsl import Constraint, ViolationResult
from nesy_tabfm.progress import get_logger, track


def run_constraints(
    constraints: list[Constraint], tables: dict[str, pd.DataFrame], progress: bool = False, desc: str = "constraints"
) -> list[ViolationResult]:
    """With ``progress=True``: a tqdm bar over the constraints plus one log line each with its
    runtime, rows checked, and violation count (some checks take seconds on multi-million-row
    tables, so silence looks like a hang)."""
    if not progress:
        return [c.check(tables) for c in constraints]

    logger = get_logger()
    results = []
    bar = track(constraints, desc=desc)
    for c in bar:
        bar.set_postfix_str(c.name[:40])
        t0 = time.time()
        r = c.check(tables)
        logger.info(
            "%-55s %-22s checked=%-10d violations=%-8d rate=%.6f (%.1fs)",
            c.name,
            r.kind,
            r.n_checked,
            r.n_violations,
            r.rate,
            time.time() - t0,
        )
        results.append(r)
    return results


def summarize(results: list[ViolationResult]) -> pd.DataFrame:
    """One row per constraint: name, kind, n_checked, n_violations, violation rate."""
    return pd.DataFrame(
        [
            {
                "name": r.name,
                "kind": r.kind,
                "n_checked": r.n_checked,
                "n_violations": r.n_violations,
                "violation_rate": r.rate,
            }
            for r in results
        ]
    )
