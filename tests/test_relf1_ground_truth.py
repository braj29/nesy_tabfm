"""Integration test: real rel-f1 data must satisfy (almost all of) our hand-specified
constraints. This validates the constraint *specs*, not just the engine -- if this test
fails, don't trust any violation rate computed against model predictions until it's fixed.

Requires network access on first run (downloads rel-f1 from the Hugging Face Hub, ~a few MB,
then cached under ~/.cache/huggingface).
"""

from __future__ import annotations

import pytest

from nesy_tabfm.constraints.check import run_constraints
from nesy_tabfm.data.relbench_loader import as_frames, get_full_db, load_rel_f1
from nesy_tabfm.specs.rel_f1 import db_level_constraints, derive_db_level_tables

relbench = pytest.importorskip("relbench")


@pytest.fixture(scope="module")
def tables():
    dataset = load_rel_f1()
    db_full = get_full_db(dataset)
    frames = as_frames(db_full)
    frames.update(derive_db_level_tables(db_full))
    return frames


def test_referential_integrity_holds_exactly(tables):
    results = [
        r for r in run_constraints(db_level_constraints(), tables) if r.kind == "referential_integrity"
    ]
    assert len(results) == 3
    for r in results:
        assert r.n_violations == 0, f"{r.name}: {r.n_violations}/{r.n_checked} violations"


def test_standings_unique_per_race_driver_holds_exactly(tables):
    [result] = [
        r
        for r in run_constraints(db_level_constraints(), tables)
        if r.name == "standings_unique_per_race_driver"
    ]
    assert result.n_violations == 0


def test_results_unique_per_race_driver_within_expected_baseline_noise(tables):
    """~0.3% baseline violation rate (85/25989 in 1950-1978) from historical F1 "shared
    drive" entries -- two drivers sharing one car, or one driver entered under two cars in
    the same nominal race -- a genuine data quirk, not a bug. See specs/rel_f1.py docstring."""
    [result] = [
        r
        for r in run_constraints(db_level_constraints(), tables)
        if r.name == "results_unique_per_race_driver"
    ]
    assert result.n_checked > 20_000
    assert result.rate < 0.01, f"results cardinality violation rate {result.rate:.3%} outside expected baseline"


def test_qualifying_before_race_holds_exactly(tables):
    [result] = [
        r for r in run_constraints(db_level_constraints(), tables) if r.name == "qualifying_before_race"
    ]
    assert result.n_violations == 0
    assert result.n_checked > 5000  # sanity: the join actually matched real rows


def test_points_non_negative_holds_exactly(tables):
    [result] = [
        r for r in run_constraints(db_level_constraints(), tables) if r.name == "points_non_negative"
    ]
    assert result.n_violations == 0


def test_season_points_consistency_within_expected_baseline_noise(tables):
    """rel-f1 has no sprint_results table, so sprint points (2021+) and one half-points race
    aren't recoverable from `results.points` alone -- see specs/rel_f1.py docstring. Real
    measured rate is ~2.5%; fail only if it drifts well outside that (data change or a
    regression in the derived-table logic)."""
    [result] = [
        r
        for r in run_constraints(db_level_constraints(), tables)
        if r.name == "season_points_consistency"
    ]
    assert result.n_checked > 1000
    assert result.rate < 0.05, f"season points violation rate {result.rate:.3%} outside expected baseline"


def test_fia_scoring_position_eligibility_holds_exactly(tables):
    """Domain-sourced (FIA regulations), not database-integrity -- see module docstring for
    the era-by-era cutoff table and the fastest-lap-bonus exception (`points == 1` exempted).
    Verified exact: 0/25989+ violations across all 74 seasons (1950-2023), once that one
    exception is accounted for -- unusually clean for a historical-data rule in this database,
    so this asserts equality, not a tolerance."""
    [result] = [
        r
        for r in run_constraints(db_level_constraints(), tables)
        if r.name == "fia_scoring_position_eligibility"
    ]
    assert result.n_checked > 20_000
    assert result.n_violations == 0, f"{result.n_violations}/{result.n_checked} violations"
