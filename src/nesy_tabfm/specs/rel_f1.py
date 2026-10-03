"""Hand-specified constraints for the rel-f1 database (RelBench).

Every constraint below was checked against real rel-f1 data before being written down here
(see ``tests/test_relf1_ground_truth.py``), and two assumptions from the original design
were corrected as a result:

1. ``driver-top3``'s target is *qualifying* top-3 in a 30-day forward window (RelBench's own
   task SQL: ``MIN(qualifying.position) <= 3``), not race-finish top-3. A window can contain
   more than one qualifying session, so "exactly 3 top-3 predictions" only holds for
   single-session windows; multi-session windows get a ``[3, 3*n_sessions]`` range instead
   (a driver who podiums in qualifying more than once in the window is only counted once).
2. ``driver-dnf`` (``statusId != 1`` in any race within the window) and ``driver-top3`` are
   about *different sessions* (race vs. qualifying) and are not mutually exclusive -- a
   driver can qualify top-3 and then DNF the race. The cross-task "dnf and top3 can't both
   be 1" denial constraint from the original design was dropped for being logically false,
   not just hard to satisfy.

Two constraints have a real, non-zero baseline violation rate on ground truth -- both are
genuine historical-data quirks, not bugs in the constraint or the engine, so the ground-truth
tests assert a small tolerance rather than exact equality:

- ``season_points_consistency`` (~2.5%): rel-f1 has no ``sprint_results`` table, so
  sprint-race points (2021+) and the 2021 Belgian GP's half-points are reflected in
  ``standings.points`` but not recoverable by summing ``results.points`` alone.
- ``results_unique_per_race_driver`` (~0.3%, 85/25989, all 1950-1978): historical F1 allowed
  "shared drives" -- two drivers sharing one car mid-race, or one driver entered under two
  cars in the same nominal race (era of the Indianapolis 500 counting toward the World
  Championship) -- producing two legitimate ``results`` rows for the same (raceId, driverId).

A third, domain-sourced constraint (not a database-integrity one -- see the research
writeup's Tier 2, "sport rulebook"): FIA points-scoring eligibility. The naive version of
this rule ("points awarded only to the top 10") is simply wrong outside 2010-present -- the
eligible-position cutoff has changed several times across F1 history (top 5 through 1959,
top 6 from 1960-2002 -- 1967 is a single-year exception, see below -- top 8 2003-2009, top 10
2010+), so ``SCORING_POSITION_CUTOFF`` encodes the real per-era cutoff rather than hardcoding
"10". Even that isn't quite enough on its own: 1950-1959 (and, it turns out, 1967 too) awarded
a 1-point bonus for fastest lap *regardless of finishing position* -- 20 rows in the 1950s and
1 in 1967 score exactly 1 point while finishing outside that era's own top-N. Excluding
``points == 1`` from the position check (real drivers who legitimately scored more than the
fastest-lap bonus alone must still respect the era's top-N cutoff) makes the refined rule
(``points > 1 -> positionOrder <= cutoff(year)``) hold **exactly**, 0 violations across all
74 seasons (1950-2023) -- unusually clean for a historical-data constraint in this database.
"""

from __future__ import annotations

import pandas as pd

from nesy_tabfm.constraints.dsl import (
    AggregateConsistency,
    Cardinality,
    Constraint,
    DenialConstraint,
    ReferentialIntegrity,
    TemporalOrder,
)
from nesy_tabfm.data.relbench_loader import as_frames

TOP3_WINDOW_DAYS = 30
POSITION_DOMAIN = (1, 39)  # observed positionOrder range across all of rel-f1's history

#: Real FIA points-scoring eligibility by era: (first_year, last_year_inclusive, top_n). See
#: module docstring for how this was verified against real rel-f1 data (0 violations,
#: 1950-2023) once the fastest-lap-bonus exception (``points == 1``) is excluded.
SCORING_POSITION_CUTOFF = [
    (1950, 1959, 5),
    (1960, 2002, 6),  # 1967 is a single-year fastest-lap-bonus exception, not a different cutoff
    (2003, 2009, 8),
    (2010, 9999, 10),
]


def _scoring_cutoff(year: int) -> int:
    for start, end, cutoff in SCORING_POSITION_CUTOFF:
        if start <= year <= end:
            return cutoff
    raise ValueError(f"no scoring cutoff defined for year {year}")


def derive_db_level_tables(db_full) -> dict[str, pd.DataFrame]:
    """Extra tables needed by the aggregate-consistency and scoring-eligibility checks,
    derived from the raw ones."""
    frames = as_frames(db_full)
    races_round = frames["races"][["raceId", "year", "round"]]

    standings_yr = frames["standings"].merge(races_round, on="raceId")
    final_idx = standings_yr.groupby(["driverId", "year"])["round"].idxmax()
    season_final_standings = standings_yr.loc[final_idx, ["driverId", "year", "points"]]

    results_with_year = frames["results"].merge(races_round[["raceId", "year"]], on="raceId")[
        ["driverId", "year", "points"]
    ]

    results_with_cutoff = frames["results"].merge(races_round[["raceId", "year"]], on="raceId")[
        ["raceId", "driverId", "year", "positionOrder", "points"]
    ]
    results_with_cutoff["scoring_cutoff"] = results_with_cutoff["year"].map(_scoring_cutoff)

    return {
        "season_final_standings": season_final_standings.reset_index(drop=True),
        "results_with_year": results_with_year,
        "results_with_cutoff": results_with_cutoff.reset_index(drop=True),
    }


def db_level_constraints() -> list[Constraint]:
    """Ground-truth constraints checked against the raw rel-f1 tables (+ the derived tables
    from ``derive_db_level_tables``). These validate the constraint specs and the data, one
    of each of the 5 taxonomy types."""
    return [
        ReferentialIntegrity("results_fk_race", "results", "raceId", "races", "raceId"),
        ReferentialIntegrity("results_fk_driver", "results", "driverId", "drivers", "driverId"),
        ReferentialIntegrity(
            "results_fk_constructor", "results", "constructorId", "constructors", "constructorId"
        ),
        Cardinality("results_unique_per_race_driver", "results", ["raceId", "driverId"], exact=1),
        Cardinality(
            "standings_unique_per_race_driver", "standings", ["raceId", "driverId"], exact=1
        ),
        AggregateConsistency(
            "season_points_consistency",
            parent="season_final_standings",
            parent_key=["driverId", "year"],
            parent_val="points",
            child="results_with_year",
            child_key=["driverId", "year"],
            child_val="points",
        ),
        TemporalOrder(
            "qualifying_before_race",
            left_table="qualifying",
            left_time_col="date",
            right_table="races",
            right_time_col="date",
            join_on="raceId",
            relation="<=",
        ),
        DenialConstraint(
            "points_non_negative",
            left="results",
            predicate=lambda df: df["points"] < 0,
        ),
        DenialConstraint(
            "fia_scoring_position_eligibility",
            left="results_with_cutoff",
            predicate=lambda df: (df["points"] > 1) & (df["positionOrder"] > df["scoring_cutoff"]),
        ),
    ]


def top3_window_bounds(
    qualifying_full: pd.DataFrame, test_dates: pd.Series, window_days: int = TOP3_WINDOW_DAYS
) -> pd.DataFrame:
    """Per-cutoff-date cardinality bounds for the top3 headline check.

    Mirrors RelBench's own ``driver-top3`` window exactly: for each cutoff ``date``, count
    distinct qualifying sessions in ``(date, date + window_days]``. Exactly one session ->
    exactly 3 top-3 predictions must exist; more than one -> between 3 and ``3 * n_sessions``
    (a repeat podium finisher in the window is only ever one prediction row, so the count
    can't reach ``3 * n_sessions`` when podiums repeat, but can't drop below 3 either).
    """
    qualifying_full = qualifying_full[["raceId", "date"]]
    rows = []
    for date in pd.to_datetime(pd.Series(test_dates).unique()):
        window = qualifying_full[
            (qualifying_full["date"] > date)
            & (qualifying_full["date"] <= date + pd.Timedelta(days=window_days))
        ]
        n_sessions = window["raceId"].nunique()
        if n_sessions == 0:
            rows.append({"date": date, "exact": 0, "min": None, "max": None})
        elif n_sessions == 1:
            rows.append({"date": date, "exact": 3, "min": None, "max": None})
        else:
            rows.append({"date": date, "exact": None, "min": 3, "max": 3 * n_sessions})
    return pd.DataFrame(rows)


def top3_cardinality_constraint() -> Cardinality:
    """Headline prediction-level constraint: exactly 3 (or ``[3, 3*n_sessions]``) drivers
    predicted top3=1 per test window. Expects ``tables`` to contain a ``"predictions"`` table
    with ``date`` and ``top3_pred`` columns, a ``"test_windows"`` table with the full set of
    ``date`` values to check, and a ``"top3_bounds"`` table from ``top3_window_bounds``."""
    return Cardinality(
        "top3_predictions_per_window",
        table="predictions",
        group_cols="date",
        row_filter=lambda df: df["top3_pred"] == 1,
        universe_table="test_windows",
        bounds_table="top3_bounds",
    )


def position_domain_bounds(results_full: pd.DataFrame, test_dates: pd.Series, window_days: int) -> pd.DataFrame:
    """Per-cutoff-date realistic max position: the largest ``positionOrder`` actually run in
    ``(date, date + window_days]``. Tighter, and more meaningful, than a fixed global bound --
    grids shrank from 30+ cars in the 1950s-60s (``POSITION_DOMAIN``'s historical max of 39)
    to ~20-24 in the modern era rel-f1's test period covers, so a global bound has little
    power to catch an unrealistic prediction. ``window_days`` should be the audited task's own
    ``timedelta`` (e.g. ``driver-position``'s is 60 days) so the window matches exactly what
    the task's own label was computed over.
    """
    rows = []
    for date in pd.to_datetime(pd.Series(test_dates).unique()):
        window = results_full[
            (results_full["date"] > date) & (results_full["date"] <= date + pd.Timedelta(days=window_days))
        ]
        max_position = window["positionOrder"].max() if len(window) else None
        rows.append({"date": date, "max_position": max_position})
    return pd.DataFrame(rows)


def position_domain_constraint(min_val: float = POSITION_DOMAIN[0]) -> DenialConstraint:
    """Headline prediction-level constraint: a ``driver-position`` regression prediction must
    be at least ``min_val`` and at most that window's actual max ``positionOrder`` (see
    ``position_domain_bounds``, not a fixed global bound). Expects a ``"predictions"`` table
    with ``position_pred`` and ``max_position`` columns (the latter from
    ``position_domain_bounds``, merged in by the caller). ``Series.between`` already returns
    False (-> flagged) for NaN/±inf, so no separate finiteness check is needed."""
    return DenialConstraint(
        "position_pred_in_domain",
        left="predictions",
        predicate=lambda df: ~df["position_pred"].between(min_val, df["max_position"]),
    )
