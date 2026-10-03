import pandas as pd

from nesy_tabfm.constraints import (
    AggregateConsistency,
    Cardinality,
    DenialConstraint,
    FunctionalDependency,
    MonotonicityConstraint,
    ReferentialIntegrity,
    TemporalOrder,
)


def test_referential_integrity_flags_dangling_fk():
    drivers = pd.DataFrame({"driverId": [1, 2, 3]})
    results = pd.DataFrame({"resultId": [10, 11, 12], "driverId": [1, 2, 4]})
    tables = {"drivers": drivers, "results": results}

    c = ReferentialIntegrity("results_fk_driver", "results", "driverId", "drivers", "driverId")
    r = c.check(tables)

    assert r.n_checked == 3
    assert r.n_violations == 1
    assert r.violations["resultId"].tolist() == [12]
    assert r.rate == 1 / 3


def test_referential_integrity_nullable_excludes_nulls():
    drivers = pd.DataFrame({"driverId": [1, 2]})
    results = pd.DataFrame({"resultId": [1, 2, 3], "driverId": [1, None, 4]})
    tables = {"drivers": drivers, "results": results}

    c = ReferentialIntegrity(
        "results_fk_driver", "results", "driverId", "drivers", "driverId", nullable=True
    )
    r = c.check(tables)

    assert r.n_checked == 2  # the null row is excluded, not counted as checked or violated
    assert r.n_violations == 1
    assert r.violations["resultId"].tolist() == [3]


def test_cardinality_exact_flags_duplicates():
    results = pd.DataFrame(
        {
            "raceId": [1, 1, 1, 2],
            "driverId": [1, 2, 2, 1],  # (raceId=1, driverId=2) appears twice
        }
    )
    c = Cardinality("results_unique_per_race_driver", "results", ["raceId", "driverId"], exact=1)
    r = c.check({"results": results})

    assert r.n_checked == 3  # 3 distinct (raceId, driverId) groups
    assert r.n_violations == 1
    assert r.violations.iloc[0][["raceId", "driverId"]].tolist() == [1, 2]


def test_cardinality_top3_per_race_with_universe_catches_zero_count_group():
    races = pd.DataFrame({"raceId": [1, 2, 3]})
    predictions = pd.DataFrame(
        {
            "raceId": [1, 1, 1, 2, 2],
            "driverId": [10, 11, 12, 13, 14],
            "top3_pred": [1, 1, 1, 1, 0],  # race 1: exactly 3 (ok); race 2: only 1 (violation)
            # race 3 has no rows at all in predictions -> 0 top3 predictions (violation)
        }
    )
    c = Cardinality(
        "exactly_3_top3_per_race",
        table="predictions",
        group_cols="raceId",
        exact=3,
        row_filter=lambda df: df["top3_pred"] == 1,
        universe_table="races",
    )
    r = c.check({"races": races, "predictions": predictions})

    assert r.n_checked == 3
    assert set(r.violations["raceId"]) == {2, 3}
    assert r.violations.set_index("raceId").loc[3, "count"] == 0


def test_cardinality_per_group_bounds_table():
    # windows spanning a variable number of qualifying sessions: exactly 3 top-3s per
    # single-session window, a wider [3, 3*n_sessions] range for multi-session windows.
    predictions = pd.DataFrame(
        {
            "date": ["d1", "d1", "d1", "d2", "d2", "d3"],
            "top3_pred": [1, 1, 1, 1, 1, 1],  # d1: 3 (ok); d2: 2 (violates exact=3); d3: 1
        }
    )
    windows = pd.DataFrame({"date": ["d1", "d2", "d3"]})
    bounds = pd.DataFrame({"date": ["d1", "d2", "d3"], "exact": [3, 3, None], "max": [None, None, 6]})

    c = Cardinality(
        "top3_per_window",
        table="predictions",
        group_cols="date",
        row_filter=lambda df: df["top3_pred"] == 1,
        universe_table="windows",
        bounds_table="bounds",
    )
    r = c.check({"predictions": predictions, "windows": windows, "bounds": bounds})

    assert r.n_checked == 3
    assert list(r.violations["date"]) == ["d2"]


def test_aggregate_consistency_flags_mismatched_sum():
    standings = pd.DataFrame(
        {"driverId": [1, 1, 2], "year": [2023, 2024, 2023], "points": [30, 40, 15]}
    )
    results = pd.DataFrame(
        {
            "driverId": [1, 1, 1, 2, 2],
            "year": [2023, 2023, 2024, 2023, 2023],
            "points": [10, 20, 40, 5, 5],  # driver 2/2023: sums to 10, standings says 15
        }
    )
    c = AggregateConsistency(
        "season_points_consistency",
        parent="standings",
        parent_key=["driverId", "year"],
        parent_val="points",
        child="results",
        child_key=["driverId", "year"],
        child_val="points",
    )
    r = c.check({"standings": standings, "results": results})

    assert r.n_checked == 3
    assert r.n_violations == 1
    row = r.violations.iloc[0]
    assert (row["driverId"], row["year"]) == (2, 2023)


def test_aggregate_consistency_missing_child_rows_treated_as_zero():
    standings = pd.DataFrame({"driverId": [9], "year": [2023], "points": [0]})
    results = pd.DataFrame({"driverId": [], "year": [], "points": []})
    c = AggregateConsistency(
        "season_points_consistency",
        parent="standings",
        parent_key=["driverId", "year"],
        parent_val="points",
        child="results",
        child_key=["driverId", "year"],
        child_val="points",
    )
    r = c.check({"standings": standings, "results": results})
    assert r.n_violations == 0


def test_temporal_order_flags_driver_racing_before_birth():
    drivers = pd.DataFrame(
        {"driverId": [1, 2], "dob": pd.to_datetime(["1990-01-01", "2021-01-01"])}
    )
    race_dates = pd.DataFrame(
        {
            "driverId": [1, 1, 2],
            "date": pd.to_datetime(["2020-01-01", "2021-01-01", "2020-06-01"]),
        }
    )
    c = TemporalOrder(
        "driver_born_before_race",
        left_table="drivers",
        left_time_col="dob",
        right_table="race_dates",
        right_time_col="date",
        join_on="driverId",
        relation="<=",
    )
    r = c.check({"drivers": drivers, "race_dates": race_dates})

    assert r.n_checked == 3
    assert r.n_violations == 1
    assert r.violations.iloc[0]["driverId"] == 2


def test_functional_dependency_flags_disagreement_with_group_majority():
    # mirrors the writeup's example: ZipCode -> City should hold, but one row has a typo'd
    # city for an otherwise-consistent zip code.
    patients = pd.DataFrame(
        {
            "patientId": [1, 2, 3, 4],
            "zip": ["10001", "10001", "10001", "20002"],
            "city": ["New York", "New York", "New Yrok", "Washington"],  # id 3: typo
        }
    )
    c = FunctionalDependency("zip_determines_city", table="patients", determinant="zip", dependent="city")
    r = c.check({"patients": patients})

    assert r.n_checked == 4
    assert r.n_violations == 1
    assert r.violations.iloc[0]["patientId"] == 3


def test_functional_dependency_holds_exactly_when_no_disagreement():
    docs = pd.DataFrame({"org": ["A", "A", "B"], "division": ["10", "10", "20"]})
    c = FunctionalDependency("org_determines_division", table="docs", determinant="org", dependent="division")
    r = c.check({"docs": docs})

    assert r.n_checked == 3
    assert r.n_violations == 0


def test_functional_dependency_composite_determinant():
    rows = pd.DataFrame(
        {
            "country": ["US", "US", "US", "DE"],
            "region": ["CA", "CA", "NY", "CA"],  # region codes aren't globally unique
            "tax_rate": [7.25, 7.25, 4.0, 19.0],
        }
    )
    c = FunctionalDependency(
        "country_region_determines_tax_rate",
        table="rows",
        determinant=["country", "region"],
        dependent="tax_rate",
    )
    r = c.check({"rows": rows})

    assert r.n_checked == 4
    assert r.n_violations == 0  # (US, CA) and (DE, CA) are different groups, no collision


def test_denial_constraint_unary_flags_out_of_domain_prediction():
    predictions = pd.DataFrame({"driverId": [1, 2, 3], "position_pred": [5.0, 25.0, float("nan")]})
    c = DenialConstraint(
        "position_pred_in_domain",
        left="predictions",
        predicate=lambda df: ~df["position_pred"].between(1, 20) | df["position_pred"].isna(),
    )
    r = c.check({"predictions": predictions})

    assert r.n_checked == 3
    assert set(r.violations["driverId"]) == {2, 3}


def test_denial_constraint_self_join_tax_bracket_example():
    # mirrors the writeup's example: no two employees in the same bracket where the one
    # with lower salary pays more tax.
    employees = pd.DataFrame(
        {
            "empId": [1, 2, 3],
            "bracket": ["A", "A", "B"],
            "salary": [50_000, 60_000, 70_000],
            "tax": [8_000, 7_000, 9_000],  # emp 1 earns less than emp 2 but pays more tax
        }
    )
    c = DenialConstraint(
        "no_tax_inversion_within_bracket",
        left="employees",
        right="employees",
        join_on="bracket",
        predicate=lambda df: (df["salary_l"] < df["salary_r"]) & (df["tax_l"] > df["tax_r"]),
        dedupe_cols=("empId_l", "empId_r"),
    )
    r = c.check({"employees": employees})

    assert r.n_checked == 1  # only one same-bracket pair exists: (1, 2)
    assert r.n_violations == 1


def test_denial_constraint_cross_table_dnf_and_top3_are_mutually_exclusive():
    dnf_preds = pd.DataFrame(
        {"driverId": [1, 2, 3], "raceId": [100, 100, 100], "dnf_pred": [1, 0, 0]}
    )
    top3_preds = pd.DataFrame(
        {"driverId": [1, 2, 3], "raceId": [100, 100, 100], "top3_pred": [1, 1, 0]}
    )
    c = DenialConstraint(
        "dnf_and_top3_mutually_exclusive",
        left="dnf_preds",
        right="top3_preds",
        join_on=["driverId", "raceId"],
        predicate=lambda df: (df["dnf_pred"] == 1) & (df["top3_pred"] == 1),
    )
    r = c.check({"dnf_preds": dnf_preds, "top3_preds": top3_preds})

    assert r.n_checked == 3
    assert r.n_violations == 1
    assert r.violations.iloc[0]["driverId"] == 1


def test_monotonicity_constraint_flags_a_prediction_that_moves_the_wrong_way():
    # id 1: risk score improved (60 -> 75) but predicted creditworthiness went DOWN -- a
    # violation. id 2: same improvement, prediction correctly went up -- fine.
    original = pd.DataFrame({"id": [1, 2], "y_pred": [0.5, 0.5]})
    perturbed = pd.DataFrame({"id": [1, 2], "y_pred": [0.4, 0.6]})

    c = MonotonicityConstraint("risk_score_monotonic", id_col="id", direction="non_decreasing")
    r = c.check({"original": original, "perturbed": perturbed})

    assert r.n_checked == 2
    assert r.n_violations == 1
    assert r.violations.iloc[0]["id"] == 1


def test_monotonicity_constraint_non_increasing_direction():
    # a "should hurt" feature (e.g. more late payments): prediction must not go UP.
    original = pd.DataFrame({"id": [1, 2], "y_pred": [0.5, 0.5]})
    perturbed = pd.DataFrame({"id": [1, 2], "y_pred": [0.6, 0.4]})

    c = MonotonicityConstraint("late_payments_monotonic", id_col="id", direction="non_increasing")
    r = c.check({"original": original, "perturbed": perturbed})

    assert r.n_violations == 1
    assert r.violations.iloc[0]["id"] == 1


def test_monotonicity_constraint_tolerance_absorbs_tiny_moves():
    original = pd.DataFrame({"id": [1], "y_pred": [0.5]})
    perturbed = pd.DataFrame({"id": [1], "y_pred": [0.5 - 1e-12]})  # float noise, not a real move

    c = MonotonicityConstraint("x", id_col="id", direction="non_decreasing")
    r = c.check({"original": original, "perturbed": perturbed})

    assert r.n_violations == 0
