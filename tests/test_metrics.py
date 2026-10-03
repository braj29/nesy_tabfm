import pandas as pd

from nesy_tabfm.constraints.dsl import ViolationResult
from nesy_tabfm.constraints.metrics import (
    instance_violation_flags,
    pairwise_instance_violation_flags,
    stratified_rate,
    temporal_buckets,
    violation_error_correlation,
)


def test_instance_violation_flags_marks_only_matching_instances():
    instances = pd.DataFrame({"raceId": [1, 1, 2, 3], "driverId": [10, 11, 12, 13]})
    violations = pd.DataFrame({"raceId": [2]})
    result = ViolationResult("c", "cardinality", n_checked=3, violations=violations)

    flags = instance_violation_flags(result, instances, join_cols=["raceId"])

    assert list(flags) == [False, False, True, False]


def test_pairwise_instance_violation_flags_marks_both_sides_of_a_violating_pair():
    instances = pd.DataFrame({"ID": [1, 2, 3, 4]})
    violations = pd.DataFrame({"ID_l": [2], "ID_r": [3]})
    result = ViolationResult("c", "denial", n_checked=5, violations=violations)

    flags = pairwise_instance_violation_flags(result, instances, id_col="ID")

    assert list(flags) == [False, True, True, False]


def test_pairwise_instance_violation_flags_all_false_when_no_violations():
    instances = pd.DataFrame({"ID": [1, 2]})
    result = ViolationResult("c", "denial", n_checked=1, violations=pd.DataFrame({"ID_l": [], "ID_r": []}))
    flags = pairwise_instance_violation_flags(result, instances, id_col="ID")
    assert list(flags) == [False, False]


def test_instance_violation_flags_all_false_when_no_violations():
    instances = pd.DataFrame({"raceId": [1, 2]})
    result = ViolationResult("c", "cardinality", n_checked=2, violations=pd.DataFrame({"raceId": []}))
    flags = instance_violation_flags(result, instances, join_cols=["raceId"])
    assert list(flags) == [False, False]


def test_temporal_buckets_splits_into_equal_sized_groups():
    ts = pd.Series(pd.date_range("2020-01-01", periods=8, freq="D"))
    buckets = temporal_buckets(ts, n_bins=4)
    counts = buckets.value_counts().sort_index()
    assert list(counts) == [2, 2, 2, 2]
    assert list(counts.index) == ["bucket_1", "bucket_2", "bucket_3", "bucket_4"]


def test_stratified_rate_computes_rate_per_group():
    flags = [True, False, True, True]
    group = pd.Series(["early", "early", "late", "late"])
    out = stratified_rate(flags, group, group_name="shift_bucket")

    early = out.set_index("shift_bucket").loc["early"]
    late = out.set_index("shift_bucket").loc["late"]
    assert early["n"] == 2 and early["n_violations"] == 1 and early["violation_rate"] == 0.5
    assert late["n"] == 2 and late["n_violations"] == 2 and late["violation_rate"] == 1.0


def test_violation_error_correlation_positive_when_violations_track_error():
    flags = [False, False, True, True]
    error = [0.1, 0.2, 5.0, 6.0]
    corr = violation_error_correlation(flags, error)
    assert corr > 0.9


def test_violation_error_correlation_nan_when_no_variance():
    flags = [False, False, False]
    error = [0.1, 0.2, 0.3]
    corr = violation_error_correlation(flags, error)
    assert pd.isna(corr)
