"""Integration test: real HELOC data must show the expected direction for each feature in
specs/heloc.py's FEATURES list. Same purpose as the other test_*_ground_truth.py files --
validate the constraint *spec* against real data before trusting any violation rate computed
against model predictions.

Requires network access on first run (downloads HELOC from OpenML, cached under
~/.openml).
"""

from __future__ import annotations

import pytest

pytest.importorskip("openml")

from nesy_tabfm.data.heloc_loader import load_heloc  # noqa: E402
from nesy_tabfm.specs.heloc import FEATURES  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return load_heloc()


def test_each_feature_correlates_with_target_in_the_expected_direction(data):
    x, y = data
    for feature in FEATURES:
        valid = x[feature.name].notna()
        corr = x.loc[valid, feature.name].corr(y[valid])
        expected_sign = 1 if feature.delta > 0 else -1
        assert corr * expected_sign > 0.1, (
            f"{feature.name}: corr={corr:.3f} does not clearly support the direction implied "
            f"by delta={feature.delta} (expected sign {expected_sign})"
        )


def test_sentinel_values_are_cleaned_to_nan(data):
    x, _ = data
    assert (x.dropna() >= 0).all().all(), "no real feature value should be negative post-cleaning"
