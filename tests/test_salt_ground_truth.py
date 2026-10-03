"""Integration test: real rel-salt data must satisfy (almost all of) our hand-specified
constraints. Same purpose as test_relf1_ground_truth.py: validate the constraint *specs*
against real data before trusting any violation rate computed against model predictions.

Requires network access on first run (downloads rel-salt from the Hugging Face Hub, cached
under ~/.cache/huggingface).
"""

from __future__ import annotations

import pytest

from nesy_tabfm.constraints.check import run_constraints
from nesy_tabfm.data.relbench_loader import as_frames, get_full_db, load_salt
from nesy_tabfm.specs.salt import db_level_constraints

relbench = pytest.importorskip("relbench")


@pytest.fixture(scope="module")
def tables():
    dataset = load_salt()
    db_full = get_full_db(dataset)
    return as_frames(db_full)


@pytest.fixture(scope="module")
def results(tables):
    """Every db-level constraint, computed once for the whole module -- several of these are
    multi-million-row self-joins, expensive enough that recomputing them per test function
    (as this file used to) meaningfully slows the suite down for no benefit, since none of the
    tests mutate `tables`."""
    return run_constraints(db_level_constraints(), tables)


def test_referential_integrity_holds_exactly(results):
    ri_results = [r for r in results if r.kind == "referential_integrity"]
    assert len(ri_results) == 3
    for r in ri_results:
        assert r.n_violations == 0, f"{r.name}: {r.n_violations}/{r.n_checked} violations"


def test_every_order_has_at_least_one_item(results):
    [result] = [r for r in results if r.name == "order_has_at_least_one_item"]
    assert result.n_checked > 400_000
    assert result.n_violations == 0


def test_items_share_plant_and_shippoint_within_order_mostly_holds(results):
    """~99.9% of within-order item pairs agree on PLANT/SHIPPINGPOINT/
    ITEMINCOTERMSCLASSIFICATION, and item-level incoterms mostly agree with the order's own
    header-level incoterms -- see specs/salt.py docstring. Not a hard rule (multi-warehouse/
    split-shipment orders, header-incoterms overrides are plausible real exceptions), so this
    asserts a small tolerance, not exact equality."""
    denial_results = {r.name: r for r in results if r.kind == "denial"}
    expected = {
        "items_share_plant_within_order",
        "items_share_shippingpoint_within_order",
        "items_share_itemincotermsclassification_within_order",
        "item_incoterms_matches_header_incoterms",
    }
    assert expected <= set(denial_results)
    for name in expected:
        r = denial_results[name]
        assert r.n_checked > 0
        assert r.rate < 0.01, f"{name}: violation rate {r.rate:.3%} outside expected baseline"


def test_incoterms_fixed_vocabulary_holds_exactly(results):
    """Domain-sourced (fixed regulatory vocabulary), not database-integrity -- see module
    docstring for how the 14-value domain (11 current ICC Incoterms-2020 codes + 2 legitimate
    predecessor codes + 1 blank sentinel) was verified against SALT-KG's raw source data.
    Every value RelBench's own encoding can take is already one of these 14, so this holds by
    construction on ground truth -- the real value is as a regression guard (a future data
    refresh or encoding change introducing a 15th value) and, more usefully, as a check on
    model *predictions* later (a classifier can't structurally violate it, but it's still the
    correct binding for this taxonomy row)."""
    vocab_results = {
        r.name: r
        for r in results
        if r.name in ("item_incoterms_in_fixed_vocabulary", "header_incoterms_in_fixed_vocabulary")
    }
    assert set(vocab_results) == {"item_incoterms_in_fixed_vocabulary", "header_incoterms_in_fixed_vocabulary"}
    for name, r in vocab_results.items():
        assert r.n_checked > 0
        assert r.n_violations == 0, f"{name}: {r.n_violations}/{r.n_checked} violations"


def test_obkg_functional_dependencies_mostly_hold(results):
    """OBKG-derived SAP enterprise-structure FDs -- see specs/salt.py docstring for sourcing
    (SALT-KG's field-to-column mapping) and the measured rate each tolerance is based on.
    Real business-config FDs, not schema-declared keys, so none is asserted exact."""
    fd_results = {r.name: r for r in results if r.kind == "functional_dependency"}
    assert set(fd_results) == {
        "salesorg_determines_division",
        "salesorg_determines_billing_company_code",
        "shippingpoint_determines_plant",
        "salesgroup_determines_salesoffice",
    }
    tight_tolerance = {
        "salesorg_determines_division": 0.0,  # measured exact: 0/500,908
        "salesorg_determines_billing_company_code": 0.0005,  # measured ~0.0006%
        "shippingpoint_determines_plant": 0.001,  # measured ~0.0585%
    }
    for name, max_rate in tight_tolerance.items():
        r = fd_results[name]
        assert r.n_checked > 0
        assert r.rate <= max_rate, f"{name}: violation rate {r.rate:.4%} outside expected baseline"

    # Weaker, but still a real SAP config rule (see docstring) -- looser tolerance than the
    # other three, not zero.
    weak = fd_results["salesgroup_determines_salesoffice"]
    assert weak.n_checked > 0
    assert weak.rate < 0.005, f"salesgroup_determines_salesoffice: violation rate {weak.rate:.4%} too high"
