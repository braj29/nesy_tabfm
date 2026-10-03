"""Which tier of the research writeup's constraint table each named constraint belongs to.

Tier 1 = DB integrity (generic relational/statistical invariants: referential integrity,
cardinality, aggregate consistency, temporal order, range/domain, functional dependency,
monotonicity). Tier 2 = domain-sourced (justified by an external rulebook/standard/business
practice: sport regulations, ERP business rules, a fixed regulatory vocabulary).

Anything not listed in ``TIER2`` is Tier 1. Note the SAP functional dependencies are Tier 1
because the writeup files the *functional dependency type* there (its own example, Hospital
ZipCode->City, is equally real-world-sourced); they could arguably move to Tier 2.
"""

from __future__ import annotations

TIER2: dict[str, str] = {
    "fia_scoring_position_eligibility": "Sport rulebook (FIA regulations)",
    "items_share_plant_within_order": "Business rule (ERP, SALT-KG)",
    "items_share_shippingpoint_within_order": "Business rule (ERP, SALT-KG)",
    "items_share_itemincotermsclassification_within_order": "Business rule (ERP, SALT-KG)",
    "item_incoterms_matches_header_incoterms": "Business rule (ERP, SALT-KG)",
    "plant_consistent_within_order": "Business rule (ERP, SALT-KG)",
    "shippingpoint_consistent_within_order": "Business rule (ERP, SALT-KG)",
    "itemincotermsclassification_consistent_within_order": "Business rule (ERP, SALT-KG)",
    "item_incoterms_pred_matches_header_incoterms_pred": "Business rule (ERP, SALT-KG)",
    "item_incoterms_in_fixed_vocabulary": "Fixed regulatory vocabulary (ICC Incoterms)",
    "header_incoterms_in_fixed_vocabulary": "Fixed regulatory vocabulary (ICC Incoterms)",
}


def tier_of(constraint_name: str) -> int:
    return 2 if constraint_name in TIER2 else 1


def tier1_only(constraints):
    return [c for c in constraints if tier_of(c.name) == 1]
