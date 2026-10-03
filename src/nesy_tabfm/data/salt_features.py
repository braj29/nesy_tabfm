"""Flat feature builders for rel-salt's 8 official autocomplete tasks: 3 item-level
(``item-plant``, ``item-shippoint``, ``item-incoterms``) and 5 header-level (``sales-office``,
``sales-group``, ``sales-payterms``, ``sales-shipcond``, ``sales-incoterms``).

Unlike rel-f1's ``flatten.py``, there's no temporal windowing here and none is needed: every
joined table is either static (``customer``, ``address``, no ``time_col``) or timestamped
*identically* to the item itself (``salesdocument.CREATIONTIMESTAMP`` always equals the
item's own -- verified in ``specs/salt.py``), so a plain join carries no leakage risk.

``build_features`` (item-level) joins each item row with its order header and shipping
customer's address. ``build_header_features`` (header-level) has no single unambiguous
ship-to address to join -- an order can have several items with different ``SHIPTOPARTY``
values -- so it's header-columns-only, a smaller feature set than the item-level one.
``build_features_for_task`` dispatches on ``entity_col`` so callers (the model adapters)
don't need to know which of the two a given task needs.
"""

from __future__ import annotations

import pandas as pd

ITEM_FEATURE_COLS = [
    "SALESDOCUMENTITEMCATEGORY",
    "PRODUCT",
    "SOLDTOPARTY",
    "SHIPTOPARTY",
    "BILLTOPARTY",
    "PAYERPARTY",
    "ITEMINCOTERMSCLASSIFICATION",
]
HEADER_FEATURE_COLS = [
    "SALESOFFICE",
    "SALESGROUP",
    "CUSTOMERPAYMENTTERMS",
    "SHIPPINGCONDITION",
    "SALESDOCUMENTTYPE",
    "SALESORGANIZATION",
    "DISTRIBUTIONCHANNEL",
    "ORGANIZATIONDIVISION",
    "BILLINGCOMPANYCODE",
    "TRANSACTIONCURRENCY",
    "HEADERINCOTERMSCLASSIFICATION",
]
ADDRESS_FEATURE_COLS = ["COUNTRY", "REGION"]


def build_features(
    instances: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    entity_col: str = "ID",
    remove_columns: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Feature matrix for ``instances`` (one row each, same order), built from the item's own
    columns, its order header (via ``SALESDOCUMENT``), and its shipping customer's address
    (via ``SHIPTOPARTY`` -> ``customer`` -> ``address``).

    ``remove_columns``: the task's own ``(table, col)`` exclusion list (RelBench's
    ``AutoCompleteTask.remove_columns`` -- e.g. predicting ``item-plant`` also removes
    ``SHIPPINGPOINT``/``ITEMINCOTERMSCLASSIFICATION`` and every header business-classification
    field, not just ``PLANT`` itself, so a model can't shortcut off another already-classified
    field). Pass the task's actual list -- omitting it silently leaks those columns back in.
    """
    removed = set(remove_columns or [])
    item_cols = [c for c in ITEM_FEATURE_COLS if ("salesdocumentitem", c) not in removed]
    header_cols = [c for c in HEADER_FEATURE_COLS if ("salesdocument", c) not in removed]
    feature_cols = item_cols + header_cols + ADDRESS_FEATURE_COLS

    item = tables["salesdocumentitem"]
    doc = tables["salesdocument"]
    customer = tables["customer"]
    address = tables["address"]

    rows = instances[[entity_col]].merge(item, on=entity_col, how="left")
    if rows["SALESDOCUMENT"].isna().any():
        missing = int(rows["SALESDOCUMENT"].isna().sum())
        raise ValueError(
            f"{missing}/{len(rows)} instances not found in the `salesdocumentitem` table passed "
            "to build_features. This left-join miss silently turns every feature column to NaN "
            "rather than raising on its own (row count is unchanged, so the existing row-count "
            "assert below doesn't catch it either) -- it previously made both LightGBM and "
            "TabPFN degenerate into constant-class predictors. The likely cause: `tables` came "
            "from get_model_visible_db(), which censors salesdocumentitem to CREATIONTIMESTAMP "
            "<= test_timestamp -- correct for a forecast task, wrong here, since a test-split "
            "AutoCompleteTask row's own CREATIONTIMESTAMP is necessarily *after* test_timestamp. "
            "Pass get_full_db() instead (see audit/salt_runner.py)."
        )
    rows = rows.merge(doc[["SALESDOCUMENT", *HEADER_FEATURE_COLS]], on="SALESDOCUMENT", how="left")

    ship_address = customer.rename(columns={"CUSTOMER": "SHIPTOPARTY"}).merge(
        address, on="ADDRESSID", how="left"
    )[["SHIPTOPARTY", *ADDRESS_FEATURE_COLS]]
    rows = rows.merge(ship_address, on="SHIPTOPARTY", how="left")

    assert len(rows) == len(instances), "join produced a different row count -- duplicate keys?"
    return rows[feature_cols].reset_index(drop=True)


def build_header_features(
    instances: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    entity_col: str = "SALESDOCUMENT",
    remove_columns: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Feature matrix for header-level tasks (``sales-office``, ``sales-group``,
    ``sales-payterms``, ``sales-shipcond``, ``sales-incoterms``): the entity is the order
    header itself, so features are just its own (non-removed) columns -- no item-level or
    address join, since an order's several items can have different ``SHIPTOPARTY`` values
    and there's no single unambiguous address to attach to the header.
    """
    removed = set(remove_columns or [])
    header_cols = [c for c in HEADER_FEATURE_COLS if ("salesdocument", c) not in removed]

    doc = tables["salesdocument"]
    merged = instances[[entity_col]].merge(doc, on=entity_col, how="left", indicator=True)
    if (merged["_merge"] == "left_only").any():
        missing = int((merged["_merge"] == "left_only").sum())
        raise ValueError(
            f"{missing}/{len(merged)} instances not found in the `salesdocument` table passed "
            "to build_header_features -- see build_features' identical check for the likely "
            "cause (a temporally-censored db; pass get_full_db(), not get_model_visible_db())."
        )

    assert len(merged) == len(instances), "join produced a different row count -- duplicate keys?"
    return merged[header_cols].reset_index(drop=True)


#: Item-level entity column (``AutoCompleteTask.entity_col`` for item-plant/item-shippoint/
#: item-incoterms) vs. header-level (the other 5 tasks) -- used by ``build_features_for_task``
#: to pick the right builder without callers needing to know the distinction themselves.
ITEM_ENTITY_COL = "ID"
HEADER_ENTITY_COL = "SALESDOCUMENT"


def build_features_for_task(
    instances: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    entity_col: str,
    remove_columns: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Dispatch to ``build_features`` or ``build_header_features`` by ``entity_col``."""
    if entity_col == ITEM_ENTITY_COL:
        return build_features(instances, tables, entity_col=entity_col, remove_columns=remove_columns)
    if entity_col == HEADER_ENTITY_COL:
        return build_header_features(instances, tables, entity_col=entity_col, remove_columns=remove_columns)
    raise ValueError(f"unknown entity_col {entity_col!r}; expected {ITEM_ENTITY_COL!r} or {HEADER_ENTITY_COL!r}")
