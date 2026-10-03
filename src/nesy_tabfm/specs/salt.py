"""Hand-specified constraints for the rel-salt database (RelBench v2, SAP-style enterprise
sales data: orders/line-items/customers/addresses).

Checked against real data before being written down here (see
``tests/test_salt_ground_truth.py``), and one assumption from the original design was
corrected as a result: the writeup's own running example ("order total = sum of its line
items") does not apply to this database -- **rel-salt has no monetary amount or quantity
columns at all** (every field is a categorical SAP business code: sales office, incoterms,
payment terms, ...). So there is no aggregate-consistency constraint here, and none is forced
in just to check a box. Likewise, ``salesdocumentitem.CREATIONTIMESTAMP`` is *always* exactly
equal to its parent ``salesdocument.CREATIONTIMESTAMP`` (verified: 2,319,540/2,319,540 rows) --
a structural artifact of how the data was extracted, not a discovered invariant -- so no
temporal-order constraint is claimed either.

What the schema *does* support, and what's used instead:

- **Cardinality**: every order has at least one line item (verified: 0/500,908 violations) --
  directly the writeup's own "every order has at least one item" example.
- **Denial constraints**: items within the same order overwhelmingly share the same
  ``PLANT``/``SHIPPINGPOINT``/``ITEMINCOTERMSCLASSIFICATION`` (verified: 99.978%/99.978%/
  99.992% -- rates as pairwise agreement among within-order item pairs, not orders themselves --
  plausibly split-shipment/multi-warehouse orders account for the small remainder, though
  that's inference, not confirmed against SALT's documentation). All three are targets of
  real official SALT tasks (``item-plant``, ``item-shippoint``, ``item-incoterms``), so this
  binds directly to predictions: do a model's per-item predictions stay internally consistent
  within an order? The reported violation rate's denominator is *item pairs within an order*,
  not orders themselves -- an order with one disagreeing item out of five contributes
  multiple violating pairs, not one. Separately, item-level and header-level incoterms
  overwhelmingly *agree with each other* too (``ITEMINCOTERMSCLASSIFICATION ==
  HEADERINCOTERMSCLASSIFICATION`` in 99.93% of items) -- real SAP practice (incoterms are set
  at header level and inherited by items unless explicitly overridden) -- which is checkable
  at the prediction level too, across two *different* tasks' predictions (``item-incoterms``
  vs. ``sales-incoterms``).

- **Functional dependencies**, sourced from SAP SALT-KG's OBKG (github.com/SAP-samples/
  salt-kg, ``data/salt-kg/salt-kg.json`` -- a field-to-column mapping across the
  ``I_SALESDOCUMENT``/``I_SALESDOCUMENTITEM``/``I_CUSTOMER``/``I_ADDRORGNAMEPOSTALADDRESS``
  CDS views this database was extracted from), not hand-invented: SAP's own enterprise-
  structure customizing assigns each organizational/master-data code to exactly one parent
  in a real fixed hierarchy, which should show up as a functional dependency in any
  extraction of real documents. Four candidates were checked against real data (row-level
  agreement with each determinant group's majority value -- see
  ``tests/test_salt_ground_truth.py``); two more plausible ones
  (``SALESORGANIZATION -> TRANSACTIONCURRENCY`` at 86.9%, ``SALESORGANIZATION ->
  DISTRIBUTIONCHANNEL`` at 98.3%) were tried and dropped for not clearing even a loose bar --
  a sales org can legitimately transact in more than one currency/channel, so those aren't
  real FDs in this data, whatever SAP's customizing screens suggest in the abstract:

  - ``SALESORGANIZATION -> ORGANIZATIONDIVISION`` (100.0000% -- exact, 34 groups)
  - ``SALESORGANIZATION -> BILLINGCOMPANYCODE`` (99.9994%, 34 groups)
  - ``SHIPPINGPOINT -> PLANT`` (99.9415%, 97 groups) -- matches SAP's real "assignment of
    shipping points to plants" config (table ``TVSTZ``): a shipping point belongs to one
    plant, though a plant is served by many shipping points, so this doesn't hold in the
    other direction (``PLANT -> SHIPPINGPOINT`` measures only 78.6%, and isn't claimed here).
  - ``SALESGROUP -> SALESOFFICE`` (99.7207%, 589 groups) -- matches SAP's "assignment of
    sales groups to a sales office" config (table ``TVKGR``); weaker than the other three
    (real, but noisier data), so its ground-truth test asserts a looser tolerance.

- **Domain-sourced (fixed regulatory vocabulary)**: RelBench's hub-hosted rel-salt exposes
  ``ITEMINCOTERMSCLASSIFICATION``/``HEADERINCOTERMSCLASSIFICATION`` as small integer codes
  (0-13, 14 distinct values) with no published int -> real-code decode -- but cross-
  referencing SALT-KG's own raw source parquet (github.com/SAP-samples/salt-kg,
  ``data/salt/I_SalesDocument.parquet``), which still carries the original SAP letter codes,
  confirms those 14 values are exactly: the 11 codes ICC's Incoterms 2020 currently defines
  (``EXW``, ``FCA``, ``CPT``, ``CIP``, ``DAP``, ``DPU``, ``DDP``, ``FAS``, ``FOB``, ``CFR``,
  ``CIF``) plus two legitimate predecessor codes from earlier Incoterms revisions this
  multi-year extract still contains (``DDU``, replaced by ``DAP`` in Incoterms 2010; ``DAT``,
  renamed ``DPU`` in Incoterms 2020) plus one blank/unspecified sentinel. Row identity between
  the two files was confirmed via a 100%-exact positional match after sorting both by
  ``CREATIONTIMESTAMP`` -- but that only proves *which raw row* a given integer code came
  from, not *which specific letter code* each of the 14 integers denotes (that mapping wasn't
  cleanly recoverable and isn't needed here). So the constraint below checks against
  RelBench's own encoded domain directly (integers 0-13), not the letter codes -- it's a
  faithful translation of "code must be in the fixed regulatory vocabulary" even without the
  decode, since by this same count-based argument every value RelBench's encoding can take is
  already one of the 14 legitimate real-world codes, no more.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from nesy_tabfm.constraints.dsl import (
    Cardinality,
    Constraint,
    DenialConstraint,
    FunctionalDependency,
    ReferentialIntegrity,
)

CONSISTENCY_TARGETS = ("PLANT", "SHIPPINGPOINT", "ITEMINCOTERMSCLASSIFICATION")


@dataclass(frozen=True)
class SaltTask:
    """One of rel-salt's 8 official autocomplete tasks."""

    task_name: str
    target_col: str
    entity_table: str  # "salesdocumentitem" (item-level) or "salesdocument" (header-level)
    entity_col: str  # "ID" or "SALESDOCUMENT"


#: All 8 official SALT-KG autocomplete targets (RelBench's rel-salt task manifest).
ALL_TASKS: list[SaltTask] = [
    SaltTask("item-plant", "PLANT", "salesdocumentitem", "ID"),
    SaltTask("item-shippoint", "SHIPPINGPOINT", "salesdocumentitem", "ID"),
    SaltTask("item-incoterms", "ITEMINCOTERMSCLASSIFICATION", "salesdocumentitem", "ID"),
    SaltTask("sales-office", "SALESOFFICE", "salesdocument", "SALESDOCUMENT"),
    SaltTask("sales-group", "SALESGROUP", "salesdocument", "SALESDOCUMENT"),
    SaltTask("sales-payterms", "CUSTOMERPAYMENTTERMS", "salesdocument", "SALESDOCUMENT"),
    SaltTask("sales-shipcond", "SHIPPINGCONDITION", "salesdocument", "SALESDOCUMENT"),
    SaltTask("sales-incoterms", "HEADERINCOTERMSCLASSIFICATION", "salesdocument", "SALESDOCUMENT"),
]

#: OBKG-derived functional dependencies (see module docstring for sourcing + empirical
#: verification), as (name, table, determinant, dependent).
FUNCTIONAL_DEPENDENCIES = [
    ("salesorg_determines_division", "salesdocument", "SALESORGANIZATION", "ORGANIZATIONDIVISION"),
    ("salesorg_determines_billing_company_code", "salesdocument", "SALESORGANIZATION", "BILLINGCOMPANYCODE"),
    ("shippingpoint_determines_plant", "salesdocumentitem", "SHIPPINGPOINT", "PLANT"),
    ("salesgroup_determines_salesoffice", "salesdocument", "SALESGROUP", "SALESOFFICE"),
]

#: The fixed regulatory vocabulary RelBench's Incoterms encoding may take -- 11 current ICC
#: Incoterms-2020 codes + 2 legitimate predecessor codes from earlier revisions + 1 blank/
#: unspecified sentinel = 14 total (see module docstring for how this was verified against
#: SALT-KG's raw source data, and why it's checked against RelBench's own 0-13 encoding
#: rather than the real letter codes directly).
VALID_INCOTERMS_CODES = frozenset(range(14))


def db_level_constraints() -> list[Constraint]:
    """Ground-truth constraints checked against the raw rel-salt tables."""
    constraints: list[Constraint] = [
        ReferentialIntegrity(
            "item_fk_salesdocument", "salesdocumentitem", "SALESDOCUMENT", "salesdocument", "SALESDOCUMENT"
        ),
        ReferentialIntegrity(
            "item_fk_soldtoparty", "salesdocumentitem", "SOLDTOPARTY", "customer", "CUSTOMER"
        ),
        ReferentialIntegrity("customer_fk_address", "customer", "ADDRESSID", "address", "ADDRESSID"),
        Cardinality(
            "order_has_at_least_one_item",
            table="salesdocumentitem",
            group_cols="SALESDOCUMENT",
            min=1,
            universe_table="salesdocument",
        ),
    ]
    for col in CONSISTENCY_TARGETS:
        constraints.append(
            DenialConstraint(
                f"items_share_{col.lower()}_within_order",
                left="salesdocumentitem",
                right="salesdocumentitem",
                join_on="SALESDOCUMENT",
                predicate=lambda df, col=col: df[f"{col}_l"] != df[f"{col}_r"],
                dedupe_cols=("ID_l", "ID_r"),
                columns=["ID", col],
            )
        )
    for name, table, determinant, dependent in FUNCTIONAL_DEPENDENCIES:
        constraints.append(FunctionalDependency(name, table=table, determinant=determinant, dependent=dependent))
    constraints.append(
        DenialConstraint(
            "item_incoterms_matches_header_incoterms",
            left="salesdocumentitem",
            right="salesdocument",
            join_on="SALESDOCUMENT",
            predicate=lambda df: df["ITEMINCOTERMSCLASSIFICATION"] != df["HEADERINCOTERMSCLASSIFICATION"],
        )
    )
    constraints.append(
        DenialConstraint(
            "item_incoterms_in_fixed_vocabulary",
            left="salesdocumentitem",
            predicate=lambda df: ~df["ITEMINCOTERMSCLASSIFICATION"].isin(VALID_INCOTERMS_CODES),
        )
    )
    constraints.append(
        DenialConstraint(
            "header_incoterms_in_fixed_vocabulary",
            left="salesdocument",
            predicate=lambda df: ~df["HEADERINCOTERMSCLASSIFICATION"].isin(VALID_INCOTERMS_CODES),
        )
    )
    return constraints


def with_order_id(predictions: pd.DataFrame, item_table: pd.DataFrame) -> pd.DataFrame:
    """Join the order id (``SALESDOCUMENT``) onto a predictions table keyed by item ``ID`` --
    the official ``item-plant``/``item-shippoint`` task tables don't carry it, but the
    consistency checks need it to group items by order."""
    return predictions.merge(item_table[["ID", "SALESDOCUMENT"]], on="ID", how="left")


def consistency_constraint(target_col: str) -> DenialConstraint:
    """Headline prediction-level constraint for ``target_col`` in ``CONSISTENCY_TARGETS``:
    within the same order, every item's predicted value should match every other item's.
    Expects a ``"predictions"`` table with ``ID``, ``SALESDOCUMENT`` (see ``with_order_id``),
    and a ``f"{target_col.lower()}_pred"`` column."""
    if target_col not in CONSISTENCY_TARGETS:
        raise ValueError(f"target_col must be one of {CONSISTENCY_TARGETS}")
    pred_col = f"{target_col.lower()}_pred"
    return DenialConstraint(
        f"{target_col.lower()}_consistent_within_order",
        left="predictions",
        right="predictions",
        join_on="SALESDOCUMENT",
        predicate=lambda df: df[f"{pred_col}_l"] != df[f"{pred_col}_r"],
        dedupe_cols=("ID_l", "ID_r"),
        columns=["ID", pred_col],
    )


#: Cross-task prediction-level checks: two *different* tasks' predictions, for the same
#: underlying entity, should respect the same OBKG-derived relationship their ground-truth
#: columns do (see FUNCTIONAL_DEPENDENCIES and the item/header incoterms db-level check
#: above). Each entry is (name, determinant task name, determinant col, dependent task name,
#: dependent col, join column shared by both tasks' entities).
CROSS_TASK_CHECKS = [
    (
        "shippingpoint_pred_determines_plant_pred",
        "item-shippoint",
        "SHIPPINGPOINT",
        "item-plant",
        "PLANT",
        "ID",
    ),
    (
        "salesgroup_pred_determines_salesoffice_pred",
        "sales-group",
        "SALESGROUP",
        "sales-office",
        "SALESOFFICE",
        "SALESDOCUMENT",
    ),
]


def cross_task_fd_constraint(
    name: str, determinant_col: str, dependent_col: str
) -> FunctionalDependency:
    """Headline cross-task check for one ``CROSS_TASK_CHECKS`` entry: among two different
    tasks' predictions for the same entities (already joined into one ``"combined"`` table by
    the caller, columns named ``f"{col.lower()}_pred"``), does the determinant task's
    prediction determine the dependent task's, the way the ground-truth columns do?"""
    return FunctionalDependency(
        name, table="combined", determinant=f"{determinant_col.lower()}_pred", dependent=f"{dependent_col.lower()}_pred"
    )


def item_header_incoterms_cross_constraint() -> DenialConstraint:
    """Headline cross-task, cross-entity-level check: ``item-incoterms``' predicted
    ``ITEMINCOTERMSCLASSIFICATION`` should match ``sales-incoterms``' predicted
    ``HEADERINCOTERMSCLASSIFICATION`` for items belonging to the same order (see db-level
    ``item_incoterms_matches_header_incoterms``). Expects a ``"combined"`` table with
    ``SALESDOCUMENT``, ``itemincotermsclassification_pred``, and
    ``headerincotermsclassification_pred`` columns (item-incoterms predictions joined to
    their order id via ``with_order_id``, then merged with sales-incoterms predictions on
    ``SALESDOCUMENT`` -- see ``audit/salt_runner.py``)."""
    return DenialConstraint(
        "item_incoterms_pred_matches_header_incoterms_pred",
        left="combined",
        predicate=lambda df: df["itemincotermsclassification_pred"] != df["headerincotermsclassification_pred"],
    )
