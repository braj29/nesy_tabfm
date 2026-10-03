"""Constraint DSL: a small set of reusable database-integrity-constraint types.

Every constraint operates on a ``tables: dict[str, pd.DataFrame]`` mapping table name ->
dataframe. The same ``check()`` code path runs whether the referenced "table" is a raw
database table or a model's prediction table plugged in under a name like
``"predictions"`` -- that substitution is what turns a DB-level integrity check into a
prediction-level audit, and is done by the caller (see ``nesy_tabfm.audit.runner``), not by
these classes.

Five concrete constraint types, matching the taxonomy in the research writeup:
ReferentialIntegrity, Cardinality, AggregateConsistency, TemporalOrder, DenialConstraint.

A sixth, MonotonicityConstraint, was added for single-table (non-relational) benchmarks
like TabArena's, where the other five have nothing to bind to (no FK graph, no natural
groups): a domain-defensible "this feature should never make the prediction worse" rule,
checked via synthetic counterfactual pairs rather than naturally co-occurring rows. It's the
same "forbidden pattern between two rows" shape as DenialConstraint, just with one row
synthesized rather than pulled from the database -- so it needs live model query access
(see nesy_tabfm.models), not just a static prediction table.

A seventh, FunctionalDependency, was added for OBKG-derived "X determines Y" business rules
(see nesy_tabfm.specs.salt) -- e.g. SAP's own enterprise-structure customizing assigns each
sales organization to exactly one billing company code. This is semantically a self-join
DenialConstraint (two rows sharing the same determinant must share the same dependent value),
but is implemented as a groupby instead: a real self-join blows up combinatorially on a
determinant with large groups (e.g. ~14,700 rows/group averaged over rel-salt's 34 sales
organizations -- ~2*10^8 pairs in that one group alone), where the row-vs-group-mode
comparison a functional dependency actually needs is linear in table size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

ColSpec = str | list[str]


def _as_list(cols: ColSpec) -> list[str]:
    return [cols] if isinstance(cols, str) else list(cols)


@dataclass
class ViolationResult:
    """Outcome of checking one constraint against one snapshot of tables."""

    name: str
    kind: str
    n_checked: int
    violations: pd.DataFrame

    @property
    def n_violations(self) -> int:
        return len(self.violations)

    @property
    def rate(self) -> float:
        if self.n_checked == 0:
            return float("nan")
        return self.n_violations / self.n_checked


class Constraint:
    """Base class. Subclasses implement ``check`` and set a ``kind`` string."""

    kind: str = "constraint"

    def __init__(self, name: str):
        self.name = name

    def check(self, tables: dict[str, pd.DataFrame]) -> ViolationResult:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class ReferentialIntegrity(Constraint):
    """Every foreign key value in ``child`` must exist as a primary key in ``parent``.

    ``fk_cols``/``pk_cols`` may be a single column name or a list (composite key). Set
    ``nullable=True`` if the foreign key is allowed to be null (nulls are excluded from
    the check rather than counted as violations, matching standard FK semantics).
    """

    kind = "referential_integrity"

    def __init__(
        self,
        name: str,
        child: str,
        fk_cols: ColSpec,
        parent: str,
        pk_cols: ColSpec,
        nullable: bool = False,
    ):
        super().__init__(name)
        self.child = child
        self.fk_cols = _as_list(fk_cols)
        self.parent = parent
        self.pk_cols = _as_list(pk_cols)
        self.nullable = nullable

    def check(self, tables: dict[str, pd.DataFrame]) -> ViolationResult:
        child_df = tables[self.child]
        parent_df = tables[self.parent]

        if self.nullable:
            child_df = child_df.dropna(subset=self.fk_cols)

        parent_keys = parent_df[self.pk_cols].drop_duplicates()

        if self.fk_cols == self.pk_cols:
            merged = child_df.merge(parent_keys, on=self.fk_cols, how="left", indicator=True)
        else:
            merged = child_df.merge(
                parent_keys,
                left_on=self.fk_cols,
                right_on=self.pk_cols,
                how="left",
                indicator=True,
            )

        mask = merged["_merge"] == "left_only"
        violations = merged.loc[mask, list(child_df.columns)].reset_index(drop=True)
        return ViolationResult(self.name, self.kind, len(child_df), violations)


class Cardinality(Constraint):
    """Rows in ``table`` (optionally filtered) must group into ``group_cols`` groups of a
    specific size.

    Set ``exact`` for an exact count, or ``min``/``max`` for a range. If ``universe_table``
    is given, its distinct ``group_cols`` values define the full population of groups to
    check (so a group with *zero* matching rows after filtering is still checked and can be
    flagged as a violation) -- essential for a check like "exactly 3 drivers predicted top3
    per race", where a race with 0 or 1 top3 predictions must still be counted.
    """

    kind = "cardinality"

    def __init__(
        self,
        name: str,
        table: str,
        group_cols: ColSpec,
        exact: int | None = None,
        min: int | None = None,  # noqa: A002
        max: int | None = None,  # noqa: A002
        row_filter: Callable[[pd.DataFrame], pd.Series] | None = None,
        universe_table: str | None = None,
        bounds_table: str | None = None,
    ):
        super().__init__(name)
        scalar_bound_given = exact is not None or min is not None or max is not None
        if bounds_table is None and not scalar_bound_given:
            raise ValueError("Cardinality requires exact, min, max, and/or bounds_table")
        if bounds_table is not None and scalar_bound_given:
            raise ValueError("pass either scalar exact/min/max or bounds_table, not both")
        self.table = table
        self.group_cols = _as_list(group_cols)
        self.exact = exact
        self.min = min
        self.max = max
        self.row_filter = row_filter
        self.universe_table = universe_table
        self.bounds_table = bounds_table

    def check(self, tables: dict[str, pd.DataFrame]) -> ViolationResult:
        df = tables[self.table]
        filtered = df[self.row_filter(df)] if self.row_filter is not None else df

        counts = (
            filtered.groupby(self.group_cols, dropna=False)
            .size()
            .rename("count")
            .reset_index()
        )

        universe_df = tables[self.universe_table] if self.universe_table else df
        universe_keys = universe_df[self.group_cols].drop_duplicates()

        merged = universe_keys.merge(counts, on=self.group_cols, how="left")
        merged["count"] = merged["count"].fillna(0).astype(int)

        if self.bounds_table is not None:
            # per-group bounds (e.g. "exactly 3 top-3 predictions per qualifying session,
            # but a test window can span a variable number of sessions") -- a group without
            # a matching bounds row isn't checkable and is dropped, same convention as
            # TemporalOrder's unmatched-row exclusion.
            bound_cols = [c for c in ("exact", "min", "max") if c in tables[self.bounds_table]]
            bounds_df = tables[self.bounds_table][self.group_cols + bound_cols]
            merged = merged.merge(bounds_df, on=self.group_cols, how="inner")

            mask = pd.Series(False, index=merged.index)
            if "exact" in bound_cols:
                has_exact = merged["exact"].notna()
                mask = mask | (has_exact & (merged["count"] != merged["exact"]))
            if "min" in bound_cols:
                has_min = merged["min"].notna()
                mask = mask | (has_min & (merged["count"] < merged["min"]))
            if "max" in bound_cols:
                has_max = merged["max"].notna()
                mask = mask | (has_max & (merged["count"] > merged["max"]))
        elif self.exact is not None:
            mask = merged["count"] != self.exact
        else:
            mask = pd.Series(False, index=merged.index)
            if self.min is not None:
                mask = mask | (merged["count"] < self.min)
            if self.max is not None:
                mask = mask | (merged["count"] > self.max)

        violations = merged.loc[mask].reset_index(drop=True)
        return ViolationResult(self.name, self.kind, len(merged), violations)


class FunctionalDependency(Constraint):
    """``determinant -> dependent`` must hold in ``table``: every row sharing the same
    ``determinant`` value(s) should share the same ``dependent`` value.

    Real business-rule FDs (as opposed to a designed schema's declared keys) are rarely
    satisfied by every last row -- data-entry exceptions and historical config changes are
    common -- so this doesn't require exact per-group uniqueness. Instead, for each
    determinant group it takes the *majority* (mode) ``dependent`` value as that group's
    "correct" value, and flags any row whose own value disagrees. A tie is broken by
    ``pandas``' ``Series.mode()`` (sorted order), which only affects *which* rows in an
    exactly-50/50 group get flagged, not the overall violation rate -- irrelevant in
    practice for the >99%-consistent FDs this is meant to check.
    """

    kind = "functional_dependency"

    def __init__(self, name: str, table: str, determinant: ColSpec, dependent: ColSpec):
        super().__init__(name)
        self.table = table
        self.determinant = _as_list(determinant)
        self.dependent = _as_list(dependent)
        if len(self.dependent) != 1:
            raise ValueError("dependent must be a single column")

    def check(self, tables: dict[str, pd.DataFrame]) -> ViolationResult:
        df = tables[self.table].reset_index(drop=True)
        dep = self.dependent[0]

        # Merge only the (determinant, dependent) columns, not every column of `df` -- `table`
        # can be wide, and some real-world loaders (e.g. RelBench's nullable Int64 pkey/fkey
        # columns) make a merge across all of a wide table's columns dramatically slower than
        # a merge of just the 2-3 needed here (empirically ~1000x on rel-salt's 2.3M-row item
        # table: this version takes ~1s where merging the full table took ~15 minutes).
        # `thin` and `df` share the same 0..n-1 index after the reset above, and a many-to-one
        # left join preserves row order/count exactly, so the resulting boolean mask still
        # lines up positionally with `df` for the final `.loc`.
        thin = df[self.determinant + [dep]]
        mode_val = (
            thin.groupby(self.determinant, dropna=False)[dep]
            .agg(lambda s: s.mode().iloc[0])
            .rename("_mode_value")
            .reset_index()
        )
        merged = thin.merge(mode_val, on=self.determinant, how="left")
        mask = (merged[dep] != merged["_mode_value"]).to_numpy()
        violations = df.loc[mask].reset_index(drop=True)
        return ViolationResult(self.name, self.kind, len(df), violations)


class AggregateConsistency(Constraint):
    """A value on the ``parent`` table must equal an aggregate of a value on ``child``,
    grouped by matching keys (e.g. ``order.total == sum(line_items.amount)`` grouped by
    order id).

    ``parent_key``/``child_key`` may be single columns or composite (list) keys, letting the
    same class express e.g. a ``(driver_id, year)`` grouping. A ``child_filter`` can restrict
    which child rows count toward the aggregate before grouping.
    """

    kind = "aggregate_consistency"

    def __init__(
        self,
        name: str,
        parent: str,
        parent_key: ColSpec,
        parent_val: str,
        child: str,
        child_key: ColSpec,
        child_val: str,
        agg: str = "sum",
        tol: float = 1e-6,
        child_filter: Callable[[pd.DataFrame], pd.Series] | None = None,
    ):
        super().__init__(name)
        self.parent = parent
        self.parent_key = _as_list(parent_key)
        self.parent_val = parent_val
        self.child = child
        self.child_key = _as_list(child_key)
        self.child_val = child_val
        self.agg = agg
        self.tol = tol
        self.child_filter = child_filter
        if len(self.parent_key) != len(self.child_key):
            raise ValueError("parent_key and child_key must have the same arity")

    def check(self, tables: dict[str, pd.DataFrame]) -> ViolationResult:
        parent_df = tables[self.parent][self.parent_key + [self.parent_val]].drop_duplicates(
            subset=self.parent_key
        )
        child_df = tables[self.child]
        if self.child_filter is not None:
            child_df = child_df[self.child_filter(child_df)]

        agg_df = (
            child_df.groupby(self.child_key)[self.child_val]
            .agg(self.agg)
            .rename("agg_value")
            .reset_index()
        )
        rename_map = dict(zip(self.child_key, self.parent_key))
        agg_df = agg_df.rename(columns=rename_map)

        merged = parent_df.merge(agg_df, on=self.parent_key, how="left")
        merged["agg_value"] = merged["agg_value"].fillna(0)

        mask = (merged[self.parent_val] - merged["agg_value"]).abs() > self.tol
        violations = merged.loc[mask].reset_index(drop=True)
        return ViolationResult(self.name, self.kind, len(merged), violations)


class TemporalOrder(Constraint):
    """For matching rows across two tables (joined on ``join_on``), a time relation must
    hold, e.g. ``drivers.dob <= races.date`` for every race a driver competed in.

    Rows that don't find a join partner in the other table are not checkable and are
    excluded from both the denominator and the violation set.
    """

    kind = "temporal_order"
    _OPS: dict[str, Callable[[pd.Series, pd.Series], pd.Series]] = {
        "<=": lambda a, b: a <= b,
        "<": lambda a, b: a < b,
        ">=": lambda a, b: a >= b,
        ">": lambda a, b: a > b,
    }

    def __init__(
        self,
        name: str,
        left_table: str,
        left_time_col: str,
        right_table: str,
        right_time_col: str,
        join_on: ColSpec,
        relation: str = "<=",
    ):
        super().__init__(name)
        if relation not in self._OPS:
            raise ValueError(f"relation must be one of {list(self._OPS)}")
        self.left_table = left_table
        self.left_time_col = left_time_col
        self.right_table = right_table
        self.right_time_col = right_time_col
        self.join_on = _as_list(join_on)
        self.relation = relation

    def check(self, tables: dict[str, pd.DataFrame]) -> ViolationResult:
        left = tables[self.left_table][self.join_on + [self.left_time_col]]
        right = tables[self.right_table][self.join_on + [self.right_time_col]]

        right_time_col = self.right_time_col
        if self.left_time_col == self.right_time_col:
            # same column name on both sides (e.g. both tables call it "date") -- rename
            # the right copy so the merge doesn't silently produce date_x/date_y instead.
            right_time_col = f"{self.right_time_col}__right"
            right = right.rename(columns={self.right_time_col: right_time_col})

        merged = left.merge(right, on=self.join_on, how="inner")

        holds = self._OPS[self.relation](merged[self.left_time_col], merged[right_time_col])
        violations = merged.loc[~holds].reset_index(drop=True)
        return ViolationResult(self.name, self.kind, len(merged), violations)


class DenialConstraint(Constraint):
    """No row (unary form) -- or no pair of rows (binary form) -- may satisfy a forbidden
    predicate.

    - Unary: pass only ``left``; ``predicate(df) -> bool mask`` flags forbidden single rows
      (e.g. a predicted position outside the valid domain).
    - Binary, two tables: pass ``left`` and ``right`` (may be the same table) plus
      ``join_on``; rows are paired via an inner join and ``predicate(joined_df)`` flags
      forbidden pairs (e.g. the same driver predicted both ``dnf=1`` and ``top3=1`` for the
      same race).
    - Binary, self-join: set ``right`` to the same table as ``left`` and pass
      ``dedupe_cols=(left_id_col, right_id_col)`` (post-merge, suffixed) so that reflexive
      pairs and symmetric duplicates are dropped, keeping only ``left_id < right_id``.

    ``columns``, for the binary form: project both sides down to ``join_on + columns`` before
    merging, instead of every column of ``left``/``right``. A self-join on a wide table blows
    up badly otherwise -- merging every column of a table this project measured at ~2.3M rows/
    13 columns took ~15-65 minutes (some of those columns are RelBench's nullable ``Int64``
    pkey/fkey dtype, which pandas merges far slower than plain ``int64``); projecting to just
    what ``predicate``/``dedupe_cols`` actually reference brought the same check under a
    second. Omit it only for a small table, or when the predicate needs columns beyond what a
    short explicit list would name.
    """

    kind = "denial"

    def __init__(
        self,
        name: str,
        left: str,
        predicate: Callable[[pd.DataFrame], pd.Series],
        right: str | None = None,
        join_on: ColSpec | None = None,
        suffixes: tuple[str, str] = ("_l", "_r"),
        dedupe_cols: tuple[str, str] | None = None,
        columns: list[str] | None = None,
    ):
        super().__init__(name)
        self.left = left
        self.predicate = predicate
        self.right = right
        self.join_on = _as_list(join_on) if join_on is not None else None
        self.suffixes = suffixes
        self.dedupe_cols = dedupe_cols
        self.columns = columns
        if self.right is not None and self.join_on is None:
            raise ValueError("join_on is required when right is set")

    def check(self, tables: dict[str, pd.DataFrame]) -> ViolationResult:
        if self.right is None:
            df = tables[self.left]
            mask = self.predicate(df)
            violations = df.loc[mask].reset_index(drop=True)
            return ViolationResult(self.name, self.kind, len(df), violations)

        left_df = tables[self.left]
        right_df = tables[self.right]
        if self.columns is not None:
            keep = list(dict.fromkeys(self.join_on + self.columns))
            left_df = left_df[keep]
            right_df = right_df[keep]
        merged = left_df.merge(
            right_df, on=self.join_on, how="inner", suffixes=self.suffixes
        )

        # dedupe_cols identifies the (left, right) row-identity columns (post-suffix) so
        # reflexive self-pairs and symmetric duplicates (a,b)/(b,a) collapse to one row.
        if self.dedupe_cols is not None:
            left_id, right_id = self.dedupe_cols
            merged = merged.loc[merged[left_id] < merged[right_id]]

        mask = self.predicate(merged)
        violations = merged.loc[mask].reset_index(drop=True)
        return ViolationResult(self.name, self.kind, len(merged), violations)


class MonotonicityConstraint(Constraint):
    """No instance's prediction may move the wrong way when one feature is nudged in the
    direction that should help (or hurt) it, holding everything else fixed -- e.g. a higher
    FICO risk-estimate score should never *decrease* predicted creditworthiness.

    Expects two tables, ``"original"`` (predictions on real rows) and ``"perturbed"``
    (predictions on the same rows after nudging one feature in the specified direction),
    row-aligned by ``id_col``, each with a ``"y_pred"`` column -- a continuous score, not a
    hard label, so a violation can register even when the nudge doesn't cross a hard
    decision boundary. ``tolerance`` absorbs float noise (default effectively zero); a
    non-zero move the wrong way beyond it is a real violation, not GBM step-function jitter.
    """

    kind = "monotonicity"

    def __init__(
        self,
        name: str,
        id_col: str,
        direction: str = "non_decreasing",
        tolerance: float = 1e-9,
    ):
        super().__init__(name)
        if direction not in ("non_decreasing", "non_increasing"):
            raise ValueError("direction must be 'non_decreasing' or 'non_increasing'")
        self.id_col = id_col
        self.direction = direction
        self.tolerance = tolerance

    def check(self, tables: dict[str, pd.DataFrame]) -> ViolationResult:
        orig = tables["original"]
        pert = tables["perturbed"]
        merged = orig.merge(pert, on=self.id_col, suffixes=("_orig", "_pert"))

        delta = merged["y_pred_pert"] - merged["y_pred_orig"]
        if self.direction == "non_decreasing":
            mask = delta < -self.tolerance
        else:
            mask = delta > self.tolerance

        violations = merged.loc[mask].reset_index(drop=True)
        return ViolationResult(self.name, self.kind, len(merged), violations)
