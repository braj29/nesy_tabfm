# SALT-KG extension: OBKG-derived constraints × TabPFN vs. LightGBM

Scope: extend the rel-f1 CVR pipeline to SALT-KG (`github.com/SAP-samples/salt-kg`), deriving
constraints from its OBKG rather than hand-inventing them, wiring up all 8 official rel-salt
autocomplete tasks, and running TabPFN (v3 backend, via the `tabpfn` package) alongside the
existing LightGBM baseline. Full sourcing and empirical verification for every constraint
lives in `src/nesy_tabfm/specs/salt.py`'s module docstring and
`tests/test_salt_ground_truth.py`; this file is the cross-model summary and the narrative the
original plan asked for.

## Two bugs found and fixed along the way

Both materially affect how to read every number below, so they're stated up front rather than
buried:

1. **A censoring bug that made both models degenerate.** `audit/salt_runner.py` was building
   model features from `get_model_visible_db()` (censored to `CREATIONTIMESTAMP <=
   test_timestamp`). That's correct for a *forecast* task, but wrong for these row-masking
   autocomplete tasks: a test-split row's own `CREATIONTIMESTAMP` is necessarily *after*
   `test_timestamp` by definition, so the censoring silently dropped every test row out of its
   own entity table before `build_features()`'s joins ran, turning every feature to `NaN`.
   Both LightGBM and TabPFN degenerated into constant-class predictors as a result. Fixed by
   switching to `get_full_db()` (matching how RelBench's own `AutoCompleteTask` treats its test
   split internally), with a permanent `ValueError` guard added to `build_features()` so this
   exact failure mode can't recur silently.
2. **A ~2-hour-per-check performance bug in the constraint engine.** The within-order
   self-joins and the `SHIPPINGPOINT -> PLANT` functional dependency check were each taking
   15-65 minutes on rel-salt's 2.3M-row item table, driven by a self-join carrying every column
   of a wide table (several of them RelBench's nullable `Int64` dtype, which pandas merges far
   slower than plain `int64`) through a large intermediate frame. Fixed by projecting both
   `DenialConstraint`'s self-join and `FunctionalDependency`'s merge down to just the columns
   actually needed before joining -- same violation counts, ~1000x faster (the full db-level
   suite now runs in under 15 seconds).

## OBKG-derived constraints (10 total, all verified against real ground truth)

Sourced from SALT-KG's OBKG (`data/salt-kg/salt-kg.json`, the field-to-column mapping across
the `I_SALESDOCUMENT`/`I_SALESDOCUMENTITEM`/`I_CUSTOMER`/`I_ADDRORGNAMEPOSTALADDRESS` CDS
views) plus real SAP enterprise-structure customizing, not hand-invented -- each was checked
against real data before being kept, and two plausible-looking FD candidates
(`SALESORGANIZATION -> TRANSACTIONCURRENCY` at 86.9%, `SALESORGANIZATION ->
DISTRIBUTIONCHANNEL` at 98.3%) were tried and dropped for not holding.

| Constraint | Type | Ground-truth violation rate |
|---|---|---|
| `SALESORGANIZATION -> ORGANIZATIONDIVISION` | functional dependency | 0.0000% (exact) |
| `SALESORGANIZATION -> BILLINGCOMPANYCODE` | functional dependency | 0.0006% |
| `SHIPPINGPOINT -> PLANT` | functional dependency | 0.0585% |
| `SALESGROUP -> SALESOFFICE` | functional dependency | 0.2793% |
| items share `PLANT` within order | denial (pairwise) | 0.0220% |
| items share `SHIPPINGPOINT` within order | denial (pairwise) | 0.0221% |
| items share `ITEMINCOTERMSCLASSIFICATION` within order | denial (pairwise) | 0.0080% |
| item incoterms == header incoterms | denial (cross-table) | 0.0733% |

Plus 3 **cross-task** checks that only exist because two different tasks' predictions can be
joined on the same entity -- these have no ground-truth violation rate (there's nothing to
violate until a model predicts both sides), only a prediction-level one (see below):
`shippingpoint_pred_determines_plant_pred`, `salesgroup_pred_determines_salesoffice_pred`,
`item_incoterms_pred_matches_header_incoterms_pred`.

(Separately, an Incoterms-fixed-vocabulary domain constraint was added afterward under the
Tier-2 "domain-sourced" work -- not part of this stage's OBKG-derivation ask, but it lives in
the same `specs/salt.py` and is worth knowing about: `ITEMINCOTERMSCLASSIFICATION`/
`HEADERINCOTERMSCLASSIFICATION` hold exactly 14 legitimate values -- the 11 current ICC
Incoterms-2020 codes plus 2 real predecessor codes from earlier revisions plus one blank
sentinel -- verified 0 violations.)

## Per-task accuracy

TabPFN ran with `ignore_pretraining_limits=True`, an 8,000-row training subsample and a
1,000-row test subsample (its officially supported limits are 10 classes / ~10,000 rows on
CPU) -- a feasibility trial, not a fully-powered baseline, and the accuracy numbers below
reflect that smaller sample. `sales-group` (502 real classes, 378 seen even in the 8,000-row
sample) hit TabPFN's **hard, unbypassable** 160-class ceiling and is excluded from TabPFN's
results, not silently dropped -- see `results/salt/tabpfn/failed_tasks.csv`.

| Task | LightGBM (full test set) | TabPFN (1,000-row sample) |
|---|---|---|
| item-plant | 99.08% | 98.9% |
| item-shippoint | **2.97%** | **96.9%** |
| item-incoterms | 99.99% | 100.0% |
| sales-office | 99.9989% | 100.0% |
| sales-group | 86.04% | *excluded (378 > 160-class ceiling)* |
| sales-payterms | 99.85% | 98.9% |
| sales-shipcond | **18.12%** | **98.8%** |
| sales-incoterms | 99.98% | 100.0% |

LightGBM is weak specifically on the two "shipping" fields, well below even a trivial
majority-class baseline for `item-shippoint` (48.2%) -- tried raising `n_estimators` 200->800
as a fix (identical accuracy both times, so it isn't an undertraining issue); plausibly
genuine unseen-customer-cohort difficulty (`SHIPTOPARTY`/`SOLDTOPARTY` values introduced only
in the temporally-later test period), but not confirmed, and flagged as such rather than
explained away.

## Prediction-level CVR

| Constraint | LightGBM | TabPFN |
|---|---|---|
| `plant_consistent_within_order` | 0.0078% (483/6,184,238) | 0.0% (0/46) |
| `shippingpoint_consistent_within_order` | **0.0%** (0/6,184,238) | 0.0% (0/46) |
| `itemincotermsclassification_consistent_within_order` | 0.0049% (303/6,184,238) | 0.0% (0/46) |
| `shippingpoint_pred_determines_plant_pred` (cross-task) | **27.29%** (109,941/402,855) | **0.5%** (5/1,000) |
| `salesgroup_pred_determines_salesoffice_pred` (cross-task) | 0.11% (102/88,942) | *n/a (sales-group excluded)* |
| `item_incoterms_pred_matches_header_incoterms_pred` (cross-task) | 0.076% (306/402,855) | 0.0% (0/8) |

## Does the plant/shippoint 0% finding replicate?

**No -- and the reason why is itself the finding.** Before the censoring-bug fix, the
originally-reported 0% CVR for plant/shippoint within-order consistency was not evidence of
good model behavior: it was an artifact of both models collapsing into constant-class
predictors, which trivially "agree with themselves" within every order. That result doesn't
replicate because it was never real.

After the fix, LightGBM's `item-shippoint` still shows an exact 0% on the *within-order*
check -- but this is now understood to be a second, subtler way the same kind of check can
mislead: the model's real accuracy on this task is 2.97%, and its predictions concentrate
heavily on a handful of classes (250,821/402,855 predictions are a single class), so items
within the same order trivially "agree" on a mostly-wrong answer without the model doing
anything resembling genuine plant/shippoint reasoning. The within-order check alone cannot
tell these two situations apart -- a genuinely well-calibrated model and a collapsed one that
happens to be internally consistent both read as 0%.

This is exactly why the OBKG's richer relational structure matters: the cross-task check
`shippingpoint_pred_determines_plant_pred` -- which only exists because `SHIPPINGPOINT ->
PLANT` is a real, independently-verified functional dependency and `item-plant`/
`item-shippoint` are two separate official tasks whose predictions can be joined -- correctly
exposes what the within-order check missed, showing LightGBM at **27.29%** violations against
TabPFN's **0.5%**. A single constraint type would have reported both models as equally
"consistent"; having several OBKG-grounded constraint types, checking different relational
structure, is what actually distinguishes a model that understands the domain from one that
doesn't.
