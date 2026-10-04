# Tier 1 (DB integrity) results

Generated 2026-10-04 15:46. Tier 1 = generic relational/statistical
invariants; the domain-sourced Tier 2 constraints are excluded here (see `specs/tiers.py`).

## 1. Ground-truth (DB-level) checks

Run fresh by `scripts/run_tier1.py` against the full real databases. These validate that each
constraint *spec* actually holds in real data before it is used to audit any model.

### rel-f1

| constraint | type | checked | violations | rate |
|---|---|---:|---:|---:|
| `results_fk_race` | referential_integrity | 26,080 | 0 | 0.0000% |
| `results_fk_driver` | referential_integrity | 26,080 | 0 | 0.0000% |
| `results_fk_constructor` | referential_integrity | 26,080 | 0 | 0.0000% |
| `results_unique_per_race_driver` | cardinality | 25,989 | 85 | 0.3271% |
| `standings_unique_per_race_driver` | cardinality | 34,124 | 0 | 0.0000% |
| `season_points_consistency` | aggregate_consistency | 3,167 | 79 | 2.4945% |
| `qualifying_before_race` | temporal_order | 9,815 | 0 | 0.0000% |
| `points_non_negative` | denial | 26,080 | 0 | 0.0000% |

### rel-salt

| constraint | type | checked | violations | rate |
|---|---|---:|---:|---:|
| `item_fk_salesdocument` | referential_integrity | 2,319,540 | 0 | 0.0000% |
| `item_fk_soldtoparty` | referential_integrity | 2,319,540 | 0 | 0.0000% |
| `customer_fk_address` | referential_integrity | 139,607 | 0 | 0.0000% |
| `order_has_at_least_one_item` | cardinality | 500,908 | 0 | 0.0000% |
| `salesorg_determines_division` | functional_dependency | 500,908 | 0 | 0.0000% |
| `salesorg_determines_billing_company_code` | functional_dependency | 500,908 | 3 | 0.0006% |
| `shippingpoint_determines_plant` | functional_dependency | 2,319,540 | 1,356 | 0.0585% |
| `salesgroup_determines_salesoffice` | functional_dependency | 500,908 | 1,399 | 0.2793% |

## 2. Prediction-level checks (Tier 1 only)

Read from the CSVs the model audits wrote; the file date is shown so stale results are visible.

### rel-f1 / lightgbm

_source: `results_cluster/rel_f1/lightgbm/prediction_violation_rates.csv`, written 2026-10-04 15:46_

| constraint | type | checked | violations | rate |
|---|---|---:|---:|---:|
| `top3_predictions_per_window` | cardinality | 30 | 4 | 13.3333% |
| `position_pred_in_domain` | denial | 760 | 0 | 0.0000% |

### rel-f1 / tabpfn-rel-local

_source: `results_cluster/rel_f1/tabpfn-rel-local/prediction_violation_rates.csv`, written 2026-10-04 15:33_

| constraint | type | checked | violations | rate |
|---|---|---:|---:|---:|
| `top3_predictions_per_window` | cardinality | 30 | 11 | 36.6667% |
| `position_pred_in_domain` | denial | 760 | 0 | 0.0000% |

### rel-salt / lightgbm

_source: `results_cluster/salt/lightgbm/prediction_violation_rates.csv`, written 2026-10-04 15:45_

| constraint | type | checked | violations | rate |
|---|---|---:|---:|---:|
| `shippingpoint_pred_determines_plant_pred` | functional_dependency | 402,855 | 106,256 | 26.3757% |
| `salesgroup_pred_determines_salesoffice_pred` | functional_dependency | 88,942 | 103 | 0.1158% |

### rel-salt / tabpfn

_source: `results_cluster/salt/tabpfn/prediction_violation_rates.csv`, written 2026-10-04 15:32_

| constraint | type | checked | violations | rate |
|---|---|---:|---:|---:|
| `shippingpoint_pred_determines_plant_pred` | functional_dependency | 1,000 | 5 | 0.5000% |

## 3. Monotonicity (HELOC)

A violation = predicted creditworthiness moved the *wrong way* when one feature was nudged in its
'should help' direction, all else fixed.

### HELOC / lightgbm

_written 2026-10-04 15:46_

| constraint | type | checked | violations | rate |
|---|---|---:|---:|---:|
| `ExternalRiskEstimate_monotonic` | monotonicity | 1,984 | 354 | 17.8427% |
| `NetFractionRevolvingBurden_monotonic` | monotonicity | 1,945 | 626 | 32.1851% |

### HELOC / tabpfn

_written 2026-10-04 15:32_

| constraint | type | checked | violations | rate |
|---|---|---:|---:|---:|
| `ExternalRiskEstimate_monotonic` | monotonicity | 1,984 | 204 | 10.2823% |
| `NetFractionRevolvingBurden_monotonic` | monotonicity | 1,945 | 384 | 19.7429% |

## 4. Context: rel-salt model accuracy behind the prediction-level numbers

A violation rate is only interpretable next to how good the predictions are.

### lightgbm

| task | target | n | accuracy |
|---|---|---:|---:|
| item-plant | PLANT | 402,855 | 99.08% |
| item-shippoint | SHIPPINGPOINT | 402,855 | 59.67% |
| item-incoterms | ITEMINCOTERMSCLASSIFICATION | 402,855 | 99.99% |
| sales-office | SALESOFFICE | 88,942 | 100.00% |
| sales-group | SALESGROUP | 88,942 | 85.61% |
| sales-payterms | CUSTOMERPAYMENTTERMS | 88,942 | 99.85% |
| sales-shipcond | SHIPPINGCONDITION | 88,942 | 11.46% |
| sales-incoterms | HEADERINCOTERMSCLASSIFICATION | 88,942 | 99.98% |

### tabpfn

| task | target | n | accuracy |
|---|---|---:|---:|
| item-plant | PLANT | 1,000 | 98.90% |
| item-shippoint | SHIPPINGPOINT | 1,000 | 96.90% |
| item-incoterms | ITEMINCOTERMSCLASSIFICATION | 1,000 | 100.00% |
| sales-office | SALESOFFICE | 1,000 | 100.00% |
| sales-payterms | CUSTOMERPAYMENTTERMS | 1,000 | 98.90% |
| sales-shipcond | SHIPPINGCONDITION | 1,000 | 98.80% |
| sales-incoterms | HEADERINCOTERMSCLASSIFICATION | 1,000 | 100.00% |

## 5. Read this before citing any number

- **Baseline noise is real.** Three ground-truth constraints are not exact and the model-level
  numbers should be read against them: `results_unique_per_race_driver` 0.33% (historical F1
  shared drives, 1950-1978), `season_points_consistency` 2.49% (rel-f1 has no sprint-results
  table), and the SAP functional dependencies 0.0006-0.28% (real business-config data, not
  schema-declared keys). A model rate near these is not evidence of a model problem.
- **TabPFN on rel-salt is a feasibility trial, not a full-power baseline.** It uses an 8,000-row
  training subsample and a 1,000-row test subsample (LightGBM uses the full ~1.6M-row train and
  ~400K/89K-row test sets), `ignore_pretraining_limits=True`, and a smaller feature set (five
  near-unique ID columns dropped). Its 1,000-row FD check covers 1,000 rows against LightGBM's
  402,855, so the two rates are not directly comparable in statistical power. `sales-group`
  (502 classes) hits TabPFN's hard 160-class ceiling and is excluded, so the
  `salesgroup_pred_determines_salesoffice_pred` check has no TabPFN row.
- **The LightGBM `shippingpoint_pred_determines_plant_pred` rate (27.29%) tracks a weak model, not
  a subtle constraint failure.** LightGBM's `item-shippoint` accuracy is 2.97%, below the 48.2%
  majority-class baseline; raising `n_estimators` 200->800 changed nothing. The cause is not
  established (unseen-customer difficulty is a hypothesis, untested).
- **rel-f1 `tabpfn-rel-local` predictions are now a clean, single run** (`predict.py --models
  tabpfn-rel-local`, started 2026-09-25 09:12, all 3 tasks written by 2026-09-27 08:12 -- the
  ~2-day spread is the machine sleeping between checks, not 2 days of compute: total CPU time
  was under an hour). Superseded the earlier mixed-provenance run (driver-dnf/driver-top3 from
  Sept 13, driver-position from a run started outside this session); the audit was re-run
  against it and the numbers were unchanged (36.7% / 11 of 30 windows for
  `top3_predictions_per_window`), so the earlier mixed files weren't actually masking a
  different result here -- but treat that as this instance, not a general guarantee.
  With only 30 test windows, 36.7% vs LightGBM's 13.3% is 11 vs 4 windows.
- **Prediction-level dates in section 2 are the date the audit last ran**, not when the model
  was trained (for rel-f1, see the previous point).
- **Timings vary a lot with machine load.** On this MacBook Air the same rel-salt TabPFN task
  took 100-960s and LightGBM `sales-payterms` 821s (vs ~105s in an earlier run) while the machine
  was under load (load average 13, swap in use, another app at 100% CPU). Per-task times in
  `results/salt/*/run.log` are real but not a benchmark.
- **HELOC TabPFN takes ~8 minutes**, not the ~2 minutes I first estimated.
