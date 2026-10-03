# RQ1 audit findings: rel-f1 / tabpfn-rel-local

## DB-level (ground truth) constraint violation rates

```
                            name                  kind  n_checked  n_violations  violation_rate
                 results_fk_race referential_integrity      26080             0        0.000000
               results_fk_driver referential_integrity      26080             0        0.000000
          results_fk_constructor referential_integrity      26080             0        0.000000
  results_unique_per_race_driver           cardinality      25989            85        0.003271
standings_unique_per_race_driver           cardinality      34124             0        0.000000
       season_points_consistency aggregate_consistency       3167            79        0.024945
          qualifying_before_race        temporal_order       9815             0        0.000000
             points_non_negative                denial      26080             0        0.000000
fia_scoring_position_eligibility                denial      26080             0        0.000000
```

## Prediction-level constraint violation rates

```
                       name        kind  n_checked  n_violations  violation_rate
top3_predictions_per_window cardinality         30            11        0.366667
    position_pred_in_domain      denial        760             0        0.000000
```

## Stratified: driver-top3 violation rate by temporal bucket

```
temporal_bucket   n  n_violations  violation_rate
       bucket_1 182             0        0.000000
       bucket_2 181             0        0.000000
       bucket_3 181            81        0.447514
       bucket_4 182           182        1.000000
```

## Stratified: driver-top3 violation rate by rare class (true top3=1)

```
 rare_class   n  n_violations  violation_rate
      False 598           211        0.352843
       True 128            52        0.406250
```

## Stratified: driver-position violation rate by temporal bucket

```
temporal_bucket   n  n_violations  violation_rate
       bucket_1 190             0             0.0
       bucket_2 190             0             0.0
       bucket_3 190             0             0.0
       bucket_4 190             0             0.0
```

## Violation rate vs. prediction error (point-biserial correlation)

- driver-top3: violation vs. (1 - accuracy): -0.034
- driver-position: violation vs. |error|: n/a (no violations, or no error variance)
