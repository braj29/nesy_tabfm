# CCD on rel-salt (lightgbm): SHIPPINGPOINT -> PLANT

402,855 test items in 88,942 orders. The ground truth breaks the item-level dependency (as learned from training items) for 1.1560% of items: 4,319 items have a shipping point the model has never seen, 0 a plant it has never seen, and 338 are genuine exceptions. It breaks the order-level constraint for 3.4202% of orders.

Joint log-losses are exact, in nats per scope, over scopes whose true labels are all among the model's classes, and on the same scopes for every method: `nll_coherent_truth` over those whose true outcome obeys the constraint (where hard conditioning lowers the loss by exactly the coherence deficit), `nll_all` over all of them (hard conditioning is infinite on the genuine exceptions). Repairs change decisions, not probabilities, so their log-loss equals the raw one.

## Model probabilities

Fitted on the other half of the orders: fold 0: T_ship=1, T_plant=1, log_delta_item=-129.2, log_delta_order=-438.8; fold 1: T_ship=1, T_plant=1, log_delta_item=-128.3, log_delta_order=-382.7. Mean coherence deficit 1.3975 nats per item, 7.8167 per order; scored 398,536 items and 86,116 orders.

### Item scope (cross-task FD)

| method | acc_shippoint | acc_plant | exact_match | cvr | n_violations | nll_coherent_truth | nll_all |
|---|---|---|---|---|---|---|---|
| B0 raw argmax | 0.90022 | 0.99092 | 0.89956 | 0.08525 | 34344 | 8895.51258 | 8888.32113 |
| B1 greedy repair | 0.90821 | 0.93634 | 0.90758 | 0.00000 | 0 | 8895.51258 | 8888.32113 |
| B2 Hamming repair | 0.90022 | 0.91308 | 0.89960 | 0.00000 | 0 | 8895.51258 | 8888.32113 |
| CCD hard (MAP) | 0.90377 | 0.91709 | 0.90314 | 0.00000 | 0 | 8894.12188 | inf |
| CCD soft (fitted delta) | 0.90377 | 0.91741 | 0.90314 | 0.00033 | 132 | 8894.12286 | 8887.03398 |

### Order scope (plant agreement + FD)

| method | acc_shippoint | acc_plant | order_exact_match | order_cvr | n_violating_orders | nll_coherent_truth | nll_all |
|---|---|---|---|---|---|---|---|
| B0 raw argmax | 0.90022 | 0.99092 | 0.79281 | 0.17452 | 15522 | 41224.76397 | 41134.23696 |
| CCD item scope only (hard) | 0.90377 | 0.91709 | 0.79917 | 0.01089 | 969 | 41218.32214 | inf |
| CCD hard (MAP) | 0.90533 | 0.91904 | 0.80185 | 0.00000 | 0 | 41216.96963 | inf |
| CCD soft (fitted delta) | 0.90380 | 0.92958 | 0.80159 | 0.00257 | 229 | 41218.16494 | 41128.64743 |

## Calibrated

Fitted on the other half of the orders: fold 0: T_ship=1e+07, T_plant=1.49, log_delta_item=-10.67, log_delta_order=-286.4; fold 1: T_ship=1e+07, T_plant=1.58, log_delta_item=-9.873, log_delta_order=-285.3. Mean coherence deficit 2.7114 nats per item, 12.5653 per order; scored 398,536 items and 86,116 orders.

### Item scope (cross-task FD)

| method | acc_shippoint | acc_plant | exact_match | cvr | n_violations | nll_coherent_truth | nll_all |
|---|---|---|---|---|---|---|---|
| B0 raw argmax | 0.90022 | 0.99092 | 0.89956 | 0.08525 | 34344 | 4.41324 | 4.41717 |
| B1 greedy repair | 0.94662 | 0.99072 | 0.94600 | 0.00000 | 0 | 4.41324 | 4.41717 |
| B2 Hamming repair | 0.90022 | 0.91308 | 0.89960 | 0.00000 | 0 | 4.41324 | 4.41717 |
| CCD hard (MAP) | 0.94670 | 0.99080 | 0.94608 | 0.00000 | 0 | 1.70352 | inf |
| CCD soft (fitted delta) | 0.94670 | 0.99080 | 0.94608 | 0.00000 | 0 | 1.70432 | 1.71547 |

### Order scope (plant agreement + FD)

| method | acc_shippoint | acc_plant | order_exact_match | order_cvr | n_violating_orders | nll_coherent_truth | nll_all |
|---|---|---|---|---|---|---|---|
| B0 raw argmax | 0.90022 | 0.99092 | 0.79281 | 0.17452 | 15522 | 20.44189 | 20.44222 |
| CCD item scope only (hard) | 0.94670 | 0.99080 | 0.86216 | 0.00013 | 12 | 7.89053 | inf |
| CCD hard (MAP) | 0.94670 | 0.99080 | 0.86224 | 0.00000 | 0 | 7.87370 | inf |
| CCD soft (fitted delta) | 0.94670 | 0.99080 | 0.86224 | 0.00000 | 0 | 8.17504 | 8.89462 |

The shipping point temperature is at the search bound 1e+07: the best single temperature flattens those probabilities to almost uniform, keeping only their ranking.

## Significance

Paired differences A - B on the same scopes, with 95% intervals from 2000 bootstrap replicates that resample whole orders (items of one order are not independent). Accuracy differences are in percentage points, log-loss differences in nats per scope. McNemar's exact test counts scopes only A or only B gets right; it assumes independent scopes, so for item-level rows the bootstrap interval is the one to trust.

| probabilities | scope | metric | comparison | difference | ci_low | ci_high | n_scopes | mcnemar_a_only | mcnemar_b_only | mcnemar_p_value |
|---|---|---|---|---|---|---|---|---|---|---|
| model probabilities | item | both correct | CCD hard (MAP) vs B0 raw argmax | 0.35795 | 0.28998 | 0.43061 | 402855 | 1,551 | 109 | 0 |
| model probabilities | item | both correct | CCD hard (MAP) vs B1 greedy repair | -0.44383 | -0.52103 | -0.36495 | 402855 | 370 | 2,158 | 6.8e-306 |
| model probabilities | item | both correct | CCD hard (MAP) vs B2 Hamming repair | 0.35447 | 0.28781 | 0.42535 | 402855 | 1,428 | 0 | 0 |
| model probabilities | item | plant correct | CCD hard (MAP) vs B0 raw argmax | -7.38380 | -7.66535 | -7.10339 | 402855 | 124 | 29,870 | 0 |
| model probabilities | order | all items correct | CCD hard (MAP) vs B0 raw argmax | 0.90396 | 0.83538 | 0.97032 | 88942 | 879 | 75 | 8.5e-175 |
| model probabilities | order | all items correct | CCD hard (MAP) vs CCD item scope only (hard) | 0.26759 | 0.23386 | 0.30357 | 88942 | 238 | 0 | 4.5e-72 |
| model probabilities | item | joint log-loss, truth obeys rule | CCD hard (MAP) vs B0 raw argmax | -1.39071 | -1.45441 | -1.32669 | 398198 |  |  |  |
| model probabilities | item | joint log-loss, all scored | CCD soft (fitted delta) vs B0 raw argmax | -1.28715 | -1.37376 | -1.18530 | 398536 |  |  |  |
| model probabilities | order | joint log-loss, truth obeys rule | CCD hard (MAP) vs B0 raw argmax | -7.79434 | -8.19202 | -7.42037 | 85900 |  |  |  |
| model probabilities | order | joint log-loss, all scored | CCD soft (fitted delta) vs B0 raw argmax | -5.58953 | -5.84245 | -5.33794 | 86116 |  |  |  |
| calibrated | item | both correct | CCD hard (MAP) vs B0 raw argmax | 4.65155 | 4.38356 | 4.92178 | 402855 | 18,848 | 109 | 0 |
| calibrated | item | both correct | CCD hard (MAP) vs B1 greedy repair | 0.00794 | 0.00395 | 0.01266 | 402855 | 32 | 0 | 4.7e-10 |
| calibrated | item | both correct | CCD hard (MAP) vs B2 Hamming repair | 4.64807 | 4.38086 | 4.92143 | 402855 | 18,821 | 96 | 0 |
| calibrated | item | plant correct | CCD hard (MAP) vs B0 raw argmax | -0.01241 | -0.02150 | -0.00326 | 402855 | 60 | 110 | 0.00015 |
| calibrated | order | all items correct | CCD hard (MAP) vs B0 raw argmax | 6.94273 | 6.77633 | 7.11028 | 88942 | 6,250 | 75 | 0 |
| calibrated | order | all items correct | CCD hard (MAP) vs CCD item scope only (hard) | 0.00787 | 0.00225 | 0.01352 | 88942 | 7 | 0 | 0.016 |
| calibrated | item | joint log-loss, truth obeys rule | CCD hard (MAP) vs B0 raw argmax | -2.70972 | -2.72632 | -2.69383 | 398198 |  |  |  |
| calibrated | item | joint log-loss, all scored | CCD soft (fitted delta) vs B0 raw argmax | -2.70170 | -2.71816 | -2.68455 | 398536 |  |  |  |
| calibrated | order | joint log-loss, truth obeys rule | CCD hard (MAP) vs B0 raw argmax | -12.56819 | -12.78136 | -12.35840 | 85900 |  |  |  |
| calibrated | order | joint log-loss, all scored | CCD soft (fitted delta) vs B0 raw argmax | -11.54760 | -11.74897 | -11.33340 | 86116 |  |  |  |
