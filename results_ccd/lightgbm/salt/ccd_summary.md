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
