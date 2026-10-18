# RQ1 audit findings: rel-salt / lightgbm

## Per-task accuracy (context, not a constraint)

```
           task                    target_col      n  accuracy
     item-plant                         PLANT 402855  0.990925
 item-shippoint                 SHIPPINGPOINT 402855  0.900222
 item-incoterms   ITEMINCOTERMSCLASSIFICATION 402855  0.999903
   sales-office                   SALESOFFICE  88942  0.999989
    sales-group                    SALESGROUP  88942  0.866677
 sales-payterms          CUSTOMERPAYMENTTERMS  88942  0.998673
 sales-shipcond             SHIPPINGCONDITION  88942  0.829439
sales-incoterms HEADERINCOTERMSCLASSIFICATION  88942  0.999809
```

## DB-level (ground truth) constraint violation rates

```
                                                name                  kind  n_checked  n_violations  violation_rate
                               item_fk_salesdocument referential_integrity    2319540             0        0.000000
                                 item_fk_soldtoparty referential_integrity    2319540             0        0.000000
                                 customer_fk_address referential_integrity     139607             0        0.000000
                         order_has_at_least_one_item           cardinality     500908             0        0.000000
                      items_share_plant_within_order                denial   32417261          7134        0.000220
              items_share_shippingpoint_within_order                denial   32417261          7155        0.000221
items_share_itemincotermsclassification_within_order                denial   32417261          2582        0.000080
                        salesorg_determines_division functional_dependency     500908             0        0.000000
            salesorg_determines_billing_company_code functional_dependency     500908             3        0.000006
                      shippingpoint_determines_plant functional_dependency    2319540          1356        0.000585
                   salesgroup_determines_salesoffice functional_dependency     500908          1399        0.002793
             item_incoterms_matches_header_incoterms                denial    2319540          1701        0.000733
                  item_incoterms_in_fixed_vocabulary                denial    2319540             0        0.000000
                header_incoterms_in_fixed_vocabulary                denial     500908             0        0.000000
```

## Prediction-level constraint violation rates

```
                                               name                  kind  n_checked  n_violations  violation_rate
                      plant_consistent_within_order                denial    6184238           426        0.000069
              shippingpoint_consistent_within_order                denial    6184238         36830        0.005955
itemincotermsclassification_consistent_within_order                denial    6184238           303        0.000049
           shippingpoint_pred_determines_plant_pred functional_dependency     402855         27005        0.067034
        salesgroup_pred_determines_salesoffice_pred functional_dependency      88942           102        0.001147
  item_incoterms_pred_matches_header_incoterms_pred                denial     402855           306        0.000760
```

## Stratified: item-plant violation rate by temporal bucket

```
temporal_bucket      n  n_violations  violation_rate
       bucket_1 100714           123        0.001221
       bucket_2 100714            23        0.000228
       bucket_3 100713            58        0.000576
       bucket_4 100714            35        0.000348
```

## Stratified: item-shippoint violation rate by temporal bucket

```
temporal_bucket      n  n_violations  violation_rate
       bucket_1 100714          2802        0.027821
       bucket_2 100714          2773        0.027533
       bucket_3 100713          2691        0.026719
       bucket_4 100714          2610        0.025915
```

## Stratified: item-incoterms violation rate by temporal bucket

```
temporal_bucket      n  n_violations  violation_rate
       bucket_1 100714            93        0.000923
       bucket_2 100714           112        0.001112
       bucket_3 100713            53        0.000526
       bucket_4 100714            35        0.000348
```

## Violation rate vs. prediction error (point-biserial correlation)

- item-plant: violation vs. (1 - accuracy): 0.088
- item-shippoint: violation vs. (1 - accuracy): 0.236
- item-incoterms: violation vs. (1 - accuracy): -0.000
