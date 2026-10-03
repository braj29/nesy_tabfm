# RQ1 audit findings: rel-salt / tabpfn

## Per-task accuracy (context, not a constraint)

```
           task                    target_col    n  accuracy
     item-plant                         PLANT 1000     0.989
 item-shippoint                 SHIPPINGPOINT 1000     0.969
 item-incoterms   ITEMINCOTERMSCLASSIFICATION 1000     1.000
   sales-office                   SALESOFFICE 1000     1.000
 sales-payterms          CUSTOMERPAYMENTTERMS 1000     0.989
 sales-shipcond             SHIPPINGCONDITION 1000     0.988
sales-incoterms HEADERINCOTERMSCLASSIFICATION 1000     1.000
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
                      plant_consistent_within_order                denial         46             0           0.000
              shippingpoint_consistent_within_order                denial         46             0           0.000
itemincotermsclassification_consistent_within_order                denial         46             0           0.000
           shippingpoint_pred_determines_plant_pred functional_dependency       1000             5           0.005
  item_incoterms_pred_matches_header_incoterms_pred                denial          8             0           0.000
```

## Stratified: item-plant violation rate by temporal bucket

```
temporal_bucket   n  n_violations  violation_rate
       bucket_1 250             0             0.0
       bucket_2 250             0             0.0
       bucket_3 250             0             0.0
       bucket_4 250             0             0.0
```

## Stratified: item-shippoint violation rate by temporal bucket

```
temporal_bucket   n  n_violations  violation_rate
       bucket_1 250             0             0.0
       bucket_2 250             0             0.0
       bucket_3 250             0             0.0
       bucket_4 250             0             0.0
```

## Stratified: item-incoterms violation rate by temporal bucket

```
temporal_bucket   n  n_violations  violation_rate
       bucket_1 250             0             0.0
       bucket_2 250             0             0.0
       bucket_3 250             0             0.0
       bucket_4 250             0             0.0
```

## Violation rate vs. prediction error (point-biserial correlation)

- item-plant: violation vs. (1 - accuracy): n/a (no violations, or no error variance)
- item-shippoint: violation vs. (1 - accuracy): n/a (no violations, or no error variance)
- item-incoterms: violation vs. (1 - accuracy): n/a (no violations, or no error variance)
