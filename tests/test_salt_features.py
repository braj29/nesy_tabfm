import pandas as pd
import pytest

from nesy_tabfm.data.salt_features import build_features


def _tiny_tables():
    item = pd.DataFrame(
        {
            "ID": [1, 2],
            "SALESDOCUMENT": [100, 100],
            "SALESDOCUMENTITEMCATEGORY": ["A", "A"],
            "PRODUCT": ["P1", "P2"],
            "SOLDTOPARTY": [10, 10],
            "SHIPTOPARTY": [10, 10],
            "BILLTOPARTY": [10, 10],
            "PAYERPARTY": [10, 10],
            "ITEMINCOTERMSCLASSIFICATION": ["EXW", "EXW"],
        }
    )
    doc = pd.DataFrame(
        {
            "SALESDOCUMENT": [100],
            "SALESOFFICE": ["OFF1"],
            "SALESGROUP": ["GRP1"],
            "CUSTOMERPAYMENTTERMS": ["NT30"],
            "SHIPPINGCONDITION": ["01"],
            "SALESDOCUMENTTYPE": ["OR"],
            "SALESORGANIZATION": ["0010"],
            "DISTRIBUTIONCHANNEL": ["10"],
            "ORGANIZATIONDIVISION": ["00"],
            "BILLINGCOMPANYCODE": ["1000"],
            "TRANSACTIONCURRENCY": ["USD"],
            "HEADERINCOTERMSCLASSIFICATION": ["EXW"],
        }
    )
    customer = pd.DataFrame({"CUSTOMER": [10], "ADDRESSID": [1000]})
    address = pd.DataFrame({"ADDRESSID": [1000], "COUNTRY": ["US"], "REGION": ["CA"]})
    return {"salesdocumentitem": item, "salesdocument": doc, "customer": customer, "address": address}


def test_build_features_happy_path_has_no_nans():
    tables = _tiny_tables()
    instances = pd.DataFrame({"ID": [1, 2]})
    features = build_features(instances, tables)

    assert len(features) == 2
    assert not features.isna().any().any()
    assert features.loc[0, "SALESOFFICE"] == "OFF1"


def test_build_features_raises_when_instance_missing_from_item_table():
    """Regression test: a temporally-censored item table (e.g. get_model_visible_db(), which
    drops test-split rows since their own CREATIONTIMESTAMP is after test_timestamp) used to
    silently turn every feature column to NaN for the missing rows instead of raising -- see
    audit/salt_runner.py's fix (use get_full_db() for autocomplete-style feature building)."""
    tables = _tiny_tables()
    # instance ID=2 isn't in the item table at all -- simulates the censored-db bug.
    tables["salesdocumentitem"] = tables["salesdocumentitem"][tables["salesdocumentitem"]["ID"] == 1]
    instances = pd.DataFrame({"ID": [1, 2]})

    with pytest.raises(ValueError, match="not found in the `salesdocumentitem` table"):
        build_features(instances, tables)
