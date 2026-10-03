import numpy as np
import pandas as pd

from nesy_tabfm.data.qualifying_features import build_features


def test_features_never_use_sessions_on_or_after_the_instance_date():
    qualifying = pd.DataFrame(
        {
            "driverId": [1, 1, 1],
            "constructorId": [10, 10, 10],
            "date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
            "position": [5, 3, 1],
        }
    )
    results = pd.DataFrame({"driverId": [], "date": [], "positionOrder": []})

    # cutoff exactly on the 2nd session's date -> strictly-before means only the 1st counts
    instances = pd.DataFrame(
        {"driverId": [1], "constructorId": [10], "raceId": [99], "date": pd.to_datetime(["2020-02-01"])}
    )
    features = build_features(instances, qualifying, results)

    assert features.loc[0, "driver_quali_avg_last5"] == 5.0


def test_unseen_entity_gets_nan_not_a_crash():
    qualifying = pd.DataFrame({"driverId": [1], "constructorId": [10], "date": pd.to_datetime(["2020-01-01"]), "position": [5]})
    results = pd.DataFrame({"driverId": [], "date": [], "positionOrder": []})
    instances = pd.DataFrame(
        {"driverId": [999], "constructorId": [10], "raceId": [1], "date": pd.to_datetime(["2020-06-01"])}
    )
    features = build_features(instances, qualifying, results)
    assert np.isnan(features.loc[0, "driver_quali_avg_last5"])
