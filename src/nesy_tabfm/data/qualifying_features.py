"""Time-respecting lag features for rel-f1's ``qualifying-position`` task: a driver's and
their constructor's recent form, computed only from rows dated strictly before the instance's
own qualifying session (per-entity binary search, same technique as the retired
``flatten.py`` used for the driver-position/dnf/top3 tasks -- see git history).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _asof_avg(history: pd.DataFrame, key_col: str, value_col: str, keys: pd.Series, dates: pd.Series, window: int) -> np.ndarray:
    """Mean of ``value_col`` over the last ``window`` rows of ``history`` (grouped by
    ``key_col``, sorted by date) strictly before each instance's own date."""
    history = history[[key_col, "date", value_col]].sort_values("date")
    grouped = {k: g for k, g in history.groupby(key_col)}
    out = np.full(len(keys), np.nan)
    for i, (key, date) in enumerate(zip(keys.to_numpy(), dates.to_numpy())):
        g = grouped.get(key)
        if g is None:
            continue
        idx = int(np.searchsorted(g["date"].to_numpy(), date, side="left"))  # strictly before
        if idx == 0:
            continue
        out[i] = g[value_col].iloc[max(0, idx - window) : idx].mean()
    return out


def build_features(instances: pd.DataFrame, qualifying: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix for ``instances`` (columns: driverId, constructorId, raceId, date),
    row-aligned."""
    dates = instances["date"]
    return pd.DataFrame(
        {
            "driver_quali_avg_last5": _asof_avg(
                qualifying, "driverId", "position", instances["driverId"], dates, 5
            ),
            "constructor_quali_avg_last5": _asof_avg(
                qualifying, "constructorId", "position", instances["constructorId"], dates, 5
            ),
            "driver_result_avg_last5": _asof_avg(
                results, "driverId", "positionOrder", instances["driverId"], dates, 5
            ),
        },
        index=instances.index,
    )
