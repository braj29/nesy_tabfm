"""CCD on rel-salt's item-level shipping-point and plant predictions, vectorized over items
and orders.

Two scopes, both from the OBKG dependency ``SHIPPINGPOINT -> PLANT``:

- **item**: one item's predicted shipping point (task ``item-shippoint``) and predicted plant
  (task ``item-plant``) must be an admissible pair (``fd_cross_task`` in ``closed_forms``);
- **order**: every item of an order shares one plant, and each item's shipping point belongs
  to it (``agreement_fd`` in ``closed_forms``).

The shipping point -> plant map ``g`` is taken from the *training* items' ground truth (each
shipping point's majority plant), never from test labels. The OBKG configuration table itself
(SAP's TVSTZ) is not part of rel-salt; on the training data ``g`` holds for 99.94% of items.

Inputs are per-class **log**-probabilities and everything is computed in log space: models
such as LightGBM put less than 1e-45 on alternative classes for most rel-salt items, so
products of probabilities underflow.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import logsumexp


def ship_to_plant(train_items: pd.DataFrame) -> pd.DataFrame:
    """Majority ``PLANT`` per ``SHIPPINGPOINT`` among training items, with its support."""
    counts = train_items.groupby(["SHIPPINGPOINT", "PLANT"]).size().rename("n").reset_index()
    counts = counts.sort_values(["SHIPPINGPOINT", "n", "PLANT"], ascending=[True, False, True])
    totals = counts.groupby("SHIPPINGPOINT")["n"].transform("sum")
    counts["share"] = counts["n"] / totals
    return counts.drop_duplicates("SHIPPINGPOINT").reset_index(drop=True)


def child_to_parent_index(ship_classes: np.ndarray, plant_classes: np.ndarray, mapping: pd.DataFrame) -> np.ndarray:
    """``g`` as indices: for each shipping-point class, the index of its plant among the plant
    model's classes, or -1 if it has no mapped plant the plant model can predict."""
    plant_pos = {p: i for i, p in enumerate(plant_classes)}
    plant_of = dict(zip(mapping["SHIPPINGPOINT"], mapping["PLANT"]))
    return np.array([plant_pos.get(plant_of.get(s, None), -1) for s in ship_classes], dtype=int)


@dataclass
class ItemFD:
    """Per-item CCD for the cross-task FD. Indices are into the ship / plant class arrays."""

    log_qC: np.ndarray
    map_ship: np.ndarray
    map_plant: np.ndarray
    log_q_map: np.ndarray  # log q of the coherent MAP pair
    incoherent_ship: np.ndarray  # best incoherent pair, for soft MAP with any delta
    incoherent_plant: np.ndarray
    log_q_incoherent: np.ndarray


def item_fd(log_ship: np.ndarray, log_plant: np.ndarray, g: np.ndarray, chunk: int = 50_000) -> ItemFD:
    """Exact CCD for ``plant = g(ship)`` on every item. ``log_ship``: ``(N, A)``,
    ``log_plant``: ``(N, P)``. O(N * (A + P)): with an exact FD the admissible plant of each
    shipping point is a single gather, and the best incoherent plant is the top or second-best
    plant."""
    n = len(log_ship)
    valid = g >= 0
    g_safe = np.where(valid, g, 0)
    out = {k: np.empty(n) for k in ("log_qC", "log_q_map", "log_q_incoherent")}
    out.update({k: np.empty(n, dtype=int) for k in ("map_ship", "map_plant", "incoherent_ship", "incoherent_plant")})
    with np.errstate(invalid="ignore"):
        for start in range(0, n, chunk):
            sl = slice(start, min(start + chunk, n))
            ls, lp = log_ship[sl].astype(float), log_plant[sl].astype(float)
            rows = np.arange(len(ls))
            coherent = np.where(valid, ls + lp[:, g_safe], -np.inf)  # (n, A): log q(s) q(g(s))
            out["log_qC"][sl] = logsumexp(coherent, axis=1)
            a = np.argmax(coherent, axis=1)
            out["map_ship"][sl] = a
            out["map_plant"][sl] = g[a]
            out["log_q_map"][sl] = coherent[rows, a]

            top2 = np.argsort(-lp, axis=1)[:, :2]
            top_idx, second_idx = top2[:, 0], top2[:, 1]
            top_val, second_val = lp[rows, top_idx], lp[rows, second_idx]
            # best plant != g(s) for each shipping point s (any plant if s maps to none)
            uses_second = valid[None, :] & (g[None, :] == top_idx[:, None])
            incoherent = ls + np.where(uses_second, second_val[:, None], top_val[:, None])
            ia = np.argmax(incoherent, axis=1)
            out["incoherent_ship"][sl] = ia
            out["incoherent_plant"][sl] = np.where(uses_second[rows, ia], second_idx, top_idx)
            out["log_q_incoherent"][sl] = incoherent[rows, ia]
    return ItemFD(**out)


@dataclass
class OrderFD:
    """Per-order CCD for plant agreement + shipping point -> plant. ``order_index[i]`` is the
    order of item ``i`` (0..n_orders-1); per-item arrays are aligned with the inputs."""

    log_qC: np.ndarray  # (n_orders,)
    map_plant: np.ndarray  # (n_orders,) plant index shared by the order
    log_q_map: np.ndarray  # (n_orders,)
    item_map_ship: np.ndarray  # (n_items,)


def group_sum(values: np.ndarray, order_index: np.ndarray, n_orders: int) -> np.ndarray:
    """Sum rows of ``values`` per order (``-inf`` entries make the sum ``-inf``)."""
    out = np.zeros((n_orders,) + values.shape[1:])
    np.add.at(out, order_index, values)
    return out


def order_agreement_fd(
    log_ship: np.ndarray, log_plant: np.ndarray, g: np.ndarray, order_index: np.ndarray, n_orders: int
) -> OrderFD:
    """Exact CCD for every order at once (``agreement_fd`` vectorized): the order-level plant
    posterior is ``prod_j q_plant_j(c) sum_{s: g(s) = c} q_ship_j(s)``, accumulated in log
    space so orders with hundreds of items do not underflow."""
    log_ship = np.asarray(log_ship, dtype=float)
    log_plant = np.asarray(log_plant, dtype=float)
    n_plant = log_plant.shape[1]
    child_sum = np.full(log_plant.shape, -np.inf)  # log sum_{s: g(s)=c} q_ship(s)
    child_max = np.full(log_plant.shape, -np.inf)  # log max_{s: g(s)=c} q_ship(s)
    for c in range(n_plant):
        cols = np.flatnonzero(g == c)
        if len(cols):
            child_sum[:, c] = logsumexp(log_ship[:, cols], axis=1)
            child_max[:, c] = log_ship[:, cols].max(axis=1)

    log_order = group_sum(log_plant + child_sum, order_index, n_orders)
    log_qC = logsumexp(log_order, axis=1)
    order_scores = group_sum(log_plant + child_max, order_index, n_orders)
    map_plant = np.argmax(order_scores, axis=1)
    log_q_map = order_scores[np.arange(n_orders), map_plant]

    allowed = g[None, :] == map_plant[order_index][:, None]  # (N, A)
    item_map_ship = np.argmax(np.where(allowed, log_ship, -np.inf), axis=1)
    return OrderFD(log_qC, map_plant, log_q_map, item_map_ship)
