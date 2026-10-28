#!/usr/bin/env python
"""Constraint-Conditioned Decoding on rel-salt, for the OBKG dependency SHIPPINGPOINT -> PLANT.

Two stages, so the expensive model run (TabPFN needs a GPU) is separate from decoding:

    uv run python scripts/run_ccd_salt.py predict --model lightgbm
    uv run python scripts/run_ccd_salt.py decode --model lightgbm

``predict`` fits the model on ``item-shippoint`` and ``item-plant`` and saves each test item's
per-class log-probabilities, plus the shipping point -> plant map from the *training* items.
``decode`` compares, on the full test split:

- raw per-cell argmax decisions (B0);
- greedy repair (B1): change the less confident of the two cells until the pair is admissible;
- Hamming-nearest repair (B2): set the plant to the shipping point's plant, ignoring confidence;
- CCD with hard conditioning (MAP over admissible pairs) and soft conditioning.

Scopes are single items (cross-task FD) and whole orders (plant agreement + FD). Everything is
run twice: on the model's own probabilities, and after per-task temperature scaling. Every
fitted parameter (temperatures, soft strengths) is cross-fitted: test orders are split in two
halves, and each half is decoded with parameters fitted on the other half only.

Log-losses are exact (no probability floor) and are averaged over the scopes whose true labels
the model can predict at all; a label outside the model's classes has zero probability under
every method. Output goes to ``<results-root>/<model>/salt/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nesy_tabfm.ccd.calibration import T_BOUNDS, fit_temperature, temper
from nesy_tabfm.ccd.core import fit_log_strength, joint_nll, soft_map_prefers_incoherent
from nesy_tabfm.ccd.salt import child_to_parent_index, group_sum, item_fd, order_agreement_fd, ship_to_plant
from nesy_tabfm.ccd.stats import cluster_bootstrap_ci, mcnemar
from nesy_tabfm.progress import get_logger, setup_run_logging, timed

REPO_ROOT = Path(__file__).resolve().parent.parent
SHIP_TASK, PLANT_TASK = "item-shippoint", "item-plant"


def out_dir(args: argparse.Namespace) -> Path:
    path = Path(args.results_root) / args.model / "salt"
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_adapter(args: argparse.Namespace):
    if args.model == "lightgbm":
        from nesy_tabfm.models.salt_lgbm import SaltLGBMAdapter

        return SaltLGBMAdapter()
    from nesy_tabfm.models.salt_tabpfn import SaltTabPFNAdapter

    kwargs = {"device": args.tabpfn_device} if args.tabpfn_device else {}
    return SaltTabPFNAdapter(
        train_sample_size=None if args.train_sample_size <= 0 else args.train_sample_size,
        test_sample_size=None,
        predict_batch_size=None if args.predict_batch_size <= 0 else args.predict_batch_size,
        seed=args.seed,
        **kwargs,
    )


def predict(args: argparse.Namespace) -> None:
    from nesy_tabfm.data.relbench_loader import get_full_db, load_salt

    out = out_dir(args)
    setup_run_logging(out)
    logger = get_logger()
    with timed("load rel-salt"):
        dataset = load_salt()
        db_full = get_full_db(dataset)
    items = db_full.table_dict["salesdocumentitem"].df[["ID", "SALESDOCUMENT", "SHIPPINGPOINT", "PLANT"]]

    adapter = build_adapter(args)
    test_ids = None
    for task_name in (SHIP_TASK, PLANT_TASK):
        task = dataset.load_task(task_name)
        with timed(f"{args.model} on {task_name}"):
            preds, classes, log_proba = adapter.fit_predict_log_proba(task, db_full)
        preds.to_parquet(out / f"{task_name}_predictions.parquet")
        # log-probabilities, not probabilities: float32 probabilities round the tiny ones to 0
        np.savez_compressed(out / f"{task_name}_log_proba.npz", classes=classes, log_proba=log_proba.astype(np.float32))
        logger.info("%s: accuracy=%.4f on n=%d, %d classes", task_name, (preds["y_true"] == preds["y_pred"]).mean(), len(preds), len(classes))
        test_ids = preds["ID"] if test_ids is None else test_ids

    train_ids = dataset.load_task(SHIP_TASK).get_table("train").df["ID"]
    mapping = ship_to_plant(items[items["ID"].isin(train_ids)])
    mapping.to_parquet(out / "ship_to_plant.parquet")
    items[items["ID"].isin(test_ids)][["ID", "SALESDOCUMENT"]].astype("int64").to_parquet(out / "test_items.parquet")
    logger.info("wrote predictions, log-probabilities and the %d-shipping-point map to %s", len(mapping), out)


# --------------------------------------------------------------------------------- decode


def load_log_proba(out: Path, task_name: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    preds = pd.read_parquet(out / f"{task_name}_predictions.parquet")
    z = np.load(out / f"{task_name}_log_proba.npz", allow_pickle=False)
    return preds, z["classes"], z["log_proba"].astype(float)


def label_index(labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    pos = {c: i for i, c in enumerate(classes)}
    return np.array([pos.get(v, -1) for v in labels], dtype=int)


def pick(log_q: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """log_q[i, idx[i]], and -inf where idx is -1 (a label the model cannot predict)."""
    vals = log_q[np.arange(len(log_q)), np.where(idx >= 0, idx, 0)]
    return np.where(idx >= 0, vals, -np.inf)


class Setup:
    """Labels, the dependency and the order structure: everything except probabilities."""

    def __init__(self, true_ship, true_plant, g, order_index, n_orders, order_fold):
        self.true_ship, self.true_plant, self.g = true_ship, true_plant, g
        self.order_index, self.n_orders, self.order_fold = order_index, n_orders, order_fold
        self.item_fold = order_fold[order_index]
        self.n = len(true_ship)

    def admissible(self, s: np.ndarray, p: np.ndarray) -> np.ndarray:
        return (s >= 0) & (p >= 0) & (self.g[np.where(s >= 0, s, 0)] == p)

    def order_ok(self, s: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Per order: every item's pair is admissible and all its plants are equal."""
        bad = np.zeros(self.n_orders, dtype=bool)
        np.logical_or.at(bad, self.order_index, ~self.admissible(s, p))
        p_min = np.full(self.n_orders, np.iinfo(np.int64).max)
        p_max = np.full(self.n_orders, np.iinfo(np.int64).min)
        np.minimum.at(p_min, self.order_index, p)
        np.maximum.at(p_max, self.order_index, p)
        return ~bad & (p_min == p_max)

    def order_all_correct(self, s: np.ndarray, p: np.ndarray) -> np.ndarray:
        wrong = np.zeros(self.n_orders, dtype=bool)
        np.logical_or.at(wrong, self.order_index, (s != self.true_ship) | (p != self.true_plant))
        return ~wrong


def greedy_repair(su: Setup, ls, lp, raw_ship, raw_plant):
    """B1: on a violating pair, change the less confident cell to its best value admissible
    with the other cell (or the other cell, if the less confident one has no such value)."""
    rows = np.arange(su.n)
    raw_ok = su.admissible(raw_ship, raw_plant)
    ship_given_plant = np.where(su.g[None, :] == raw_plant[:, None], ls, -np.inf)
    alt_ship = np.argmax(ship_given_plant, axis=1)
    has_alt_ship = np.isfinite(ship_given_plant[rows, alt_ship])
    can_fix_plant = su.g[raw_ship] >= 0
    plant_less_confident = lp[rows, raw_plant] <= ls[rows, raw_ship]
    change_plant = ~raw_ok & can_fix_plant & (plant_less_confident | ~has_alt_ship)
    change_ship = ~raw_ok & ~change_plant & has_alt_ship
    return np.where(change_ship, alt_ship, raw_ship), np.where(change_plant, su.g[raw_ship], raw_plant)


def hamming_repair(su: Setup, raw_ship, raw_plant):
    """B2: a Hamming-nearest coherent pair, chosen without confidence: keep the shipping point
    and set its plant."""
    fix = ~su.admissible(raw_ship, raw_plant) & (su.g[raw_ship] >= 0)
    return raw_ship, np.where(fix, su.g[raw_ship], raw_plant)


ITEM_METHODS = ["B0 raw argmax", "B1 greedy repair", "B2 Hamming repair", "CCD hard (MAP)", "CCD soft (fitted delta)"]
ORDER_METHODS = ["B0 raw argmax", "CCD item scope only (hard)", "CCD hard (MAP)", "CCD soft (fitted delta)"]
NLL_OF = {"B0 raw argmax": "raw", "B1 greedy repair": "raw", "B2 Hamming repair": "raw",
          "CCD hard (MAP)": "hard", "CCD soft (fitted delta)": "soft", "CCD item scope only (hard)": "item_hard"}


def evaluate(su: Setup, log_ship: np.ndarray, log_plant: np.ndarray, calibrate: bool) -> dict:
    """Decode every test item and order, each half with parameters fitted on the other half."""
    n, n_orders, rows = su.n, su.n_orders, np.arange(su.n)
    item_dec = {m: [np.empty(n, dtype=int), np.empty(n, dtype=int)] for m in ITEM_METHODS}
    order_dec = {m: [np.empty(n, dtype=int), np.empty(n, dtype=int)] for m in ORDER_METHODS}
    item_nll = {k: np.empty(n) for k in ("raw", "hard", "soft")}
    order_nll = {k: np.empty(n_orders) for k in ("raw", "hard", "soft", "item_hard")}
    item_deficit, order_deficit = np.empty(n), np.empty(n_orders)
    true_violated = ~su.admissible(su.true_ship, su.true_plant)
    order_true_violated = ~su.order_ok(su.true_ship, su.true_plant)
    raw_ship, raw_plant = np.argmax(log_ship, axis=1), np.argmax(log_plant, axis=1)  # temperature-invariant
    raw_order_ok = su.order_ok(raw_ship, raw_plant)
    folds = []

    for k in (0, 1):
        fit_i, ev_i = su.item_fold != k, su.item_fold == k
        fit_o, ev_o = su.order_fold != k, su.order_fold == k
        if calibrate:
            t_ship = fit_temperature(log_ship[fit_i], su.true_ship[fit_i])
            t_plant = fit_temperature(log_plant[fit_i], su.true_plant[fit_i])
            ls, lp = temper(log_ship, t_ship), temper(log_plant, t_plant)
        else:
            t_ship = t_plant = 1.0
            ls, lp = log_ship, log_plant
        fd = item_fd(ls, lp, su.g)
        of = order_agreement_fd(ls, lp, su.g, su.order_index, n_orders)

        raw_nll = -(pick(ls, su.true_ship) + pick(lp, su.true_plant))  # inf outside the model's classes
        o_raw_nll = group_sum(raw_nll, su.order_index, n_orders)
        scored_i, scored_o = np.isfinite(raw_nll), np.isfinite(o_raw_nll)
        fi, fo = fit_i & scored_i, fit_o & scored_o
        d_item = fit_log_strength(true_violated[fi], fd.log_qC[fi])  # log(delta)
        d_order = fit_log_strength(order_true_violated[fo], of.log_qC[fo])
        folds.append({"fold": k, "t_shippoint": t_ship, "t_plant": t_plant, "log_delta_item": d_item, "log_delta_order": d_order})
        for name, t in (("shipping point", t_ship), ("plant", t_plant)):
            if calibrate and not (T_BOUNDS[0] * 1.01 < t < T_BOUNDS[1] * 0.99):
                get_logger().warning("fold %d: %s temperature %.3g is at the search bound %s", k, name, t, T_BOUNDS)

        soft_inc = soft_map_prefers_incoherent(fd.log_q_map, fd.log_q_incoherent, d_item)
        decisions = {
            "B0 raw argmax": (raw_ship, raw_plant),
            "B1 greedy repair": greedy_repair(su, ls, lp, raw_ship, raw_plant),
            "B2 Hamming repair": hamming_repair(su, raw_ship, raw_plant),
            "CCD hard (MAP)": (fd.map_ship, fd.map_plant),
            "CCD soft (fitted delta)": (
                np.where(soft_inc, fd.incoherent_ship, fd.map_ship),
                np.where(soft_inc, fd.incoherent_plant, fd.map_plant),
            ),
        }
        # Order-level soft MAP: for delta <= 1 the best incoherent assignment is the raw argmax
        # when that is incoherent (otherwise the raw argmax is coherent and is the MAP itself).
        log_q_raw = group_sum(ls[rows, raw_ship] + lp[rows, raw_plant], su.order_index, n_orders)
        o_soft = soft_map_prefers_incoherent(of.log_q_map, np.where(raw_order_ok, -np.inf, log_q_raw), d_order)
        keep_raw = o_soft[su.order_index]
        ccd_plant = of.map_plant[su.order_index]
        order_decisions = {
            "B0 raw argmax": (raw_ship, raw_plant),
            "CCD item scope only (hard)": (fd.map_ship, fd.map_plant),
            "CCD hard (MAP)": (of.item_map_ship, ccd_plant),
            "CCD soft (fitted delta)": (np.where(keep_raw, raw_ship, of.item_map_ship), np.where(keep_raw, raw_plant, ccd_plant)),
        }
        for table, decs in ((item_dec, decisions), (order_dec, order_decisions)):
            for m, (s, p) in decs.items():
                table[m][0][ev_i], table[m][1][ev_i] = s[ev_i], p[ev_i]

        with np.errstate(invalid="ignore"):
            hard_item = joint_nll(raw_nll, true_violated, fd.log_qC, -np.inf)
            item_nll["raw"][ev_i] = raw_nll[ev_i]
            item_nll["hard"][ev_i] = hard_item[ev_i]
            item_nll["soft"][ev_i] = joint_nll(raw_nll, true_violated, fd.log_qC, d_item)[ev_i]
            order_nll["raw"][ev_o] = o_raw_nll[ev_o]
            order_nll["hard"][ev_o] = joint_nll(o_raw_nll, order_true_violated, of.log_qC, -np.inf)[ev_o]
            order_nll["soft"][ev_o] = joint_nll(o_raw_nll, order_true_violated, of.log_qC, d_order)[ev_o]
            order_nll["item_hard"][ev_o] = group_sum(hard_item, su.order_index, n_orders)[ev_o]
        item_deficit[ev_i] = -fd.log_qC[ev_i]
        order_deficit[ev_o] = -of.log_qC[ev_o]

    label = "calibrated" if calibrate else "model probabilities"
    scored_i, scored_o = np.isfinite(item_nll["raw"]), np.isfinite(order_nll["raw"])

    # Log-loss is compared on the same scopes for every method: (1) scopes whose true outcome
    # obeys the constraint, where all methods are finite and Proposition 1 applies; (2) all
    # scored scopes, where hard conditioning is infinite on the genuine exceptions.
    coherent_i = scored_i & ~true_violated
    coherent_o = scored_o & ~order_true_violated

    def nll_cols(x: np.ndarray, scored: np.ndarray, coherent: np.ndarray) -> tuple[float, float]:
        return float(x[coherent].mean()), float(x[scored].mean())

    item_rows = []
    for m in ITEM_METHODS:
        s, p = item_dec[m]
        ok = su.admissible(s, p)
        on_coherent, on_all = nll_cols(item_nll[NLL_OF[m]], scored_i, coherent_i)
        item_rows.append({
            "probabilities": label, "method": m,
            "acc_shippoint": float((s == su.true_ship).mean()), "acc_plant": float((p == su.true_plant).mean()),
            "exact_match": float(((s == su.true_ship) & (p == su.true_plant)).mean()),
            "cvr": float((~ok).mean()), "n_violations": int((~ok).sum()),
            "nll_coherent_truth": on_coherent, "nll_all": on_all,
        })
    order_rows = []
    for m in ORDER_METHODS:
        s, p = order_dec[m]
        ok = su.order_ok(s, p)
        on_coherent, on_all = nll_cols(order_nll[NLL_OF[m]], scored_o, coherent_o)
        order_rows.append({
            "probabilities": label, "method": m,
            "acc_shippoint": float((s == su.true_ship).mean()), "acc_plant": float((p == su.true_plant).mean()),
            "order_exact_match": float(su.order_all_correct(s, p).mean()),
            "order_cvr": float((~ok).mean()), "n_violating_orders": int((~ok).sum()),
            "nll_coherent_truth": on_coherent, "nll_all": on_all,
        })
    stats = {
        "probabilities": label,
        "folds": folds,
        "n_scored_items": int(scored_i.sum()),
        "n_scored_items_coherent_truth": int(coherent_i.sum()),
        "n_scored_orders": int(scored_o.sum()),
        "n_scored_orders_coherent_truth": int(coherent_o.sum()),
        "mean_item_coherence_deficit_nats": float(item_deficit[scored_i].mean()),
        "mean_order_coherence_deficit_nats": float(order_deficit[scored_o].mean()),
        "mean_item_model_violation_prob": float(-np.expm1(-item_deficit[scored_i]).mean()),
    }
    per_scope = {
        "item_dec": item_dec, "order_dec": order_dec, "item_nll": item_nll, "order_nll": order_nll,
        "scored_i": scored_i, "coherent_i": coherent_i, "scored_o": scored_o, "coherent_o": coherent_o,
    }
    return {"item": item_rows, "order": order_rows, "stats": stats, "per_scope": per_scope}


def significance(su: Setup, res: dict, n_boot: int, seed: int) -> list[dict]:
    """Paired comparisons between decoders on the same scopes: the difference A - B with a
    95% interval from a bootstrap over orders, and McNemar's test for accuracy."""
    ps = res["per_scope"]
    label = res["stats"]["probabilities"]
    item_cluster, order_cluster = su.order_index, np.arange(su.n_orders)
    rows = []

    def add(scope, metric, a_name, b_name, a_vals, b_vals, cluster, n_clusters, mask=None, correct=False):
        est, lo, hi = cluster_bootstrap_ci(
            a_vals.astype(float) - b_vals.astype(float), cluster, n_clusters, mask, n_boot, seed
        )
        row = {"probabilities": label, "scope": scope, "metric": metric, "comparison": f"{a_name} vs {b_name}",
               "difference": est, "ci_low": lo, "ci_high": hi,
               "n_scopes": int(len(a_vals) if mask is None else np.sum(mask))}
        if correct:
            row.update({f"mcnemar_{k}": v for k, v in mcnemar(a_vals, b_vals, mask).items()})
        rows.append(row)

    def both(m):
        s, p = ps["item_dec"][m]
        return (s == su.true_ship) & (p == su.true_plant)

    def plant(m):
        return ps["item_dec"][m][1] == su.true_plant

    def order_exact(m):
        return su.order_all_correct(*ps["order_dec"][m])

    hard, soft = "CCD hard (MAP)", "CCD soft (fitted delta)"
    for b_name in ("B0 raw argmax", "B1 greedy repair", "B2 Hamming repair"):
        add("item", "both correct", hard, b_name, both(hard), both(b_name), item_cluster, su.n_orders, correct=True)
    add("item", "plant correct", hard, "B0 raw argmax", plant(hard), plant("B0 raw argmax"), item_cluster, su.n_orders, correct=True)
    for b_name in ("B0 raw argmax", "CCD item scope only (hard)"):
        add("order", "all items correct", hard, b_name, order_exact(hard), order_exact(b_name), order_cluster, su.n_orders, correct=True)
    nll_i, nll_o = ps["item_nll"], ps["order_nll"]
    add("item", "joint log-loss, truth obeys rule", hard, "B0 raw argmax", nll_i["hard"], nll_i["raw"], item_cluster, su.n_orders, ps["coherent_i"])
    add("item", "joint log-loss, all scored", soft, "B0 raw argmax", nll_i["soft"], nll_i["raw"], item_cluster, su.n_orders, ps["scored_i"])
    add("order", "joint log-loss, truth obeys rule", hard, "B0 raw argmax", nll_o["hard"], nll_o["raw"], order_cluster, su.n_orders, ps["coherent_o"])
    add("order", "joint log-loss, all scored", soft, "B0 raw argmax", nll_o["soft"], nll_o["raw"], order_cluster, su.n_orders, ps["scored_o"])
    return rows


def md_table(df: pd.DataFrame) -> str:
    def fmt(v):
        return f"{v:.5f}" if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def decode(args: argparse.Namespace) -> None:
    out = out_dir(args)
    setup_run_logging(out)
    logger = get_logger()

    ship_preds, ship_classes, log_ship = load_log_proba(out, SHIP_TASK)
    plant_preds, plant_classes, log_plant = load_log_proba(out, PLANT_TASK)
    pos = pd.Index(plant_preds["ID"]).get_indexer(ship_preds["ID"])
    if (pos < 0).any():
        raise SystemExit("item-shippoint and item-plant test items differ")
    plant_preds, log_plant = plant_preds.iloc[pos].reset_index(drop=True), log_plant[pos]
    items = pd.read_parquet(out / "test_items.parquet").set_index("ID")
    order_ids = items.loc[ship_preds["ID"], "SALESDOCUMENT"].to_numpy()
    g = child_to_parent_index(ship_classes, plant_classes, pd.read_parquet(out / "ship_to_plant.parquet"))

    uniq_orders, order_index = np.unique(order_ids, return_inverse=True)
    order_fold = np.random.default_rng(args.seed).integers(0, 2, size=len(uniq_orders))
    su = Setup(
        label_index(ship_preds["y_true"].to_numpy(), ship_classes),
        label_index(plant_preds["y_true"].to_numpy(), plant_classes),
        g, order_index, len(uniq_orders), order_fold,
    )

    seen = (su.true_ship >= 0) & (su.true_plant >= 0)
    other_plant = seen & ~su.admissible(su.true_ship, su.true_plant)
    data = {
        "model": args.model,
        "n_items": su.n,
        "n_orders": su.n_orders,
        "n_shippoint_classes": len(ship_classes),
        "n_plant_classes": len(plant_classes),
        "ground_truth_item_violation_rate": float((~su.admissible(su.true_ship, su.true_plant)).mean()),
        "ground_truth_order_violation_rate": float((~su.order_ok(su.true_ship, su.true_plant)).mean()),
        "items_true_shippoint_outside_model_classes": int((su.true_ship < 0).sum()),
        "items_true_plant_outside_model_classes": int(((su.true_plant < 0) & (su.true_ship >= 0)).sum()),
        "items_genuine_exception_both_labels_known": int(other_plant.sum()),
    }
    logger.info("%s", json.dumps(data))

    results = []
    for calibrate in (False, True):
        with timed(f"decode, {'calibrated' if calibrate else 'model'} probabilities"):
            results.append(evaluate(su, log_ship, log_plant, calibrate))
    with timed(f"significance ({args.n_boot} bootstrap replicates over orders)"):
        sig_table = pd.DataFrame([r for res in results for r in significance(su, res, args.n_boot, args.seed)])
    sig_table.to_csv(out / "ccd_significance.csv", index=False)
    item_table = pd.DataFrame([r for res in results for r in res["item"]])
    order_table = pd.DataFrame([r for res in results for r in res["order"]])
    item_table.to_csv(out / "ccd_item_scope.csv", index=False)
    order_table.to_csv(out / "ccd_order_scope.csv", index=False)
    (out / "ccd_stats.json").write_text(json.dumps({"data": data, "variants": [r["stats"] for r in results]}, indent=2))

    lines = [
        f"# CCD on rel-salt ({args.model}): SHIPPINGPOINT -> PLANT",
        "",
        f"{su.n:,} test items in {su.n_orders:,} orders. The ground truth breaks the item-level dependency (as "
        f"learned from training items) for {data['ground_truth_item_violation_rate']:.4%} of items: "
        f"{data['items_true_shippoint_outside_model_classes']:,} items have a shipping point the model has never "
        f"seen, {data['items_true_plant_outside_model_classes']:,} a plant it has never seen, and "
        f"{data['items_genuine_exception_both_labels_known']:,} are genuine exceptions. It breaks the order-level "
        f"constraint for {data['ground_truth_order_violation_rate']:.4%} of orders.",
        "",
        "Joint log-losses are exact, in nats per scope, over scopes whose true labels are all among the model's "
        "classes, and on the same scopes for every method: `nll_coherent_truth` over those whose true outcome obeys "
        "the constraint (where hard conditioning lowers the loss by exactly the coherence deficit), `nll_all` over "
        "all of them (hard conditioning is infinite on the genuine exceptions). Repairs change decisions, not "
        "probabilities, so their log-loss equals the raw one.",
    ]
    for res in results:
        st = res["stats"]
        folds = "; ".join(
            f"fold {f['fold']}: T_ship={f['t_shippoint']:.3g}, T_plant={f['t_plant']:.3g}, "
            f"log_delta_item={f['log_delta_item']:.4g}, log_delta_order={f['log_delta_order']:.4g}"
            for f in st["folds"]
        )
        lines += [
            "",
            f"## {st['probabilities'].capitalize()}",
            "",
            f"Fitted on the other half of the orders: {folds}. Mean coherence deficit "
            f"{st['mean_item_coherence_deficit_nats']:.4f} nats per item, {st['mean_order_coherence_deficit_nats']:.4f} "
            f"per order; scored {st['n_scored_items']:,} items and {st['n_scored_orders']:,} orders.",
            "",
            "### Item scope (cross-task FD)",
            "",
            md_table(pd.DataFrame(res["item"]).drop(columns="probabilities")),
            "",
            "### Order scope (plant agreement + FD)",
            "",
            md_table(pd.DataFrame(res["order"]).drop(columns="probabilities")),
        ]
        if any(f["log_delta_order"] > 0 for f in st["folds"]):
            lines += ["", "Order-level soft decisions assume delta <= 1 and are approximate here."]
        at_bound = sorted({
            name for f in st["folds"] for name, t in (("shipping point", f["t_shippoint"]), ("plant", f["t_plant"]))
            if t >= T_BOUNDS[1] * 0.99
        })
        if at_bound and st["probabilities"] == "calibrated":
            lines += ["", f"The {' and '.join(at_bound)} temperature is at the search bound {T_BOUNDS[1]:g}: the best "
                      "single temperature flattens those probabilities to almost uniform, keeping only their ranking."]
    shown = sig_table.copy()
    for col in ("difference", "ci_low", "ci_high"):
        acc = ~shown["metric"].str.startswith("joint")
        shown.loc[acc, col] = shown.loc[acc, col] * 100  # percentage points
    for col in ("mcnemar_a_only", "mcnemar_b_only"):
        shown[col] = shown[col].map(lambda v: "" if pd.isna(v) else f"{int(v):,}")
    shown["mcnemar_p_value"] = shown["mcnemar_p_value"].map(lambda v: "" if pd.isna(v) else f"{v:.2g}")
    lines += [
        "",
        "## Significance",
        "",
        f"Paired differences A - B on the same scopes, with 95% intervals from {args.n_boot} bootstrap "
        "replicates that resample whole orders (items of one order are not independent). Accuracy "
        "differences are in percentage points, log-loss differences in nats per scope. McNemar's exact "
        "test counts scopes only A or only B gets right; it assumes independent scopes, so for "
        "item-level rows the bootstrap interval is the one to trust.",
        "",
        md_table(shown),
    ]
    (out / "ccd_summary.md").write_text("\n".join(lines) + "\n")
    logger.info("\n%s", "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["predict", "decode"])
    parser.add_argument("--model", required=True, choices=["lightgbm", "tabpfn"])
    parser.add_argument("--results-root", default=str(REPO_ROOT / "results_ccd"))
    parser.add_argument("--tabpfn-device", default=None)
    parser.add_argument("--train-sample-size", type=int, default=8000)
    parser.add_argument("--predict-batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0, help="TabPFN sampling seed and cross-fitting fold seed.")
    parser.add_argument("--n-boot", type=int, default=2000, help="Bootstrap replicates for significance.")
    args = parser.parse_args()
    predict(args) if args.stage == "predict" else decode(args)


if __name__ == "__main__":
    main()
