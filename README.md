# nesy-tabfm

Neurosymbolic audit of tabular/relational foundation models against database integrity
constraints. First implementation of RQ1 from the "Neurosymbolic Methods for Tabular and
Relational Foundation Models" research direction: **do foundation models for structured data
respect database constraints?**

Constraint types audited: referential integrity, cardinality, aggregate consistency, temporal
order, denial constraints, functional dependency, monotonicity. First benchmark: `rel-f1`
(RelBench). Model predictions come from
[RelArena-α](https://github.com/PriorLabs/relarena) (`lightgbm`, `tabpfn-rel-local` — see
`relarena_predict/`) and cluster-trained RelGNN (see `scripts/cluster/`); this repo itself
never trains a model, only audits predictions someone else produced. See
`src/nesy_tabfm/specs/rel_f1.py` for exactly which constraints bind to which of rel-f1's 3
official tasks, and why (two assumptions from the original design were corrected after
checking real data — documented there, not swept under the rug), including a domain-sourced
FIA points-scoring-eligibility rule (verified exact across all 74 F1 seasons, once the real
era-by-era cutoff and a documented fastest-lap-bonus exception are accounted for).

Second benchmark: `rel-salt` (SAP SALT-KG, RelBench v2) — see
`src/nesy_tabfm/specs/salt.py` and `results/salt/findings.md`. All 8 official autocomplete
tasks are audited (unlike rel-f1, this runs in-process against `relbench>=3.0` directly, no
separate prediction-producing environment needed): LightGBM and TabPFN (the plain `tabpfn`
package, v3 backend — a from-scratch feasibility trial, not RelArena's `tabpfn-rel-local`,
which turned out to structurally cap out at binary/regression tasks and can't run rel-salt's
genuinely multiclass targets at all). Its constraints are OBKG-derived — sourced from
SALT-KG's own field-to-column mapping (`github.com/SAP-samples/salt-kg`) and real SAP
enterprise-structure customizing, empirically verified against ground truth rather than
hand-invented (two plausible-looking functional dependencies were tried and dropped for not
holding). `results/salt/findings.md` also documents two real bugs this work found and fixed:
a temporal-censoring bug that silently made every model's predictions degenerate into a
single constant class (invalidating an earlier "0% violations" result), and a ~1000x
constraint-engine performance bug in the DSL's self-joins.

## Setup

```bash
uv sync --extra dev
```

Producing predictions to audit happens in separate environments, on purpose (see each for
why): `relarena_predict/` (RelArena-α's CPU baselines, runs on this machine) and
`scripts/cluster/` (RelGNN, needs a GPU). This repo's own venv only ever reads their output
parquet files — it never needs `torch`, `tabpfn`, or a pinned old `relbench` itself.

## Layout

- `src/nesy_tabfm/constraints/` — the constraint DSL and checker (reusable core; also used by
  RQ2/RQ3 later).
- `src/nesy_tabfm/specs/rel_f1.py`, `specs/salt.py`, `specs/heloc.py` — hand-specified
  constraints per database.
- `src/nesy_tabfm/data/relbench_loader.py` — RelBench loading (ground-truth tables + task
  metadata).
- `src/nesy_tabfm/models/precomputed.py` — adapter for predictions produced outside this repo
  (rel-f1's RelArena-α / cluster runs): reads a directory of `<task-name>_predictions.parquet`
  files into the standardized prediction schema.
- `src/nesy_tabfm/models/salt_lgbm.py`, `salt_tabpfn.py` — in-process rel-salt model adapters
  (this repo's own `relbench>=3.0` + plain LightGBM/TabPFN; no separate environment needed).
- `src/nesy_tabfm/audit/` — orchestration: run a model, check constraints, compute violation-rate
  metrics stratified by temporal shift and rare class, write a report.
- `relarena_predict/` — separate `uv` project that runs RelArena-α's baselines on rel-f1 and
  writes prediction parquets in the schema this repo expects.

## Running the audit

```bash
uv run pytest                                          # constraint engine unit tests
uv run pytest tests/test_relf1_ground_truth.py          # real rel-f1 ground-truth sanity check
uv run pytest tests/test_salt_ground_truth.py           # real rel-salt ground-truth sanity check
```

### rel-f1

Get predictions first (`relarena_predict/README.md` or `scripts/cluster/README.md`), then:

```bash
uv run python scripts/run_audit.py --model lightgbm --predictions relarena_predict/results/lightgbm
uv run python scripts/run_audit.py --model tabpfn-rel-local --predictions relarena_predict/results/tabpfn-rel-local
uv run python scripts/run_audit.py --model relgnn --predictions <path/to/cluster/output>
```

writes `results/rel_f1/<model>/` with violation-rate CSVs and a `findings.md` summary.

### rel-salt

Runs entirely in this repo's own venv — no separate prediction-producing environment:

```bash
uv run python scripts/run_salt_audit.py --model lightgbm
uv run python scripts/run_salt_audit.py --model tabpfn
```

writes `results/salt/<model>/` (per-task predictions, accuracy, violation-rate CSVs, a
`findings.md`); see `results/salt/findings.md` for the cross-model summary and narrative.
TabPFN is a CPU feasibility trial (`ignore_pretraining_limits=True`, an 8,000-row training
subsample) — it can't run any task whose real class count exceeds its hard 160-class ceiling
(`sales-group`, 502 classes, is skipped and reported in `failed_tasks.csv` rather than
aborting the whole run).

On ICF, run TabPFN through Slurm rather than at the login-node prompt. Start with:

```bash
sbatch scripts/cluster/icf_tabpfn_smoke.sbatch
sbatch scripts/cluster/icf_full_experiment.sbatch
```

See `scripts/cluster/icf_tabpfn.md`.

See `/Users/rayb/.claude/plans/groovy-wobbling-rabbit.md` for the full design writeup.
