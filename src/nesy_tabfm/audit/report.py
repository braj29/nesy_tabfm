"""Render a short markdown findings summary from an audit run's outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _table(df: pd.DataFrame) -> str:
    return "```\n" + df.to_string(index=False) + "\n```"


def write_findings(
    out_dir: Path,
    dataset_name: str,
    model_name: str,
    db_level: pd.DataFrame,
    prediction_level: pd.DataFrame,
    stratified: dict[str, pd.DataFrame],
    correlations: dict[str, float],
    accuracy: pd.DataFrame | None = None,
) -> Path:
    lines = [f"# RQ1 audit findings: {dataset_name} / {model_name}", ""]

    if accuracy is not None:
        lines += ["## Per-task accuracy (context, not a constraint)", "", _table(accuracy), ""]
    lines += ["## DB-level (ground truth) constraint violation rates", "", _table(db_level), ""]
    lines += ["## Prediction-level constraint violation rates", "", _table(prediction_level), ""]

    for key, df in stratified.items():
        lines += [f"## Stratified: {key}", "", _table(df), ""]

    lines += ["## Violation rate vs. prediction error (point-biserial correlation)", ""]
    for k, v in correlations.items():
        lines.append(f"- {k}: {v:.3f}" if pd.notna(v) else f"- {k}: n/a (no violations, or no error variance)")

    out_path = Path(out_dir) / "findings.md"
    out_path.write_text("\n".join(lines) + "\n")
    return out_path
