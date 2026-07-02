"""Evidently-based drift detection between the reference set and served traffic."""

from dataclasses import dataclass, field

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from readmit.features.build import FEATURES


@dataclass
class DriftResult:
    share_drifted: float
    drifted_columns: list[str] = field(default_factory=list)
    report_html: str = ""


def _align(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict to model features present in both frames, ids as strings."""
    cols = [c for c in FEATURES if c in df.columns]
    out = df[cols].copy()
    for col in ("admission_type_id", "discharge_disposition_id", "admission_source_id"):
        if col in out.columns:
            out[col] = out[col].astype(str)
    return out


def compute_drift(reference: pd.DataFrame, current: pd.DataFrame) -> DriftResult:
    """Run Evidently data drift on the shared feature columns."""
    ref, cur = _align(reference), _align(current)
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)

    summary = next(
        m["result"] for m in report.as_dict()["metrics"] if m["metric"] == "DatasetDriftMetric"
    )
    table = next(
        m["result"] for m in report.as_dict()["metrics"] if m["metric"] == "DataDriftTable"
    )
    drifted = [name for name, info in table["drift_by_columns"].items() if info["drift_detected"]]
    return DriftResult(
        share_drifted=float(summary["share_of_drifted_columns"]),
        drifted_columns=drifted,
        report_html=report.get_html(),
    )
