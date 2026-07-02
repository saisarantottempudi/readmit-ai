"""Prefect monitoring flow: drift report on served traffic, retrain on breach."""

import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from prefect import flow, task

from readmit.api import scoring_log
from readmit.config import DRIFT_SHARE_THRESHOLD, PROCESSED_DIR, REPORTS_DIR
from readmit.monitoring.drift import DriftResult, compute_drift

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@task
def load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = pd.read_parquet(PROCESSED_DIR / "reference.parquet")
    current = scoring_log.read()
    return reference, current


@task
def run_drift(reference: pd.DataFrame, current: pd.DataFrame) -> DriftResult:
    return compute_drift(reference, current)


@task
def write_report(result: DriftResult, out_dir: Path = REPORTS_DIR) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"drift_{datetime.now(UTC):%Y-%m-%d_%H%M}.html"
    path.write_text(result.report_html)
    return str(path)


@task
def maybe_retrain(result: DriftResult, threshold: float) -> bool:
    if result.share_drifted <= threshold:
        logger.info("Drift share %.2f within threshold %.2f", result.share_drifted, threshold)
        return False
    logger.warning(
        "Drift share %.2f exceeds threshold %.2f (columns: %s) — triggering retrain",
        result.share_drifted,
        threshold,
        result.drifted_columns,
    )
    from flows.train_flow import train  # deferred import avoids circular flow registration

    outcome = train()
    logger.info("Retrain outcome: %s", outcome)
    return True


@flow(name="readmit-monitor")
def monitor(threshold: float = DRIFT_SHARE_THRESHOLD) -> dict:
    reference, current = load_frames()
    if current.empty or len(current) < 50:
        logger.info("Scoring log has %d rows (<50) — skipping drift check", len(current))
        return {"skipped": True, "rows": len(current)}

    result = run_drift(reference, current)
    report_path = write_report(result)
    retrained = maybe_retrain(result, threshold)
    return {
        "skipped": False,
        "share_drifted": result.share_drifted,
        "drifted_columns": result.drifted_columns,
        "report": report_path,
        "retrained": retrained,
    }


if __name__ == "__main__":
    print(monitor())
