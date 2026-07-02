"""Append-only parquet log of served predictions, consumed by drift monitoring."""

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from readmit.config import SCORING_LOG


def append(features: dict, risk_score: float, path: Path = SCORING_LOG) -> None:
    row = pd.DataFrame(
        [{**features, "risk_score": risk_score, "ts": datetime.now(UTC).isoformat()}]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        row = pd.concat([pd.read_parquet(path), row], ignore_index=True)
    row.to_parquet(path, index=False)


def read(path: Path = SCORING_LOG) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
