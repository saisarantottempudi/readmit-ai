"""Central configuration for paths, URIs and model settings."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("READMIT_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = Path(os.getenv("READMIT_REPORTS_DIR", PROJECT_ROOT / "reports"))
SCORING_LOG = DATA_DIR / "scoring_log.parquet"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file://{PROJECT_ROOT / 'mlruns'}")
MODEL_NAME = "readmit"
PRODUCTION_ALIAS = "production"

RANDOM_SEED = 42
REFERENCE_SIZE = 5_000
DRIFT_SHARE_THRESHOLD = float(os.getenv("READMIT_DRIFT_THRESHOLD", "0.3"))

UCI_DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)
