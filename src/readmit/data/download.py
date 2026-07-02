"""Download and cache the UCI Diabetes 130-US hospitals dataset."""

import io
import logging
import zipfile
from pathlib import Path

import requests

from readmit.config import RAW_DIR, UCI_DATASET_URL

logger = logging.getLogger(__name__)

CSV_NAME = "diabetic_data.csv"


def fetch_raw(dest: Path = RAW_DIR, url: str = UCI_DATASET_URL) -> Path:
    """Download the dataset zip and extract the CSV. Cached: skips if already present."""
    dest.mkdir(parents=True, exist_ok=True)
    csv_path = dest / CSV_NAME
    if csv_path.exists():
        logger.info("Raw data already present at %s", csv_path)
        return csv_path

    logger.info("Downloading dataset from %s", url)
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        member = next(n for n in zf.namelist() if n.endswith(CSV_NAME))
        with zf.open(member) as src, open(csv_path, "wb") as out:
            out.write(src.read())

    logger.info("Extracted %s", csv_path)
    return csv_path
