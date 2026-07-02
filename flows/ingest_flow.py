"""Prefect ingest flow: download → validate → clean → split → write parquet."""

import logging

import pandas as pd
from prefect import flow, task

from readmit.data.clean import clean
from readmit.data.download import fetch_raw
from readmit.data.schema import validate_raw
from readmit.data.split import make_splits, write_splits

logging.basicConfig(level=logging.INFO)


@task(retries=2, retry_delay_seconds=30)
def download() -> str:
    return str(fetch_raw())


@task
def load_and_validate(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return validate_raw(df)


@task
def clean_and_split(df: pd.DataFrame) -> list[str]:
    splits = make_splits(clean(df))
    return [str(p) for p in write_splits(splits)]


@flow(name="readmit-ingest")
def ingest() -> list[str]:
    csv_path = download()
    df = load_and_validate(csv_path)
    return clean_and_split(df)


if __name__ == "__main__":
    print(ingest())
