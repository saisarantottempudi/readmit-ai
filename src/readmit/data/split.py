"""Train/test/reference splitting. Reference is the frozen drift baseline."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from readmit.config import PROCESSED_DIR, RANDOM_SEED, REFERENCE_SIZE
from readmit.data.clean import TARGET


def make_splits(
    df: pd.DataFrame,
    seed: int = RANDOM_SEED,
    test_size: float = 0.2,
    reference_size: int = REFERENCE_SIZE,
) -> dict[str, pd.DataFrame]:
    """Stratified train/test split plus a reference sample drawn from train."""
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df[TARGET])
    reference = train.sample(n=min(reference_size, len(train)), random_state=seed)
    return {
        "train": train.reset_index(drop=True),
        "test": test.reset_index(drop=True),
        "reference": reference.reset_index(drop=True),
    }


def write_splits(splits: dict[str, pd.DataFrame], out_dir: Path = PROCESSED_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, frame in splits.items():
        path = out_dir / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        paths.append(path)
    return paths
