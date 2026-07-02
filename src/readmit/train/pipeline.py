"""Model pipeline: preprocessing + XGBoost classifier."""

import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from readmit.config import RANDOM_SEED
from readmit.data.clean import TARGET
from readmit.features.build import FEATURES, make_preprocessor


def make_model(scale_pos_weight: float = 1.0) -> Pipeline:
    return Pipeline(
        [
            ("pre", make_preprocessor()),
            (
                "clf",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="auc",
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select model features (as strings for categoricals) and target."""
    X = df[FEATURES].copy()
    for col in ("admission_type_id", "discharge_disposition_id", "admission_source_id"):
        X[col] = X[col].astype(str)
    return X, df[TARGET]


def imbalance_weight(y: pd.Series) -> float:
    positives = int(y.sum())
    return (len(y) - positives) / max(positives, 1)
