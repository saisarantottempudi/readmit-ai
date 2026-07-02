"""Model evaluation metrics."""

import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    proba = model.predict_proba(X)[:, 1]
    return {
        "auc": float(roc_auc_score(y, proba)),
        "average_precision": float(average_precision_score(y, proba)),
        "brier": float(brier_score_loss(y, proba)),
    }
