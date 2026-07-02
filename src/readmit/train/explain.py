"""SHAP explanations for the trained pipeline."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


def shap_explainer(model: Pipeline) -> shap.TreeExplainer:
    return shap.TreeExplainer(model.named_steps["clf"])


def global_importance(model: Pipeline, X: pd.DataFrame, sample: int = 2_000) -> dict[str, float]:
    """Mean |SHAP| per feature, descending, on a sample of X."""
    sample_df = X.sample(n=min(sample, len(X)), random_state=0)
    transformed = model.named_steps["pre"].transform(sample_df)
    names = model.named_steps["pre"].get_feature_names_out()
    values = shap_explainer(model).shap_values(transformed)
    importance = np.abs(values).mean(axis=0)
    order = np.argsort(importance)[::-1]
    return {str(names[i]): float(importance[i]) for i in order}


def top_factors(model: Pipeline, X_row: pd.DataFrame, k: int = 3) -> list[dict[str, str | float]]:
    """Top-k signed SHAP contributions for a single row."""
    transformed = model.named_steps["pre"].transform(X_row)
    names = model.named_steps["pre"].get_feature_names_out()
    values = shap_explainer(model).shap_values(transformed)[0]
    order = np.argsort(np.abs(values))[::-1][:k]
    return [{"feature": str(names[i]), "impact": float(values[i])} for i in order]


def write_importance(importance: dict[str, float], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(importance, indent=2))
    return path
