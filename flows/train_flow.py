"""Prefect training flow: fit → evaluate → log to MLflow → register → promote."""

import logging
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient
from prefect import flow, task

from readmit.config import MLFLOW_TRACKING_URI, MODEL_NAME, PROCESSED_DIR
from readmit.train.evaluate import evaluate
from readmit.train.explain import global_importance, write_importance
from readmit.train.pipeline import imbalance_weight, make_model, prepare_xy
from readmit.train.promote import promote_if_better

logging.basicConfig(level=logging.INFO)


@task
def load_split(name: str, data_dir: str) -> pd.DataFrame:
    return pd.read_parquet(Path(data_dir) / f"{name}.parquet")


@task
def fit_and_log(train_df: pd.DataFrame, test_df: pd.DataFrame, mlflow_uri: str) -> dict:
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("readmit")

    X_train, y_train = prepare_xy(train_df)
    X_test, y_test = prepare_xy(test_df)

    model = make_model(scale_pos_weight=imbalance_weight(y_train))

    with mlflow.start_run() as run:
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        mlflow.log_params(
            {
                "model": "xgboost",
                "n_estimators": model.named_steps["clf"].n_estimators,
                "max_depth": model.named_steps["clf"].max_depth,
                "scale_pos_weight": round(model.named_steps["clf"].scale_pos_weight, 3),
                "train_rows": len(X_train),
            }
        )
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as tmp:
            importance_path = write_importance(
                global_importance(model, X_test), Path(tmp) / "shap_importance.json"
            )
            mlflow.log_artifact(str(importance_path))

        info = mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name=MODEL_NAME,
            input_example=X_test.head(2),
            skops_trusted_types=[
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
            ],
        )
        return {
            "run_id": run.info.run_id,
            "version": info.registered_model_version,
            "auc": metrics["auc"],
        }


@task
def promote(result: dict, mlflow_uri: str) -> bool:
    client = MlflowClient(tracking_uri=mlflow_uri)
    return promote_if_better(client, result["version"], result["auc"])


@flow(name="readmit-train")
def train(data_dir: str = str(PROCESSED_DIR), mlflow_uri: str = MLFLOW_TRACKING_URI) -> dict:
    train_df = load_split("train", data_dir)
    test_df = load_split("test", data_dir)
    result = fit_and_log(train_df, test_df, mlflow_uri)
    promoted = promote(result, mlflow_uri)
    return {**result, "promoted": promoted}


if __name__ == "__main__":
    print(train())
