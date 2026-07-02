"""Loads and caches the production model from the MLflow registry."""

import logging

import mlflow
from sklearn.pipeline import Pipeline

from readmit.config import MLFLOW_TRACKING_URI, MODEL_NAME, PRODUCTION_ALIAS

logger = logging.getLogger(__name__)


class ModelStore:
    """Holds the current production model; call load() at startup or to refresh."""

    def __init__(self, tracking_uri: str = MLFLOW_TRACKING_URI):
        self.tracking_uri = tracking_uri
        self.model: Pipeline | None = None
        self.version: str = "none"

    def load(self) -> bool:
        """Load models:/<name>@production. Returns True on success."""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            client = mlflow.MlflowClient(tracking_uri=self.tracking_uri)
            mv = client.get_model_version_by_alias(MODEL_NAME, PRODUCTION_ALIAS)
            self.model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{PRODUCTION_ALIAS}")
            self.version = str(mv.version)
            logger.info("Loaded %s version %s", MODEL_NAME, self.version)
            return True
        except Exception:
            logger.exception("No production model available")
            self.model = None
            self.version = "none"
            return False

    @property
    def ready(self) -> bool:
        return self.model is not None
