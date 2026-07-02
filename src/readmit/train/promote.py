"""Champion/challenger promotion logic for the MLflow model registry."""

import logging

from mlflow import MlflowClient

from readmit.config import MODEL_NAME, PRODUCTION_ALIAS

logger = logging.getLogger(__name__)


def should_promote(challenger_auc: float, champion_auc: float | None) -> bool:
    """Promote when there is no champion, or the challenger is strictly better."""
    if champion_auc is None:
        return True
    return challenger_auc > champion_auc


def get_champion_auc(client: MlflowClient, model_name: str = MODEL_NAME) -> float | None:
    """AUC of the current production model, or None if no production alias exists."""
    try:
        version = client.get_model_version_by_alias(model_name, PRODUCTION_ALIAS)
    except Exception:
        return None
    if version.run_id is None:
        return None
    run = client.get_run(version.run_id)
    return run.data.metrics.get("auc")


def promote_if_better(client: MlflowClient, version: str, challenger_auc: float) -> bool:
    """Set the production alias to `version` if the challenger beats the champion."""
    champion_auc = get_champion_auc(client)
    if should_promote(challenger_auc, champion_auc):
        client.set_registered_model_alias(MODEL_NAME, PRODUCTION_ALIAS, version)
        logger.info(
            "Promoted version %s (auc=%.4f, previous champion=%s)",
            version,
            challenger_auc,
            champion_auc,
        )
        return True
    logger.info(
        "Challenger auc=%.4f did not beat champion auc=%.4f — keeping champion",
        challenger_auc,
        champion_auc,
    )
    return False
