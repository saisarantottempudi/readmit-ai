"""FastAPI service exposing the production readmission model."""

import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from readmit.api import scoring_log
from readmit.api.model import ModelStore
from readmit.api.schemas import Factor, PatientEncounter, PredictionResponse, risk_band
from readmit.features.build import FEATURES
from readmit.train.explain import top_factors

PREDICTIONS = Counter("readmit_predictions_total", "Predictions served", ["band"])
LATENCY = Histogram("readmit_prediction_latency_seconds", "Prediction latency")

store = ModelStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.load()
    yield


app = FastAPI(title="readmit-ai", version="0.1.0", lifespan=lifespan)


def encounter_frame(encounter: PatientEncounter) -> pd.DataFrame:
    frame = pd.DataFrame([encounter.model_dump()])
    for col in ("admission_type_id", "discharge_disposition_id", "admission_source_id"):
        frame[col] = frame[col].astype(str)
    return frame[FEATURES]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_ready": store.ready, "model_version": store.version}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/reload")
def reload_model() -> dict:
    """Refresh the production model from the registry (called after retraining)."""
    ok = store.load()
    return {"reloaded": ok, "model_version": store.version}


@app.post("/predict", response_model=PredictionResponse)
def predict(encounter: PatientEncounter) -> PredictionResponse:
    model = store.model
    if model is None:
        raise HTTPException(status_code=503, detail="No production model available")

    started = time.perf_counter()
    frame = encounter_frame(encounter)
    score = float(model.predict_proba(frame)[0, 1])
    factors = top_factors(model, frame)
    band = risk_band(score)
    LATENCY.observe(time.perf_counter() - started)
    PREDICTIONS.labels(band=band).inc()

    scoring_log.append(encounter.model_dump(), score)

    return PredictionResponse(
        risk_score=round(score, 4),
        risk_band=band,
        top_factors=[Factor(feature=str(f["feature"]), impact=float(f["impact"])) for f in factors],
        model_version=store.version,
    )
