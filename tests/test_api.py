import numpy as np
import pytest
from fastapi.testclient import TestClient

from readmit.api import main
from readmit.api.schemas import risk_band
from readmit.data.clean import clean
from readmit.train.pipeline import imbalance_weight, make_model, prepare_xy

VALID_PAYLOAD = {
    "race": "Caucasian",
    "gender": "Female",
    "age": "[70-80)",
    "admission_type_id": 1,
    "discharge_disposition_id": 1,
    "admission_source_id": 7,
    "diag_group": "circulatory",
    "max_glu_serum": "missing",
    "A1Cresult": ">7",
    "insulin": "Steady",
    "change": "Ch",
    "diabetesMed": "Yes",
    "time_in_hospital": 6,
    "num_lab_procedures": 45,
    "num_procedures": 1,
    "num_medications": 18,
    "number_outpatient": 0,
    "number_emergency": 1,
    "number_inpatient": 2,
    "number_diagnoses": 9,
}


@pytest.fixture
def client_with_model(raw_frame, tmp_path, monkeypatch):
    cleaned = clean(raw_frame)
    X, y = prepare_xy(cleaned)
    model = make_model(scale_pos_weight=imbalance_weight(y))
    model.fit(X, y)
    monkeypatch.setattr(main.store, "model", model)
    monkeypatch.setattr(main.store, "version", "test-1")
    monkeypatch.setattr(main.scoring_log, "SCORING_LOG", tmp_path / "scoring.parquet")
    monkeypatch.setattr(
        main.scoring_log,
        "append",
        lambda features, risk_score, path=tmp_path / "scoring.parquet": None,
    )
    return TestClient(main.app)


@pytest.fixture
def client_without_model(monkeypatch):
    monkeypatch.setattr(main.store, "model", None)
    monkeypatch.setattr(main.store, "version", "none")
    return TestClient(main.app)


class TestPredict:
    def test_happy_path(self, client_with_model):
        response = client_with_model.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200
        body = response.json()
        assert 0.0 <= body["risk_score"] <= 1.0
        assert body["risk_band"] in {"low", "medium", "high"}
        assert len(body["top_factors"]) == 3
        assert body["model_version"] == "test-1"

    def test_invalid_payload_422(self, client_with_model):
        bad = {**VALID_PAYLOAD, "gender": "Robot"}
        assert client_with_model.post("/predict", json=bad).status_code == 422

    def test_missing_field_422(self, client_with_model):
        bad = {k: v for k, v in VALID_PAYLOAD.items() if k != "age"}
        assert client_with_model.post("/predict", json=bad).status_code == 422

    def test_no_model_503(self, client_without_model):
        assert client_without_model.post("/predict", json=VALID_PAYLOAD).status_code == 503


class TestHealthAndMetrics:
    def test_health_reports_model_state(self, client_without_model):
        body = client_without_model.get("/health").json()
        assert body["status"] == "ok"
        assert body["model_ready"] is False

    def test_metrics_exposes_prometheus_format(self, client_with_model):
        client_with_model.post("/predict", json=VALID_PAYLOAD)
        text = client_with_model.get("/metrics").text
        assert "readmit_predictions_total" in text


class TestRiskBand:
    @pytest.mark.parametrize(
        ("score", "band"),
        [(0.0, "low"), (0.29, "low"), (0.3, "medium"), (0.59, "medium"), (0.6, "high"), (1.0, "high")],
    )
    def test_boundaries(self, score, band):
        assert risk_band(score) == band


class TestScoringLog:
    def test_append_and_read_roundtrip(self, tmp_path):
        from readmit.api import scoring_log

        path = tmp_path / "log.parquet"
        scoring_log.append(VALID_PAYLOAD, 0.42, path=path)
        scoring_log.append(VALID_PAYLOAD, 0.55, path=path)
        df = scoring_log.read(path)
        assert len(df) == 2
        assert np.allclose(df["risk_score"], [0.42, 0.55])
