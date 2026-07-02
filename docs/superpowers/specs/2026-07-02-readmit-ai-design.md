# readmit-ai — Design Spec

Date: 2026-07-02
Status: Approved

## Purpose

Production-grade, open-source MLOps platform predicting 30-day hospital readmission
risk. Demonstrates the full automated ML lifecycle — ingest → validate → train →
register → serve → monitor → retrain — targeted at NHS-adjacent employers (trusts,
health-tech, insurers) where readmission reduction is a tracked national metric
(~£2.5B/yr cost to the NHS).

Portfolio positioning: healthcare-claims-glm proves statistics depth, FinSight proves
infra depth; readmit-ai proves MLOps lifecycle depth (registry, drift, champion/
challenger, automated retraining).

## Constraints

- 100% open source, $0 cost, runs locally via `docker compose up`
- No cloud dependency; deployable anywhere Docker runs
- Real public data: UCI "Diabetes 130-US hospitals" dataset (101,766 encounters)
- Interview-demoable in ~5 minutes (including drift → retrain loop via simulator)

## Architecture

Seven services under Docker Compose:

| Service    | Role                                             |
|------------|--------------------------------------------------|
| api        | FastAPI serving `/predict`, `/health`, `/metrics`|
| mlflow     | Tracking server + model registry                 |
| prefect    | Orchestration server + worker                    |
| postgres   | Backend store for MLflow and Prefect             |
| prometheus | Scrapes API metrics                              |
| grafana    | Dashboards (latency, prediction dist., drift)    |
| reports    | Static server for Evidently HTML drift reports   |

Data flow:

1. **Ingest flow (Prefect)** — download raw CSV → pandera schema validation →
   cleaning → feature engineering → train/test/reference parquet splits. Reference
   set frozen as drift baseline.
2. **Training flow (Prefect)** — sklearn preprocessing Pipeline + XGBoost classifier,
   class-imbalance handling, probability calibration, SHAP importance artifacts.
   Logged to MLflow. Champion/challenger: promote to `Production` alias only if
   test AUC beats current champion.
3. **Serving** — FastAPI loads `models:/readmit@production`, returns risk score,
   risk band, and top SHAP contributors (clinician explainability). Prometheus
   middleware. All predictions logged to a scoring log.
4. **Monitoring flow (Prefect, scheduled)** — Evidently compares scoring log vs
   reference: data drift + prediction drift. Writes HTML report, pushes drift
   metrics. Drift share above threshold → triggers training flow via Prefect API.
5. **Simulator** — `scripts/simulate_traffic.py` sends normal then drifted traffic
   to demo the full loop live.

## Components

- `src/readmit/data/` — download, schema (pandera), cleaning, splitting
- `src/readmit/features/` — feature engineering shared by train and serve
- `src/readmit/train/` — training pipeline, evaluation, promotion logic
- `src/readmit/api/` — FastAPI app, pydantic schemas, model loader, middleware
- `src/readmit/monitoring/` — drift job, report generation, retrain trigger
- `flows/` — Prefect flow definitions (ingest, train, monitor)
- `docker/` — Dockerfiles + service configs (prometheus.yml, grafana provisioning)
- `tests/` — pytest: unit (features, promotion), API (httpx), data validation

## Error handling

- Pandera validation failures abort ingest flow with actionable report
- API returns 503 if no Production model available; 422 on schema violations
- Drift job degrades gracefully when scoring log is empty (skip, log)
- Promotion never demotes champion on challenger failure

## Quality bar

- Tests target ~85% coverage on core logic (features, promotion, API)
- CI: GitHub Actions — ruff, mypy, pytest, docker build, Trivy scan, push to ghcr.io
- Docs: README (architecture diagram, quickstart), MODEL_CARD.md (intended use,
  bias/limitations — NHS governance angle), RUNBOOK.md, MIT license, pre-commit

## Out of scope (v1)

- Cloud/Kubernetes deployment (possible v2, reuse FinSight k3s pattern)
- Authentication on the API
- Feature store service (parquet files suffice)
- Deep-learning models
