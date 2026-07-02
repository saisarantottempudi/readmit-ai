# readmit-ai Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Open-source MLOps platform predicting 30-day hospital readmission with automated train→register→serve→monitor→retrain lifecycle, one `docker compose up`.

**Architecture:** Prefect flows orchestrate ingest/train/monitor; MLflow tracks runs and holds the model registry with champion/challenger promotion; FastAPI serves the Production model with SHAP explanations and Prometheus metrics; Evidently detects drift against a frozen reference set and triggers retraining.

**Tech Stack:** Python 3.12, pandas, pandera, scikit-learn, XGBoost, SHAP, MLflow 2.x, Prefect 3.x, FastAPI, Evidently, Prometheus, Grafana, Docker Compose, GitHub Actions.

## Global Constraints

- 100% open source, $0 cost, no cloud dependency (spec: "runs locally via `docker compose up`")
- Dataset: UCI "Diabetes 130-US hospitals" (101,766 encounters)
- Package layout: `src/readmit/` importable package; flows in `flows/`
- Model registry name: `readmit`; production alias: `production`
- Line length 100, ruff + mypy clean, pytest green before every commit
- Commit after every task; update Brain wiki after every task (user requirement)

---

### Task 1: Repo scaffold

**Files:** Create `pyproject.toml`, `.gitignore`, `Makefile`, `LICENSE` (MIT), `README.md` (stub), `.pre-commit-config.yaml`, `src/readmit/__init__.py`, `tests/__init__.py`, `docker/.gitkeep`, `flows/.gitkeep`

**Produces:** installable package `readmit` (`pip install -e ".[dev]"`), `make test lint fmt` targets.

- [ ] Write pyproject with deps: pandas, pandera, scikit-learn, xgboost, shap, mlflow, prefect, fastapi, uvicorn, evidently, prometheus-client, httpx, pyarrow; dev: pytest, pytest-cov, ruff, mypy, pre-commit
- [ ] `pip install -e ".[dev]"` succeeds; `pytest` collects 0 tests, exits 0 (`--co` check)
- [ ] Commit `chore: scaffold readmit-ai project`

### Task 2: Data layer

**Files:** Create `src/readmit/data/download.py`, `src/readmit/data/schema.py`, `src/readmit/data/clean.py`, `src/readmit/data/split.py`, `src/readmit/config.py`; Test `tests/test_schema.py`, `tests/test_clean.py`, `tests/test_split.py`

**Interfaces (produces):**
- `download.fetch_raw(dest: Path) -> Path` — downloads/caches UCI zip, returns CSV path
- `schema.RawSchema` — pandera DataFrameSchema for raw CSV; `schema.validate_raw(df) -> pd.DataFrame`
- `clean.clean(df: pd.DataFrame) -> pd.DataFrame` — target `readmitted_30d` (1 if `readmitted == "<30"`), drop leakage/id cols, map `?`→NA, collapse rare diagnosis codes to ICD-9 chapter groups
- `split.make_splits(df, seed=42) -> dict[str, pd.DataFrame]` — keys `train`, `test`, `reference` (reference = sample of train, frozen drift baseline); writes parquet to `data/processed/`

- [ ] TDD each module: failing test → implement → pass (tests use small synthetic frames, no network)
- [ ] Commit `feat: data layer with pandera validation and splits`

### Task 3: Features + training + promotion

**Files:** Create `src/readmit/features/build.py`, `src/readmit/train/pipeline.py`, `src/readmit/train/evaluate.py`, `src/readmit/train/promote.py`, `flows/train_flow.py`; Test `tests/test_features.py`, `tests/test_promote.py`, `tests/test_pipeline.py`

**Interfaces:**
- `build.FEATURES: list[str]`, `build.make_preprocessor() -> ColumnTransformer`
- `pipeline.make_model() -> sklearn.Pipeline` (preprocessor + XGBClassifier, `scale_pos_weight` for imbalance, calibrated)
- `evaluate.evaluate(model, X, y) -> dict[str, float]` (auc, ap, brier)
- `promote.should_promote(challenger_auc: float, champion_auc: float | None) -> bool` — True if champion None or challenger strictly better
- `train_flow.train(data_dir: str, mlflow_uri: str)` — Prefect flow: load parquet → fit → evaluate → log to MLflow → register `readmit` → set alias `production` iff `should_promote`; logs SHAP summary artifact

- [ ] TDD promote + features on synthetic data; smoke-test full pipeline on 500-row sample with local `mlflow` file store
- [ ] Commit `feat: training pipeline with MLflow registry and champion promotion`

### Task 4: Serving API

**Files:** Create `src/readmit/api/main.py`, `src/readmit/api/schemas.py`, `src/readmit/api/model.py`, `src/readmit/api/logging_.py`; Test `tests/test_api.py`

**Interfaces:**
- `schemas.PatientEncounter` — pydantic model mirroring `build.FEATURES`
- `schemas.PredictionResponse` — `risk_score: float`, `risk_band: Literal["low","medium","high"]`, `top_factors: list[Factor]`, `model_version: str`
- `model.ModelStore.load()` — loads `models:/readmit@production`; API returns 503 when absent
- Endpoints: `POST /predict`, `GET /health`, `GET /metrics` (prometheus), predictions appended to `data/scoring_log.parquet`

- [ ] Tests with dummy model injected via dependency override: 200 happy path, 422 bad payload, 503 no model, bands boundaries (`<0.3 low`, `<0.6 medium`, else high)
- [ ] Commit `feat: FastAPI serving with SHAP factors and Prometheus metrics`

### Task 5: Monitoring + auto-retrain

**Files:** Create `src/readmit/monitoring/drift.py`, `flows/monitor_flow.py`, `scripts/simulate_traffic.py`; Test `tests/test_drift.py`

**Interfaces:**
- `drift.compute_drift(reference: pd.DataFrame, current: pd.DataFrame) -> DriftResult` (`share_drifted: float`, `drifted_columns: list[str]`, `report_html: str`)
- `monitor_flow.monitor(threshold: float = 0.3)` — empty scoring log → skip; else Evidently report → write `reports/drift_<date>.html` → if `share_drifted > threshold` trigger train flow (Prefect run_deployment; direct call fallback)
- `simulate_traffic.py --drift` shifts age/num_medications distributions

- [ ] TDD drift on synthetic shifted frames (assert drift detected when shifted, not when identical)
- [ ] Commit `feat: Evidently drift monitoring with auto-retrain trigger`

### Task 6: Containerisation

**Files:** Create `docker/Dockerfile.api`, `docker/Dockerfile.jobs`, `docker-compose.yml`, `docker/prometheus.yml`, `docker/grafana/provisioning/...` (datasource + dashboard json), `.env.example`

**Services:** api:8000, mlflow:5000 (postgres backend), prefect:4200 (+worker), postgres, prometheus:9090, grafana:3000, reports (nginx static on `reports/`).

- [ ] Multi-stage builds, non-root user, healthchecks; `docker compose config` valid; `docker compose up` smoke: `/health` 200, mlflow UI up
- [ ] Commit `feat: docker compose stack with observability`

### Task 7: CI/CD + docs + publish

**Files:** Create `.github/workflows/ci.yml`, `README.md` (full: badges, mermaid architecture diagram, 5-min quickstart, demo script), `docs/MODEL_CARD.md`, `docs/RUNBOOK.md`

- [ ] CI: ruff → mypy → pytest → docker build → Trivy → push ghcr on main
- [ ] Model card: intended use, dataset provenance, metrics, bias/limitations, NHS governance framing
- [ ] Create public GitHub repo `saisarantottempudi/readmit-ai`, push, verify CI
- [ ] Commit `docs: README, model card, runbook` + `ci: GitHub Actions pipeline`

---

Self-review: spec sections all mapped (data→T2, train→T3, serve→T4, monitor→T5, compose→T6, CI/docs→T7); interfaces consistent (`readmit` registry name, `production` alias, `build.FEATURES` shared by T3/T4).
