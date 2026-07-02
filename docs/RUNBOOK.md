# Runbook — readmit-ai

## Start / stop

```bash
cp .env.example .env        # optional; defaults work
docker compose up -d --build
docker compose down          # keep volumes
docker compose down -v       # wipe data, registry, dashboards
```

First boot: the `jobs` container downloads the UCI dataset, runs the ingest flow,
trains the first model and promotes it. Allow ~3–5 minutes, then:

```bash
curl -s localhost:8000/health          # expect model_ready: true
curl -s -X POST localhost:8000/reload  # force API to pick up a new champion
```

## Service map

| URL | Service |
|-----|---------|
| http://localhost:8000/docs | Prediction API (OpenAPI UI) |
| http://localhost:5001 | MLflow tracking + registry |
| http://localhost:4200 | Prefect UI |
| http://localhost:9090 | Prometheus |
| http://localhost:3000 | Grafana (anonymous viewer enabled) |
| http://localhost:8081 | Evidently drift reports |

## Demo the drift → retrain loop

```bash
# venv on host: pip install -e .
python scripts/simulate_traffic.py --n 100            # normal traffic
python scripts/simulate_traffic.py --n 100 --drift    # older, sicker population
docker compose exec jobs python -m flows.monitor_flow # or wait for the scheduled run
curl -s -X POST localhost:8000/reload                 # load new champion if promoted
```

Expected: monitor flow reports `share_drifted > 0.3`, triggers the training flow,
champion/challenger gate decides promotion, drift report appears at :8081.

## Common failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| `/predict` returns 503 | First training not finished, or no champion | `docker compose logs jobs`; wait, then `POST /reload` |
| mlflow container restart loop | postgres not ready / wiped volume mid-flight | `docker compose down && docker compose up -d` |
| Monitor flow always skips | fewer than 50 rows in scoring log | send more traffic |
| API serves stale model after retrain | API caches the champion in memory | `POST /reload` |

## Retention & privacy

The scoring log stores request features and scores in the `shared-data` volume —
synthetic/demo traffic only. Wipe with `docker compose down -v`.
