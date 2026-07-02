.PHONY: install fmt lint test train serve up down simulate

install:
	pip install -e ".[dev]"

fmt:
	ruff format src tests flows scripts
	ruff check --fix src tests flows scripts

lint:
	ruff check src tests flows scripts
	mypy src

test:
	pytest --cov=src/readmit --cov-report=term-missing

train:
	python -m flows.train_flow

serve:
	uvicorn readmit.api.main:app --host 0.0.0.0 --port 8000 --reload

up:
	docker compose up -d --build

down:
	docker compose down -v

simulate:
	python scripts/simulate_traffic.py
