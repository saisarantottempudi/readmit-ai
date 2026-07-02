#!/bin/sh
# Create separate databases for MLflow and Prefect on first boot.
set -e
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    CREATE DATABASE mlflow;
    CREATE DATABASE prefect;
EOSQL
