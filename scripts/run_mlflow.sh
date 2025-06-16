#!/usr/bin/env bash
mlflow ui \
--backend-store-uri mlflow/mlruns \
--default-artifact-root mlflow/artifacts \
--host 0.0.0.0 \
--port 5001