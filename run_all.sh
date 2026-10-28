#!/usr/bin/env bash
# Train + register the champion model, then run the three drift scenarios.
set -e
export MLFLOW_DISABLE_AGENT_HINT=1
python src/train.py
python src/drift.py
echo "Done. Start the UI with: mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000"
