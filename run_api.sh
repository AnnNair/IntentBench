#!/usr/bin/env bash
# Local serving entrypoint -- replaces a Docker CMD (no Docker in this
# environment; see README.md "Reproducibility without Docker").
set -e
source .venv/bin/activate
export MODEL_ADAPTER_PATH=${MODEL_ADAPTER_PATH:-outputs/best}
export BASE_MODEL=${BASE_MODEL:-google/gemma-2-2b-it}
uvicorn serving.app:app --host 0.0.0.0 --port 8000
