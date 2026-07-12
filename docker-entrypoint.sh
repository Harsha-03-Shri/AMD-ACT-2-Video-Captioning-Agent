#!/bin/sh
set -e

if [ "$MODE" = "server" ]; then
    echo "Starting in SERVER mode (Render / API) on port ${PORT:-8000}"
    exec uvicorn app_server:app --host 0.0.0.0 --port "${PORT:-8000}"
else
    echo "Starting in BATCH mode"
    echo "Reading tasks from: ${INPUT_PATH:-/input/tasks.json}"
    echo "Writing results to: ${OUTPUT_PATH:-/output/results.json}"
    exec python -m app.main
fi
