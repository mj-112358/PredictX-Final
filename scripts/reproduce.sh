#!/bin/sh
set -eu
python scripts/evaluate.py
python scripts/benchmark_refined.py
python scripts/benchmark_model_loading.py
python -m pytest -q
