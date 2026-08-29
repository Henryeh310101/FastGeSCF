#!/usr/bin/env bash
set -euo pipefail

python -m compileall -q framework.py registration.py project_config.py model datasets py_utils experiments
python scripts/smoke_test.py
