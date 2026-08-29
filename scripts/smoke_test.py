"""FastGeSCF smoke checks for imports and experiment CLIs."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODULES = [
    "framework",
    "registration",
    "model.CD_model",
    "datasets.changesim",
    "datasets.vl_cmu_cd",
    "datasets.pscd",
    "py_utils.vpr",
]

HELP_COMMANDS = [
    ["experiments/train.py", "--help"],
    ["experiments/evaluate_datasets.py", "--help"],
    ["experiments/infer_pair.py", "--help"],
    ["experiments/video_change_detection.py", "--help"],
    ["experiments/temporal_alignment.py", "--help"],
]


def check_imports() -> None:
    for module_name in MODULES:
        importlib.import_module(module_name)
        print(f"import ok: {module_name}")


def check_help_commands() -> None:
    for command in HELP_COMMANDS:
        result = subprocess.run(
            [sys.executable, *command],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            raise SystemExit(f"CLI failed: {' '.join(command)}")
        print(f"cli ok: {' '.join(command)}")


def main() -> None:
    check_imports()
    check_help_commands()
    print("smoke checks passed")


if __name__ == "__main__":
    main()
