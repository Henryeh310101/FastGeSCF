"""Shared paths and runtime helpers for FastGeSCF experiments."""

from __future__ import annotations

import os
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = Path(os.environ.get("FASTGESCF_CHECKPOINT_DIR", PROJECT_ROOT / "checkpoints"))
RESULTS_DIR = Path(os.environ.get("FASTGESCF_RESULTS_DIR", PROJECT_ROOT / "results"))
DATA_ROOT = os.environ.get("FASTGESCF_DATA_ROOT")


DATASET_ROOT_ENV = {
    "VL_CMU_CD": "FASTGESCF_VL_CMU_CD_ROOT",
    "vl-cmu-cd": "FASTGESCF_VL_CMU_CD_ROOT",
    "diff-vl-cmu-cd": "FASTGESCF_VL_CMU_CD_ROOT",
    "PSCD": "FASTGESCF_PSCD_ROOT",
    "pscd": "FASTGESCF_PSCD_ROOT",
    "St Lucia": "FASTGESCF_ST_LUCIA_ROOT",
    "St-Lucia": "FASTGESCF_ST_LUCIA_ROOT",
    "Nordland": "FASTGESCF_NORDLAND_ROOT",
    "SF-XL": "FASTGESCF_SF_XL_ROOT",
    "ChangeSim": "FASTGESCF_CHANGESIM_TEST_ROOT",
    "changesim": "FASTGESCF_CHANGESIM_TRAIN_ROOT",
}

DATASET_SUBDIRS = {
    "VL_CMU_CD": "VL-CMU-CD-binary255",
    "vl-cmu-cd": "VL-CMU-CD-binary255",
    "diff-vl-cmu-cd": "VL-CMU-CD-binary255",
    "PSCD": "pscd_test",
    "pscd": "1024x224",
    "St Lucia": "St Lucia",
    "St-Lucia": "St Lucia",
    "Nordland": "Nordland",
    "SF-XL": "SF-XL",
    "ChangeSim": "Query_Seq_Test",
    "changesim": "Query_Seq_Train",
}


def resolve_device(requested: str | None = None) -> str:
    if requested is None or requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def require_file(path: str | Path, label: str) -> Path:
    path = resolve_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def require_dir(path: str | Path, label: str) -> Path:
    path = resolve_path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def dataset_root(dataset: str, override: str | None = None) -> Path:
    if override:
        return require_dir(override, f"{dataset} dataset root")

    env_name = DATASET_ROOT_ENV.get(dataset)
    if env_name and os.environ.get(env_name):
        return require_dir(os.environ[env_name], f"{dataset} dataset root")

    if DATA_ROOT:
        candidate = Path(DATA_ROOT) / DATASET_SUBDIRS.get(dataset, dataset)
        return require_dir(candidate, f"{dataset} dataset root")

    relative = PROJECT_ROOT / "data" / DATASET_SUBDIRS.get(dataset, dataset)
    return require_dir(relative, f"{dataset} dataset root")


def ensure_project_root_on_path() -> None:
    import sys

    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
