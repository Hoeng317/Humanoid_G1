"""Canonical paths and configuration loading for the local pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parent
PIPELINE_ROOT = PACKAGE_ROOT.parent
ACCAD_ROOT = PIPELINE_ROOT.parent
PROJECT_ROOT = ACCAD_ROOT.parent

DEFAULT_CONFIG = PIPELINE_ROOT / "config.yaml"
DEFAULT_WORK_ROOT = PIPELINE_ROOT / "work"
DEFAULT_MODEL_ROOT = PIPELINE_ROOT / "models"
DEFAULT_VENDOR_ROOT = PIPELINE_ROOT / "vendor"
DEFAULT_B3_INPUT = ACCAD_ROOT / "Female1Walking_c3d" / "B3_-_walk1_stageii.npz"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config and resolve supported relative paths locally."""

    config_path = Path(path).expanduser().resolve() if path else DEFAULT_CONFIG
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    config["_config_path"] = str(config_path)
    return config


def local_path(value: str | Path, *, base: Path = PIPELINE_ROOT) -> Path:
    """Resolve a config path relative to the pipeline, never to the shell CWD."""

    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def assert_output_is_local(path: str | Path) -> Path:
    """Reject writes outside ACCAD/_g1_pipeline."""

    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(PIPELINE_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"Output must stay inside {PIPELINE_ROOT}; received {resolved}"
        ) from exc
    return resolved

