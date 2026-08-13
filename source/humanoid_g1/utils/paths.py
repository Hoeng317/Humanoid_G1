"""Centralized project paths; no user-specific absolute paths are embedded."""

from __future__ import annotations

import os
from pathlib import Path


def _discover_root() -> Path:
    configured = os.environ.get("HUMANOID_G1_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    # source/humanoid_g1/utils/paths.py -> project root
    return Path(__file__).resolve().parents[3]


ROOT = _discover_root()
CONFIGS = ROOT / "configs"
THIRD_PARTY = ROOT / "third_party"
WORKSPACE = ROOT / "workspace"
ARTIFACTS = WORKSPACE / "artifacts"
LOGS = WORKSPACE / "logs"
CONTRACT_PATH = CONFIGS / "robot" / "g1_29dof.yaml"
URDF_PATH = THIRD_PARTY / "unitree_ros" / "robots" / "g1_description" / "g1_29dof_rev_1_0.urdf"
MUJOCO_MODEL_PATH = THIRD_PARTY / "unitree_mujoco" / "unitree_robots" / "g1" / "g1_29dof.xml"
MUJOCO_SCENE_PATH = THIRD_PARTY / "unitree_mujoco" / "unitree_robots" / "g1" / "scene_29dof.xml"


def relative_to_root(path: Path | str) -> str:
    """Return a stable project-relative path when possible."""
    value = Path(path).resolve()
    try:
        return str(value.relative_to(ROOT))
    except ValueError:
        return str(value)
