"""Run manifests and content hashing."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import hashlib
import json
import os
import platform
import subprocess

from humanoid_g1.utils.paths import ROOT


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def create_manifest(run_dir: Path, command: Iterable[str], config_path: Path | None = None) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(command),
        "cwd": str(Path.cwd()),
        "project_root": str(ROOT),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "isaaclab_git": git_revision(ROOT.parent),
        "unitree_rl_lab_git": git_revision(ROOT / "third_party" / "unitree_rl_lab"),
        "unitree_ros_git": git_revision(ROOT / "third_party" / "unitree_ros"),
        "unitree_mujoco_git": git_revision(ROOT / "third_party" / "unitree_mujoco"),
        "unitree_sdk2_git": git_revision(ROOT / "third_party" / "unitree_sdk2"),
        "config": str(config_path) if config_path else None,
        "config_sha256": sha256(config_path) if config_path and config_path.is_file() else None,
        "environment": {
            key: os.environ.get(key)
            for key in ("CUDA_VISIBLE_DEVICES", "HUMANOID_G1_ROOT")
            if os.environ.get(key) is not None
        },
    }
    output = run_dir / "run_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output

