#!/usr/bin/env python3
"""Fast, side-effect-free project and dependency audit."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
import json
import platform
import shutil
import subprocess
import sys

from humanoid_g1.assets.joint_contract import load_joint_contract
from humanoid_g1.utils.config import load_experiment
from humanoid_g1.utils.paths import ARTIFACTS, MUJOCO_SCENE_PATH, ROOT, URDF_PATH


def command_output(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return (result.stdout or result.stderr).strip()


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def main() -> int:
    checks: list[tuple[str, bool, str]] = []
    contract = load_joint_contract()
    try:
        contract.validate_urdf()
        contract.validate_mujoco()
        checks.append(("joint contract", True, "URDF and MuJoCo orders match 29-DoF contract"))
    except Exception as exc:
        checks.append(("joint contract", False, str(exc)))
    try:
        load_experiment("configs/experiments/g1_flat_ppo.yaml")
        checks.append(("experiment config", True, "composition and validation passed"))
    except Exception as exc:
        checks.append(("experiment config", False, str(exc)))
    checks.extend(
        [
            ("official G1 URDF", URDF_PATH.is_file(), str(URDF_PATH)),
            ("official MuJoCo scene", MUJOCO_SCENE_PATH.is_file(), str(MUJOCO_SCENE_PATH)),
            (
                "local Unitree MuJoCo binary",
                (ROOT / ".local" / "unitree_mujoco" / "bin" / "unitree_mujoco").is_file(),
                str(ROOT / ".local" / "unitree_mujoco" / "bin" / "unitree_mujoco"),
            ),
            (
                "C++ LibTorch controller",
                (ROOT / "deploy" / "generated" / "g1_controller").is_file(),
                str(ROOT / "deploy" / "generated" / "g1_controller"),
            ),
            ("Isaac Lab launcher", (ROOT.parent / "isaaclab.sh").is_file(), str(ROOT.parent / "isaaclab.sh")),
            ("CUDA driver", shutil.which("nvidia-smi") is not None, command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]) if shutil.which("nvidia-smi") else "nvidia-smi not found"),
        ]
    )
    print("Unitree G1 research environment doctor")
    print(f"  project : {ROOT}")
    print(f"  host    : {platform.platform()}")
    print(f"  python  : {sys.version.split()[0]}")
    print(f"  packages: torch={package_version('torch')}, rsl-rl-lib={package_version('rsl-rl-lib')}, "
          f"onnx={package_version('onnx')}, mujoco-python={package_version('mujoco')}")
    for label, passed, detail in checks:
        print(f"  [{'OK' if passed else 'FAIL'}] {label}: {detail}")
    report = {
        "project_root": str(ROOT),
        "python": sys.version,
        "packages": {name: package_version(name) for name in ("torch", "rsl-rl-lib", "onnx", "mujoco")},
        "checks": [{"name": a, "passed": b, "detail": c} for a, b, c in checks],
    }
    output = ARTIFACTS / "reports" / "doctor.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if all(item[1] for item in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
