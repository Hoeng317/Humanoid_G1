#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
isaaclab_root="$(cd "${project_root}/.." && pwd)"
# shellcheck source=dependency_lock.env
source "${project_root}/scripts/setup/dependency_lock.env"

check_only=false
build_native=false
run_tests=true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-only) check_only=true ;;
    --with-native) build_native=true ;;
    --skip-tests) run_tests=false ;;
    -h|--help)
      cat <<'EOF'
Usage: bootstrap.sh [--check-only] [--with-native] [--skip-tests]

  --check-only   verify locked dependencies without cloning or building
  --with-native  additionally build MuJoCo/SDK2 and the C++ controller
  --skip-tests   skip the fast source-level unit tests
EOF
      exit 0
      ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

actual_isaaclab_commit="$(git -C "${isaaclab_root}" rev-parse HEAD 2>/dev/null || true)"
if [[ "${actual_isaaclab_commit}" != "${ISAACLAB_COMMIT}" ]]; then
  echo "ERROR: Isaac Lab commit is ${actual_isaaclab_commit:-missing}." >&2
  echo "Expected ${ISAACLAB_COMMIT} (Isaac Lab ${ISAACLAB_VERSION})." >&2
  echo "Use docs/REINSTALL_FROM_GITHUB_KO.md to create an exact installation." >&2
  exit 1
fi

python_bin="${isaaclab_root}/_isaac_sim/python.sh"
if [[ ! -x "${python_bin}" ]]; then
  echo "ERROR: Isaac Sim ${ISAACSIM_VERSION} must be linked at ${isaaclab_root}/_isaac_sim." >&2
  exit 1
fi
version_file="${isaaclab_root}/_isaac_sim/VERSION"
if [[ ! -f "${version_file}" ]]; then
  echo "ERROR: Isaac Sim VERSION file is missing: ${version_file}" >&2
  exit 1
fi
actual_isaacsim_version="$(head -n 1 "${version_file}" | cut -d- -f1)"
if [[ "${actual_isaacsim_version}" != "${ISAACSIM_VERSION}" ]]; then
  echo "ERROR: Isaac Sim version is ${actual_isaacsim_version}; expected ${ISAACSIM_VERSION}." >&2
  exit 1
fi
echo "[OK] Isaac Lab ${ISAACLAB_COMMIT} / Isaac Sim ${actual_isaacsim_version}"

if [[ "${check_only}" == true ]]; then
  "${project_root}/scripts/setup/fetch_third_party.sh" --check-only
else
  "${project_root}/scripts/setup/fetch_third_party.sh"
fi

export HUMANOID_G1_ROOT="${project_root}"
export PYTHONPATH="${project_root}/source${PYTHONPATH:+:${PYTHONPATH}}"
"${python_bin}" - "${ISAACSIM_PYTHON_VERSION}" "${PYTORCH_VERSION}" \
  "${RSL_RL_VERSION}" "${ONNX_VERSION}" <<'PY'
from importlib import metadata
import sys

expected_python, expected_torch, expected_rsl, expected_onnx = sys.argv[1:]
actual = {
    "Python": ".".join(map(str, sys.version_info[:3])),
    "PyTorch": metadata.version("torch"),
    "RSL-RL": metadata.version("rsl-rl-lib"),
    "ONNX": metadata.version("onnx"),
}
expected = {
    "Python": expected_python,
    "PyTorch": expected_torch,
    "RSL-RL": expected_rsl,
    "ONNX": expected_onnx,
}
wrong = [f"{name}={actual[name]} (expected {value})" for name, value in expected.items() if actual[name] != value]
if wrong:
    raise SystemExit("ERROR: runtime version mismatch: " + ", ".join(wrong))
print("[OK] " + ", ".join(f"{name}={value}" for name, value in actual.items()))
PY
"${python_bin}" -c "import humanoid_g1; print('humanoid_g1 import: OK')"
"${python_bin}" "${project_root}/scripts/inspect/verify_joint_contract.py" >/dev/null

if [[ "${run_tests}" == true ]]; then
  "${python_bin}" -m pytest -q -m "not integration and not hardware and not sim2sim" \
    "${project_root}/tests"
fi

if [[ "${build_native}" == true ]]; then
  if [[ "${check_only}" == true ]]; then
    echo "ERROR: --check-only and --with-native cannot be combined." >&2
    exit 2
  fi
  "${project_root}/scripts/setup/build_mujoco.sh"
  "${project_root}/scripts/setup/build_controller.sh"
  "${python_bin}" "${project_root}/scripts/setup/doctor.py"
fi

"${project_root}/scripts/setup/collect_versions.sh"
echo "Bootstrap complete. The shared Isaac Sim Python environment was not modified."
if [[ "${build_native}" == false ]]; then
  echo "Native Sim2Sim/deployment binaries were not built; rerun with --with-native if needed."
fi
