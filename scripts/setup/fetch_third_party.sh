#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=dependency_lock.env
source "${project_root}/scripts/setup/dependency_lock.env"

mode=fetch
if [[ "${1:-}" == "--check-only" ]]; then
  mode=check
  shift
fi
if [[ $# -ne 0 ]]; then
  echo "Usage: $0 [--check-only]" >&2
  exit 2
fi

command -v git >/dev/null || {
  echo "ERROR: git is required." >&2
  exit 1
}

ensure_repository() {
  local name="$1"
  local repository="$2"
  local expected_commit="$3"
  local destination="${project_root}/third_party/${name}"

  if [[ ! -d "${destination}/.git" ]]; then
    if [[ "${mode}" == "check" ]]; then
      echo "[MISSING] ${name}: ${destination}" >&2
      return 1
    fi
    if [[ -e "${destination}" ]] && [[ -n "$(find "${destination}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
      echo "ERROR: ${destination} exists but is not a Git repository." >&2
      return 1
    fi
    mkdir -p "${destination}"
    git -C "${destination}" init --quiet
    git -C "${destination}" remote add origin "${repository}"
    git -C "${destination}" fetch --depth 1 origin "${expected_commit}"
    git -C "${destination}" checkout --quiet --detach FETCH_HEAD
  fi

  local actual_commit
  actual_commit="$(git -C "${destination}" rev-parse HEAD)"
  if [[ "${actual_commit}" != "${expected_commit}" ]]; then
    if [[ -n "$(git -C "${destination}" status --porcelain)" ]]; then
      echo "ERROR: ${name} has local changes and the wrong commit; refusing to overwrite it." >&2
      return 1
    fi
    if [[ "${mode}" == "check" ]]; then
      echo "[WRONG COMMIT] ${name}: ${actual_commit}, expected ${expected_commit}" >&2
      return 1
    fi
    git -C "${destination}" fetch --depth 1 origin "${expected_commit}"
    git -C "${destination}" checkout --quiet --detach "${expected_commit}"
    actual_commit="$(git -C "${destination}" rev-parse HEAD)"
  fi

  if [[ -n "$(git -C "${destination}" status --porcelain)" ]]; then
    echo "ERROR: ${name} contains local modifications; dependency must remain pristine." >&2
    return 1
  fi
  if [[ "${actual_commit}" != "${expected_commit}" ]]; then
    echo "ERROR: ${name} commit verification failed." >&2
    return 1
  fi
  echo "[OK] ${name} ${actual_commit}"
}

mkdir -p "${project_root}/third_party"
ensure_repository unitree_rl_lab "${UNITREE_RL_LAB_REPOSITORY}" "${UNITREE_RL_LAB_COMMIT}"
ensure_repository unitree_ros "${UNITREE_ROS_REPOSITORY}" "${UNITREE_ROS_COMMIT}"
ensure_repository unitree_mujoco "${UNITREE_MUJOCO_REPOSITORY}" "${UNITREE_MUJOCO_COMMIT}"
ensure_repository unitree_sdk2 "${UNITREE_SDK2_REPOSITORY}" "${UNITREE_SDK2_COMMIT}"
ensure_repository unitree_sdk2_python "${UNITREE_SDK2_PYTHON_REPOSITORY}" "${UNITREE_SDK2_PYTHON_COMMIT}"

echo "Third-party dependency ${mode} completed."
