#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
output="${project_root}/workspace/logs/setup/VERSIONS.txt"
mkdir -p "$(dirname "${output}")"
{
  date -Iseconds
  uname -a
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true
  for repo in unitree_rl_lab unitree_ros unitree_mujoco unitree_sdk2 unitree_sdk2_python; do
    git -C "${project_root}/third_party/${repo}" rev-parse HEAD 2>/dev/null | sed "s/^/${repo}: /"
  done
  git -C "${project_root}/.." rev-parse HEAD 2>/dev/null | sed 's/^/IsaacLab: /'
} > "${output}"
echo "${output}"
