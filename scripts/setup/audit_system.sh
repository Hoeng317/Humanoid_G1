#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export HUMANOID_G1_ROOT="${project_root}"
export PYTHONPATH="${project_root}/source${PYTHONPATH:+:${PYTHONPATH}}"
output="${project_root}/workspace/logs/setup/system_audit.log"
mkdir -p "$(dirname "${output}")"
{
  date --iso-8601=seconds
  uname -a
  cat /etc/os-release
  nvidia-smi
  nvcc --version || true
  python3 --version
  git --version
  cmake --version
  gcc --version | head -n 1
  g++ --version | head -n 1
  free -h
  df -h /home/hoeng
  git -C "${project_root}/.." status --short
  git -C "${project_root}/.." branch --show-current
  git -C "${project_root}/.." rev-parse HEAD
  "${project_root}/../_isaac_sim/python.sh" "${project_root}/scripts/setup/doctor.py" "$@"
} 2>&1 | tee "${output}"
echo "Audit log: ${output}"
