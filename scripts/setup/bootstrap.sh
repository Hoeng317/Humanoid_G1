#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export HUMANOID_G1_ROOT="${project_root}"
export PYTHONPATH="${project_root}/source${PYTHONPATH:+:${PYTHONPATH}}"
python_bin="${project_root}/../_isaac_sim/python.sh"
"${python_bin}" -c "import humanoid_g1; print('humanoid_g1 import: OK')"
"${project_root}/scripts/setup/collect_versions.sh"
echo "Bootstrap complete without modifying the shared Isaac Sim Python environment."
