#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
torch_prefix="$("${project_root}/../_isaac_sim/python.sh" -c 'import torch; print(torch.utils.cmake_prefix_path)')"
"${project_root}/scripts/setup/build_local_sdk.sh"
cmake -S "${project_root}/deploy/cpp" -B "${project_root}/.local/build-controller" -G Ninja \
  -DCMAKE_PREFIX_PATH="${project_root}/.local/unitree_sdk2;${torch_prefix}"
cmake --build "${project_root}/.local/build-controller"
echo "G1 C++ controller: ${project_root}/deploy/generated/g1_controller"

