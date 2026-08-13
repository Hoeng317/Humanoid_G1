#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
sdk_source="${project_root}/third_party/unitree_sdk2"
sdk_prefix="${project_root}/.local/unitree_sdk2"
cmake -S "${sdk_source}" -B "${project_root}/.local/build-unitree-sdk2" -G Ninja \
  -DBUILD_EXAMPLES=OFF -DCMAKE_INSTALL_PREFIX="${sdk_prefix}"
cmake --build "${project_root}/.local/build-unitree-sdk2"
cmake --install "${project_root}/.local/build-unitree-sdk2"
echo "Unitree SDK2 local prefix: ${sdk_prefix}"

