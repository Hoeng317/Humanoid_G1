#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mujoco_version="3.3.6"
archive="${project_root}/.local/downloads/mujoco-${mujoco_version}-linux-x86_64.tar.gz"
expected_sha256="049204172901afad251070385a6badf46d795ebe47403d093f8469557eeeab5a"
mkdir -p "${project_root}/.local/downloads" "${project_root}/.local/mujoco"
if [[ ! -f "${archive}" ]]; then
  curl -L --fail --retry 3 -o "${archive}" \
    "https://github.com/google-deepmind/mujoco/releases/download/${mujoco_version}/mujoco-${mujoco_version}-linux-x86_64.tar.gz"
fi
echo "${expected_sha256}  ${archive}" | sha256sum --check
if [[ ! -f "${project_root}/.local/mujoco/lib/libmujoco.so.${mujoco_version}" ]]; then
  tar -xzf "${archive}" -C "${project_root}/.local/mujoco" --strip-components=1
fi

if [[ ! -f "${project_root}/.local/glfw/lib/libglfw3.a" ]]; then
  if [[ ! -d "${project_root}/.local/src/glfw/.git" ]]; then
    git clone --depth 1 --branch 3.4 https://github.com/glfw/glfw.git "${project_root}/.local/src/glfw"
  fi
  cmake -S "${project_root}/.local/src/glfw" -B "${project_root}/.local/build-glfw" -G Ninja \
    -DCMAKE_INSTALL_PREFIX="${project_root}/.local/glfw" \
    -DGLFW_BUILD_EXAMPLES=OFF -DGLFW_BUILD_TESTS=OFF -DGLFW_BUILD_DOCS=OFF
  cmake --build "${project_root}/.local/build-glfw"
  cmake --install "${project_root}/.local/build-glfw"
fi

"${project_root}/scripts/setup/build_local_sdk.sh"
cmake -S "${project_root}/deploy/sim2sim" -B "${project_root}/.local/build-unitree-mujoco" -G Ninja \
  -DCMAKE_PREFIX_PATH="${project_root}/.local/unitree_sdk2;${project_root}/.local/glfw"
cmake --build "${project_root}/.local/build-unitree-mujoco"
echo "Unitree MuJoCo: ${project_root}/.local/unitree_mujoco/bin/unitree_mujoco"
