#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=dependency_lock.env
source "${project_root}/scripts/setup/dependency_lock.env"
mujoco_version="${MUJOCO_VERSION}"
archive="${project_root}/.local/downloads/mujoco-${mujoco_version}-linux-x86_64.tar.gz"
expected_sha256="${MUJOCO_ARCHIVE_SHA256}"
mkdir -p "${project_root}/.local/downloads" "${project_root}/.local/mujoco"
if [[ ! -f "${archive}" ]]; then
  curl -L --fail --retry 3 -o "${archive}" \
    "https://github.com/google-deepmind/mujoco/releases/download/${mujoco_version}/mujoco-${mujoco_version}-linux-x86_64.tar.gz"
fi
echo "${expected_sha256}  ${archive}" | sha256sum --check
if [[ ! -f "${project_root}/.local/mujoco/lib/libmujoco.so.${mujoco_version}" ]]; then
  tar -xzf "${archive}" -C "${project_root}/.local/mujoco" --strip-components=1
fi

if [[ ! -d "${project_root}/.local/src/glfw/.git" ]]; then
  mkdir -p "${project_root}/.local/src/glfw"
  git -C "${project_root}/.local/src/glfw" init --quiet
  git -C "${project_root}/.local/src/glfw" remote add origin "${GLFW_REPOSITORY}"
  git -C "${project_root}/.local/src/glfw" fetch --depth 1 origin "${GLFW_COMMIT}"
  git -C "${project_root}/.local/src/glfw" checkout --quiet --detach FETCH_HEAD
fi
actual_glfw_commit="$(git -C "${project_root}/.local/src/glfw" rev-parse HEAD)"
[[ "${actual_glfw_commit}" == "${GLFW_COMMIT}" ]] || {
  echo "ERROR: GLFW commit ${actual_glfw_commit}; expected ${GLFW_COMMIT}." >&2
  exit 1
}
[[ -z "$(git -C "${project_root}/.local/src/glfw" status --porcelain)" ]] || {
  echo "ERROR: GLFW source contains local modifications." >&2
  exit 1
}

if [[ ! -f "${project_root}/.local/glfw/lib/libglfw3.a" ]]; then
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
