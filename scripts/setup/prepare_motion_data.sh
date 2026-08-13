#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
pipeline_root="${project_root}/ACCAD/_g1_pipeline"
accad_root="${project_root}/ACCAD"
python_bin="${project_root}/../_isaac_sim/python.sh"
# shellcheck source=dependency_lock.env
source "${project_root}/scripts/setup/dependency_lock.env"

model_source=""
check_only=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smplx-model)
      [[ $# -ge 2 ]] || { echo "ERROR: --smplx-model needs a path." >&2; exit 2; }
      model_source="$2"
      shift 2
      ;;
    --check-only)
      check_only=true
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Usage: prepare_motion_data.sh [--smplx-model /path/to/SMPLX_NEUTRAL.npz] [--check-only]

Before running, extract the licensed AMASS ACCAD SMPL-X release directly into
this repository's ACCAD/ directory. The script never downloads or commits the
licensed files. If --smplx-model is omitted, it looks for the official
locked-head archive at:
  ACCAD/smplx_lockedhead_20230207/models_lockedhead/smplx/SMPLX_NEUTRAL.npz
EOF
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

[[ -x "${python_bin}" ]] || {
  echo "ERROR: Isaac Sim Python is missing: ${python_bin}" >&2
  exit 1
}

stageii_count="$(find "${accad_root}" -path "${pipeline_root}" -prune -o -type f -name '*_stageii.npz' -print | wc -l)"
stagei_count="$(find "${accad_root}" -path "${pipeline_root}" -prune -o -type f -name '*_stagei.npz' -print | wc -l)"
if [[ "${stageii_count}" -ne "${ACCAD_EXPECTED_STAGEII_COUNT}" ]]; then
  echo "ERROR: found ${stageii_count} ACCAD stage-II files; expected ${ACCAD_EXPECTED_STAGEII_COUNT}." >&2
  echo "Extract the AMASS ACCAD SMPL-X release directly below ${accad_root}." >&2
  exit 1
fi
if [[ "${stagei_count}" -ne "${ACCAD_EXPECTED_STAGEI_COUNT}" ]]; then
  echo "ERROR: found ${stagei_count} ACCAD stage-I files; expected ${ACCAD_EXPECTED_STAGEI_COUNT}." >&2
  exit 1
fi
echo "[OK] ACCAD files: stage-II=${stageii_count}, stage-I=${stagei_count}"

accad_aggregate_sha="$(
  find "${accad_root}" -path "${pipeline_root}" -prune -o -type f \
    \( -name '*_stagei.npz' -o -name '*_stageii.npz' \) -print \
    | LC_ALL=C sort \
    | while IFS= read -r path; do
        digest="$(sha256sum "${path}" | awk '{print $1}')"
        relative="${path#${accad_root}/}"
        printf '%s  %s\n' "${digest}" "${relative}"
      done \
    | sha256sum \
    | awk '{print $1}'
)"
if [[ "${accad_aggregate_sha}" != "${ACCAD_STAGE_FILES_AGGREGATE_SHA256}" ]]; then
  echo "ERROR: ACCAD file-set checksum differs from the release used by this project." >&2
  echo "actual=${accad_aggregate_sha}" >&2
  echo "expected=${ACCAD_STAGE_FILES_AGGREGATE_SHA256}" >&2
  exit 1
fi
echo "[OK] ACCAD aggregate checksum ${accad_aggregate_sha}"

default_model="${accad_root}/smplx_lockedhead_20230207/models_lockedhead/smplx/SMPLX_NEUTRAL.npz"
model_link="${pipeline_root}/models/smplx/SMPLX_NEUTRAL.npz"
if [[ -z "${model_source}" ]] && [[ -f "${default_model}" ]]; then
  model_source="${default_model}"
fi
if [[ -z "${model_source}" ]] && [[ -e "${model_link}" ]]; then
  model_source="$(realpath "${model_link}")"
fi
if [[ -z "${model_source}" ]] || [[ ! -f "${model_source}" ]]; then
  echo "ERROR: licensed SMPL-X neutral locked-head model was not found." >&2
  echo "Pass --smplx-model /absolute/path/to/SMPLX_NEUTRAL.npz." >&2
  exit 1
fi
model_source="$(realpath "${model_source}")"
actual_model_sha="$(sha256sum "${model_source}" | awk '{print $1}')"
if [[ "${actual_model_sha}" != "${SMPLX_NEUTRAL_LOCKEDHEAD_SHA256}" ]]; then
  echo "ERROR: SMPL-X model checksum differs from the version used by this project." >&2
  echo "actual=${actual_model_sha}" >&2
  echo "expected=${SMPLX_NEUTRAL_LOCKEDHEAD_SHA256}" >&2
  exit 1
fi
echo "[OK] SMPL-X neutral locked-head checksum ${actual_model_sha}"

if [[ "${check_only}" == false ]]; then
  mkdir -p "$(dirname "${model_link}")"
  if [[ -e "${model_link}" ]] || [[ -L "${model_link}" ]]; then
    linked_target="$(realpath "${model_link}" 2>/dev/null || true)"
    if [[ "${linked_target}" != "${model_source}" ]]; then
      echo "ERROR: ${model_link} already points to another file; refusing to replace it." >&2
      exit 1
    fi
  else
    ln -s "${model_source}" "${model_link}"
  fi
  "${python_bin}" -m pip install \
    --disable-pip-version-check --no-deps --require-hashes --upgrade \
    --target "${pipeline_root}/vendor" \
    -r "${pipeline_root}/requirements-motion.txt"
fi

export PYTHONDONTWRITEBYTECODE=1
"${python_bin}" "${pipeline_root}/run.py" audit >/dev/null
"${python_bin}" "${pipeline_root}/run.py" preflight >/dev/null
echo "[OK] ACCAD Gate 1 audit and SMPL-X Gate 2 preflight passed."
echo "Motion-data preparation completed. Generated files remain ignored by Git."
