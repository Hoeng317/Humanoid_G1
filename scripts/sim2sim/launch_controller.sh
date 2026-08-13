#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
controller="${project_root}/deploy/generated/g1_controller"
policy=""
operator_mode="idle"

usage() {
  cat <<'EOF'
Usage: launch_controller.sh --policy PATH [--idle|--stand|--run]

Runs the G1 C++ controller only on loopback DDS domain 1.  Idle is the
default and does not create a LowCmd publisher.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy) policy="${2:?missing value for --policy}"; shift 2 ;;
    --idle) operator_mode="idle"; shift ;;
    --stand) operator_mode="stand"; shift ;;
    --run) operator_mode="run"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown controller option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${policy}" ]]; then
  echo "--policy is required" >&2
  exit 2
fi
policy="$(realpath "${policy}")"
bundle="$(dirname "${policy}")"
if [[ "$(basename "${policy}")" != "policy.pt" ]]; then
  echo "Expected an exported policy named policy.pt: ${policy}" >&2
  exit 2
fi
if [[ ! -x "${controller}" ]]; then
  echo "C++ controller is not built; run scripts/setup/build_controller.sh" >&2
  exit 1
fi
(
  cd "${bundle}"
  sha256sum --check --quiet SHA256SUMS
)

args=(--policy "${policy}" --sim --interface lo --domain-id 1)
case "${operator_mode}" in
  stand) args+=(--stand) ;;
  run) args+=(--run) ;;
esac

exec "${controller}" "${args[@]}"
