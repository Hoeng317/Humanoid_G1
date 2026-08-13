#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
simulator="${project_root}/.local/unitree_mujoco/bin/unitree_mujoco"
controller_launcher="${project_root}/scripts/sim2sim/launch_controller.sh"
scene="${project_root}/third_party/unitree_mujoco/unitree_robots/g1/scene_29dof.xml"
report_dir="${project_root}/workspace/artifacts/reports"
policy=""
operator_mode="idle"
duration_s=0
use_xvfb=0
sim_pid=""
controller_pid=""
started_at="$(date --iso-8601=seconds)"
mkdir -p "${report_dir}"
run_stamp="$(date +%Y%m%d_%H%M%S)"
sim_log="${report_dir}/sim2sim_${run_stamp}_mujoco.log"
controller_log="${report_dir}/sim2sim_${run_stamp}_controller.log"
report="${report_dir}/SIM2SIM_${run_stamp}.md"

usage() {
  cat <<'EOF'
Usage: ./humanoid_G1/g1.sh sim2sim --policy PATH [options]

Options:
  --idle          Receive LowState only; never create LowCmd publisher (default)
  --stand         Explicitly move to the default standing pose
  --run           Explicitly enter POLICY_RUNNING after the stand transition
  --duration SEC  Stop automatically after SEC seconds (0: until Ctrl-C)
  --xvfb          Use a virtual X display for automated/headless validation

Simulation is hard-restricted to interface lo and DDS domain 1.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy) policy="${2:?missing value for --policy}"; shift 2 ;;
    --idle) operator_mode="idle"; shift ;;
    --stand) operator_mode="stand"; shift ;;
    --run) operator_mode="run"; shift ;;
    --duration) duration_s="${2:?missing value for --duration}"; shift 2 ;;
    --xvfb) use_xvfb=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown sim2sim option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${policy}" ]]; then
  echo "--policy is required" >&2
  exit 2
fi
if ! [[ "${duration_s}" =~ ^[0-9]+$ ]]; then
  echo "--duration must be a non-negative integer" >&2
  exit 2
fi
policy="$(realpath "${policy}")"

export HUMANOID_G1_ROOT="${project_root}"
export PYTHONPATH="${project_root}/source${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${project_root}/.local/unitree_sdk2/lib:${project_root}/.local/mujoco/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

"${project_root}/../_isaac_sim/python.sh" \
  "${project_root}/scripts/inspect/verify_joint_contract.py" >/dev/null
(
  cd "$(dirname "${policy}")"
  sha256sum --check --quiet SHA256SUMS
)
if [[ ! -x "${simulator}" ]]; then
  echo "Unitree MuJoCo is not built; run scripts/setup/build_mujoco.sh" >&2
  exit 1
fi
if [[ ! -f "${scene}" ]]; then
  echo "Official G1 29-DoF scene not found: ${scene}" >&2
  exit 1
fi

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  if [[ -n "${controller_pid}" ]] && kill -0 "${controller_pid}" 2>/dev/null; then
    kill -TERM "${controller_pid}" 2>/dev/null || true
    for _ in 1 2 3 4 5; do
      kill -0 "${controller_pid}" 2>/dev/null || break
      sleep 0.2
    done
    kill -KILL "${controller_pid}" 2>/dev/null || true
  fi
  if [[ -n "${sim_pid}" ]] && kill -0 "${sim_pid}" 2>/dev/null; then
    kill -TERM "${sim_pid}" 2>/dev/null || true
    wait "${sim_pid}" 2>/dev/null || true
  fi
  local controller_state="not-started"
  local low_state="not-observed"
  local low_cmd="not-created"
  if [[ -f "${controller_log}" ]]; then
    grep -q "valid G1 LowState received" "${controller_log}" && low_state="received"
    grep -q "POLICY_RUNNING" "${controller_log}" && controller_state="POLICY_RUNNING"
    grep -q "remaining IDLE" "${controller_log}" && controller_state="IDLE"
    if [[ "${operator_mode}" != "idle" ]] && [[ "${low_state}" == "received" ]]; then
      low_cmd="enabled-after-explicit-${operator_mode}"
    fi
  fi
  {
    echo "# MuJoCo Sim2Sim run"
    echo
    echo "- Started: ${started_at}"
    echo "- Finished: $(date --iso-8601=seconds)"
    echo "- Policy: ${policy}"
    echo "- Robot/scene: Unitree G1 / scene_29dof.xml"
    echo "- DDS: domain 1, interface lo"
    echo "- Requested operator mode: ${operator_mode}"
    echo "- LowState: ${low_state}"
    echo "- Controller state: ${controller_state}"
    echo "- LowCmd: ${low_cmd}"
    echo "- Launcher exit code: ${exit_code}"
    echo "- MuJoCo log: ${sim_log}"
    echo "- Controller log: ${controller_log}"
  } > "${report}"
  echo "Sim2Sim report: ${report}"
  exit "${exit_code}"
}
trap cleanup EXIT INT TERM

sim_args=("${simulator}" --domain_id 1 --network lo --robot g1 --scene "${scene}")
if [[ "${use_xvfb}" -eq 1 ]]; then
  xvfb-run -a "${sim_args[@]}" >"${sim_log}" 2>&1 &
else
  "${sim_args[@]}" >"${sim_log}" 2>&1 &
fi
sim_pid=$!
sleep 2
if ! kill -0 "${sim_pid}" 2>/dev/null; then
  echo "MuJoCo exited during startup; see ${sim_log}" >&2
  exit 1
fi

"${controller_launcher}" --policy "${policy}" "--${operator_mode}" >"${controller_log}" 2>&1 &
controller_pid=$!

if [[ "${duration_s}" -gt 0 ]]; then
  deadline=$((SECONDS + duration_s))
  while (( SECONDS < deadline )); do
    kill -0 "${sim_pid}" 2>/dev/null || { echo "MuJoCo exited unexpectedly" >&2; exit 1; }
    if [[ "${operator_mode}" != "idle" ]]; then
      kill -0 "${controller_pid}" 2>/dev/null || {
        wait "${controller_pid}" || true
        echo "Controller exited unexpectedly; see ${controller_log}" >&2
        exit 1
      }
    fi
    sleep 1
  done
else
  echo "MuJoCo PID ${sim_pid}, controller PID ${controller_pid}. Press Ctrl-C to stop safely."
  wait "${sim_pid}"
fi
