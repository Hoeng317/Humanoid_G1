#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
isaaclab_root="$(cd "${project_root}/.." && pwd)"
python_bin="${isaaclab_root}/_isaac_sim/python.sh"
isaac_python=("${isaaclab_root}/isaaclab.sh" -p)

export HUMANOID_G1_ROOT="${project_root}"
export PYTHONPATH="${project_root}/source${PYTHONPATH:+:${PYTHONPATH}}"
if [[ "${TERM:-dumb}" == "dumb" ]]; then
  export TERM=xterm
fi
cd "${project_root}"

usage() {
  cat <<'EOF'
Usage: ./humanoid_G1/g1.sh COMMAND [options]

Commands:
  doctor         Audit dependencies, assets, configs, and versions
  inspect        Offline G1/joint inspection (`--live` starts Isaac)
  simulate       G1 physics modes: default_pose/free_fall/sine/random_action
  stability      10,000-step default-pose stability acceptance
  train          RSL-RL PPO training
  play           Play a checkpoint
  evaluate       Quantitative checkpoint evaluation
  export         TorchScript/ONNX deployment bundle and golden vectors
  sim2sim        Loopback-only MuJoCo launcher
  deploy-check   Offline observation → inference → target dry-run
  deploy-real    Guarded physical entry point (cannot bypass safety flags)
  test           Unit tests; pass --integration for Isaac tests
  setup          Import check and version collection (shared env unchanged)
EOF
}

command_name="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${command_name}" in
  doctor)
    exec "${python_bin}" scripts/setup/doctor.py "$@"
    ;;
  inspect)
    if [[ "${1:-}" == "--live" ]]; then
      shift
      exec "${isaac_python[@]}" scripts/inspect/inspect_task.py "$@"
    fi
    "${python_bin}" scripts/inspect/verify_joint_contract.py "$@"
    exec "${python_bin}" scripts/inspect/inspect_robot.py
    ;;
  simulate)
    exec "${isaac_python[@]}" scripts/simulation/simulate.py "$@"
    ;;
  stability)
    exec "${isaac_python[@]}" scripts/simulation/simulate.py --headless --mode default_pose --steps 10000 "$@"
    ;;
  train)
    exec "${isaac_python[@]}" scripts/rsl_rl/train.py "$@"
    ;;
  play)
    exec "${isaac_python[@]}" scripts/rsl_rl/play.py "$@"
    ;;
  evaluate)
    exec "${isaac_python[@]}" scripts/rsl_rl/evaluate.py "$@"
    ;;
  export)
    exec "${python_bin}" scripts/rsl_rl/export_policy.py "$@"
    ;;
  sim2sim)
    exec scripts/sim2sim/launch_mujoco.sh "$@"
    ;;
  deploy-check)
    exec "${python_bin}" scripts/deployment/deploy_check.py "$@"
    ;;
  deploy-real)
    exec "${python_bin}" scripts/deployment/deploy_real.py "$@"
    ;;
  test)
    if [[ "${1:-}" == "--integration" ]]; then
      shift
      exec "${isaac_python[@]}" -m pytest -m integration "$@"
    fi
    exec "${python_bin}" -m pytest -m "not integration and not hardware and not sim2sim" "$@"
    ;;
  setup)
    exec scripts/setup/bootstrap.sh "$@"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown command: ${command_name}" >&2
    usage >&2
    exit 2
    ;;
esac
