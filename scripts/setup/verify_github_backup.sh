#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${project_root}/../_isaac_sim/python.sh"
with_motion=false
if [[ "${1:-}" == "--with-motion" ]]; then
  with_motion=true
  shift
fi
if [[ $# -ne 0 ]]; then
  echo "Usage: $0 [--with-motion]" >&2
  exit 2
fi

cd "${project_root}"
[[ -x "${python_bin}" ]] || { echo "ERROR: missing ${python_bin}" >&2; exit 1; }

if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "ERROR: the source repository has uncommitted or untracked changes:" >&2
  git status --short >&2
  exit 1
fi

current_branch="$(git branch --show-current)"
[[ "${current_branch}" == "main" ]] || {
  echo "ERROR: expected branch main, found ${current_branch:-detached}." >&2
  exit 1
}

head_commit="$(git rev-parse HEAD)"
remote_commit="$(git ls-remote origin refs/heads/main | awk '{print $1}')"
[[ -n "${remote_commit}" ]] || { echo "ERROR: origin/main was not found." >&2; exit 1; }
[[ "${head_commit}" == "${remote_commit}" ]] || {
  echo "ERROR: local HEAD ${head_commit} differs from origin/main ${remote_commit}." >&2
  exit 1
}
echo "[OK] local main equals GitHub origin/main: ${head_commit}"

required_files=(
  README.md
  docs/REINSTALL_FROM_GITHUB_KO.md
  scripts/setup/dependency_lock.env
  scripts/setup/fetch_third_party.sh
  scripts/setup/prepare_motion_data.sh
  ACCAD/_g1_pipeline/config.yaml
  ACCAD/_g1_pipeline/correspondence.yaml
  ACCAD/_g1_pipeline/accad_g1/retarget.py
  ACCAD/_g1_pipeline/accad_g1/tracking_task.py
  configs/robot/g1_29dof.yaml
  source/humanoid_g1/tasks/locomotion/g1_env_cfg.py
)
for path in "${required_files[@]}"; do
  git ls-files --error-unmatch "${path}" >/dev/null
done
echo "[OK] required implementation and reinstall files are tracked"

if git ls-files | grep -E '(^|/)([^/]+_stagei{1,2}\.npz|SMPLX_NEUTRAL\.npz)$' >/dev/null; then
  echo "ERROR: licensed ACCAD/SMPL-X data is tracked by main." >&2
  exit 1
fi
echo "[OK] licensed ACCAD/SMPL-X data is excluded from main"

git fsck --no-dangling >/dev/null
"${project_root}/scripts/setup/bootstrap.sh" --check-only --skip-tests

export PYTHONDONTWRITEBYTECODE=1
"${python_bin}" -m pytest -q -p no:cacheprovider \
  -m "not integration and not hardware and not sim2sim" "${project_root}/tests"

temporary_root="$(mktemp -d -t humanoid-g1-verify.XXXXXXXX)"
cleanup() {
  case "${temporary_root}" in
    /tmp/humanoid-g1-verify.*) rm -rf -- "${temporary_root}" ;;
    *) echo "WARNING: refused to remove unexpected temporary path ${temporary_root}" >&2 ;;
  esac
}
trap cleanup EXIT
"${python_bin}" -m pytest -q -p no:cacheprovider \
  --basetemp="${temporary_root}/pytest" \
  "${project_root}/ACCAD/_g1_pipeline/tests"

if [[ "${with_motion}" == true ]]; then
  "${project_root}/scripts/setup/prepare_motion_data.sh" --check-only
fi

echo
echo "BACKUP VERIFIED: source HEAD is on GitHub and the reproducibility checks passed."
echo "External datasets and generated experiment outputs are intentionally not required."
