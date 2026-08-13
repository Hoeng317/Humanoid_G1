# Unitree G1 29-DoF 연구 환경

> 로컬 폴더를 삭제한 뒤 다시 설치하려면
> [GitHub 완전 복원 가이드](docs/REINSTALL_FROM_GITHUB_KO.md)를 먼저 보십시오.
> 자체 구현은 `main`에, 선택적인 과거 실행 결과는 `artifacts` 브랜치에 있습니다.

이 프로젝트의 로봇은 **공식 Unitree G1 29-DoF rev_1_0**입니다. 모델은
`unitree_ros`의 공식 URDF, Sim2Sim은 `unitree_mujoco`의 공식
`g1_29dof.xml/scene_29dof.xml`을 사용합니다. 손은 구매 옵션에 따라 구성이 달라
29-DoF body locomotion action에서 제외하고 별도 subsystem으로 유지합니다.

처음 코드를 수정하려면 [PROJECT_MAP.md](PROJECT_MAP.md)에서 목적별 파일 위치를
확인하고, [한국어 코드베이스 상세 가이드](docs/CODEBASE_GUIDE_KO.md)를 읽으십시오.
baseline과 분리된 사용자 실험 시작점은
`configs/experiments/g1_custom_ppo.yaml`입니다.

모든 자체 구현과 빌드 산출물은 이 저장소 안에만 있습니다. 바깥의 Isaac Lab,
SeRT 및 기존 RSL-RL 코드는 수정하지 않습니다. 경로는 실행 시 저장소 위치에서
계산하므로 `/home/hoeng` 계정에 종속되지 않습니다.

## 새 clone 최소 복원

Isaac Sim 5.1.0과 정확한 Isaac Lab 2.3.2 commit을 설치한 뒤:

```bash
cd /path/to/IsaacLab
git clone https://github.com/Hoeng317/Humanoid_G1.git humanoid_G1
cd humanoid_G1
./scripts/setup/bootstrap.sh
```

이 명령은 GitHub에 포함하지 않은 Unitree 공식 저장소를 고정 commit으로 복원하고,
import·joint contract·unit test를 검사합니다. ACCAD/SMPL-X는 각 사이트의 라이선스에
동의해 별도로 받아야 하며 다음 명령으로 배치와 checksum을 확인합니다.

```bash
./scripts/setup/prepare_motion_data.sh \
  --smplx-model /path/to/models_lockedhead/smplx/SMPLX_NEUTRAL.npz
```

전체 전처리, 리타게팅, Isaac 검증, PPO 재학습 순서는
[복원 가이드](docs/REINSTALL_FROM_GITHUB_KO.md)에 명령 단위로 고정되어 있습니다.

로컬 폴더를 삭제하기 직전에는 `./scripts/setup/verify_github_backup.sh --with-motion`을
실행하십시오. GitHub `main`과 local HEAD가 같고 테스트가 모두 통과해야
`BACKUP VERIFIED`를 출력합니다.

## 학습·평가 산출물

PPO 체크포인트, ONNX 정책, TensorBoard 로그, 평가 보고서와 시뮬레이션 영상은
소스 코드와 분리해 GitHub의
[`artifacts` 브랜치](https://github.com/Hoeng317/Humanoid_G1/tree/artifacts)에
Git LFS로 보관합니다. GitHub 화면이 `main` 브랜치로 되어 있으면 이 산출물은
보이지 않으므로, 브랜치 선택 메뉴에서 `artifacts`를 선택해야 합니다.

```bash
git clone --single-branch --branch artifacts \
  git@github.com:Hoeng317/Humanoid_G1.git Humanoid_G1_Artifacts
cd Humanoid_G1_Artifacts
git lfs pull
```

## 가장 먼저 실행할 순서

아래 명령은 clone을 둔 Isaac Lab 루트(예: `/path/to/IsaacLab`)에서 실행합니다.

```bash
# 1. 설치·asset·GPU·joint contract 확인
./humanoid_G1/g1.sh doctor

# 2. 공식 모델과 29개 관절 순서 확인(Isaac 실행 없음)
./humanoid_G1/g1.sh inspect

# 3. 환경 1-step smoke
./humanoid_G1/g1.sh simulate \
  --config configs/sim/g1_debug.yaml \
  --mode hold-pose --num-envs 1 --steps 1 --headless

# 4. GUI에서 유지 자세 확인(창을 닫을 때까지는 --steps 0)
./humanoid_G1/g1.sh simulate \
  --config configs/sim/g1_debug.yaml \
  --mode hold-pose --num-envs 1 --steps 0
```

관절 하나만 작은 진폭으로 움직이려면 다음처럼 실행합니다.

```bash
./humanoid_G1/g1.sh simulate --mode sine-joint-test \
  --joint left_knee_joint --steps 500
```

## 학습 → 평가 → export

```bash
# 32 env × 2 iteration 파이프라인 smoke
./humanoid_G1/g1.sh train \
  --config configs/experiments/g1_smoke.yaml \
  --headless --run-name my_smoke

# 평지 baseline
./humanoid_G1/g1.sh train \
  --config configs/experiments/g1_flat_ppo.yaml \
  --num-envs 4096 --seed 42 --headless

# 체크포인트부터 이어서 학습
./humanoid_G1/g1.sh train \
  --config configs/experiments/g1_flat_ppo.yaml \
  --resume workspace/logs/rsl_rl/<EXPERIMENT>/<RUN_ID>/model_<N>.pt \
  --headless

# 재생/정량 평가
./humanoid_G1/g1.sh play \
  --config configs/experiments/g1_flat_ppo.yaml \
  --checkpoint /absolute/path/model_N.pt
./humanoid_G1/g1.sh evaluate \
  --config configs/experiments/g1_flat_ppo.yaml \
  --checkpoint /absolute/path/model_N.pt

# TorchScript + ONNX + contract + golden vector bundle
./humanoid_G1/g1.sh export \
  --config configs/experiments/g1_flat_ppo.yaml \
  --checkpoint /absolute/path/model_N.pt \
  --run-id <RUN_ID>
```

과거에 검증했던 smoke checkpoint는
`workspace/logs/rsl_rl/g1_smoke/2026-07-30_14-22-00_acceptance_smoke2/model_1.pt`,
export bundle은 `workspace/artifacts/policies/acceptance_smoke2/`였습니다. 해당 실행
산출물은 `main` 복원의 필수 항목이 아닙니다. 이것은 실행 경로 검증용 2-iteration
정책이지, 보행 성능이 확보된 배포 정책이 아닙니다.

## MuJoCo Sim2Sim

```bash
# 기본은 IDLE: LowState만 확인하며 LowCmd publisher를 만들지 않음
./humanoid_G1/g1.sh sim2sim \
  --policy workspace/artifacts/policies/<RUN_ID>/policy.pt

# 명시적으로 stand/run을 요청한 경우만 LowCmd 활성화
./humanoid_G1/g1.sh sim2sim \
  --policy workspace/artifacts/policies/<RUN_ID>/policy.pt --stand
./humanoid_G1/g1.sh sim2sim \
  --policy workspace/artifacts/policies/<RUN_ID>/policy.pt --run
```

Sim2Sim은 `lo`, DDS domain `1`로 강제됩니다. 실제 G1의 domain `0`과 섞을 수
없습니다. 자동 검증에는 `--duration 10 --xvfb`를 사용할 수 있습니다.

## 배포 전 검사와 실물 진입

```bash
./humanoid_G1/g1.sh deploy-check \
  --policy workspace/artifacts/policies/<RUN_ID>/policy.pt
```

실물 명령은 다음 조건을 모두 충족하기 전에는 publisher조차 생성하지 않습니다:
`--real`, 위험 확인, 명시적 non-loopback interface, domain 0, 확정된 hardware
profile/motor index, 유효한 joint contract와 정책 checksum, 정상 G1 LowState,
그리고 별도의 `--stand` 또는 `--run`.

```bash
./humanoid_G1/g1.sh deploy-real \
  --interface eth0 \
  --policy workspace/artifacts/policies/<RUN_ID>/policy.pt \
  --real --acknowledge-hardware-risk --stand
```

현재 `configs/robot/g1_variant.yaml`은 `verified: false`이므로 위 명령은 의도대로
거부됩니다. 실제 로봇 수령 후에도 [SIM2REAL.md](docs/SIM2REAL.md)의 단계별
commissioning을 먼저 완료해야 합니다.

## 구현 위치

- `configs/`: robot/sim/terrain/randomization/PPO/deploy 조합형 YAML
- `source/humanoid_g1/assets/`: URDF asset과 단일 joint contract
- `source/humanoid_g1/tasks/locomotion/`: ManagerBasedRLEnv flat/rough/play task
- `source/humanoid_g1/deployment/`: Python observation/action/safety dry-run
- `deploy/cpp/`: LibTorch + Unitree SDK2 실시간 제어기
- `scripts/`: inspect, simulation, RSL-RL, Sim2Sim, deployment 진입점
- `workspace/artifacts/contracts/`: joint 및 actor/critic observation 계약
- `third_party/`: commit이 고정된 공식 Unitree 저장소

상세 구성은 [ARCHITECTURE.md](docs/ARCHITECTURE.md), 설정 변경법은
[CONFIGURATION.md](docs/CONFIGURATION.md), 실제 검증 결과와 보류 항목은
[FINAL_IMPLEMENTATION_REPORT.md](docs/FINAL_IMPLEMENTATION_REPORT.md)에 있습니다.
