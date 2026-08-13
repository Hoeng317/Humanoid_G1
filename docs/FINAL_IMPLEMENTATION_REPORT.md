# Final implementation report

## 1. 구현 결과

`humanoid_G1`을 공식 Unitree G1 29-DoF 기반의 독립 연구 프로젝트로
재구성했습니다. 기존 demo source/config 백업은 이후 2026-08-07 구조 정리에서
제거했습니다. 프로젝트 바깥 Isaac Lab,
SeRT/SERT, shared PyTorch/RSL-RL 환경은 수정하거나 upgrade하지 않았습니다.

주요 module:

- official URDF 기반 `G1_29DOF_CFG`와 implicit PD actuator
- Isaac–policy–MuJoCo–SDK2 단일 joint contract
- flat/rough/play `ManagerBasedRLEnv`
- YAML composition/validation 및 CLI override
- RSL-RL PPO train/resume/play/evaluate/export
- Python deployment dry-run과 C++ LibTorch controller
- project-local Unitree SDK2, MuJoCo, GLFW build
- loopback/domain-separated Sim2Sim launcher
- real-mode state machine, watchdog, checksum/profile/network guard

공식 upstream repository는 모두 clean 상태이며 수정하지 않았습니다. Locomotion
MDP port와 adaptation은 `source/humanoid_g1`에 분리했습니다.

## 2. 확정된 robot/deployment contract

- Robot: Unitree G1 29-DoF rev_1_0
- Asset: `third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf`
- MuJoCo: `third_party/unitree_mujoco/unitree_robots/g1/g1_29dof.xml`
- Body action: 29, hand 제외
- Actor observation: 96 × history 5 = 480
- Critic observation: 99 × history 5 = 495
- Actor base linear velocity: 제외
- Action: normalized joint-position target, scale 0.25 rad
- Physics/policy: 200/50 Hz (`dt=0.005`, decimation 4)
- Policy→SDK: `[0,6,12,1,7,13,2,8,14,3,9,15,22,4,10,16,23,5,11,17,24,18,25,19,26,20,27,21,28]`
- Normalization: 현재 baseline은 identity이며 exported module contract에 포함

각 관절의 이름, index, limit, default, kp/kd는
`workspace/artifacts/contracts/g1_29dof_joint_contract.{json,md}`에 있습니다. 값은 공식
URDF와 Unitree RL Lab baseline에서 가져오고 validator로 대조했습니다.

## 3. 실제 실행 결과

|명령/검증|결과|근거|
|---|---|---|
|`g1.sh doctor`|PASS|URDF/MuJoCo contract, config, GPU, local binaries|
|`g1.sh inspect`|PASS|29 joints, official G1 source, cross-engine order|
|Isaac 1-env 1-step smoke|PASS|`workspace/artifacts/reports/smoke_1step.json`|
|live task contract|PASS|action 29, actor/critic 480/495|
|32-env, 2-iteration PPO|PASS|`workspace/logs/rsl_rl/g1_smoke/2026-07-30_14-22-00_acceptance_smoke2/model_1.pt`|
|TorchScript/ONNX export|PASS|`workspace/artifacts/policies/acceptance_smoke2/`|
|Python→TorchScript golden 100|PASS|max abs error 0|
|TorchScript→C++ golden 100|PASS|max abs error `5.96046e-08`, tolerance `1e-5`|
|C++ controller build|PASS|`deploy/generated/g1_controller`|
|MuJoCo DDS Sim2Sim 10 s|PASS|LowState→POLICY_RUNNING→LowCmd→STOPPING/damping|
|deployment dry-run|PASS|480→29 inference/target, DDS publisher false|
|real-mode negative guard|PASS|unverified hardware profile에서 publisher 생성 전 거부|

Sim2Sim 통과 report:
`workspace/artifacts/reports/SIM2SIM_20260730_144924.md`.

## 4. 성능/자원

- GPU: RTX 4090 24 GB
- PPO smoke: 32 environments, 2 iterations
- Exported actor: 480→128→64→29 (smoke preset)
- Python dry-run inference: 약 1.3 ms (이 실행의 단발 측정)
- C++ golden tolerance: `1e-5`, measured `5.96e-08`
- C++ real-time gate: inference 10 ms, loop work 30 ms

이번 변경에서는 사용자의 후속 요청에 따라 긴 benchmark를 추가 실행하지
않았습니다. 따라서 안정된 simulation FPS, peak VRAM, baseline iteration time은
확정 수치로 기록하지 않습니다.

## 5. 명확한 보류/미완료 항목

사용자가 “1-step 실행 확인만 하고 긴 디버깅은 나중에”라고 범위를 변경했습니다.
이에 따라 다음은 완료로 주장하지 않습니다.

- 기존 32-env/10,000-step hold-pose 시도는 crash/NaN은 없었지만
  `bad_orientation` 종료 5,216회로 acceptance 실패했습니다.
  `workspace/artifacts/reports/stability_default_pose_32env_10000.json`에 보존했습니다.
- ankle/default-pose balance tuning과 10,000-step 재검증은 중단했습니다.
- full 1500/3000-iteration 보행 학습 및 보행 품질 평가는 실행하지 않았습니다.
- 여러 scenario evaluation, resume 1-iteration, GUI video는 이번 최종 smoke에서
  재실행하지 않았습니다.
- latency/packet-drop/action-delay randomization은 YAML/hook에 정의했지만 현재
  Isaac action buffer에 직접 주입되지 않습니다.
- 실제 G1 LowState, motor index, firmware, network, remote emergency stop,
  support-rig 시험은 실물이 없어 미검증입니다.

현재 `acceptance_smoke2` 정책은 pipeline용이며 물리 로봇 배포 후보가 아닙니다.

## 6. 외부 코드 보호

- Isaac Lab HEAD는 작업 전후
  `2210934acca1a2f2401d541874163406b7ca8b53`로 동일합니다.
- root worktree에는 작업 시작 전부터 `.gitignore`, `.vscode`, SeRT asset 등
  다수의 사용자 변경/삭제가 있었습니다. 이를 복구·정리·stage하지 않았습니다.
- 이 작업에서 생성/수정한 경로는 `humanoid_G1/**`뿐입니다.
- 공식 Unitree 5개 clone은 모두 `git status --porcelain` 0건입니다.

## 7. 다음 권장 단계

먼저 `docs/CONFIGURATION.md`를 기준으로 default pose/ankle gain/contact를
진단해 hold-pose 10,000-step을 통과시킵니다. 이후 debug training, baseline
training, scenario evaluation, Sim2Sim 장시간 regression 순서로 진행합니다.
실물 배포는 마지막에 `docs/SIM2REAL.md` commissioning을 완료한 뒤 시작합니다.
