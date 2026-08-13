# Humanoid G1 코드베이스 상세 가이드

## 1. 현재 프로젝트가 학습하는 것

이 프로젝트는 Unitree G1 29-DoF의 joint-position policy를 PPO로 학습합니다.
정책의 목표는 자동 생성되는 `(vx, vy, wz)` 속도 명령을 따라가면서 넘어지지 않고
걷는 것입니다.

현재 `hold-pose` simulation은 학습 policy가 아닙니다. action 29개를 모두 0으로
보내 기본 관절 자세만 유지합니다. floating base의 무게중심을 되돌리는 feedback
policy가 없으므로 작은 자세 오차가 누적되면 쓰러질 수 있습니다.

## 2. 실행 진입점

### Shell

- `train.sh`: `g1.sh train`으로 전달하는 호환 wrapper
- `play.sh`: `g1.sh play`로 전달
- `simulate.sh`: `g1.sh simulate`로 전달
- `check_setup.sh`: `g1.sh doctor`로 전달
- `g1.sh`: 모든 명령을 실제 Python/C++ 진입점으로 분기

따라서 일반적으로 wrapper 내부를 수정할 필요는 없습니다. 새 CLI 명령을 만들
때만 `g1.sh`에 case를 추가합니다.

### Python training

`scripts/rsl_rl/train.py`의 흐름은 다음과 같습니다.

1. CLI와 experiment YAML을 읽음
2. Isaac Sim `AppLauncher` 시작
3. Gym task 등록
4. environment config와 RSL-RL config 조합
5. `ManagerBasedRLEnv` 생성
6. `RslRlVecEnvWrapper`로 vector environment 변환
7. `OnPolicyRunner.learn()`으로 rollout/PPO update
8. `workspace/logs/rsl_rl/<experiment>/<run>/model_N.pt` 저장

`scripts/rsl_rl/common.py`는 train/play/evaluate가 공유하는 config와 checkpoint
탐색 코드입니다. `source/humanoid_g1/utils/config.py`는 여러 YAML을 실제 Isaac
config object에 적용합니다.

## 3. Robot과 관절

### 공식 asset

- URDF 경로와 articulation 생성: `source/humanoid_g1/assets/g1_asset.py`
- authoritative joint contract: `configs/robot/g1_29dof.yaml`
- 순서/limit 검증: `source/humanoid_g1/assets/joint_contract.py`

29 action은 다리 12, 허리 3, 팔 14개 관절입니다. 손은 포함되지 않습니다.

`g1_asset.py`에서 수정할 수 있는 값:

- `init_state.pos`: 초기 몸통 높이
- `init_state.joint_pos`: 초기/기본 관절 자세
- actuator `stiffness`, `damping`: Isaac implicit PD gain
- effort/velocity simulation limit
- self-collision과 solver iteration

주의: 기본 각도, limit, kp/kd를 contract YAML과 asset Python 양쪽에서 서로 다르게
바꾸면 Isaac, export, MuJoCo, 실기 controller가 불일치합니다. 관절 계약 변경 후에는
반드시 `g1.sh inspect`를 실행해야 합니다.

## 4. Environment와 MDP

중심 파일은 `source/humanoid_g1/tasks/locomotion/env_cfg.py`입니다.

### `RobotSceneCfg`

robot, terrain, contact sensor, height scanner, light를 생성합니다. 거친 지형의 실제
generator 모양도 현재 이 파일 상단에 정의되어 있습니다.

### `ActionsCfg`

정책 출력 29개를 관절 위치 offset으로 변환합니다.

```text
q_target = q_default + 0.25 × action
```

정책은 50 Hz, physics는 200 Hz이므로 같은 목표를 physics step 4번 동안 적용합니다.

### `ObservationsCfg`

Actor 한 frame:

```text
base angular velocity 3
projected gravity 3
velocity command 3
joint position relative 29
joint velocity 29
previous action 29
= 96
```

5-frame history를 연결해 actor 480차원이 됩니다. Critic은 base linear velocity
3개가 추가되어 frame당 99, 전체 495차원입니다.

### `RewardsCfg`

어떤 reward 함수를 어떤 sensor/body/joint에 적용하는지 조합합니다. 실제 수학식은
다음 두 위치에 있습니다.

- Isaac Lab 기본 reward: `isaaclab.envs.mdp`
- G1 custom reward: `source/humanoid_g1/mdp/rewards.py`

행동을 바꾸는 가장 안전한 순서:

1. `configs/experiments/g1_custom_ppo.yaml`에서 기존 reward weight만 변경
2. 새 수학식이 필요하면 `mdp/rewards.py`에 tensor 함수를 추가
3. `env_cfg.py`의 `RewardsCfg`에 `RewTerm`으로 등록
4. custom experiment의 `rewards:`에 weight 추가
5. smoke 학습 후 짧은 custom 학습으로 비교

예를 들어 더 오래 서 있게 만들려면 orientation, base height, action rate, joint
deviation과 termination을 먼저 검토합니다. 단순히 `alive`만 크게 올리면 걷지 않고
버티는 local optimum이 생길 수 있습니다.

### `TerminationsCfg`

pelvis contact, 낮은 몸통, 큰 roll/pitch, 관절 limit에서 episode를 종료합니다.
로봇이 넘어지면 환경은 자동 reset되고 PPO는 실패한 transition도 학습합니다.

### `EventCfg`와 randomization

마찰, base mass/COM, actuator gain, joint friction, reset velocity, 외부 push를
randomize합니다. `configs/randomization/sim2real.yaml`의 모든 항목이 아직 연결된
것은 아닙니다. latency, packet drop, motor strength, action scale 등은 추가 구현이
필요합니다.

## 5. 정책 네트워크와 PPO를 수정하는 곳

설정을 세 층으로 분리했습니다.

### 정책 MLP: `configs/policy/`

`mlp_custom.yaml`에서 actor/critic hidden dimension, activation, exploration noise,
normalization을 바꿉니다. 이 설정은 신경망의 모양을 바꿉니다.

현재 실제 Python class는 RSL-RL의 `ActorCritic`입니다. custom CNN/Transformer는
새 local module과 custom runner를 함께 만들고 train/play/evaluate/export에 동일한
factory를 연결해야 합니다.

### PPO: `configs/algorithm/`

`ppo_custom.yaml`에서 learning rate, entropy, epoch, mini-batch, gamma, lambda,
clip/KL을 수정합니다. 이는 신경망 모양이 아니라 업데이트 방법을 바꿉니다.

### 학습 실행: `configs/train/`

`ppo_custom.yaml`에서 rollout step, 총 iteration, checkpoint 간격을 바꿉니다.
이 파일이 사용할 policy와 algorithm YAML을 선택합니다.

### 최상위 실험: `configs/experiments/`

`g1_custom_ppo.yaml`에서 robot/sim/terrain/train/randomization 조합, command 범위,
reward weight, env 수와 seed를 결정합니다.

## 6. 로봇 행동을 수정하는 방법

### 정지 균형

검토 순서:

1. `g1_asset.py`의 default pose와 초기 높이
2. 발 collision/contact와 몸통 COM
3. ankle/hip/knee kp/kd
4. orientation/base-height/action-rate reward
5. fall termination threshold

`hold-pose` 통과는 static PD pose의 물리 검증이고, 학습 policy의 동적 균형과는
별개입니다.

### 더 빠르게 걷기

`g1_custom_ppo.yaml`의 command range를 조금씩 넓히고 velocity tracking reward와
gait/feet reward의 균형을 조정합니다. 처음부터 큰 속도를 주면 reset만 반복할 수
있습니다.

### 새로운 command

현재 command 구현은 `source/humanoid_g1/mdp/commands/velocity_command.py`이고
`env_cfg.py`의 `CommandsCfg`에서 등록됩니다. 방향, 목표 위치, 동작 ID 같은 command를
추가하려면 command generator와 observation term을 함께 추가합니다.

### observation 변경

camera, height scan, phase, command ID 등을 추가할 수 있지만 actor 차원이 바뀝니다.
다음도 모두 함께 갱신해야 합니다.

- actor/critic observation schema
- Python `ObservationAdapter`
- export template와 metadata
- C++ `ObservationHistory`
- checkpoint와 golden vectors

이 작업 후 기존 checkpoint는 사용할 수 없습니다.

### action 변경

torque control, residual action, arm/leg 분리 등을 구현하면 Isaac `ActionsCfg`, Python
`ActionAdapter`, C++ `bounded_targets`, joint contract를 동시에 변경해야 합니다.

현재 C++ deployment는 policy action에 step당 0.15 rate limit을 추가하지만 Isaac
학습 action에는 같은 hard limiter가 없습니다. 실기 이전에 이 차이를 없애거나 학습
중에도 동일 limiter/delay를 적용해야 합니다.

## 7. 사전학습 데이터

### 바로 가능한 것

`data/pretrained/model_N.pt`처럼 완전히 동일한 contract의 RSL-RL checkpoint는
`--resume`으로 이어서 학습할 수 있습니다.

### 추가 구현이 필요한 것

- motion imitation: `data/motions` loader + reference tracking reward
- behavior cloning: `data/demonstrations` loader + supervised pretrain stage
- teacher/student: teacher checkpoint loader + distillation runner
- 실제 로그 학습: `data/recordings`를 policy order/480 observation으로 변환

현재 `workspace/artifacts/contracts/isaac_actor_observations.npz`는 export golden-vector 입력이지
학습 dataset이 아닙니다.

## 8. 이미지와 카메라

문서나 결과 이미지는 `data/media/images`, 영상은 `data/media/videos`, 그래프는
`data/media/plots`에 정리합니다.

이미지를 policy 입력으로 사용하려면 별도 작업이 필요합니다.

1. scene에 Isaac camera sensor 추가
2. observation term에 RGB/depth 연결
3. CNN/vision encoder policy 구현
4. vector observation과 image observation의 batching 정의
5. play/evaluate/export/deployment에 camera contract 연결

단순히 이미지 파일을 `data/media/images`에 넣는 것만으로 학습에 사용되지는 않습니다.

## 9. 평가, export와 결과

- `scripts/rsl_rl/play.py`: checkpoint GUI/video 재생
- `scripts/rsl_rl/evaluate.py`: episode return/action saturation JSON 평가
- `scripts/rsl_rl/export_policy.py`: TorchScript/ONNX/golden vector bundle 생성
- `workspace/logs/rsl_rl/`: 학습 checkpoint와 TensorBoard
- `workspace/artifacts/contracts/`: joint/observation 계약
- `workspace/artifacts/reports/`: doctor/stability/evaluation/Sim2Sim 결과
- `workspace/artifacts/policies/`: 배포 bundle

학습 결과를 배포하려면 `model_N.pt`를 직접 C++에 주는 것이 아니라 반드시 export한
`policy.pt`와 checksum bundle을 사용합니다.

## 10. Deployment

Python contract:

- `deployment/observation_adapter.py`: LowState → 480 observation
- `deployment/action_adapter.py`: 29 action → bounded SDK target
- `deployment/policy_runtime.py`: checksum 및 TorchScript inference
- `deployment/safety_state_machine.py`: 상태/자세/시간 안전 검사
- `deployment/real_guard.py`: 실기 publisher 생성 전 hard gate

C++ controller는 `deploy/cpp/`에 있습니다. 현재 policy command는 C++에서
`(0, 0, 0)`으로 고정되어 있어 `--run`만으로 전진 명령을 줄 수 없습니다. joystick,
keyboard 또는 network command provider를 연결하려면 C++ `send_policy()`에 전달되는
command source와 clamp/watchdog를 구현해야 합니다.

## 11. 권장 개발 순서

```bash
# 1. 설정과 contract
./humanoid_G1/g1.sh doctor
./humanoid_G1/g1.sh inspect

# 2. static pose 물리 확인
./humanoid_G1/g1.sh simulate --mode hold-pose --num-envs 1 --steps 1000 --headless

# 3. 코드 경로만 확인하는 2-iteration smoke
./humanoid_G1/g1.sh train --config configs/experiments/g1_smoke.yaml --headless

# 4. 사용자 실험
./humanoid_G1/g1.sh train \
  --config configs/experiments/g1_custom_ppo.yaml \
  --num-envs 1024 --max-iterations 1000 --headless --run-name first_custom

# 5. 재생/평가
./humanoid_G1/g1.sh play --config configs/experiments/g1_custom_ppo.yaml \
  --checkpoint /absolute/path/model_N.pt
./humanoid_G1/g1.sh evaluate --config configs/experiments/g1_custom_ppo.yaml \
  --checkpoint /absolute/path/model_N.pt
```

한 번에 policy 구조, reward, command, randomization을 모두 바꾸지 말고 experiment
이름과 run name을 분리하여 한 요소씩 비교하는 것이 좋습니다.

## 12. 이번 폴더 정리에서 바뀐 것

- `configs/policy/`: Actor/Critic 구조를 train YAML에서 분리
- `configs/algorithm/`: PPO update 설정 분리
- `configs/train/`: rollout 설정과 policy/algorithm 참조만 유지
- `g1_custom_ppo` 및 custom policy/algorithm/train preset 추가
- `data/`: motions/demonstrations/recordings/pretrained 작업 공간 추가
- `data/media/`: images/videos/plots 작업 공간 추가
- 사용되지 않던 custom policy placeholder는 정리하고 실제 MLP 설정은 `configs/policy/`에 유지
- `PROJECT_MAP.md`: 목적별 수정 위치 빠른 지도 추가

기존 `g1.sh train/play/evaluate/export/simulate` 명령, task ID, log/artifact 경로는
변경하지 않았습니다.
