# Humanoid G1 프로젝트 지도

처음 코드를 수정한다면 이 문서와 `docs/CODEBASE_GUIDE_KO.md`부터 읽습니다.

## 가장 자주 수정할 위치

|목적|먼저 수정할 파일|더 깊게 수정할 코드|
|---|---|---|
|나만의 실험 만들기|`configs/experiments/g1_custom_ppo.yaml`|`source/humanoid_g1/tasks/locomotion/env_cfg.py`|
|정책 MLP 크기 변경|`configs/policy/mlp_custom.yaml`|완전한 custom class는 별도 runner/export 연결 필요|
|PPO 학습 방식 변경|`configs/algorithm/ppo_custom.yaml`|`scripts/rsl_rl/train.py`|
|iteration/저장 주기 변경|`configs/train/ppo_custom.yaml`|`scripts/rsl_rl/train.py`|
|로봇이 배우는 행동 변경|experiment의 reward weight|`source/humanoid_g1/mdp/rewards.py`|
|정책 입력 변경|`tasks/locomotion/env_cfg.py`의 `ObservationsCfg`|deployment observation adapter/C++도 함께 변경|
|관절 action 변경|`tasks/locomotion/env_cfg.py`의 `ActionsCfg`|deployment action adapter/C++도 함께 변경|
|기본 자세·PD gain 변경|`source/humanoid_g1/assets/g1_asset.py`|`configs/robot/g1_29dof.yaml` contract|
|속도 명령 범위 변경|experiment의 `commands`|`source/humanoid_g1/mdp/commands/`|
|모션/시연 데이터 추가|`data/motions`, `data/demonstrations`|loader와 imitation reward를 새로 연결|
|이미지·영상 정리|`data/media/images`, `data/media/videos`, `data/media/plots`|camera policy는 sensor/encoder 구현 필요|

## 폴더 구조

```text
humanoid_G1/
├── configs/
│   ├── experiments/       # 한 번의 실험을 조합하는 최상위 YAML
│   ├── policy/            # Actor/Critic 네트워크 모양
│   ├── algorithm/         # PPO update 설정
│   ├── train/             # rollout/iteration/save와 policy 선택
│   ├── robot/             # 29-DoF contract, gain, safety
│   ├── sim/               # Isaac physics 설정
│   ├── terrain/           # 지형 preset
│   ├── randomization/     # domain randomization
│   └── deploy/            # 실기/Sim2Sim 설정
├── source/humanoid_g1/
│   ├── assets/            # G1 URDF를 Isaac articulation으로 생성
│   ├── tasks/locomotion/  # scene/action/observation/reward/termination 조합
│   ├── mdp/               # reward/observation/command 함수
│   ├── deployment/        # Python observation/action/safety contract
│   ├── interfaces/        # command/sensor/safety 확장 hook
│   └── utils/             # config composition, path, 재현성
├── scripts/
│   ├── rsl_rl/            # train/play/evaluate/export 진입점
│   ├── simulation/        # 물리 및 안정성 시험
│   ├── deployment/        # dry-run/실기 guarded launcher
│   ├── sim2sim/           # MuJoCo launcher/comparison
│   ├── inspect/           # asset/joint/task 검사
│   └── setup/             # dependency/build/doctor
├── data/
│   ├── motions/
│   │   └── README.md      # G1 reference-motion schema
│   ├── demonstrations/    # imitation/behavior cloning 시연 데이터
│   ├── pretrained/        # 사전학습 weight 및 메타데이터
│   ├── recordings/        # 센서·로봇 녹화 데이터
│   └── media/
│       ├── images/        # 문서·reference 이미지
│       ├── videos/        # 보존할 재생 영상
│       └── plots/         # reward/evaluation 그래프
├── workspace/
│   ├── logs/rsl_rl/       # 자동 생성되는 checkpoint/TensorBoard
│   └── artifacts/         # contract, report, exported policy
├── deploy/
│   ├── cpp/               # SDK2 + LibTorch 실시간 controller
│   ├── sim2sim/           # Unitree MuJoCo 빌드 정의
│   └── generated/         # 빌드된 controller
├── tests/                 # unit/integration test
├── third_party/           # 고정된 공식 Unitree source/assets
└── docs/                  # 설계, 학습, Sim2Sim/Sim2Real 문서
```

## 실행 경로

`train.sh`, `play.sh`, `simulate.sh`는 짧은 호환 wrapper이고 실제 명령 분기는
`g1.sh`에 있습니다.

```text
train.sh
  → g1.sh train
  → scripts/rsl_rl/train.py
  → configs/experiments/*.yaml 조합
  → Gym task 등록
  → ManagerBasedRLEnv
  → RSL-RL OnPolicyRunner + PPO
  → workspace/logs/rsl_rl/.../model_N.pt
```

실제 호출되는 시뮬레이션 파일은 `scripts/simulation/simulate.py`입니다.
`scripts/simulate.py`라는 경로는 현재 실행 진입점이 아닙니다.

## 사용자 실험 시작

```bash
./humanoid_G1/g1.sh train \
  --config configs/experiments/g1_custom_ppo.yaml \
  --num-envs 1024 --max-iterations 1000 --headless \
  --run-name first_custom
```

먼저 `configs/experiments/g1_custom_ppo.yaml`에서 reward와 command를 바꾸고,
네트워크는 `configs/policy/mlp_custom.yaml`, PPO는
`configs/algorithm/ppo_custom.yaml`에서 독립적으로 바꿉니다.
