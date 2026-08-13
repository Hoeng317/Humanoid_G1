# 2026-08-07 폴더 구조 정리

## 목표

실행 코드, 학습 설정, 데이터, 배포, 생성 결과를 상위 목적별로 묶고
캐시·실패 로그·빈 placeholder를 제거했다.

## 이동한 항목

| 이전 | 현재 | 용도 |
|---|---|---|
| `artifacts/` | `workspace/artifacts/` | contract, 보고서, export policy |
| `logs/` | `workspace/logs/` | checkpoint, TensorBoard, setup log |
| `media/` | `data/media/` | 선별한 이미지·영상·그래프 |
| `sim2sim/` | `deploy/sim2sim/` | Unitree MuJoCo 빌드 정의 |

## 제거한 항목

- `.pytest_cache/`, Python `__pycache__/`
- `_legacy_backup/`, `.pretrained_checkpoints/`
- 빈 `checkpoints/`, `outputs/`, `deploy/python/`, `tests/sim2sim/`
- 모델이 없던 실패 RSL-RL run과 빈 SKRL 로그
- 이동 전 CMake cache인 `.local/build-unitree-mujoco/`
- 빈 custom-policy placeholder인 `source/humanoid_g1/policies/`

`third_party/`의 빈 `__init__.py`는 Python package 표식이므로 유지했다.
`.local/`은 빌드된 SDK/MuJoCo 실행 환경, `third_party/`는 의존 소스와
로봇 asset이므로 유지했다.

## 새 상위 구조

- `configs/`: 실험·policy·PPO·robot·simulation 설정
- `source/`: 로봇 task, MDP, observation/action, deployment Python 코드
- `scripts/`: train/play/export/simulation/setup 실행 진입점
- `data/`: 모션, 시연, pretrained weight, 녹화, media
- `workspace/`: 실행 중 생성되는 로그·checkpoint·보고서·policy
- `deploy/`: Python/C++ 배포 contract, 생성 binary, MuJoCo 빌드
- `tests/`: unit/integration 검증
- `third_party/`: Unitree 의존 소스와 asset
- `docs/`: 학습·배포·코드 구조 문서

전체 항목은 `PROJECT_MAP.md`에서 확인한다.
