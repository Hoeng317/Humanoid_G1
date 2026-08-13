# ACCAD → Unitree G1 파이프라인 운영 가이드

이 폴더는 ACCAD/AMASS 사람 모션을 G1 29-DoF reference로 리타게팅하고,
Isaac Lab에서 PPO tracking·평가·영상·export까지 실행하는 독립 작업 공간이다.
신규 코드와 산출물은 모두 `ACCAD/` 내부에만 둔다.

교수 제출용 상세 결과는 [`../FINAL_REPORT_KO.md`](../FINAL_REPORT_KO.md)를 본다.

## 현재 최종 상태

| 단계 | 상태 | 결과 |
|---|---|---|
| Gate 1: 데이터 감사 | PASS | Stage-II 252, alias 3, 고유 249 |
| Gate 2: Human 50 Hz | PASS | SMPL-X FK/contact/coordinate 검증 |
| Gate 3: G1 v27 | PASS | 249/249, 79,518 frame |
| Gate 4: Isaac ingestion | PASS | B3 kinematic RMSE `3.56e-7 m` |
| v28 dynamic filter | 완료 | train 103, validation 2, TEST 미구축 |
| v7 PPO 학습 | 완료 | 1,000 update, 27,852,800 transition |
| v7 train acceptance | **FAIL** | 평가한 네 후보 중 선택 `model_998`, 48/103 |
| v7 B3 physics replay | completion PASS | 609/609; MP4/export 생성, 최종 영상 gate 아님 |
| v7 checkpoint validation | 미실행 | train gate 실패; split은 과거 진단에 이미 사용 |
| Male1 TEST | 동적/정책 봉인 | v27 처리 완료, v28 build와 정책 평가 없음 |
| CPU regression | PASS | 295 passed (기존 288 + TensorBoard 7) |

중요: train gate까지 구현과 실행은 완료됐지만 holdout branch는 실행하지 않았고,
범용 추종 정책은 acceptance를 통과하지 못했다.
`model_998.pt`는 train-only 연구 checkpoint이며 production/일반화 정책이 아니다.

## TensorBoard로 현재 결과 확인

기존 결과를 다시 계산하지 않고 다음 세 run으로 요약한다.

| run | 사용자가 확인할 내용 |
|---|---|
| `01_data` | 252/249/3 파일 accounting, train/validation/test, 길이·FPS·품질 분포, Stage-II 차원 |
| `02_retargeting` | 249개 G1 기구학 검증 분포, joint/contact/FK 한계, 현재 B3 관절·접촉 궤적 |
| `03_physics_policy` | Isaac ingestion, v28/v29 필터, 선택 v7 학습곡선, 48/103 acceptance |

대시보드 생성은 읽기 전용 post-processing이다. 원본/retarget/정책을 수정하거나
TEST 수치 NPZ와 TEST 정책 branch를 열지 않는다.

```bash
cd /home/hoeng/IsaacLab

PYTHONDONTWRITEBYTECODE=1 ./_isaac_sim/python.sh \
  humanoid_G1/ACCAD/_g1_pipeline/run.py tensorboard --force

./_isaac_sim/python.sh -m tensorboard.main \
  --logdir humanoid_G1/ACCAD/_g1_pipeline/work/tensorboard/accad_g1_overview \
  --port 6006
```

브라우저에서 `http://localhost:6006`을 연다. `Scalars`는 핵심 값과 시간곡선,
`Distributions`/`Histograms`는 249개 모션의 분포, `Text`는 차원·split·각 지표의
정확한 의미를 보여준다. 기존 대시보드를 교체할 때만 `--force`를 사용한다.

주의: `02_retargeting`의 Pinocchio FK 오차는 G1 archive 내부 일관성이지
human↔G1 의미 보존 오차가 아니다. `03_physics_policy`의 v7 값은 train 결과이며
validation/TEST 일반화 결과가 아니다.

## 빠른 확인

모든 명령은 저장소 루트에서 실행한다.

```bash
cd /home/hoeng/IsaacLab
```

### 기존 Gate 스냅샷

`run.py status`의 Gate-5는 현재 v7이 아니라 과거 v5 validation 0/2 실패를 표시한다.
v7 결과는 `work/reports/v7_train_selection_outcome.json`을 확인한다.

```bash
PYTHONDONTWRITEBYTECODE=1 ./_isaac_sim/python.sh \
  humanoid_G1/ACCAD/_g1_pipeline/run.py status
```

### B3 retarget를 Isaac G1에 직접 로드

```bash
TERM=xterm ./isaaclab.sh -p \
  humanoid_G1/ACCAD/_g1_pipeline/run_isaac.py kinematic \
  --headless --device cuda:0 --seed 42 --max-frames 0 \
  --input humanoid_G1/ACCAD/_g1_pipeline/work/g1/Female1Walking_c3d/B3_-_walk1_g1_50hz.npz \
  --output humanoid_G1/ACCAD/_g1_pipeline/work/reports/B3_-_walk1_isaac_kinematic.json \
  --video \
  --video-path humanoid_G1/ACCAD/_g1_pipeline/work/media/B3_-_walk1_isaac_kinematic.mp4
```

### 선택 정책의 103개 train 모션 평가

```bash
TERM=xterm ./isaaclab.sh -p \
  humanoid_G1/ACCAD/_g1_pipeline/evaluate_tracking_suite.py \
  --headless --device cuda:0 \
  --checkpoint humanoid_G1/ACCAD/_g1_pipeline/work/training/runs/2026-08-09_18-55-56_v7_zerovel_objv4_framezero100_lr1e5_clip010_ep3_resume_model299_env1024_h32_u700/model_998.pt \
  --manifest humanoid_G1/ACCAD/_g1_pipeline/work/dynamic/g1_train.json \
  --split train \
  --reference-root humanoid_G1/ACCAD/_g1_pipeline/work/dynamic/g1 \
  --max-steps 0 --physics-warmup-steps 100 --seed 42 \
  --export none \
  --evaluation-name v7_continuation_model998_train_selection
```

### B3 정책 영상과 JIT/ONNX export

```bash
TERM=xterm ./isaaclab.sh -p \
  humanoid_G1/ACCAD/_g1_pipeline/play_tracking.py \
  --headless --device cuda:0 \
  --manifest humanoid_G1/ACCAD/_g1_pipeline/work/dynamic/g1_train.json \
  --split train \
  --reference-root humanoid_G1/ACCAD/_g1_pipeline/work/dynamic/g1 \
  --checkpoint humanoid_G1/ACCAD/_g1_pipeline/work/training/runs/2026-08-09_18-55-56_v7_zerovel_objv4_framezero100_lr1e5_clip010_ep3_resume_model299_env1024_h32_u700/model_998.pt \
  --num-envs 1 --motion-id 34 --steps 609 --episodes 0 \
  --physics-warmup-steps 100 --seed 42 \
  --video --video-length 609 --export both \
  --evaluation-name v7_model998_train_b3_success_video_export
```

### CPU 회귀 테스트

```bash
PYTHONDONTWRITEBYTECODE=1 ./_isaac_sim/python.sh -m pytest -q \
  -p no:cacheprovider \
  --basetemp=humanoid_G1/ACCAD/_g1_pipeline/work/pytest_final_verification \
  --junitxml=humanoid_G1/ACCAD/_g1_pipeline/work/reports/cpu_regression_junit.xml \
  humanoid_G1/ACCAD/_g1_pipeline/tests
```

현재 결과는 `295 passed`, failure/error/skipped 0이다. 기존 파이프라인 288개에
TensorBoard dashboard 전용 테스트 7개가 추가됐다.

## 데이터 흐름

```text
ACCAD *_stageii.npz
  → inventory / checksum / subject split
  → SMPL-X FK + contact + ground + 50 Hz
  → Human canonical archive
  → sequence-level G1 retarget v27
  → independent float64 FK / limit / contact validation
  → v27 train / validation / TEST manifests
  → dynamic prefilter / uniform retime v28
  → v28 train 103 / validation 2 / TEST sealed
  → Isaac Lab 200 Hz physics / 50 Hz control
  → 29-D reference-residual PPO
  → deterministic train selection
  → train PASS일 때만 validation
  → validation PASS일 때만 TEST build/evaluation/finalizer
```

## 폴더 구조

```text
_g1_pipeline/
├── README.md
├── config.yaml
├── correspondence.yaml
├── accad_g1/                     구현 package
├── tests/                        CPU regression
├── run.py                        Gate 상태/전처리
├── run_parallel_batch.py         v27 전체 batch
├── build_gate3_manifests.py      검증 archive 승격
├── build_dynamic_retime.py       v28 처리
├── build_contact_feasibility.py  v29 audit-only
├── build_contact_aware_repair.py v30 production-disabled
├── run_isaac.py                  kinematic/PD replay
├── train_tracking.py             PPO 학습
├── evaluate_tracking_suite.py    전체 결정론 평가
├── play_tracking.py              단일 모션/영상/export
└── work/
    ├── manifests/                Gate-1/3 manifest
    ├── human/                    canonical human 50 Hz
    ├── g1/                       immutable v27
    ├── dynamic/                  v28 train/validation
    ├── contact_v29/              contact proxy audit
    ├── quarantine/               v30 prototype 격리
    ├── reports/                  보고서/JUnit/outcome
    ├── media/                    kinematic/PD 영상
    └── training/
        ├── runs/                 checkpoint/TensorBoard/contract
        └── evaluations/          suite/video/export
```

## 무엇을 수정하면 되는가

| 수정 목적 | 파일 |
|---|---|
| 데이터/split/기본 경로 | `config.yaml`, `accad_g1/data.py` |
| Human FK/contact/50 Hz | `accad_g1/human.py` |
| Human↔G1 joint/body 대응 | `correspondence.yaml` |
| Retarget objective/constraint | `accad_g1/retarget.py` |
| G1 독립 검증 | `accad_g1/g1_validation.py` |
| Dynamic retime/filter | `accad_g1/dynamic_retime.py` |
| Policy observation/action/reward/termination | `accad_g1/tracking_task.py` |
| PPO 실행/초기화/resume | `train_tracking.py`, `accad_g1/training_io.py` |
| Acceptance metric | `accad_g1/evaluation_metrics.py` |
| 결정론 suite | `evaluate_tracking_suite.py` |
| 영상/JIT/ONNX | `play_tracking.py` |
| Success 최종 봉인 | `accad_g1/final_artifacts.py` |

평가 결과를 본 뒤 acceptance threshold를 낮추면 안 된다.

## 현재 v7 제어 계약

```text
policy contract:
  accad_g1_tracking_v7_zero_velocity_target_train_selected

training objective:
  accad_g1_training_objective_v4_strong_dense_divergence_action_margins

actor / critic / action:
  293 / 505 / 29

future horizons:
  0.00, 0.04, 0.08, 0.16 s

action:
  a = clip(policy, -1, 1)
  a = rate_limit(a, previous, 0.15 per control step)
  q_target = clamp(q_ref + 0.25 * a, safety limits)
  dq_target = 0
```

v6의 full reference-velocity target은 train 성능을 악화시켜 v7에서 0으로 되돌렸다.
v5 terminal failure penalty의 dt 이중 적용도 수정해 설정값 `-10`이 정확한 one-shot
impulse가 되게 했다.

## 선택 정책과 증거

선택 checkpoint:

```text
work/training/runs/
2026-08-09_18-55-56_v7_zerovel_objv4_framezero100_lr1e5_clip010_ep3_resume_model299_env1024_h32_u700/
model_998.pt
```

SHA-256:

```text
84be6f1af3e859e74cf7b79b3b5333ff74e830e39c72d8cc0aa0fd17a2ff521f
```

Train 결과:

- clean success: 48/103
- mean completion: 0.689659
- fall/divergence: 22/35
- joint MAE: 0.087921 rad
- root/body position error: 0.105449/0.111446 m
- pooled contact F1: 0.961539
- weighted raw clip: 0.086704
- joint-limit/nonfinite termination: 0/0

기계 판독용 선택 보고서:

```text
work/reports/v7_train_selection_outcome.json
SHA-256 0f70f7ad2755c165dd4624eca35d0a467770122b1c95abb436d451ae5acd849e
```

B3 영상/export:

```text
work/training/evaluations/
2026-08-09_19-33-43_v7_model998_train_b3_success_video_export/
```

- `evaluation_summary.json`: 609/609 clean motion-end
- `videos/rl-video-step-0.mp4`: 1280×720, 50 Hz; train 수행/decode 증거
- `exported/policy.pt`: TorchScript 293→29 finite inference PASS
- `exported/policy.onnx`: checker/reference inference PASS

MP4의 changed-frame-pair fraction은 `0.023064`로 최종 held-out 영상 기준 `0.05`보다
낮다. 따라서 이 파일을 최종 대표 영상 PASS로 표현하지 않는다.

## Validation/TEST 봉인 규칙

현재 v7 train acceptance가 실패했으므로 다음을 실행하지 않는다.

- v7 checkpoint validation suite
- v28 TEST dynamic build
- held-out TEST suite
- 대표 TEST 영상
- success `final_artifact_manifest.json`
- success `final_completion.json`

`run.py status`의 Gate-5 terminal outcome은 과거 v5 validation 0/2 실패를 checksum으로
보존한다. v7은 train 단계에서 멈췄으므로 이 historical v5 outcome을 v7 validation
결과로 해석하면 안 된다. Validation split 자체도 과거 반복 진단에 사용돼 unbiased
holdout이 아니다. Male1 TEST는 v27 처리까지만 완료됐고 동적 build/정책 평가는 봉인됐다.

## v29/v30 주의

- `contact_v29/`는 audit-only다. production reference를 publish하지 않는다.
- `contact_aware_repair.py`는 연구 구현을 보존하지만 공식 v30 builder는 비활성이다.
- `build_contact_aware_repair.py`는 항상 `production_disabled`, exit code 2다.
- prototype은 `work/quarantine/contact_v30_prototypes/`에만 있다.
- v30을 train/validation/TEST 성공 증거로 사용하지 않는다.

## 최종 해석

현재 폴더는 다음을 추가 구현하지 않고 재현할 수 있다.

- ACCAD audit 및 249개 G1 v27 retarget
- Isaac G1 ingestion
- v28 train policy 학습/resume
- 결정론 suite와 acceptance
- MP4/JIT/ONNX export
- fail-closed validation/TEST 경계

다만 103/103 추종 성공을 만들려면 contact-constrained retargeting, multi-contact
curriculum, teacher/student imitation 또는 actuator/action redesign 같은 별도 연구가
필요하다. 현재 결과를 “정책 성공”으로 표현하지 않는다.
