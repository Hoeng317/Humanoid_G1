# GitHub에서 Humanoid G1 연구 환경 완전 복원하기

이 문서는 로컬 `humanoid_G1/` 폴더를 삭제한 뒤 GitHub `main` 브랜치에서
**자체 구현을 다시 설치하는 기준 절차**다. 경로는 예시일 뿐이며 설치 스크립트는
사용자의 홈 디렉터리나 계정 이름을 가정하지 않는다.

## 1. GitHub에 있는 것과 없는 것

GitHub `main`에는 다음 재현 재료가 들어 있다.

- G1 29-DoF asset/joint contract와 Isaac Lab task
- 시뮬레이션, PPO 학습·평가·export 진입점
- ACCAD 감사, SMPL-X 복원, G1 리타게팅과 검증 구현
- 리타게팅 대응표와 모든 설정 YAML
- CPU/Isaac 테스트와 설치·검증 스크립트
- 외부 저장소의 URL과 정확한 Git commit lock
- 연구 설명서와 당시 결과를 해석하기 위한 보고서

다음 항목은 의도적으로 `main`에 없다.

- AMASS의 ACCAD `*_stagei.npz`, `*_stageii.npz`
- 라이선스가 적용되는 `SMPLX_NEUTRAL.npz`
- `third_party/`에 다시 clone할 Unitree 원본 저장소
- `.local/` native build, 캐시, 임시 파일
- ACCAD 전처리 결과, PPO checkpoint, TensorBoard log, 영상

마지막 항목은 구현에 필수적인 소스가 아니다. 기존 실행 결과 일부는 선택적으로
`artifacts` 브랜치에 보관되어 있지만, 새 학습을 시작할 때 받을 필요는 없다.

## 2. 고정된 기준 환경

정확한 값은 `scripts/setup/dependency_lock.env`가 단일 기준이다.

| 구성 요소 | 기준 |
|---|---|
| Isaac Lab | 2.3.2, commit `2210934acca1a2f2401d541874163406b7ca8b53` |
| Isaac Sim | 5.1.0 |
| Python | Isaac Sim 포함 Python 3.11.13 |
| PyTorch | 2.7.0+cu128 |
| RSL-RL | 3.1.2 |
| G1 | Unitree G1 29-DoF rev_1_0 |
| SMPL-X Python | 0.1.28, 파이프라인 내부 `vendor/` 설치 |

다른 최신 버전으로 바로 올리면 API와 물리 결과가 달라질 수 있다. 먼저 위 버전으로
복원한 뒤 별도 브랜치에서 업그레이드해야 한다.

## 3. Isaac Sim과 Isaac Lab 설치

Ubuntu와 NVIDIA GPU 드라이버를 준비하고, NVIDIA 공식 문서에 따라 Isaac Sim
5.1.0 standalone을 설치한다.

- Isaac Sim 5.1.0: <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_workstation.html>
- Isaac Lab 2.3.2 설치 문서: <https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/index.html>

예를 들어 Isaac Sim을 `${HOME}/isaacsim`에 설치했다면 Isaac Lab은 다음처럼 정확한
commit으로 준비한다.

```bash
cd "${HOME}"
git clone https://github.com/isaac-sim/IsaacLab.git IsaacLab
cd IsaacLab
git checkout 2210934acca1a2f2401d541874163406b7ca8b53

ln -s "${HOME}/isaacsim" _isaac_sim
./isaaclab.sh -i rsl_rl
./_isaac_sim/python.sh -m pip install onnx==1.20.1
```

이미 같은 Isaac Lab commit과 `_isaac_sim` 링크가 있다면 이 단계는 건너뛴다.
`_isaac_sim/python.sh`가 실제로 실행 가능해야 한다.

## 4. 이 저장소와 Unitree 의존성 복원

반드시 Isaac Lab 바로 아래에 로컬 이름 `humanoid_G1`으로 clone한다.

```bash
cd "${HOME}/IsaacLab"
git clone https://github.com/Hoeng317/Humanoid_G1.git humanoid_G1
cd humanoid_G1

# Unitree 5개 저장소를 고정 commit으로 받고 import/contract/unit test 확인
./scripts/setup/bootstrap.sh
```

이 명령은 `third_party/`를 자동으로 만들며 기존 Isaac Sim Python 환경에 package를
설치하지 않는다. 모든 Python 진입점은 `PYTHONPATH=source`를 자체 설정한다.

MuJoCo Sim2Sim과 실물용 C++ controller까지 다시 빌드하려면 Ubuntu build 도구를
설치한 뒤 다음을 실행한다.

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build curl

cd "${HOME}/IsaacLab/humanoid_G1"
./scripts/setup/bootstrap.sh --with-native
```

이때 MuJoCo 3.3.6 archive의 SHA-256과 GLFW·Unitree Git commit이 자동 검증된다.

## 5. ACCAD와 SMPL-X를 다시 받는 방법

두 파일 묶음은 로그인과 라이선스 동의가 필요하므로 스크립트가 대신 다운로드하지
않는다.

1. <https://amass.is.tue.mpg.de/>에 로그인한다.
2. ACCAD의 **SMPL-X 형식**을 다운로드한다.
3. 압축을 풀어 20개 subject/category 폴더를 새 clone의 `ACCAD/` 바로 아래 둔다.
4. <https://smpl-x.is.tue.mpg.de/>에 로그인한다.
5. **SMPL-X with removed head bun (NPZ)**, 즉 locked-head 묶음을 다운로드한다.
6. `SMPLX_NEUTRAL.npz`의 경로를 아래 스크립트에 넘긴다.

올바른 배치 예시는 다음과 같다.

```text
humanoid_G1/ACCAD/
├── Female1General_c3d/
│   └── ..._stageii.npz
├── Female1Walking_c3d/
├── Male1General_c3d/
├── Male2MartialArtsKicks_c3d/
├── s001/
├── ...                         총 stage-II 252개, stage-I 20개
└── _g1_pipeline/               GitHub에서 받은 구현 코드
```

데이터와 모델을 배치한 뒤 다음 한 명령으로 수량·모델 checksum·Python package·Gate
1/2 준비 상태를 검증한다.

```bash
cd "${HOME}/IsaacLab/humanoid_G1"
./scripts/setup/prepare_motion_data.sh \
  --smplx-model "/다운로드를/푼/경로/models_lockedhead/smplx/SMPLX_NEUTRAL.npz"
```

현재와 똑같은 locked-head 모델의 SHA-256은
`43d8f3a1375d7c5baae207870a5d51def0f7e6b507df709b4937598b5e7d965d`다.
252개 Stage-II와 20개 Stage-I의 경로·개별 hash를 합친 ACCAD 집계 SHA-256은
`1d7a5803fc106ae5bee0d05b74fc04457418c4cd931e1d53d54beea60cc5f8f6`다.
다른 배포본이나 모델이 들어오면 조용히 계속하지 않고 실패한다.

## 6. 리타게팅을 처음부터 다시 생성

명령은 새 clone 루트에서 실행한다.

### 6.1 한 개 B3 보행으로 전체 연결 확인

```bash
cd "${HOME}/IsaacLab/humanoid_G1"

# ACCAD 감사 → SMPL-X prerequisite → B3 human reconstruction/validation
../_isaac_sim/python.sh ACCAD/_g1_pipeline/run.py all

# canonical human → G1 29-DoF retarget → 독립 Pinocchio 검증
../_isaac_sim/python.sh ACCAD/_g1_pipeline/run.py retarget
../_isaac_sim/python.sh ACCAD/_g1_pipeline/run.py validate-g1
```

### 6.2 ACCAD 전체 249개 고유 모션 처리

252개 Stage-II 중 checksum이 같은 alias 3개를 제거하므로 고유 입력은 249개다.

```bash
# Gate 1 manifest 생성
../_isaac_sim/python.sh ACCAD/_g1_pipeline/run.py audit

# 여러 process로 SMPL-X reconstruction + G1 retarget + offline validation
../_isaac_sim/python.sh ACCAD/_g1_pipeline/run_parallel_batch.py \
  --num-shards 4 --max-workers 4 \
  --splits train validation test

# 완료된 v27 archive를 독립 재검증한 뒤 split manifest 승격
../_isaac_sim/python.sh ACCAD/_g1_pipeline/build_gate3_manifests.py

# train에서 동적 scale 정책을 고정하고 validation에 그대로 적용
../_isaac_sim/python.sh ACCAD/_g1_pipeline/build_dynamic_retime.py --split train
../_isaac_sim/python.sh ACCAD/_g1_pipeline/build_dynamic_retime.py --split validation
```

`test`의 dynamic build는 최종 정책을 고르기 전에는 실행하지 않는다. 이것은 누락이
아니라 test leakage를 막기 위한 fail-closed 설계다.

## 7. Isaac Lab에서 G1 적용 확인

먼저 일반 G1 asset과 제어 경로를 확인한다.

```bash
./g1.sh inspect
./g1.sh simulate --mode hold-pose --num-envs 1 --steps 1 --headless
```

그다음 B3 retarget archive를 Isaac G1 관절에 직접 써서, 저장된 reference가 simulator
asset과 일치하는지 검사한다.

```bash
TERM=xterm ../isaaclab.sh -p ACCAD/_g1_pipeline/run_isaac.py kinematic \
  --headless --device cuda:0 --seed 42 --max-frames 0 \
  --input ACCAD/_g1_pipeline/work/g1/Female1Walking_c3d/B3_-_walk1_g1_50hz.npz \
  --output ACCAD/_g1_pipeline/work/reports/B3_-_walk1_isaac_kinematic.json
```

이 단계는 **학습이 아니다**. G1 관절 reference를 운동학적으로 재생하고 asset
ingestion 오차를 측정하는 검증이다.

## 8. PPO를 새로 학습

먼저 작은 smoke run으로 환경, observation 293차원, action 29차원, checkpoint 저장을
확인한다.

```bash
TERM=xterm ../isaaclab.sh -p ACCAD/_g1_pipeline/train_tracking.py \
  --headless --device cuda:0 \
  --manifest ACCAD/_g1_pipeline/work/dynamic/g1_train.json \
  --split train \
  --reference-root ACCAD/_g1_pipeline/work/dynamic/g1 \
  --max-motions 2 --num-envs 32 --max-iterations 2 \
  --seed 42 --run-name restore_smoke
```

smoke가 정상일 때 환경 수·iteration·PPO hyperparameter를 연구 설정으로 늘린다.
이 저장소는 코드와 설정을 재현하지만, 제외하기로 한 기존 checkpoint를 자동 복원하지
않는다. GPU 병렬 물리와 PPO의 수치 비결정성 때문에 새 학습 결과가 과거 checkpoint와
비트 단위로 같다고 보장하지도 않는다.

## 9. 최종 복원 검증

소스와 외부 코드가 준비된 상태에서:

```bash
cd "${HOME}/IsaacLab/humanoid_G1"
./scripts/setup/bootstrap.sh --check-only
./scripts/setup/prepare_motion_data.sh --check-only

../_isaac_sim/python.sh -m pytest -q -p no:cacheprovider \
  ACCAD/_g1_pipeline/tests
```

GitHub clone만 확인할 때는 첫 번째 명령까지만 실행하면 된다. 두 번째와 ACCAD test는
외부 motion data를 다시 받은 뒤 실행한다.

## 10. 기존 결과가 필요할 때만 artifacts 받기

```bash
git clone --single-branch --branch artifacts \
  https://github.com/Hoeng317/Humanoid_G1.git Humanoid_G1_Artifacts
cd Humanoid_G1_Artifacts
git lfs pull
```

이는 참고 결과 복구용이다. 새로운 전처리와 학습에는 필요하지 않다.

## 11. 삭제 전 의미

`main`이 원격과 같고 working tree가 깨끗하면 자체 구현은 GitHub에 보존된 것이다.
ACCAD와 SMPL-X 원본, `.local`, `third_party`, `work`는 다시 내려받거나 생성할 수 있다.
따라서 이 문서의 검증을 통과한 뒤 로컬 `humanoid_G1/`을 삭제해도 자체 구현 소스는
잃지 않는다.

현재 설치를 실제로 삭제하기 직전에는 다음 명령 하나를 사용한다.

```bash
cd /path/to/IsaacLab/humanoid_G1
./scripts/setup/verify_github_backup.sh --with-motion
```

이 명령은 local `main`과 GitHub `origin/main`의 commit 일치, 깨끗한 working tree,
필수 파일 추적 여부, raw data 제외, 외부 dependency commit, root/ACCAD unit test,
ACCAD/SMPL-X 준비 상태를 모두 검사한다. 마지막에 `BACKUP VERIFIED`가 출력되어야 한다.
