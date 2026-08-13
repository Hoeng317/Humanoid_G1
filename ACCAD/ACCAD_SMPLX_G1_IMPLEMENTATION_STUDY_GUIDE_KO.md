# ACCAD·SMPL-X 인간 모션에서 Unitree G1 물리 추종까지

## 구현 내용과 이론을 함께 이해하기 위한 학습용 해설서

- 문서 기준일: 2026-08-13
- 작업 경로: `/home/hoeng/IsaacLab/humanoid_G1/ACCAD`
- 대상 로봇: Unitree G1, 29 policy DoF
- 시뮬레이터: Isaac Sim 5.1 / Isaac Lab 0.54.2
- 정책: reference-conditioned residual PPO
- 목적: 코드를 읽기 전에 전체 데이터 흐름, 수학적 의미, 구현 결과와 한계를 이해한다.

---

## 0. 먼저 알아야 할 결론

현재 프로젝트가 한 일은 다음 한 문장으로 요약할 수 있다.

> ACCAD의 SMPL-X 인간 모션 파라미터를 사람의 3차원 관절 운동으로 복원하고, 이를 Unitree G1의 29개 관절 reference로 리타게팅한 다음, Isaac Lab 물리 환경에서 그 reference를 추종하도록 PPO residual policy를 학습했다.

여기에는 서로 다른 세 종류의 계산이 들어 있다.

1. **SMPL-X 인간 복원**
   - 입력은 인간의 자세와 체형 파라미터다.
   - 출력은 시간에 따른 인간 관절 위치·회전·속도·접촉 정보다.
   - 이것은 강화학습이 아니다.

2. **Human-to-G1 retargeting**
   - 입력은 복원된 인간 운동이다.
   - 출력은 G1의 29개 관절각과 floating-base reference다.
   - 이것은 미분 가능한 순방향 운동학과 수치 최적화를 이용한다.
   - 이것도 강화학습이 아니다.

3. **Isaac Lab PPO tracking**
   - 입력은 G1 reference와 현재 시뮬레이션 상태다.
   - 출력은 reference 관절각에 더할 29차원 residual이다.
   - 이것이 강화학습 단계다.

세 단계를 하나로 혼동하면 다음과 같은 잘못된 설명을 하게 된다.

- “PPO가 ACCAD 파일을 직접 읽어서 사람 동작을 배웠다.” → 정확하지 않다.
- “리타게팅이 통과했으므로 G1이 넘어지지 않는다.” → 정확하지 않다.
- “영상에서 격투 자세를 했으므로 정책이 격투라는 의미를 이해한다.” → 정확하지 않다.

정확한 관계는 다음과 같다.

```text
ACCAD Stage-II
  인간 자세·체형 파라미터
        ↓
SMPL-X forward kinematics
  인간 3D 관절·회전·접촉
        ↓
Human→G1 sequence retargeting
  G1 29-DoF reference trajectory
        ↓
Isaac Lab reference-conditioned PPO
  물리 오차를 보정하는 residual controller
        ↓
걷기·펀치·격투 자세 등 reference별 동작
```

현재 파이프라인은 실제로 끝까지 실행되었지만, 최종 정책은 전체 합격 상태가 아니다.

- Stage-II 파일: 252개 발견
- checksum이 동일한 alias: 3개
- 고유 모션: 249개
- 기구학적 G1 retarget 및 독립 검증: 249/249 통과
- 동적 prefilter 후 PPO train reference: 103개
- 선택 정책의 deterministic train 완주: 48/103
- 평균 completion: 0.6897
- 따라서 현재 정책은 **train-selected 연구 checkpoint**이며 범용·일반화 정책이 아니다.

---

# Part I. ACCAD와 SMPL-X

## 1. ACCAD는 무엇인가

ACCAD는 사람의 다양한 전신 동작을 motion capture로 기록한 데이터셋이다. 현재 내려받은 데이터에는 다음과 같은 범주의 동작이 포함되어 있다.

- 서기, 흔들기, 주변 보기
- 걷기, 뒤로 걷기, 방향 전환
- 달리기와 정지·전환
- 앉기, 숙이기, 눕기, 기어가기
- 물체를 줍거나 드는 동작
- 손짓과 대화 제스처
- 격투 준비 자세와 스탠스 전환
- 펀치, 훅, 크로스
- 킥, 회전 발차기
- 회피와 블록 동작

중요한 점은 ACCAD가 “로봇 명령 데이터”가 아니라 “사람의 운동 데이터”라는 것이다. 사람과 G1은 다음이 다르다.

- 신체 길이
- 관절 개수
- 관절 축
- 관절 가동 범위
- 발의 크기와 접촉 형상
- 질량과 관성
- actuator 출력
- 균형을 유지하는 방식

따라서 ACCAD의 사람 동작을 G1에 그대로 복사할 수 없다. ACCAD는 이 프로젝트에서 다음 역할을 한다.

> “로봇이 어떤 모양의 전신 동작을 따라야 하는가”를 제공하는 원천 데이터다.

ACCAD 자체는 다음을 제공하지 않는다.

- G1 관절각
- G1 torque
- G1 motor command
- G1에서의 균형 제어
- G1의 접촉력
- PPO의 정답 action

이 부족한 부분을 SMPL-X 복원, retargeting, 물리 학습이 차례로 채운다.

---

## 2. AMASS, ACCAD, Stage-II의 관계

AMASS는 여러 motion-capture 데이터셋을 공통한 parametric body 표현으로 정리한 데이터 모음이다. ACCAD는 그 안에 포함되는 원본 motion collection 중 하나다. 이 프로젝트에서 사용하는 `*_stageii.npz`는 ACCAD 동작이 SMPL-X 형식으로 fitting된 결과다. 따라서 세 이름의 관계를 다음처럼 이해하면 된다.

```text
ACCAD
  실제 motion-capture 동작 collection AMASS
  ACCAD 등 여러 collection을 공통 body-model 표현으로 정리한 체계 Stage-II NPZ
  특정 동작을 SMPL-X 파라미터 시계열로 저장한 파일
```

Stage-II 파일은 영상 파일도 아니고, 단순한 3D joint CSV도 아니다. 파일 안에는 “SMPL-X body를 어떤 체형과 어떤 관절 회전으로 놓을 것인가”가 들어 있다.

---

## 3. Stage-II NPZ에 무엇이 들어 있는가

한 동작의 프레임 수를 \(T\)라고 하자. 현재 구현이 엄격하게 검사하는 주요 필드는 다음과 같다.

| 필드 | 형태 | 의미 |
|---|---:|---|
| `trans` | \(T\times3\) | 사람 body root의 전역 translation |
| `root_orient` | \(T\times3\) | root의 axis-angle 회전 |
| `pose_body` | \(T\times63\) | 21개 body joint의 axis-angle |
| `pose_hand` | \(T\times90\) | 왼손 45 + 오른손 45 |
| `pose_jaw` | \(T\times3\) | 턱 회전 |
| `pose_eye` | \(T\times6\) | 양쪽 눈 회전 |
| `poses` | \(T\times165\) | 위 pose 성분을 합친 전체 pose |
| `betas` | \(16\) | 사람 체형 계수 |
| `num_betas` | scalar | beta 개수, 현재 계약은 16 |
| `mocap_frame_rate` | scalar | 원본 sampling rate |
| `mocap_time_length` | scalar | 모션 시간 길이 |
| `gender` | scalar text | 원본 metadata의 성별 표현 |
| `surface_model_type` | scalar text | 현재 계약에서는 `smplx` |

### 3.1 Axis-angle이란 무엇인가

Axis-angle은 3차원 벡터 하나로 회전을 표현한다. 벡터를 \(\boldsymbol\omega\in\mathbb R^3\)라고 하면 다음과 같이 해석한다.

- 벡터 방향: 회전축
- 벡터 길이 \(\|\boldsymbol\omega\|\): 회전각, radian

예를 들어 \([0,0,\pi/2]\)는 Z축을 중심으로 90도 회전하는 것을 뜻한다. 단, axis-angle 성분을 프레임 사이에서 단순 선형 보간하면 회전의 shortest path가 보장되지 않는다. 그래서 현재 구현은 다음 순서를 사용한다.

1. axis-angle을 quaternion으로 변환한다.
2. quaternion shortest-path SLERP를 수행한다.
3. 필요하면 다시 axis-angle로 변환한다.

### 3.2 Beta가 의미하는 것

`betas`는 동작이 아니라 체형을 나타낸다.

예를 들어 beta 변화는 다음과 같은 body shape 차이에 영향을 준다.

- 키와 팔다리 비율
- 몸통 폭
- 신체 부피 분포

Pose는 같은데 beta가 다르면 관절 회전은 같더라도 실제 joint 위치와 mesh 형태가 달라질 수 있다. 그래서 사람 joint 위치를 정확히 복원하려면 pose뿐 아니라 beta와 SMPL-X model이 함께 필요하다.

### 3.3 `poses` 일관성 검사

현재 구현은 `poses`가 다음 성분을 정확히 이어 붙인 것인지 검사한다.

\[
\text{poses}
=
[\text{root},\text{body},\text{jaw},\text{eyes},\text{hands}]
\]

별도 필드와 `poses`가 다르면 잘못된 Stage-II 파일로 판단한다. 이는 단순 shape 검사보다 강한 검증이다. 파일의 배열 크기가 맞더라도, 내부 의미가 서로 충돌하면 다음 단계로 보내지 않는다.

---

## 4. 왜 SMPL-X model 파일이 별도로 필요한가

Stage-II NPZ에는 body model 전체가 들어 있지 않다. Stage-II는 다음과 같은 “명령값”을 제공한다.

- 체형 계수
- 관절 회전
- root translation

하지만 이 명령값으로 실제 3D 관절 위치를 만들려면 다음 정보가 추가로 필요하다.

- template body
- kinematic parent tree
- shape blend shape
- joint regressor
- skinning 관련 model parameters

이 정보가 SMPL-X model 파일에 들어 있다. 현재 다운로드된 model 경로는 다음과 같다.

`ACCAD/smplx_lockedhead_20230207/models_lockedhead/smplx/`

이 폴더에는 다음 파일이 존재한다.

- `SMPLX_FEMALE.npz`
- `SMPLX_MALE.npz`
- `SMPLX_NEUTRAL.npz`
- `md5sums.txt`

현재 파이프라인이 실제로 고정해서 사용하는 파일은 `SMPLX_NEUTRAL.npz`다. 즉, male/female 파일이 존재하더라도 현재 human reconstruction은 neutral model을 사용한다. 또한 다음 환경을 고정한다.

- Python package: `smplx==0.1.28`
- model type: SMPL-X
- model gender: neutral
- PCA hand pose: 사용하지 않음
- beta 개수: 16
- flat hand mean: 사용

`smplx_lockedhead_20230207`은 다운로드된 model distribution의 폴더 이름이다.

현재 구현은 이 이름만 믿지 않고, 실제 `SMPLX_NEUTRAL.npz`의 필수 key·dtype·finite 여부와 checksum을 검사한다. 따라서 “lockedhead 파일을 받았으니 자동으로 맞다”가 아니라 다음을 증명한다.

1. 필요한 model 파일이 실제로 있다.
2. 파일 형식이 SMPL-X model 계약과 맞는다.
3. 요구한 Python package 버전과 맞는다.
4. 이후 산출물에 사용 model의 SHA-256을 기록한다.

---

## 5. 원본 ACCAD 감사가 한 일

원본 감사는 `accad_g1/data.py`의 책임이다. 감사의 목적은 “동작이 멋있어 보이는가”가 아니다. 목적은 모든 원본 파일의 존재, 구조, 시간축, 수치 안정성, 중복 관계를 기록하는 것이다.

### 5.1 파일 accounting

실제 감사 결과는 다음과 같다.

| 항목 | 결과 |
|---|---:|
| 발견한 Stage-I 파일 | 20 |
| 발견한 Stage-II 파일 | 252 |
| checksum-identical alias | 3 |
| 고유하고 구조적으로 사용 가능한 Stage-II | 249 |

alias는 이름은 다르지만 byte checksum이 같은 파일이다. 같은 motion을 두 번 학습 데이터로 세지 않기 위해 canonical 파일 하나만 유지한다. 세 중복 그룹은 walking·running 파일에서 발견되었고, 제외 사실과 canonical 경로를 manifest에 기록했다.

### 5.2 구조 검사

각 파일에 대해 다음을 검사한다.

- 필수 배열 존재
- 예상 shape
- 모든 값이 finite인지
- frame 수가 일치하는지
- FPS가 양수인지
- duration이 `frame_count / fps`와 맞는지
- beta가 16개인지
- surface model type이 SMPL-X인지
- pose 구성 요소와 full pose가 일치하는지

### 5.3 시간적 품질 검사

다음 현상도 기록한다.

- 한 프레임 사이에 translation이 비정상적으로 점프하는가
- translation speed가 비정상적으로 큰가
- root 또는 body rotation의 geodesic 변화가 큰가
- 완전히 같은 frame이 과도하게 반복되는가

현재 설정에는 품질 flag가 있다고 무조건 자동 삭제하는 정책을 사용하지 않는다. 대신 구조적으로 위험한 파일과 checksum 중복은 제외하고, 품질 신호는 보고서에 남긴다.

### 5.4 보안·재현성 경계

NPZ는 `allow_pickle=False`로 읽는다. 이유는 일부 AMASS 파일에 object dtype marker metadata가 들어 있을 수 있기 때문이다. 현재 파이프라인은 필요한 numeric field만 신뢰하며, 알 수 없는 object field를 역직렬화하지 않는다. 모든 입력과 주요 산출물에는 SHA-256을 기록한다. 따라서 나중에 같은 이름의 파일이 바뀌면 checksum chain이 끊어져 탐지할 수 있다.

---

## 6. Train·validation·TEST 분할

분할은 clip을 무작위로 섞는 방식이 아니라 subject 단위로 수행했다.

| Split | Subject | 고유 모션 수 |
|---|---|---:|
| train | Female1, Male2 | 215 |
| validation | s001, s007, s008, s009, s011 | 5 |
| TEST | Male1 | 29 |

### 6.1 왜 subject-disjoint가 필요한가

같은 사람이 거의 같은 방식으로 수행한 clip이 train과 TEST에 동시에 있으면 정책이 performer 특성을 기억했을 가능성을 배제하기 어렵다. Subject-disjoint split은 최소한 다음을 묻기 위한 장치다.

> 학습에 사용한 사람과 다른 사람의 motion reference에도 같은 방법이 작동하는가?

### 6.2 현재 split의 한계

ACCAD에는 큰 named actor 수가 많지 않다. 그래서 validation은 5개로 매우 작고, category 분포도 균형적이지 않다. 예를 들어 격투 동작은 대부분 Male2에 있어 train에 집중되어 있다. 따라서 다음을 구분해야 한다.

- 분할이 subject-disjoint라는 사실
- validation이 충분히 크고 균형적이라는 주장

첫 번째는 맞지만 두 번째는 아니다. 또한 validation split은 과거 정책 진단에 반복 사용되어 pristine holdout이라고 부를 수 없다. 현재 v7 정책의 validation suite와 TEST 정책 평가는 실행하지 않았다.

---

## 7. SMPL-X를 이용한 canonical human reconstruction

이 단계는 `accad_g1/human.py`에 구현되어 있다. 목표는 Stage-II의 body parameters를 다음처럼 명시적인 human motion으로 바꾸는 것이다.

\[
\mathcal H_t=
\{p_j(t),R_j(t),v_j(t),\omega_j(t),c_k(t)\}
\]

여기서 다음 기호를 사용한다.

- \(p_j\): human joint의 world position
- \(R_j\): human joint의 world orientation
- \(v_j\): linear velocity
- \(\omega_j\): angular velocity
- \(c_k\): heel/toe contact label

### 7.1 Forward kinematics

Forward kinematics, 줄여서 FK는 관절 회전과 skeleton 구조를 이용하여 각 joint의 전역 위치·회전을 계산하는 과정이다. 부모 joint의 전역 transform을 \(T_{p(j)}\), 자식의 local transform을 \(T_j^{local}\)이라고 하면 다음과 같다.

\[
T_j^{world}=T_{p(j)}^{world}T_j^{local}
\]

root부터 자식 방향으로 이 연산을 반복하면 전체 skeleton의 전역 자세가 나온다. SMPL-X model은 55개의 kinematic joint parent 관계를 제공한다. 현재 구현은 parent table에 다음 문제가 없는지 검사한다.

- joint 수가 정확히 55인지
- parent index가 범위 안인지
- cycle이 없는지
- root가 정확히 하나인지

### 7.2 숫자 index가 아니라 semantic name 사용

사람 joint mapping을 단순히 “index 10은 발일 것이다”라고 추정하지 않는다. 고정한 SMPL-X package의 upstream joint-name table에서 다음 semantic name을 확인한다.

- pelvis
- hip, knee, ankle
- foot, heel, big toe, small toe
- wrist
- head

필수 이름이 없으면 복원을 중지한다. 이는 package 버전 변화로 joint 순서가 바뀌어도 조용히 잘못된 발과 손을 사용하는 일을 막는다.

### 7.3 원본 sampling rate에서 먼저 접촉 계산

접촉처럼 짧은 사건은 먼저 50 Hz로 낮춘 뒤 찾으면 사라질 수 있다. 그래서 구현 순서는 다음과 같다.

1. 원본 FPS에서 SMPL-X FK를 수행한다.
2. 원본 FPS의 heel/toe 궤적에서 ground와 contact를 추정한다.
3. 그 뒤 pose와 translation을 50 Hz로 재표본화한다.
4. 50 Hz에서 FK를 다시 수행한다.

이 순서를 사용하면 짧은 발 접촉 전환을 더 잘 보존할 수 있다.

### 7.4 Ground baseline 추정

원본 motion에는 바닥이 항상 정확히 \(z=0\)으로 고정되어 있지 않을 수 있다. 또한 calibration이나 fitting 때문에 전체 몸이 천천히 위아래로 drift할 수 있다. 현재 구현은 heel/toe 중 천천히 움직이는 후보를 이용하여 시간에 따른 ground baseline을 추정한다. 이 baseline은 다음 원칙을 따른다.

- 점프 중인 발을 바닥으로 간주하지 않는다.
- 너무 빠른 vertical 변화는 바닥 drift로 따라가지 않는다.
- 신뢰할 수 없는 비행 구간은 주변 baseline을 연결한다.
- 실제 사람의 점프 높이를 없애지 않고 common-mode floor drift만 제거한다.

정규화된 root translation은 다음 개념을 따른다.

\[
z_{normalized}(t)=z_{source}(t)-h_{ground}(t)
\]

### 7.5 Heel/toe contact hysteresis

Contact는 한 개 threshold로 켰다 껐다 하지 않는다. 한 threshold만 사용하면 경계에서 contact가 프레임마다 진동하는 chatter가 발생할 수 있다. 현재 구현의 기본 hysteresis 개념은 다음과 같다.

- contact 진입: 높이 0.04 m 미만, 속도 0.35 m/s 미만
- contact 유지 종료: 높이 0.07 m 초과 또는 속도 0.60 m/s 초과
- 너무 짧은 true run: 제거

즉, contact를 시작하는 조건과 끝내는 조건이 다르다. 또한 다음 네 channel을 별도로 유지한다.

- left heel
- left toe
- right heel
- right toe

이렇게 해야 뒤꿈치부터 닿고 앞꿈치가 나중에 닿는 보행 접촉을 표현할 수 있다.

### 7.6 50 Hz 재표본화

G1 control contract는 50 Hz다. 따라서 모든 human motion을 일정한 0.02초 간격으로 변환한다. Translation은 downsampling 전에 anti-alias 처리를 하고 보간한다. 회전은 quaternion SLERP를 사용한다. Contact boolean은 target sample 주변 원본 frame들의 다수결을 사용하며, 동률일 때는 가장 가까운 원본 frame을 따른다. 최종 human archive에는 다음이 저장된다.

- root 위치·quaternion·선속도·각속도
- named human joint 위치
- local/global joint quaternion
- joint 선속도·각속도
- heel/toe 위치와 contact
- ground correction
- 원본 Stage-II 경로와 checksum
- SMPL-X model 경로·버전·checksum
- 좌표계와 단위

이 archive는 아직 G1 motion이 아니다. 사람 skeleton을 명시적으로 복원한 중간 표현이다.

---

# Part II. Human motion을 G1로 retargeting

## 8. Retargeting이 필요한 이유: embodiment gap

Embodiment는 “움직임을 수행하는 물리적 몸”을 뜻한다. 사람과 G1의 embodiment가 다르기 때문에 같은 joint angle을 복사하는 방식은 의미가 없다. 예를 들어 사람의 shoulder joint와 G1의 shoulder는 다음이 다르다.

- 회전축 방향
- joint 수
- 팔 길이
- elbow와 wrist의 연결 구조
- 가동 범위

다리도 마찬가지다. 사람의 발 위치를 따라가려면 G1 hip·knee·ankle이 협력해야 하지만, 사람 관절각과 G1 관절각 사이에는 일대일 대응이 없다. 따라서 retargeting은 다음 문제로 정의한다.

> 인간 동작의 중요한 의미—골반 이동, 무릎·발·손 위치, 몸통 방향, 접촉 순서—를 가능한 한 보존하면서 G1의 유효한 관절 궤적을 찾는다.

---

## 9. G1의 29 DoF와 floating base

현재 policy가 제어하는 G1 관절은 29개다. 여기에는 대략 다음 관절군이 포함된다.

- 양쪽 hip pitch/roll/yaw
- 양쪽 knee
- 양쪽 ankle pitch/roll
- waist yaw/roll/pitch
- 양쪽 shoulder pitch/roll/yaw
- 양쪽 elbow
- 양쪽 wrist roll/pitch/yaw

29 DoF는 actuated joint 수다.

시뮬레이션에서 골반 base는 world에 고정되어 있지 않은 floating base다. 따라서 전체 상태에는 관절 29개 외에도 root의 위치와 방향, 선속도와 각속도가 존재한다. Retarget reference도 다음 두 부분을 함께 갖는다.

1. G1 joint trajectory \(q_{ref}(t)\in\mathbb R^{29}\)
2. Root trajectory \(p_{root}(t),R_{root}(t)\)

PPO action은 29개 joint residual만 출력하지만, root reference는 observation과 reward를 통해 간접적으로 추종한다.

---

## 10. Human↔G1 semantic correspondence

대응 관계는 `_g1_pipeline/correspondence.yaml`에 정의되어 있다. 숫자 index가 아니라 human semantic joint와 G1 link 이름을 연결한다.

### 10.1 Position target

현재 주요 position 대응은 다음과 같다.

| G1 link | Human target | 상대 가중치 |
|---|---|---:|
| left/right knee link | left/right knee | 4.0 |
| left/right ankle roll link | left/right ankle | 2.0 |
| left/right elbow link | left/right elbow | 2.0 |
| left/right wrist yaw link | left/right wrist | 4.0 |

손과 무릎은 동작 의미를 크게 결정하므로 상대적으로 높은 가중치를 갖는다.

### 10.2 Orientation target

몸통 방향은 다음 대응을 사용한다.

- G1 torso link ↔ human spine3
- orientation weight: 1.5

### 10.3 Tracking body

Reference와 policy 평가에는 14개 G1 body를 사용한다.

- pelvis
- 양쪽 hip roll
- 양쪽 knee
- 양쪽 ankle roll
- torso
- 양쪽 shoulder roll
- 양쪽 elbow
- 양쪽 wrist yaw

이는 관절각만 맞고 실제 손·발·몸통 위치가 크게 어긋나는 해를 감지하기 위한 것이다.

---

## 11. 좌표계 정렬

Human reconstruction archive는 right-handed Z-up world를 사용한다. G1의 canonical local convention은 다음과 같다.

- X: forward
- Y: left
- Z: up

SMPL-X model-local 축 의미와 G1 local 축 의미가 다르기 때문에 고정된 orthonormal basis mapping을 사용한다. 이 mapping은 다음을 검사한다.

- \(M^TM=I\): 축들이 서로 직교하고 길이가 1인지
- \(\det(M)>0\): 오른손 좌표계를 보존하는지

G1 root orientation은 개념적으로 다음과 같이 만든다.

\[
R_{root}^{G1}=R_{root}^{human}M
\]

여기서 \(M\)은 human-local과 G1-local 사이의 basis transform이다. 이 과정을 생략하면 forward·left·up의 의미가 뒤섞여 로봇이 옆으로 눕거나 반대 방향을 바라볼 수 있다.

---

## 12. Morphology scale

사람과 G1의 다리 길이가 다르므로 위치 target을 그대로 사용하면 발이나 무릎 위치를 맞출 수 없다. 현재 구현은 사람의 pose-dependent pelvis 높이를 그대로 scale 기준으로 사용하지 않는다. 그 이유는 눕기·기어가기 motion에서 pelvis 높이가 작아져 잘못된 scale이 나오기 때문이다. 대신 다음 rigid bone-chain 길이를 이용한다.

\[
L_{human}
=
\|p_{pelvis}-p_{hip}\|
+\|p_{hip}-p_{knee}\|
+\|p_{knee}-p_{ankle}\|
\]

좌우와 전체 frame의 median을 사용해 pose 변화에 덜 민감한 사람 leg height를 얻는다. 동작에 충분한 upright support sample이 있으면 pelvis-to-sole 높이를 추가로 활용한다. G1 leg height는 G1 default pose의 URDF FK와 sole 위치에서 계산한다. 최종 scale은 다음 개념을 따른다.

\[
s=\operatorname{clip}
\left(
\frac{L_{G1}}{L_{human}},0.70,1.10
\right)
\]

Human joint의 pelvis-relative vector에 이 scale을 곱한다.

\[
p_j^{target}
=p_{root}^{G1}
+s\left(p_j^{human}-p_{pelvis}^{human}\right)
\]

Root XY의 첫 프레임은 원점에 맞춘다. 따라서 원본 capture 공간의 임의 전역 위치가 G1 reference에 그대로 남지 않는다.

---

## 13. Contact와 stance는 다른 개념이다

이 구분은 현재 구현에서 매우 중요하다.

### 13.1 Contact

Contact는 human heel/toe가 ground 근처에 있다는 기하학적 label이다. 예를 들어 다음도 contact가 될 수 있다.

- 발을 끌고 가는 순간
- 발끝으로 pivot하는 순간
- crawling 중 발이 바닥 가까이 있는 순간

### 13.2 Stance

Stance는 실제로 고정 anchor처럼 취급할 수 있는 더 엄격한 접촉이다. 현재 stance가 되려면 다음 성질을 함께 본다.

- human contact가 true
- pelvis가 load-bearing posture에 충분히 높음
- contact point가 국소적으로 느리게 움직임
- 일정 frame 이상 지속됨
- target drift가 제한 이내임

주요 설정은 다음과 같다.

- stance speed threshold: 0.18 m/s
- target drift limit: 0.025 m
- 최소 stance 길이: 3 frame
- 한 anchor의 최대 길이: 50 frame
- 최소 pelvis-height ratio: 0.45

너무 긴 anchor는 중간에 재-anchor 경계를 넣는다. 이유는 인간 fitting의 작은 장기 drift를 하나의 절대 고정점으로 강제하면 전체 sequence가 왜곡될 수 있기 때문이다. 최종 G1 archive의 stance는 한 번 더 G1 support geometry에서 정제한다. 따라서 최종 stance는 다음 관계를 갖는다.

\[
\text{G1 output stance}
\subseteq
\text{preliminary load-bearing human stance}
\subseteq
\text{human geometric contact}
\]

---

## 14. Retargeting 최적화 변수

한 sequence의 frame 수를 \(T\)라고 하자. 최적화 변수는 크게 두 종류다.

1. 모든 frame의 G1 joint angle

\[
Q=\{q_t\in\mathbb R^{29}\}_{t=0}^{T-1}
\]

2. 모든 frame의 제한된 root position correction

\[
\Delta P=\{\Delta p_t\in\mathbb R^3\}_{t=0}^{T-1}
\]

Root orientation은 human root와 basis mapping에서 정해지며 이 optimizer가 자유롭게 바꾸지 않는다. Joint angle 초기값은 G1 default pose다. Root correction 초기값은 0이다.

### 14.1 Joint limit를 구조적으로 만족시키는 방법

Optimizer가 unconstrained 변수 \(u_t\)를 갱신하고, 실제 joint angle은 tanh로 decode한다.

\[
q_t=c+h\tanh(u_t)
\]

여기서 다음과 같다.

- \(c\): 허용 joint interval의 중심
- \(h\): 허용 joint interval의 half range

이 구조에서는 \(\tanh(u)\in(-1,1)\)이므로 실제 \(q_t\)가 interval 밖으로 나가지 않는다. 현재 interval은 hard limit 전체를 쓰지 않는다.

- hard range의 90% 사용
- 양끝에서 0.02 rad margin 추가

즉, 경계에 딱 붙는 궤적보다 actuator와 수치 오차에 여유가 있는 궤적을 만든다.

### 14.2 Root correction 제한

Root reference를 완전히 자유롭게 움직이면 손·발 target을 쉽게 맞출 수 있지만 사람 motion의 이동 의미가 사라질 수 있다. 그래서 correction trust region을 둔다.

- X/Y correction: 각각 최대 0.05 m
- 일반 root Z correction: 최대 0.30 m
- non-upright motion Z correction: 최대 0.40 m

Root correction도 tanh로 제한한다.

---

## 15. Retargeting 목적함수

전체 목적함수는 여러 항의 가중합이다.

\[
\mathcal L
=
\mathcal L_{body}
+\mathcal L_{foot}
+\mathcal L_{contact}
+\mathcal L_{stance}
+\mathcal L_{ground}
+\mathcal L_{smooth}
+\mathcal L_{regularization}
\]

### 15.1 Body position loss

대응된 knee, ankle, elbow, wrist의 위치 오차를 최소화한다.

\[
\mathcal L_{position}
=
\sum_jw_j
\frac1T\sum_t
\|p_j^{G1}(q_t)-p_j^{target}(t)\|^2
\]

이 항이 인간 동작의 팔다리 모양을 G1에서 보존한다.

### 15.2 Torso orientation loss

G1 torso rotation과 human spine3 기반 target rotation을 맞춘다. 현재 구현은 rotation matrix 성분의 제곱 오차를 사용한다.

\[
\mathcal L_{orientation}
=w_R\operatorname{mean}
\left(\|R_{torso}^{G1}-R_{spine3}^{target}\|_F^2\right)
\]

### 15.3 Foot position loss

양발 sole의 위치를 morphology-scaled human sole target에 맞춘다. 가중치는 16.0이다.

### 15.4 Contact position loss

Heel/toe contact가 true인 frame에서 G1 support representative를 human-derived contact target에 맞춘다. 가중치는 512.0으로 크다. 이 항은 contact가 아닌 swing foot에 잘못된 ground anchor를 강제하지 않는다.

### 15.5 Stance ground loss

Contact support sphere center가 ground 위 collision radius 높이에 있도록 한다. 가중치는 400.0이다.

### 15.6 Stance lock loss

연속된 stance frame에서 heel/toe representative의 XY 이동을 줄인다.

\[
\mathcal L_{lock}
=w_{lock}
\operatorname{mean}_{stance}
\|p_{t+1}^{xy}-p_t^{xy}\|^2
\]

가중치는 2048.0이다. 이 항이 foot skating을 줄이는 핵심 중 하나다.

### 15.7 Penetration loss

G1 발 support geometry가 floor 아래로 내려가면 제곱 벌점을 준다.

\[
\mathcal L_{penetration}
=w_p
\operatorname{mean}
\left[
\max(r_{foot}-z_{support},0)^2
\right]
\]

가중치는 10000.0으로 매우 크다.

### 15.8 Stance orientation loss

발 전체가 지지 중일 때 ankle/foot 방향이 ground와 평행하고 human heading을 따르도록 한다. 가중치는 10.0이다.

### 15.9 Nominal pose regularization

필요 이상으로 G1 default pose에서 멀어지지 않도록 한다.

\[
\mathcal L_{nominal}
=w_n\operatorname{mean}\|q_t-q_{default}\|^2
\]

가중치는 0.012다. 이 값은 tracking target보다 훨씬 작으며, target이 모호한 자유도를 안정시키는 역할을 한다.

### 15.10 Joint velocity와 acceleration smoothness

Joint velocity proxy는 다음과 같다.

\[
\dot q_t\approx\frac{q_{t+1}-q_t}{\Delta t}
\]

Joint acceleration smoothness는 두 번째 차분을 사용한다.

\[
q_{t+2}-2q_{t+1}+q_t
\]

현재 가중치는 다음과 같다.

- joint velocity smoothness: 2.0
- joint acceleration smoothness: 20.0
- joint velocity limit excess: 2.0
- velocity limit factor: URDF limit의 0.9

### 15.11 Root correction smoothness

Root correction 자체와 그 1차·2차 차분도 벌점으로 준다.

- root correction magnitude: 1.0
- root correction velocity: 1.0
- root correction acceleration: 2.0

이 항이 frame별 IK 결과가 흔들리는 것을 막고 sequence 전체를 연속적으로 만든다.

---

## 16. 왜 frame별 IK가 아니라 sequence optimization인가

각 frame을 독립적으로 풀면 그 순간의 손·발 위치는 잘 맞출 수 있다. 하지만 인접 frame 사이에서 서로 다른 local minimum을 선택하면 다음 문제가 생긴다.

- 관절각 jitter
- 큰 joint velocity
- 비현실적인 joint acceleration
- stance foot sliding
- policy가 추종하기 어려운 reference

현재 구현은 모든 frame의 \(q_t\)와 root correction을 한 번에 최적화한다. 그래서 시간 차분 loss와 stance lock을 직접 걸 수 있다. 이것이 `torch_global_sequence_adam`이라는 solver 이름의 의미다. 주요 optimizer 설정은 다음과 같다.

- optimizer: Adam
- iteration: 900
- 초기 learning rate: 0.035
- cosine annealing
- gradient norm clipping: 10
- seed: 42
- 기본 device: CUDA
- 기본 optimization dtype: float32

최종 FK serialization은 별도의 CPU float64 G1 kinematics로 다시 계산한다. 이는 optimizer의 float32 endpoint noise와 저장값 검증 오차를 줄이기 위한 것이다.

---

## 17. 최종 ground projection

Soft penetration loss가 있다고 해서 모든 발 support point가 수치적으로 정확히 ground 위에 있다는 보장은 없다. 그래서 최적화 후 final CPU float64 FK에서 각 frame의 최저 support height를 다시 계산한다. 필요한 최소 Z shift는 다음과 같다.

\[
\Delta z_t
=
\max
\left(
r_{foot}+m-\min_k z_{support,k}(t),0
\right)
\]

여기서 다음과 같다.

- \(r_{foot}\): collision support radius
- \(m\): ground margin, 현재 0.0005 m

이 correction은 root Z만 올린다. 따라서 joint angle과 body-relative pose는 바꾸지 않으면서 floor half-space를 만족시킨다. 필요 correction이 trust region을 비정상적으로 크게 넘으면 fail-closed 한다.

---

## 18. G1 reference archive에 저장되는 값

Retarget 결과는 `accad_g1_reference_v1` archive다. 주요 배열은 다음과 같다.

| 배열 | 의미 |
|---|---|
| `joint_pos` | policy order의 \(T\times29\) 관절각 |
| `joint_pos_sdk` | SDK/URDF order로 재배열한 관절각 |
| `joint_vel` | 관절속도 |
| `joint_acc` | 관절가속도 proxy |
| `root_pos_w` | world root position |
| `root_quat_wxyz` | world root quaternion |
| `root_lin_vel_w` | root 선속도 |
| `root_ang_vel_w` | root 각속도 |
| `body_pos_w` | G1 body world position |
| `body_quat_wxyz` | G1 body world orientation |
| `body_lin_vel_w` | body 선속도 |
| `body_ang_vel_w` | body 각속도 |
| `left/right_foot_contact` | 발 단위 contact |
| heel/toe contact | 네 개 contact channel |
| heel/toe stance | 최종 G1 기준 stance channel |
| support positions | G1 발 support geometry |

추가로 다음 provenance를 저장한다.

- source human archive checksum
- source Stage-II checksum
- G1 joint contract checksum
- G1 URDF checksum
- correspondence checksum
- solver version과 iteration
- morphology scale
- 단위·좌표계·quaternion order

이 정보는 잘못된 관절 순서나 다른 URDF 결과가 조용히 섞이는 것을 막는다.

---

## 19. Retarget 결과를 어떻게 검증했는가

생성 코드가 자기 결과를 자기 방식으로 다시 확인하는 것만으로는 충분하지 않다. 같은 bug가 생성과 검사에 동시에 있으면 둘이 서로 일치할 수 있기 때문이다. 그래서 `accad_g1/g1_validation.py`가 별도의 검증 경로를 사용한다.

### 19.1 검증 항목

- archive schema와 필수 field
- frame 수와 50 Hz time contract
- quaternion 순서 `wxyz`
- policy joint order와 SDK/URDF joint order permutation
- source·model·contract·URDF checksum
- hard joint position limit
- soft joint position limit
- joint velocity limit
- stored derivative와 재계산 derivative
- stored body FK와 독립 FK
- heel/toe contact 구성
- contact geometry와 ground penetration
- stance point 속도와 drift
- NaN/Inf

### 19.2 생성과 검증의 분리

Retargeter는 처음 archive를 만들 때 `gate3_complete=false`로 저장한다. 독립 검증이 통과한 archive만 manifest에서 승격한다. 즉, “파일을 생성했다”와 “사용 가능한 reference로 승인했다”가 다르다.

### 19.3 실제 결과

| Split | 모션 | Frame | 시간 |
|---|---:|---:|---:|
| train | 215 | 66,076 | 1,317.22 s |
| validation | 5 | 1,952 | 38.94 s |
| TEST | 29 | 11,490 | 229.22 s |
| 전체 | 249 | 79,518 | 1,585.38 s |

249개 고유 모션 모두 canonical human 처리, G1 retarget, 독립 기구학 검증까지 통과했다.

여기서 말하는 PASS는 다음 뜻이다.

> 저장된 G1 궤적이 지정한 관절 순서·URDF·joint limit·FK·ground/contact 형식에 맞는 유효한 수치 궤적이다.

다음 뜻은 아니다.

> G1이 중력과 접촉이 있는 시뮬레이션에서 이 궤적을 반드시 수행할 수 있다.

---

## 20. v28 dynamic prefilter

기구학적으로 유효한 249개를 바로 PPO에 모두 넣지 않았다. 먼저 `dynamic_retime.py`에서 conservative dynamic prefilter를 수행했다.

### 20.1 검사한 값

- segment joint velocity
- root linear/angular speed
- body speed
- joint acceleration proxy
- 양발이 모두 떨어진 연속 flight 구간

Joint velocity factor의 개념은 다음과 같다.

\[
s_v
=
\max_{t,j}
\frac{|\dot q_{t,j}|}{0.8\dot q_{limit,j}}
\]

Acceleration은 URDF의 공식 limit가 아니라 velocity limit에 기반한 보수적 proxy다. 모든 factor의 최대가 1보다 크면 uniform slowdown이 필요한 후보가 된다.

### 20.2 왜 flight motion을 무조건 느리게 하지 않는가

시간을 늘리면 지상 motion의 속도와 acceleration을 낮출 수 있다. 하지만 점프의 체공 시간을 늘리면 중력 아래에서 같은 궤적이 더 쉬워지는 것이 아니라 오히려 물리적으로 다른 동작이 된다. 그래서 연속 flight가 3 frame 이상인데 추가 slowdown이 필요한 clip은 quarantine한다.

### 20.3 결과

| Split | v27 입력 | v28 승격 | 미승격 |
|---|---:|---:|---:|
| train | 215 | 103 | 112 |
| validation | 5 | 2 | 3 |
| TEST | 29 | 미구축 | 봉인 |

PPO는 이 103개 train reference를 사용했다. 이 103개에는 걷기뿐 아니라 일부 펀치, 발차기, 격투 스탠스, 블록·회피 동작도 포함된다. 그러나 v28 승격도 물리 수행의 충분조건은 아니다. 이는 velocity·acceleration·flight에 관한 사전 필터일 뿐, contact force와 균형을 직접 풀지 않는다.

### 20.4 v29·v30의 의미

v29 offline contact proxy audit에서는 train 103개 중 3개만 eligible이었다. 이는 많은 reference가 centroidal/contact 관점에서 어렵다는 신호지만, proxy이므로 물리 ground truth로 사용하지 않는다. v30 contact-aware repair는 prototype까지 구현했지만 검증 chain을 통과하지 못해 production 경로에서 비활성화했다. 따라서 현재 정책은 공식적으로 v28 reference를 사용한다.

---

# Part III. Isaac Sim·Isaac Lab에 적용한 방법

## 21. Isaac Sim과 Isaac Lab의 역할

두 이름은 같은 것이 아니다.

### 21.1 Isaac Sim

Isaac Sim은 다음을 제공하는 시뮬레이터다.

- PhysX rigid-body dynamics
- 충돌과 접촉
- 중력
- articulation joint
- sensor와 rendering
- USD scene

### 21.2 Isaac Lab

Isaac Lab은 Isaac Sim 위에서 로봇학·강화학습 환경을 구성하는 framework다. 현재 프로젝트에서는 다음을 담당한다.

- G1 scene 생성
- 병렬 환경 생성
- observation manager
- action manager
- reward manager
- termination manager
- event/randomization manager
- RSL-RL PPO 연결
- 영상과 policy export

즉, Isaac Sim이 물리 엔진이라면 Isaac Lab은 그 물리 엔진을 RL 문제로 조직하는 층이다.

---

## 22. 세 가지 재생 모드를 구분해야 한다

### 22.1 Kinematic replay

Kinematic replay는 archive의 root와 joint state를 매 frame articulation에 직접 쓴다. 개념적으로 다음과 같다.

\[
q(t)\leftarrow q_{ref}(t),\qquad
p_{root}(t)\leftarrow p_{ref}(t)
\]

이때 robot이 실제 torque로 균형을 만든 것이 아니다. 용도는 다음과 같다.

- 관절 순서 확인
- quaternion 규약 확인
- URDF body 이름 확인
- archive FK와 Isaac articulation FK 일치 확인
- 화면상 동작 형태 확인

B3 walk v27 archive 381 frame을 직접 적용했을 때 다음 결과를 얻었다.

- body position RMSE: \(3.56\times10^{-7}\) m
- body orientation RMSE: \(7.09\times10^{-7}\) rad
- geometric contact F1: 0.93473

이는 ingestion이 정확하다는 강한 증거지만 동역학 성공 증거는 아니다.

### 22.2 Open-loop PD tracking

Open-loop PD는 root를 매 frame teleport하지 않고 reference joint target만 actuator에 전달한다. Feedback policy 없이 다음을 수행하는 baseline이다.

\[
q_{target}(t)=q_{ref}(t)
\]

동일 B3 계열 reference에서 첫 fall은 0.705초에 발생했다. 이 결과가 보여 주는 것은 다음이다.

- archive를 읽는 문제는 해결되었다.
- reference joint angle을 그대로 주는 것만으로는 균형이 유지되지 않는다.
- feedback correction 또는 더 물리적으로 feasible한 reference가 필요하다.

### 22.3 Closed-loop PPO tracking

현재 최종 영상에서 사용하는 방식이다. Policy는 매 control step마다 다음을 반복한다.

1. 현재 실제 robot state를 읽는다.
2. 현재·미래 reference와의 오차를 계산한다.
3. 29차원 residual action을 출력한다.
4. reference에 residual을 더해 joint target을 만든다.
5. PD actuator와 PhysX가 다음 state를 만든다.
6. 새 state를 다시 관측한다.

이것이 closed loop인 이유는 action이 미리 고정된 sequence가 아니라 매 순간 실제 오차에 따라 달라지기 때문이다.

---

## 23. 시간 계약

| 항목 | 값 |
|---|---:|
| Reference FPS | 50 Hz |
| Policy/control FPS | 50 Hz |
| Control dt | 0.02 s |
| Physics FPS | 200 Hz |
| Physics dt | 0.005 s |
| Decimation | 4 |

Policy action 하나는 네 번의 physics substep 동안 적용된다.

```text
Policy step 1회: 20 ms
  ├─ PhysX step 1: 5 ms
  ├─ PhysX step 2: 5 ms
  ├─ PhysX step 3: 5 ms
  └─ PhysX step 4: 5 ms
```

Reference FPS와 control FPS가 모두 50 Hz이므로 기본적으로 reference frame 하나가 control transition 하나에 대응한다. Motion library는 FPS와 환경 step dt가 맞지 않으면 실행을 중지한다.

---

## 24. Reference가 simulation command가 되는 과정

G1 archive를 motion library가 읽어 다음 시간의 reference를 제공한다.

\[
x_{ref}(t)=
\{q,\dot q,p_{root},R_{root},v_{root},\omega_{root},p_{body},R_{body},c_{foot}\}
\]

Policy는 파일 이름의 `walk`, `punch`, `stance` 문자열을 읽지 않는다. Policy가 보는 것은 숫자 reference다. 예를 들어 걷기와 격투 스탠스의 차이는 다음 숫자에서 나타난다.

- 미래 joint angle
- root 이동 방향과 속도
- torso orientation
- 손과 팔의 reference
- 왼발·오른발 contact timing
- motion phase

따라서 동작 선택은 다음 구조다.

```text
사용자가 motion ID 또는 reference 파일 선택
        ↓
Motion library가 해당 reference trajectory 제공
        ↓
같은 PPO policy가 reference-conditioned residual 출력
        ↓
선택한 reference에 대응하는 동작이 나타남
```

즉, 현재 구조는 동작마다 별도 policy를 하나씩 쓰는 구조가 아니다.

103개 reference를 조건으로 받는 하나의 공통 policy다.

---

## 25. Actor observation 293차원

Actor는 배포 시 실제 action을 결정하는 network다. 현재 observation은 총 293차원이다.

| 항목 | 차원 | 의미 |
|---|---:|---|
| Future joint-position error | 116 | 4 horizon × 29 joint |
| Future root-position error | 9 | 양의 미래 horizon 3개 × xyz |
| Future root-orientation error | 18 | 3 horizon × rotation 6D |
| Current joint-velocity error | 29 | reference와 실제 속도 차이 |
| Current root-orientation error | 6 | rotation 6D |
| Current root-position error | 3 | pelvis local frame |
| Root linear-velocity error | 3 | pelvis local frame |
| Target root linear velocity | 3 | pelvis local frame |
| Motion phase | 2 | sine/cosine |
| Root angular-velocity error | 3 | pelvis local frame |
| Target root angular velocity | 3 | pelvis local frame |
| Projected gravity | 3 | body orientation/tilt 정보 |
| Measured joint position | 29 | default pose 기준 |
| Measured joint velocity | 29 | proprioception |
| Predicted contacts | 8 | 4 horizon × 양발 |
| Last residual action | 29 | 이전 제어 출력 |
| 합계 | 293 |  |

### 25.1 Future preview

Future horizon은 다음과 같다.

\[
h\in\{0.00,0.04,0.08,0.16\}\text{ s}
\]

Joint error는 네 horizon을 모두 사용한다. Root future preview는 현재 root error와 중복을 피하기 위해 0.04, 0.08, 0.16초만 사용한다. Preview가 필요한 이유는 접촉 전환에 미리 대응하기 위해서다. 예를 들어 0.08초 뒤 왼발이 swing으로 바뀐다는 것을 알면, policy는 현재부터 weight shift를 준비할 수 있다.

### 25.2 Pelvis-local representation

Root 위치·속도 오차는 world 좌표 그대로 넣기보다 현재 측정 pelvis frame으로 변환한다.

\[
e_p^{local}
=
R_{actual}^{T}
(p_{ref}-p_{actual})
\]

이렇게 하면 같은 동작을 world의 다른 위치나 yaw에서 시작해도 observation 의미가 비슷해진다.

### 25.3 Rotation 6D

Orientation error는 quaternion 네 성분을 그대로 쓰지 않고 rotation matrix의 첫 두 column으로 만든 6D representation을 사용한다. Quaternion은 \(q\)와 \(-q\)가 같은 rotation을 나타내므로 표현에 sign discontinuity가 있다. Rotation 6D는 neural network가 더 연속적인 orientation 입력을 받도록 돕는다.

---

## 26. Privileged critic observation 505차원

Critic은 state value \(V(s)\)를 추정한다. Critic은 Actor의 293차원에 212차원을 추가로 받는다.

| 추가 항목 | 차원 |
|---|---:|
| 14 body position error | 42 |
| 14 body orientation error | 84 |
| 14 body linear-velocity error | 42 |
| 14 body angular-velocity error | 42 |
| 실제 양발 contact | 2 |
| 추가 합계 | 212 |

\[
293+212=505
\]

이 구조를 asymmetric actor-critic이라고 부른다.

- Actor: 실행 시 사용할 수 있는 293차원
- Critic: 학습 안정화를 위한 privileged 505차원
- 실제 inference/export: Actor만 사용

Critic에 더 많은 정보를 주더라도 Actor가 그 privileged 값을 inference 때 요구하지 않는다.

---

## 27. Policy action과 G1 actuator

Policy 출력은 29차원 normalized residual이다.

\[
a_t\in\mathbb R^{29}
\]

### 27.1 Raw action clipping

먼저 action을 안전 범위로 제한한다.

\[
a_t^{clip}=\operatorname{clip}(a_t,-1,1)
\]

Wrapper가 raw action을 미리 숨기지 않기 때문에 evaluator는 policy가 얼마나 자주 ±1을 요청했는지 측정할 수 있다.

### 27.2 Residual rate limit

직전 normalized residual과의 차이를 control step당 ±0.15로 제한한다.

\[
r_t
=r_{t-1}
+\operatorname{clip}
(a_t^{clip}-r_{t-1},-0.15,0.15)
\]

Residual scale이 0.25 rad이므로 residual 부분의 한 step 최대 변화는 다음과 같다.

\[
0.25\times0.15=0.0375\text{ rad}
\]

이는 residual contribution의 변화 제한이다. 시간에 따라 움직이는 \(q_{ref}\) 자체까지 포함한 전체 target 변화가 0.0375 rad로 제한된다는 뜻은 아니다.

### 27.3 Reference-residual position target

실제 position target은 다음과 같다.

\[
q_t^{target}
=
\operatorname{clamp}_{safety}
\left(q_t^{ref}+0.25r_t\right)
\]

Policy action이 0이면 다음이 된다.

\[
q_t^{target}=q_t^{ref}
\]

따라서 reference가 기본 안무를 제공하고 policy가 그 주변의 correction을 학습한다.

### 27.4 Velocity target

현재 v7은 다음을 사용한다.

\[
\dot q_t^{target}=0
\]

v6에서 full reference velocity feed-forward를 넣었지만 train 성능이 악화되어 v7에서 0으로 되돌렸다. 이는 “모든 로봇에서 velocity feed-forward가 나쁘다”는 일반 결론이 아니다. 현재 실험 조건에서 선택된 결과다.

### 27.5 PD actuator

Action term은 torque를 직접 출력하지 않는다. Position·velocity target을 Isaac의 implicit actuator에 전달한다. 개념적인 torque는 다음과 같다.

\[
\tau_t
\approx
K_p(q_t^{target}-q_t)
+K_d(\dot q_t^{target}-\dot q_t)
\]

현재 \(\dot q_t^{target}=0\)이므로 damping은 실제 joint velocity를 줄이는 방향으로 작용한다. 최종 torque는 actuator effort limit와 PhysX dynamics의 영향을 받는다.

---

## 28. Reward 설계

PPO가 최대화하는 것은 “사람처럼 보이는가”라는 추상 문장이 아니라 수치 reward의 기대 누적합이다.

\[
J(\theta)
=
\mathbb E_{\pi_\theta}
\left[
\sum_t\gamma^tr_t
\right]
\]

### 28.1 Tracking kernel

대부분의 tracking reward는 다음 형태다.

\[
r_{track}
=
\exp
\left(
-\frac{\operatorname{MSE}(e)}{\sigma^2}
\right)
\]

오차 RMSE가 \(\sigma\)와 같으면 reward term은 약 \(e^{-1}=0.368\)이다.

### 28.2 양의 reward

| 항목 | Weight | \(\sigma\) |
|---|---:|---:|
| Joint position tracking | 2.0 | 0.30 rad |
| Joint velocity tracking | 0.50 | 2.0 rad/s |
| Root position tracking | 1.0 | 0.25 m |
| Root orientation tracking | 0.50 | 0.35 rad |
| Root linear velocity tracking | 0.75 | 1.0 m/s |
| Root angular velocity tracking | 0.25 | 2.0 rad/s |
| 14-body position tracking | 2.0 | 0.20 m |
| 14-body orientation tracking | 0.75 | 0.50 rad |
| Foot-contact timing match | 0.30 | binary match |
| Alive | 0.25 | 실제 v7 override |

Joint angle만 맞추는 것이 아니라 root·body·contact를 함께 보상하는 이유는 다음과 같다.

- Joint angle은 비슷하지만 base가 밀릴 수 있다.
- Root가 비슷하지만 팔·발 endpoint가 다를 수 있다.
- Pose가 비슷하지만 잘못된 발이 ground에 닿을 수 있다.

### 28.3 음의 reward와 regularization

| 항목 | Weight | 목적 |
|---|---:|---|
| Feet slide | -0.20 | stance 중 발 미끄러짐 억제 |
| Raw action excess | -0.25 | ±1 밖의 policy 요청 억제 |
| Action boundary margin | -2.0 | |action| 0.9부터 경계 접근 억제 |
| Limiter request gap | -1.0 | rate limiter가 거부하는 요청 억제 |
| Residual magnitude | -0.02 | reference에서 과도한 보정 억제 |
| Residual rate | -0.05 | 급격한 correction 변화 억제 |
| Divergence risk | -20.0 | hard failure 전 조기 경고 |
| Target clamp | -0.10 | joint safety clamp 의존 억제 |
| Joint torque | -2e-5 | 에너지·과도한 torque 억제 |
| Joint acceleration | -2.5e-7 | 거친 motion 억제 |
| Joint position limit | -2.0 | soft limit 밖 상태 억제 |
| Joint velocity limit | -0.10 | soft velocity limit 90% 초과 억제 |
| Terminal failure | -10.0 | non-timeout failure one-shot |

### 28.4 Divergence-risk barrier

Hard failure threshold에 도달한 뒤에만 벌점을 주면 policy가 위험 상태를 미리 피하기 어렵다. 그래서 threshold의 60%부터 제곱형 위험 벌점을 증가시킨다. Hard threshold는 다음과 같다.

- root distance: 0.80 m
- 평균 29-joint absolute error: 1.0 rad
- 평균 14-body distance: 0.50 m

세 위험을 합하지 않고 가장 큰 normalized risk를 사용한다.

### 28.5 Terminal penalty의 dt 문제

Isaac RewardManager는 dense reward에 control dt 0.02를 곱한다. 초기 구현에서는 설정한 `-10`이 다시 dt를 받아 실효 약 `-0.2`가 되는 문제가 있었다. 현재 구현은 termination indicator를 step dt로 나누어 integration 뒤 정확히 one-shot `-10`이 되도록 수정했다. 이것은 reward 숫자를 적어 놓는 것과 실제 simulator에서 적용되는 값이 다를 수 있음을 보여 주는 중요한 사례다.

---

## 29. Episode termination과 성공 정의

### 29.1 정상 종료

- Motion end
- Global time-out

Motion end는 reference 마지막에 도달했다는 뜻이다.

### 29.2 실패 종료

#### Fall

다음 조건 중 하나가 발생하면 fall이다.

- 실제 pelvis가 reference pelvis보다 0.25 m 넘게 낮음
- root orientation error가 1.0 rad 초과
- 서 있는 reference에서 pelvis contact force가 50 N 초과

Pelvis contact는 모든 motion에 무조건 fall로 적용하지 않는다. Reference pelvis 높이가 0.50 m 이상이고 tilt가 0.80 rad 이하인 경우에만 “서 있는 목표”로 본다. 그래서 눕기·구르기 reference의 의도된 pelvis contact를 무조건 낙상으로 오판하지 않는다.

#### Tracking divergence

- root distance > 0.80 m
- 평균 joint error > 1.0 rad
- 평균 tracking-body distance > 0.50 m

#### Joint limit

- hard joint position limit에서 tolerance 0.01 rad 초과
- joint speed가 soft velocity limit의 1.10배 초과

#### Non-finite

- root state 또는 joint position/velocity에 NaN·Inf 발생

### 29.3 Clean success

정확한 성공은 다음과 같다.

> Motion end가 발생했고 같은 transition에서 어떤 non-timeout failure도 발생하지 않았다.

Motion end와 fall이 같은 step에 동시에 발생하면 성공으로 세지 않는다.

---

## 30. PPO가 실제로 학습한 것

Policy는 다음 조건부 함수다.

\[
a_t=\pi_\theta(o_t,x_{ref,t:t+H})
\]

학습 대상은 neural-network parameter \(\theta\)다. Policy가 학습한 것은 다음 관계다.

> 현재 G1이 reference에서 이 정도 벗어나 있고 곧 이런 자세와 접촉이 올 때, reference에 어떤 관절 residual을 더하면 미래 누적 reward가 커지는가?

Policy가 직접 학습하지 않은 것은 다음과 같다.

- ACCAD 파일 parser
- SMPL-X FK
- Human-to-G1 correspondence
- Retarget optimizer
- 동작 이름의 언어적 의미
- 새로운 reference 자체를 생성하는 generative model

즉, 이것은 motion generation policy가 아니라 motion tracking policy다.

---

## 31. PPO update 원리

학습 중 Actor는 대각 Gaussian distribution에서 action을 sampling한다.

\[
a_t\sim\pi_\theta(a_t|o_t)
\]

Critic은 value를 추정한다.

\[
V_\phi(s_t)\approx
\mathbb E\left[\sum_{k=0}^{\infty}\gamma^kr_{t+k}\right]
\]

### 31.1 GAE advantage

TD residual은 다음과 같다.

\[
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
\]

GAE는 다음처럼 여러 step의 residual을 합친다.

\[
\hat A_t
=
\sum_{l=0}^{T-t-1}
(\gamma\lambda)^l\delta_{t+l}
\]

현재 값은 다음과 같다.

- \(\gamma=0.99\)
- \(\lambda=0.95\)

### 31.2 PPO clipped objective

Old policy와 new policy의 action probability ratio는 다음과 같다.

\[
\rho_t(\theta)
=
\frac{\pi_\theta(a_t|o_t)}
{\pi_{\theta_{old}}(a_t|o_t)}
\]

PPO는 다음 clipped objective를 사용한다.

\[
L^{clip}
=
\mathbb E_t
\left[
\min
\left(
\rho_t\hat A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t
\right)
\right]
\]

현재 \(\epsilon=0.1\)이다. Policy update가 한 번에 너무 크게 변하는 것을 제한하는 장치다.

### 31.3 Network와 optimizer

| 구성 | 실제 설정 |
|---|---|
| Actor | 293 → 512 → 512 → 256 → 29 |
| Critic | 505 → 512 → 512 → 256 → 1 |
| Activation | ELU |
| Optimizer | Adam |
| Learning rate | 1e-5 fixed |
| PPO epoch | 3 |
| Minibatch | 4 |
| Value loss coefficient | 1.0 |
| Clipped value loss | 사용 |
| Entropy coefficient | 0.0 |
| Maximum gradient norm | 1.0 |
| Actor/Critic observation normalization | 사용 |

Trainable parameter 수는 약 1,205,307개다.

---

## 32. 103개 motion을 학습에 공급한 방법

현재 production training은 `balanced-fixed` motion assignment를 사용했다. Environment \(i\)가 담당하는 motion은 다음과 같다.

\[
m_i=i\bmod103
\]

Continuation에서 environment 수가 1,024개이므로 각 motion은 9개 또는 10개 environment를 받는다. 장점은 다음과 같다.

- 103개 중 학습에서 빠지는 motion이 없다.
- 짧은 clip만 과도하게 선택되는 uniform-episode sampling 문제를 줄인다.
- rollout transition exposure를 정확히 기록할 수 있다.

각 environment는 자신에게 배정된 motion을 episode가 끝날 때마다 다시 시작한다. 최종 v7에서는 frame-zero start probability가 1.0이다. 즉, training episode는 motion 중간이 아니라 시작 frame에서 출발한다. 이 선택은 full-clip 시작 성공을 직접 학습하는 대신, 임의 phase에서의 recovery 다양성은 줄인다.

---

## 33. Randomness와 robustness 설정

현재 run에서 모든 randomness가 꺼진 것은 아니다. 정확한 상태는 다음과 같다.

| 항목 | Training v7 |
|---|---|
| Gaussian policy action sampling | 사용 |
| Actor observation corruption/noise | 사용 |
| Critic corruption | 사용하지 않음 |
| Reset joint/root state noise | 0 |
| Random start phase | 사용하지 않음, frame 0 |
| Global XY alignment | 랜덤 범위 사용 |
| Global yaw alignment | 랜덤 범위 사용 |
| Friction randomization | 사용하지 않음 |
| Mass randomization | 사용하지 않음 |
| Actuator gain randomization | 사용하지 않음 |
| External push event | 사용하지 않음 |
| Random initial episode length | 사용하지 않음 |

따라서 다음 표현은 부정확하다.

> “Domain randomization까지 적용해 sim-to-real robustness를 확보했다.”

현재는 observation noise는 있으나 물리 parameter domain randomization은 꺼져 있다.

---

## 34. 최종 v7 학습 계보와 학습량

최종 정책은 완전한 random initialization에서 v7을 시작하지 않았다. 학습 계보는 다음과 같다.

```text
기존 v5 tracking checkpoint
  ActorCritic + normalizer + learned log_std 이전
  optimizer와 iteration counter는 새로 시작
        ↓
v7 초기 적응
  512 env × 32 step × 300 update
        ↓ model_299
v7 continuation
  optimizer와 counter까지 full resume
  1024 env × 32 step × 700 update
        ↓ model_998
```

v7 objective에서 추가한 transition 수는 다음과 같다.

\[
512\times32\times300
=4,915,200
\]

\[
1024\times32\times700
=22,937,600
\]

\[
\text{v7 합계}
=27,852,800
\]

따라서 정확한 표현은 다음과 같다.

> 기존 tracking 정책을 초기값으로 사용하여 v7 제어·reward 계약에서 1,000 update, 약 2,785만 transition을 추가 학습했다.

다음 표현은 정확하지 않다.

> 2,785만 transition만으로 모든 것을 처음부터 학습했다.

---

## 35. 걷기·펀치·격투 자세가 나타나는 원리

### 35.1 걷기

걷기 reference에는 시간에 따라 다음 패턴이 들어 있다.

- root 전진
- 좌우 다리 swing/stance 교대
- knee와 ankle 궤적
- 양발 contact timing
- torso와 팔의 동반 운동

Policy는 이 숫자 패턴을 condition으로 받아 균형 보정 residual을 출력한다. 그래서 `B3_-_walk1` reference를 선택하면 걷기 형태가 나온다.

### 35.2 펀치

펀치 reference에는 다음이 들어 있다.

- shoulder·elbow·wrist reference 변화
- torso 회전
- root 또는 stance 변화
- 양발 지지 timing

Policy가 “body hook”이라는 단어를 이해하는 것은 아니다. 팔과 torso의 미래 reference를 보고 해당 궤적을 물리적으로 따라가는 것이다.

### 35.3 격투 스탠스

Stance-switch reference는 다음을 지정한다.

- 어느 발이 앞에 오는가
- 좌우 foot contact가 언제 전환되는가
- hip·knee·ankle의 자세
- torso와 팔의 guard 형태
- root 이동과 회전

Policy는 같은 network를 사용하지만 reference가 다르므로 다른 행동이 나타난다.

### 35.4 이것이 가능한 범위

현재 시스템은 다음을 할 수 있다.

- manifest 안의 reference를 motion ID 또는 path로 선택
- 선택한 reference를 같은 PPO policy에 공급
- 성공 가능한 train clip에서 걷기·펀치·스탠스 등을 closed-loop physics로 재생

현재 시스템이 아직 제공하지 않는 것은 다음과 같다.

- “왼쪽으로 걸어” 같은 언어 명령 이해
- 임의 목표점으로 가는 high-level planner
- reference 없이 새로운 격투 조합 생성
- 보지 못한 모든 motion에 대한 일반화 보장
- 상대 로봇의 공격에 반응하는 interactive fighting policy

현재의 격투 동작은 opponent-aware combat가 아니라 recorded motion reference tracking이다.

---

## 36. 실제로 확인된 세 가지 예

선택 checkpoint는 `model_998.pt`다. Deterministic train suite에서 다음 세 clip은 clean motion-end를 기록했다.

### 36.1 Motion ID 34: 걷기

- Reference: `Female1Walking_c3d/B3_-_walk1`
- Frame: 609
- Clean motion-end: 성공
- Joint MAE: 약 0.0974 rad
- Root position error: 약 0.1095 m
- Tracking-body position error: 약 0.1163 m
- Foot-contact match: 약 0.9072

이 clip은 별도 B3 physics replay에서도 609/609 step을 완주했다.

### 36.2 Motion ID 58: body-hook 동작

- Reference: `Male2MartialArtsPunches_c3d/E11_-_body_hook_right_t2`
- Reference frame: 197
- Clean motion-end: 성공
- Joint MAE: 약 0.1198 rad
- Root position error: 약 0.0867 m
- Tracking-body position error: 약 0.1043 m
- Foot-contact match: 약 0.8611

이 결과는 해당 train reference를 끝까지 물리 추종했다는 뜻이다. 실제 상대에게 유효한 punch force를 냈다는 검증은 아니다.

### 36.3 Motion ID 70: stance switch

- Reference: `Male2MartialArtsStances_c3d/D15_-_switch_stance`
- Reference frame: 237
- Clean motion-end: 성공
- Joint MAE: 약 0.0755 rad
- Root position error: 약 0.0530 m
- Tracking-body position error: 약 0.0759 m
- Foot-contact match: 약 0.9622

이 예가 “같은 policy가 다른 reference에 조건부로 반응한다”는 것을 보여 준다.

---

## 37. 전체 정책 평가

평가는 다음 조건을 고정했다.

- train manifest 103개 checksum 확인
- motion당 environment 하나
- frame 0 시작
- first episode만 집계
- deterministic policy mean action
- random start 비활성
- observation/state noise 비활성
- domain randomization 비활성
- physics warm-up 100 step 후 authoritative frame-0 reset
- seed 42

### 37.1 사전 acceptance 기준과 결과

| 기준 | 요구 | model_998 | 판정 |
|---|---:|---:|---|
| Clean completion | 103/103 | 48/103 | FAIL |
| Mean completion | ≥ 0.90 | 0.68966 | FAIL |
| Joint MAE | ≤ 0.20 rad | 0.08792 rad | PASS |
| Root position error | ≤ 0.20 m | 0.10545 m | PASS |
| Body position error | ≤ 0.15 m | 0.11145 m | PASS |
| Contact F1 | ≥ 0.80 | 0.96154 | PASS |
| Weighted raw-action clip | ≤ 0.05 | 0.08670 | FAIL |
| Joint-limit termination | 0 | 0 | PASS |
| Non-finite termination | 0 | 0 | PASS |

Termination count는 다음과 같다.

- Clean motion end: 48
- Fall signal: 22
- Tracking divergence signal: 35

두 episode에서는 fall과 divergence가 같은 step에 동시에 발생했기 때문에 단순 합이 103보다 클 수 있다.

### 37.2 낮은 MAE와 낮은 성공률이 동시에 나오는 이유

Error metric은 episode가 종료되기 전까지의 sample로 계산된다. 실패 motion은 큰 오차가 계속 커지기 전에 termination된다. 따라서 낮은 MAE만 보면 다음 편향이 생긴다.

> 끝까지 추종하지 못한 어려운 후반 구간이 metric에 포함되지 않는다.

이를 truncation 또는 survivorship bias로 해석할 수 있다. 그래서 다음을 함께 봐야 한다.

- clean success
- completion ratio
- tracking error
- termination cause
- action saturation

### 37.3 Contact F1의 한계

Contact F1 0.9615는 양발의 binary contact timing이 많이 일치했다는 뜻이다. 다음이 맞다는 뜻은 아니다.

- contact force가 적절함
- center of pressure가 support polygon 안에 있음
- friction cone을 만족함
- 발이 미끄러지지 않음
- actuator torque가 충분함

따라서 높은 contact F1과 동역학 실패는 동시에 가능하다.

---

## 38. 무엇이 완료되었고 무엇이 남았는가

### 38.1 완료된 것

- ACCAD Stage-II 252개 전수 accounting
- checksum alias 제거 후 249개 고유 motion 확정
- subject-disjoint split과 manifest
- SMPL-X neutral model prerequisite와 checksum 검증
- Stage-II strict numeric parser
- 원본 FPS human FK와 contact 추정
- 50 Hz canonical human archive
- morphology/contact-aware G1 sequence retarget
- G1 29-DoF joint/URDF/coordinate contract
- 249/249 independent kinematic validation
- Isaac kinematic ingestion 검증
- open-loop PD baseline
- v28 dynamic prefilter
- 103-motion PPO training environment
- 29-D reference-residual action
- 293-D Actor / 505-D privileged Critic
- reward·termination·deterministic evaluation
- v7 1,000 update와 checkpoint 선택
- 성공 train clip의 closed-loop physics replay
- TorchScript·ONNX export 경로
- TensorBoard·report·checksum evidence

### 38.2 완료되지 않은 것

- Train 103/103 안정 추종
- v7 validation suite 통과
- Male1 held-out TEST policy 평가
- multi-seed 통계
- 물리 domain randomization 기반 robustness
- real G1 배포 검증
- opponent-aware fighting
- arbitrary new-motion generalization
- contact-feasible retarget의 완성된 production solver

현재 상태를 가장 정확하게 표현하면 다음과 같다.

> 데이터 처리, retarget, Isaac ingestion, PPO 학습과 평가의 전체 소프트웨어 경로는 구현·실행되었다. 일부 걷기·펀치·스탠스 reference는 실제 physics에서 완주하지만, 전체 train acceptance와 일반화는 아직 실패 상태다.

---

# Part IV. 구현 위치를 이해하기 위한 지도

## 39. 파일별 책임

| 파일 | 공부할 질문 |
|---|---|
| `_g1_pipeline/config.yaml` | 데이터 경로, split, 품질 기준, 50 Hz 계약은 무엇인가 |
| `_g1_pipeline/correspondence.yaml` | 사람의 어떤 부위를 G1 어디에 맞추는가 |
| `accad_g1/data.py` | 원본을 어떤 기준으로 감사하고 split하는가 |
| `accad_g1/human.py` | SMPL-X parameters를 3D human motion으로 어떻게 복원하는가 |
| `accad_g1/g1_kinematics.py` | G1 URDF FK와 support geometry를 어떻게 계산하는가 |
| `accad_g1/retarget.py` | 어떤 변수와 loss로 G1 reference를 찾는가 |
| `accad_g1/g1_validation.py` | 생성 결과를 독립적으로 어떻게 검증하는가 |
| `accad_g1/dynamic_retime.py` | PPO 전에 어떤 motion을 늦추거나 격리하는가 |
| `accad_g1/motion_library.py` | 여러 reference를 runtime tensor로 어떻게 제공하는가 |
| `accad_g1/tracking_task.py` | observation, action, reward, termination이 무엇인가 |
| `train_tracking.py` | PPO를 어떻게 시작·초기화·resume하는가 |
| `evaluate_tracking_suite.py` | 103개 정책 결과를 어떤 계약으로 판정하는가 |
| `play_tracking.py` | checkpoint physics replay와 영상/export는 어떻게 하는가 |

### 39.1 데이터 산출물 위치

| 위치 | 내용 |
|---|---|
| `ACCAD/*_c3d`, `ACCAD/s*` | 원본 Stage-II motion |
| `ACCAD/smplx_lockedhead_20230207` | licensed SMPL-X model distribution |
| `_g1_pipeline/work/manifests` | 원본 감사와 split manifest |
| `_g1_pipeline/work/human` | canonical 50 Hz human archive |
| `_g1_pipeline/work/g1` | v27 G1 retarget archive |
| `_g1_pipeline/work/dynamic/g1` | v28 PPO 입력 reference |
| `_g1_pipeline/work/training/runs` | PPO run, checkpoint, TensorBoard, runtime contract |
| `_g1_pipeline/work/training/evaluations` | suite 결과, 영상, export |
| `_g1_pipeline/work/reports` | gate 결과와 machine-readable outcome |

---

## 40. 전체 데이터 lineage

한 motion이 최종 policy 입력이 되기까지 다음 provenance chain을 가진다.

```text
Stage-II NPZ
  source path + SHA-256
        ↓
Canonical human NPZ
  Stage-II checksum
  SMPL-X model checksum
  50 Hz human FK/contact
        ↓
G1 v27 NPZ
  human checksum
  G1 contract checksum
  URDF checksum
  correspondence checksum
        ↓
Independent Gate-3 manifest
  FK/limit/contact 검증 결과
        ↓
G1 v28 NPZ
  dynamic policy checksum
  retime scale와 audit 결과
        ↓
Training runtime contract
  manifest checksum
  observation/action/reward 계약
        ↓
Checkpoint
  SHA-256와 evaluation 결과
```

이 chain의 목적은 “파일이 있다”를 넘어 다음을 답하기 위한 것이다.

- 어느 원본에서 왔는가
- 어느 body model을 썼는가
- 어느 URDF와 joint order를 썼는가
- 어떤 retarget 설정을 썼는가
- 어떤 학습 환경과 checkpoint로 평가했는가

---

# Part V. 공부할 때 자주 혼동하는 용어

## 41. 핵심 용어 사전

### Motion capture

사람이 실제로 움직일 때 marker·camera 등으로 측정한 운동 정보다.

### Parametric body model

적은 수의 shape·pose parameter로 사람의 mesh와 skeleton을 생성하는 model이다.

### SMPL-X

몸, 손, 얼굴을 포함하는 parametric human body model이다.

### Forward kinematics

관절 회전과 skeleton parent 관계로 body/joint의 전역 pose를 계산하는 과정이다.

### Inverse kinematics

원하는 손·발·body 위치를 만족하는 관절각을 찾는 문제다. 현재 retargeter는 단일 frame IK보다 넓은 sequence optimization이다.

### Retargeting

한 embodiment의 동작 의미를 다른 embodiment의 유효한 관절 궤적으로 변환하는 과정이다.

### Embodiment gap

사람과 로봇처럼 몸의 구조·크기·자유도·물리가 다른 데서 생기는 차이다.

### Reference trajectory

시간에 따라 robot이 따라야 할 목표 관절·root·body·contact 궤적이다.

### Kinematic validity

관절 범위와 FK, 좌표계 관점에서 궤적이 일관적이라는 뜻이다.

### Dynamic feasibility

중력·관성·접촉·마찰·torque 제약 아래 실제로 수행 가능하다는 뜻이다. Kinematic validity가 dynamic feasibility를 보장하지 않는다.

### Open loop

실제 오차를 action 결정에 반영하지 않고 미리 정한 command를 적용하는 방식이다.

### Closed loop

현재 실제 상태와 목표 오차를 측정하여 action을 계속 수정하는 방식이다.

### Residual policy

전체 command가 아니라 기존 reference나 controller에 더할 correction을 출력하는 policy다.

### Proprioception

Joint position·velocity처럼 robot 자신의 내부 운동 상태를 나타내는 감각이다.

### Privileged information

학습 중 Critic에는 줄 수 있지만 실제 Actor 배포 입력에는 넣지 않는 추가 정보다.

### Contact

Body가 ground와 닿아 있거나 닿아야 한다는 상태다.

### Stance

Contact 중에서도 load-bearing하고 상대적으로 고정된 support로 취급할 수 있는 상태다.

### Foot skating

발이 ground에 닿아 있어야 하는데 수평 방향으로 미끄러지는 현상이다.

### PPO

Old policy와 new policy의 확률비를 clip하여 급격한 policy update를 제한하는 on-policy actor-critic 알고리즘이다.

### Transition

한 환경의 한 step에서 얻은 \((s_t,a_t,r_t,s_{t+1})\) sample이다.

### Checkpoint

Actor, Critic, normalization, action distribution parameter, optimizer state 등을 저장한 학습 시점이다.

### Completion ratio

Reference 전체 길이 중 episode가 진행한 비율이다.

### Clean motion-end

Reference 마지막까지 도달하면서 동일 step의 물리 실패가 없었던 성공이다.

### Generalization

학습에 직접 사용하지 않은 subject·motion·조건에서도 policy가 작동하는 성질이다. 현재 v7 결과로는 아직 주장할 수 없다.

---

## 42. 최종 핵심 요약

> Retargeting은 동작의 G1 목표 궤적을 만들고, PPO는 그 목표를 중력과 접촉이 있는 환경에서 따라가기 위한 feedback residual을 학습한다.

현재 전체 데이터·retarget·Isaac ingestion·PPO 경로는 구현되었고 걷기, body-hook, stance-switch를 포함한 일부 train motion은 clean completion을 기록했다. 그러나 전체 결과는 48/103이며 v7 validation과 held-out TEST를 통과한 상태가 아니다.
