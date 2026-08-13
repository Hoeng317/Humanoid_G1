# ACCAD → Unitree G1 Motion Pre-training Pipeline

> 프로젝트: /home/hoeng/IsaacLab/humanoid_G1  
> 로봇: Unitree G1 g1_29dof_rev_1_0, body 29-DoF  
> 시뮬레이터/학습: Isaac Lab + RSL-RL PPO  
> 기준일: 2026-08-08  
> 상태: 검토 반영 설계서. 아직 없는 script와 task는 [미구현]으로 표시한다.

## 0. 목표와 산출물

ACCAD stageii.npz의 SMPL-X 인간 모션을 G1이 물리 환경에서 안정적으로 추적할 수 있는 controller로 변환한다.
~~~text
1. ACCAD 선별·검증 | ↓ | 2. SMPL-X Human Reconstruction | ↓ | 3. Human → G1 Kinematic Retargeting | ↓ | 4. Isaac Physics Motion Tracking | ↓ | 5. Sim2Sim/Real Gap 축소 + Downstream Transfer
~~~
첫 번째 산출물은 reference-conditioned G1 motion tracker다.
~~~text
ACCAD | → Human kinematics | → G1 reference library | → Reference-conditioned G1 Motion Tracker
~~~
Reference 없이 자연스러운 동작을 자율 생성하는 motion prior는 별도 단계다.
~~~text
G1 Motion Tracker | → latent skill / teacher-student / AMP / VAE | → Reusable G1 Motion Prior
~~~
따라서 tracker와 prior를 같은 산출물로 부르지 않는다.

## 0.1 현재 상태

원본 데이터:
~~~text
/home/hoeng/IsaacLab/humanoid_G1/ACCAD | stageii NPZ : 252개 | stagei NPZ  : 20개 | source FPS  : 120Hz | scan error  : 0개
~~~
현재 완료:
- 전체 NPZ scan과 key/shape 검사
- inventory report: workspace/artifacts/reports/accad_inventory.json
- 첫 clip: ACCAD/Female1Walking_c3d/B3_-_walk1_stageii.npz
현재 미준비 또는 미구현:
- licensed SMPL-X neutral body model
- smplx Python dependency
- Human FK reconstruction
- G1 retargeter
- G1 motion-tracking task
- Motion reference deployment runtime
- Motion observation/export/C++ contract

## 0.2 핵심 원칙

1. ACCAD는 읽기 전용 원본으로 유지한다.
2. 파생 데이터는 data/motions 아래에 저장한다.
3. 보고서는 workspace/artifacts/reports에 저장한다.
4. pose_body를 G1 joint angle로 직접 복사하지 않는다.
5. Human → G1은 task-space, morphology, contact-aware optimization으로 처리한다.
6. G1 canonical joint order는 g1_29dof.yaml의 policy_index다.
7. Human quality/contact는 원본 120Hz에서 계산한다.
8. G1 reference와 control은 현재 제어 계약인 50Hz로 만든다.
9. Kinematic replay와 action reachability 통과 전 PPO를 시작하지 않는다.
10. 학습과 Python/C++ deployment의 action/observation 의미를 일치시킨다.
11. Nominal tracking 성공 후 domain randomization을 점진적으로 추가한다.

## 0.3 공통 데이터 계약

~~~text
coordinate system : right-handed, Z-up | position unit     : meter | angle unit        : radian | quaternion order  : wxyz | human source FPS  : 120Hz | G1 reference FPS  : 50Hz | G1 control FPS    : 50Hz | G1 joint order    : policy_index
~~~
모든 파생 NPZ에 다음 metadata를 넣는다.
~~~text
schema_version | coordinate_system | quaternion_order | position_unit | angle_unit | fps | source_file | source_checksum_sha256 | processing_version
~~~
Object array와 pickle 의존 metadata를 피하고 문자열·숫자 NumPy array를 사용한다.
---

# 1. ACCAD 데이터 선별·검증

## 1.1 목표

252개 stageii motion을 검사하고 train/validation/test split을 확정한다. stagei neutral body file은 motion sequence 학습 대상에서 제외한다.
첫 end-to-end clip:
~~~text
ACCAD/Female1Walking_c3d/B3_-_walk1_stageii.npz
~~~

## 1.2 필수 검사

~~~text
required : trans, root_orient, pose_body, betas | expected : trans[T,3], root_orient[T,3], pose_body[T,63] | optional : pose_hand[T,90], pose_jaw[T,3], pose_eye[T,6], metadata
~~~
검사 항목:
- 시간축 frame 수 T 일치
- NaN/Inf
- zero-length 또는 너무 짧은 motion
- mocap_frame_rate 양수
- frame/FPS와 duration 정합성
- translation jump
- SO(3) geodesic rotation jump
- duplicated frame
- SHA-256 checksum
- subject/category/motion-name parsing
Axis-angle 벡터를 직접 빼서 rotation jump를 판단하지 않는다. Rotation matrix 또는 quaternion의 geodesic distance를 사용한다.

## 1.3 Manifest

~~~text
data/motions/manifests/ | ├── accad_inventory.json | ├── accad_quality.json | ├── accad_train.json | ├── accad_validation.json | ├── accad_test.json | └── accad_excluded.json
~~~
필드:
~~~text
source_file, checksum_sha256, subject, category, motion_name | fps, num_frames, duration_sec, gender, valid, exclude_reason
~~~
기본 split은 subject-disjoint다. ACCAD subject 수가 적으므로 category imbalance도 같이 보고한다. Motion-disjoint split은 보조 평가용으로 유지할 수 있다.

## 1.4 Gate 1

- [ ] stageii 252개 checksum/manifest
- [ ] subject/category mapping
- [ ] train/validation/test split
- [ ] 제외 이유 기록
- [x] B3 walk1 key/shape/FPS 확인
- [x] scan read error 0
---

# 2. SMPL-X Human Reconstruction

## 2.1 목표

~~~text
trans + root_orient + pose_body + betas | → SMPL-X FK @120Hz | → human joint/body kinematics | → ground/contact/quality | → root/pose resampling | → SMPL-X FK @50Hz | → canonical human motion NPZ
~~~

## 2.2 준비물

Licensed body model:
~~~text
data/motions/body_models/ | └── smplx/ | └── SMPLX_NEUTRAL.npz
~~~
현재 ACCAD inventory의 gender는 모두 neutral이다. 첫 milestone에는 neutral model만 사용한다.
공용 Isaac Python에 plain pip install을 무판으로 실행하지 않는다. smplx 버전을 고정한 optional dependency를 만들고 Isaac Python import smoke부터 한다.

## 2.3 정확한 SMPL-X 입력 계약

현재 데이터:
~~~text
pose_body [T,63] = 21 body joints × axis-angle 3 | pose_hand [T,90] = left 45 + right 45 | betas     [16] | gender    neutral
~~~
Model 생성 조건:
~~~python
model = smplx.create(
    model_path="data/motions/body_models",
    model_type="smplx",
    gender="neutral",
    use_pca=False,
    num_betas=16,
)
~~~
Hand 분리:
~~~python
left_hand_pose = pose_hand[:, :45]
right_hand_pose = pose_hand[:, 45:90]
~~~
use_pca=True 기본 설정에 45차원 axis-angle을 그대로 넣지 않는다. 현재 G1은 손가락 DoF가 없으므로 hand pose는 retarget target에서 제외하지만 원본 경로와 metadata는 보존한다.

## 2.4 FK와 orientation

SMPL-X standard output이 모든 joint global orientation을 완성된 schema로 제공한다고 가정하지 않는다.
- local rotation: root_orient와 pose_body axis-angle에서 계산
- global rotation: SMPL-X parent tree를 따라 누적
- quaternion: wxyz로 변환
- sign continuity: angular velocity 계산 전에 보정
~~~text
if dot(q[t], q[t-1]) < 0: | q[t] = -q[t]
~~~
SMPL-X package 버전에 따른 joint index 차이를 고려한다. 고정 숫자를 암묵적으로 사용하지 않고 joint mapper와 joint_names를 검증한다.

## 2.5 좌표계

B3 walk1 샘플은 trans[:,2]가 pelvis 높이이고 trans[:,:2]가 바닥 이동이므로 Z-up으로 보인다. 모든 파일에 고정 basis rotation을 무조건 적용하지 않는다.
~~~text
source coordinate audit | → 필요한 경우에만 basis transform B 적용 | → position:    p_dst = B p_src | → orientation: R_dst = B R_src B^T | → B를 metadata에 기록
~~~

## 2.6 Ground와 contact

Ground를 전체 minimum z 하나로 결정하지 않는다. Heel/toe 또는 foot keypoint의 낮은 높이와 낮은 속도를 같이 사용한다.
~~~text
contact enter : height < h_enter AND speed < v_enter | contact exit  : height > h_exit  OR speed > v_exit
~~~
Hysteresis, minimum-contact duration, isolated-frame removal을 적용한다.

## 2.7 120Hz → 50Hz

권장 순서:
~~~text
120Hz pose/root | → SMPL-X FK @120Hz | → quality/contact/ground 추정 | → root translation anti-alias filtering | → root/local rotation SO(3) smoothing + SLERP to 50Hz | → contact majority/nearest resampling | → SMPL-X FK @50Hz 재실행 | → 최종 50Hz velocity 재계산
~~~
3D joint position을 관절별로 독립 선형 보간하지 않는다. Resampled pose로 FK를 다시 수행해 bone length를 보존한다.
Angular velocity는 relative quaternion 또는 rotation-log로 계산한다. Boundary는 one-sided difference, 내부는 central difference를 사용한다. World/local frame을 필드 이름에 명시한다.

## 2.8 저장 범위와 schema

모든 frame의 mesh vertices를 영구 저장하지 않는다. 필요한 joint/body pose, velocity, contact, metadata만 저장한다. Mesh는 debug 시 chunk로 생성한다.
~~~text
data/motions/human_joints/accad/Female1Walking_c3d/ | └── B3_-_walk1_human_50hz.npz
~~~
~~~text
fps                         scalar = 50 | root_position_w             [T,3] | root_quaternion_wxyz        [T,4] | root_linear_velocity_w      [T,3] | root_angular_velocity_w     [T,3] | joint_position_w            [T,J,3] | joint_quaternion_wxyz       [T,J,4] | joint_local_quaternion_wxyz [T,J,4] | joint_linear_velocity_w     [T,J,3] | joint_angular_velocity_w    [T,J,3] | left_heel_contact           [T] | left_toe_contact            [T] | right_heel_contact          [T] | right_toe_contact           [T] | joint_names                 [J] | betas                       [16] | gender                      scalar | source_fps                  scalar = 120 | source_file                 scalar | source_checksum_sha256      scalar | schema_version              scalar
~~~

## 2.9 Gate 2

- [ ] licensed SMPLX_NEUTRAL.npz 준비
- [ ] pinned smplx import smoke
- [ ] use_pca=False, num_betas=16 검증
- [ ] B3 walk1 120Hz FK
- [ ] human joint name/index 검증
- [ ] Z-up/ground/contact 검증
- [ ] quaternion continuity/velocity 검증
- [ ] pose resampling 후 50Hz FK
- [ ] 50Hz human NPZ schema 검증
---

# 3. Human → G1 Kinematic Retargeting

## 3.1 목표

~~~text
canonical human motion | → morphology-normalized task-space target | → contact-aware sequence optimization | → G1 root + 29-DoF reference @50Hz
~~~
Human과 G1은 topology, joint axis, DoF, limit, link length, foot geometry, pelvis/shoulder width가 다르므로 joint angle을 직접 복사하지 않는다.

## 3.2 Joint/body 계약

Canonical joint order:
~~~text
configs/robot/g1_29dof.yaml :: joints[*].policy_index
~~~
Policy/SDK reorder:
~~~text
source/humanoid_g1/assets/joint_contract.py
~~~
Human ↔ G1 target:
| Human | G1 |
|---|---|
| pelvis | pelvis |
| torso/chest | torso_link |
| knee | left/right knee link |
| ankle/foot | ankle-roll link + foot frame |
| shoulder | shoulder-roll link |
| elbow | elbow link |
| wrist | wrist-yaw link |
| head | torso/head virtual direction |
정확한 human index와 G1 link 이름은 correspondence.yaml에 명시하고 startup에서 검증한다.

## 3.3 Kinematics/solver 결정 Gate

구현 전에 하나를 선택하고 버전을 고정한다.
- Pinocchio + SciPy/CasADi
- PyTorch differentiable FK
- Isaac articulation/Jacobian
선택 기준:
~~~text
offline/headless 실행 | URDF와 joint order 일치 | floating-base FK | Jacobian/gradient | joint hard bound | batch/window optimization | reproducible dependency
~~~
단일-frame IK와 이전-frame warm start만 사용하지 않는다. 1~2초 sliding window와 overlap으로 velocity, acceleration, contact를 sequence 단위로 최적화한다.

## 3.4 Optimization

Variable:
~~~text
root_position[t]     [3] | root_orientation[t]  SO(3) local update | joint_position[t]    [29]
~~~
Objective:
~~~text
pelvis position/orientation | torso orientation | feet position/orientation | wrist position/orientation | knee/elbow structure | head direction | joint/velocity bounds | temporal velocity/acceleration | stance-foot lock/contact | ground penetration | self-collision proxy | nominal-pose regularization
~~~
Walking의 초기 priority:
~~~text
pelvis → feet/contact → torso → hands → joint regularization
~~~
Human endpoint 절대 XYZ를 그대로 사용하지 않는다. Human limb direction을 G1 link length와 pelvis/shoulder width에 맞게 재구성한다.

## 3.5 Root/contact refinement

- XY: human displacement를 G1 stride scale에 맞춤
- Z: G1 nominal pelvis와 foot geometry에서 결정
- yaw: human heading 유지
- roll/pitch: G1 feasible range로 제한
- stance: foot XY/yaw/height drift 억제
- contact transition: G1 foot geometry로 재검증
Human contact를 G1 contact에 무조건 복사하지 않는다.

## 3.6 G1 output

~~~text
data/motions/retargeted/accad/Female1Walking_c3d/ | └── B3_-_walk1_g1_50hz.npz
~~~
~~~text
fps                    scalar = 50 | joint_pos              [T,29] policy order | joint_vel              [T,29] policy order | joint_acc              [T,29] policy order | root_pos_w             [T,3] | root_quat_w            [T,4] | root_lin_vel_w         [T,3] | root_ang_vel_w         [T,3] | body_pos_w             [T,B,3] | body_quat_w            [T,B,4] | body_lin_vel_w         [T,B,3] | body_ang_vel_w         [T,B,3] | left_foot_contact      [T] | right_foot_contact     [T] | joint_names            [29] | body_names             [B] | source_file            scalar | source_checksum_sha256 scalar | retarget_version       scalar | schema_version         scalar
~~~
body tensor는 body_names에 기록된 G1 link order로 G1 FK를 다시 실행해 만든다.
## 3.7 Action reachability Gate

현재 action 계약:
~~~text
q_target = q_default + 0.25 * action | training clip_actions = 1.0
~~~
각 frame/joint에서 계산:
~~~python
required_action = (q_ref - q_default) / action_scale
~~~
Report:
~~~text
max(abs(required_action)) | abs(required_action) > 1 비율 | 관절별 saturation 비율 | reference joint velocity/acceleration
~~~
범위를 자주 넘으면 다음 중 하나를 선택한다.
1. 관절군별 action scale
2. 더 넓은 clip과 safety validation
3. q_target = q_ref + residual_scale × action
4. motion 제외 또는 time scaling
선택한 action 의미를 Isaac task, Python adapter, exporter, C++ controller에 모두 반영한다.
## 3.8 Gate 3

- [ ] 29 joint policy order/name 일치
- [ ] hard/soft position-limit 통과
- [ ] velocity-limit 통과 또는 time scaling
- [ ] foot penetration/skating 허용 범위
- [ ] pelvis/feet/torso/wrist quality report
- [ ] action reachability report
- [ ] Isaac name-based kinematic replay
- [ ] root/joint sign과 좌표계 검증
Report:
~~~text
workspace/artifacts/reports/retargeting/<motion>_report.json
~~~
---
# 4. Isaac Physics Motion Tracking

## 4.1 목표

~~~text
reference + phase/future target + current state | → motion-tracking policy | → 29-D action | → action filter/PD target | → Isaac physics
~~~
## 4.2 코드 구조

~~~text
source/humanoid_g1/ | ├── motion/motion_library.py | ├── mdp/motion/ | │   ├── commands.py | │   ├── observations.py | │   ├── rewards.py | │   ├── events.py | │   └── terminations.py | └── tasks/motion_tracking/ | ├── __init__.py | ├── env_cfg.py | └── rsl_rl_cfg.py
~~~
third_party/unitree_rl_lab의 mimic 구현은 참고만 하고 vendor source는 수정하지 않는다. 해당 single-motion loader는 control step마다 frame index를 하나씩 증가시키므로, 재사용 시 reference가 정확히 50Hz이거나 time-based interpolation이어야 한다.
## 4.3 Motion library

~~~text
manifest/split loading | clip/time/phase sampling | batch reference query | time interpolation | per-motion weight | failure-bin adaptive sampling
~~~
실제 retarget library 용량과 query latency를 profile한 뒤 packed GPU tensor 또는 cache/streaming을 선택한다.
## 4.4 Reference alignment

각 reset에서 reference를 environment origin/current robot에 맞춘다.
~~~text
reference initial pelvis XY/yaw | → current robot pelvis XY/yaw alignment | → aligned root/body reference
~~~
Root-relative/heading-relative 변환은 actor observation뿐 아니라 tracking reward에도 똑같이 적용한다.
## 4.5 Observation

Actor 후보:
~~~text
reference joint position/velocity or error | reference root orientation/velocity | phase sin/cos | future reference at 0/40/80/160ms | base angular velocity | projected gravity | current joint position/velocity | measurable contact | last action
~~~
실기에서 안정적으로 측정할 수 없는 base linear velocity, full body state, contact truth는 critic privileged observation으로 분리한다.
Policy input 확정 후 다음 contract를 새로 만든다.
~~~text
actor term order and scale | history/future order | observation dimension | normalization | reference convention
~~~
현재 locomotion 480-d adapter를 motion tracker에 그대로 사용하지 않는다.
## 4.6 Action training/deployment parity

현재 deployment adapter:
~~~text
policy clip       = 1.0 | action delta      = 0.15 per control step | action scale      = 0.25 | max target change = 0.0375 rad per control step
~~~
기존 locomotion training에는 같은 hard delta filter가 없다. Motion tracking은 다음 중 하나를 선택해 training과 deployment를 맞춘다.
1. training action term에 같은 hard delta filter
2. actuator delay/rate limit를 환경 상태로 모델링
3. 실기 limit 재설계 후 safety 재검증
Action-rate reward만으로 hard runtime limit가 동일해진다고 가정하지 않는다.
## 4.7 Reward/termination

Positive:
~~~text
joint pose/velocity tracking | aligned pelvis/root tracking | relative body pose/orientation tracking | feet/wrist tracking | contact matching
~~~
Penalty:
~~~text
foot sliding | premature swing-foot contact | undesired body contact | joint-limit proximity | torque/energy/power | action rate/saturation | joint acceleration
~~~
Exponential reward는 weight뿐 아니라 std/scale도 함께 기록한다.
Termination:
~~~text
aligned anchor height/orientation error | feet/wrist body error | illegal contact | invalid joint state | NaN/Inf | motion end
~~~
## 4.8 Initialization과 curriculum

Reset:
~~~text
motion id | start time/phase | root yaw | root pose/velocity perturbation | joint pose/velocity perturbation
~~~
Curriculum:
~~~text
B3 single walk + small noise | → B3 + random phase/stronger noise | → similar walking clips | → stand/walk/turn | → validated ACCAD motions | → progressive domain randomization
~~~
## 4.9 Evaluation

~~~text
completion ratio | fall rate/survival time | joint position/velocity RMSE | pelvis/body/feet/wrist error | contact accuracy/F1 | foot skating distance | mean/peak torque and power | action saturation/rate-limit activation | joint-limit margin | per-motion success | unseen-subject/clip result
~~~
## 4.10 Gate 4

- [ ] 1-env reset/step/reference/reward debug
- [ ] 16-env, 10-iteration checkpoint smoke
- [ ] 256-env B3 full-sequence tracking
- [ ] random-phase recovery
- [ ] action reachability/rate-limit 통과
- [ ] fall/contact/skating/tracking report
- [ ] multi-walk generalization
---
# 5. Embodiment Gap 축소·Downstream Transfer

## 5.1 두 gap

~~~text
A. Human kinematics ↔ G1 physical dynamics | B. Isaac ↔ MuJoCo/Real G1 dynamics
~~~
A는 retargeting과 tracking, B는 model randomization, latency, Sim2Sim으로 줄인다.
## 5.2 Progressive domain randomization

Nominal tracking 성공 후 small → medium → deployment 범위로 늘린다.
~~~text
mass, CoM, inertia | joint stiffness/damping/friction | motor strength/bias | ground friction/restitution | IMU/joint noise | action/observation delay | control jitter | external push | small terrain perturbation
~~~
## 5.3 Reference runtime

Reference-conditioned tracker를 실기에서 실행하려면 policy 외에 다음이 필요하다.
~~~text
MotionReferencePlayer | 50Hz phase/time clock | reference interpolation | start-pose transition | phase reset/resynchronization | motion end/loop handling | abort/recovery | motion schema validation
~~~
Isaac command manager, Python deployment, C++ controller가 같은 reference query를 생성해야 한다.
## 5.4 Export/deployment contract

현재 480-d velocity-locomotion contract와 motion-tracking contract를 섞지 않는다.
추가 구현:
~~~text
MotionObservationAdapter | MotionReferencePlayer | motion policy exporter | motion observation contract JSON | golden observation/action vectors | Python/C++ parity test | C++ reference observation builder
~~~
일치 항목:
~~~text
joint order/sign | action semantics/scale/clip/delta | PD gains | observation order/scale/history/future | quaternion convention | reference alignment | control timing/latency
~~~
## 5.5 Isaac → MuJoCo → Real

~~~text
Isaac nominal tracking | → progressive domain randomization | → policy/reference bundle export | → offline Python/C++ parity | → MuJoCo Sim2Sim | → suspended low-gain test | → guarded start-pose/standing | → small-amplitude motion | → limited walking | → full validated motion
~~~
Safety:
~~~text
joint/velocity/torque limits | action saturation/rate limit | PD gains | watchdog | emergency stop | fall/tilt detection | reference abort/recovery
~~~
## 5.6 Downstream 선택지

A. Reference-conditioned low-level tracker
~~~text
high-level motion id/phase/command | → pretrained tracker | → G1 actions
~~~
B. Teacher-student distillation
Rich reference/privileged teacher를 deployment observation student로 distill한다.
C. Latent motion skill
~~~text
motion library → encoder → latent z → low-level policy
~~~
D. AMP/VAE prior
Tracking library를 discriminator 또는 generative motion prior에 재사용한다.
E. Selective transfer
Observation dimension이 다르면 전체 checkpoint resume을 하지 않는다. Shape가 같은 encoder/hidden layer/action head만 명시적으로 transfer한다.
## 5.7 Gate 5

- [ ] Python/Isaac/C++ reference runtime parity
- [ ] motion observation/action golden-vector parity
- [ ] domain-randomized tracking
- [ ] MuJoCo Sim2Sim
- [ ] latency/PD/contact 차이 report
- [ ] low-gain hardware safety gate
- [ ] tracker/distillation/latent/AMP 중 downstream 방식 결정
---
# 6. 최종 폴더 구조

~~~text
humanoid_G1/ | ├── ACCAD/                              # 읽기 전용 원본 | ├── data/motions/ | │   ├── manifests/ | │   ├── body_models/smplx/ | │   ├── human_joints/accad/ | │   ├── retargeted/accad/ | │   └── motion_library/ | ├── source/humanoid_g1/ | │   ├── motion/ | │   │   ├── amass_loader.py | │   │   ├── smplx_fk.py | │   │   ├── rotations.py | │   │   ├── coordinates.py | │   │   ├── contacts.py | │   │   ├── filters.py | │   │   ├── resample.py | │   │   ├── schema.py | │   │   ├── correspondence.py | │   │   ├── g1_kinematics.py | │   │   ├── retarget_loss.py | │   │   ├── retargeter.py | │   │   ├── contact_refinement.py | │   │   └── motion_library.py | │   ├── mdp/motion/ | │   │   ├── commands.py | │   │   ├── observations.py | │   │   ├── rewards.py | │   │   ├── events.py | │   │   └── terminations.py | │   └── tasks/motion_tracking/ | │       ├── __init__.py | │       ├── env_cfg.py | │       └── rsl_rl_cfg.py | ├── scripts/motions/ | │   ├── visualize_amass.py             # 현재 구현 | │   ├── build_accad_manifest.py | │   ├── extract_smplx_joints.py | │   ├── validate_human_motion.py | │   ├── retarget_human_to_g1.py | │   ├── validate_g1_reference.py | │   ├── replay_g1_reference.py | │   ├── compare_human_g1_motion.py | │   └── batch_retarget_accad.py | ├── workspace/artifacts/reports/       # JSON/Markdown report | └── data/media/ | ├── plots/ | └── videos/
~~~
---
# 7. Milestones

## Milestone 1: Human reconstruction

~~~text
B3 stageii | → SMPL-X neutral/use_pca=False/16 betas FK @120Hz | → coordinate/contact/quality | → root/pose resampling | → FK @50Hz | → B3 human 50Hz NPZ
~~~
## Milestone 2: G1 retargeting

~~~text
B3 human 50Hz | → morphology/contact-aware sliding-window optimization | → action reachability | → Isaac kinematic replay | → B3 G1 50Hz NPZ
~~~
## Milestone 3: Physics tracking

~~~text
B3 G1 50Hz | → 1-env debug | → 16-env PPO smoke | → 256-env single-motion tracking | → multi-walk tracker
~~~
## Milestone 4: Multi-motion tracker

~~~text
validated ACCAD clips | → packed motion library | → stand/walk/turn | → broad whole-body tracking
~~~
## Milestone 5: Deployment/prior

~~~text
reference runtime + motion observation contract | → domain randomization | → Sim2Sim/Real | → distillation/latent/AMP 선택
~~~
## 7.1 실행 예정 interface

다음 명령은 관련 script를 구현한 후 사용할 목표 CLI다.
~~~bash
cd /home/hoeng/IsaacLab/humanoid_G1
# [미구현] Manifest
../_isaac_sim/python.sh scripts/motions/build_accad_manifest.py \
  --root ACCAD \
  --output data/motions/manifests
# [미구현] SMPL-X reconstruction
../_isaac_sim/python.sh scripts/motions/extract_smplx_joints.py \
  --input ACCAD/Female1Walking_c3d/B3_-_walk1_stageii.npz \
  --model-root data/motions/body_models \
  --output data/motions/human_joints/accad/Female1Walking_c3d/B3_-_walk1_human_50hz.npz
# [미구현] Human → G1
../_isaac_sim/python.sh scripts/motions/retarget_human_to_g1.py \
  --input data/motions/human_joints/accad/Female1Walking_c3d/B3_-_walk1_human_50hz.npz \
  --output data/motions/retargeted/accad/Female1Walking_c3d/B3_-_walk1_g1_50hz.npz
# [미구현] G1 validation
../_isaac_sim/python.sh scripts/motions/validate_g1_reference.py \
  --motion data/motions/retargeted/accad/Female1Walking_c3d/B3_-_walk1_g1_50hz.npz
# [미구현] Isaac kinematic replay
../isaaclab.sh -p scripts/motions/replay_g1_reference.py \
  --motion data/motions/retargeted/accad/Female1Walking_c3d/B3_-_walk1_g1_50hz.npz
# [미구현] PPO smoke
./g1.sh train \
  --config configs/experiments/g1_motion_tracking_smoke.yaml \
  --num-envs 16 --max-iterations 10 --headless \
  --run-name b3_walk1_smoke
~~~
---
# 8. Go/No-Go Checklist

## Data

- [ ] stageii checksum/manifest/split
- [ ] coordinate/FPS/shape validation
- [ ] raw ACCAD 불변성
## Human

- [ ] licensed neutral SMPL-X model
- [ ] use_pca=False, 16 betas, hand split
- [ ] 120Hz FK/contact/quality
- [ ] resampling 후 50Hz FK
- [ ] joint/global-local rotation schema
## Retarget

- [ ] solver/backend/version 결정
- [ ] sliding-window/contact-aware optimization
- [ ] policy joint order
- [ ] joint/velocity/action reachability
- [ ] name-based Isaac replay
## Tracking

- [ ] aligned observation/reward
- [ ] actor/critic observation 분리
- [ ] training/deployment action rate parity
- [ ] single-motion completion
- [ ] multi-motion evaluation
## Deployment/Prior

- [ ] reference player/phase/start/end/abort
- [ ] motion observation/export/C++ parity
- [ ] domain randomization/Sim2Sim
- [ ] hardware safety gate
- [ ] tracker와 prior 산출물 구분
---
# 9. 지금 당장 할 일

현재 다음 milestone은 SMPL-X Human Reconstruction Gate다.
1. licensed SMPLX_NEUTRAL.npz를 data/motions/body_models/smplx에 준비
2. 버전 고정 smplx optional dependency 구성
3. source/humanoid_g1/motion/amass_loader.py 구현
4. source/humanoid_g1/motion/rotations.py 구현
5. source/humanoid_g1/motion/smplx_fk.py 구현
6. B3 walk1 120Hz FK 검증
7. contact/ground/resampling 후 50Hz FK 재실행
8. B3 walk1 human 50Hz NPZ schema report 생성
이 Gate를 통과하기 전에 G1 retargeting optimizer와 PPO를 동시에 만들지 않는다.
<!-- compressed document: exactly 600 lines -->
