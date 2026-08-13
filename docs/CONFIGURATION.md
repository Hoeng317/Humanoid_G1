# Configuration

`configs/experiments/*.yaml` 하나가 robot, simulation, terrain, training,
randomization, deployment YAML을 조합합니다. `training`은 다시 `policy`와
`algorithm` YAML을 참조합니다. CLI 값은 조합 후 마지막에 적용되며 각 학습 run의
`params/experiment_resolved.yaml`에 완전히 해석된 설정으로 저장됩니다.

- `configs/policy/`: Actor/Critic 신경망 구조
- `configs/algorithm/`: PPO update hyperparameter
- `configs/train/`: rollout/iteration/save 및 policy/algorithm 선택
- `configs/experiments/`: reward/command와 모든 component의 최상위 조합

사용자 실험은 baseline 대신 `g1_custom_ppo.yaml`, `mlp_custom.yaml`,
`ppo_custom.yaml`을 수정합니다.

주요 override:

```text
--task --config --num-envs --seed --max-iterations --device --headless
--video --checkpoint --resume --terrain --command-range --randomization --run-name
```

유효성 검사는 simulator 시작 전에 환경 수, dt/decimation, 29 action, 480/495
observation, command/randomization 범위, reward finite 여부, policy period,
real-mode hardware/interface를 검사합니다.

## 기본 timing/action

- physics: 200 Hz (`dt=0.005`)
- policy: 50 Hz (`decimation=4`, `policy_dt=0.02`)
- action: normalized joint position target `[-1,1]^29`
- target: `q_default + 0.25 × action`
- deploy action delta: policy step당 0.15
- target: URDF limit에서 0.05 rad 안쪽으로 clip

## Reward

|config key|의미|기본 weight|
|---|---|---:|
|track_lin_vel_xy|yaw frame 선속도 추종|1.0|
|track_ang_vel_z|yaw 각속도 추종|0.5|
|alive|생존|0.15|
|base_linear_velocity|수직 선속도 억제|-2.0|
|base_angular_velocity|roll/pitch 각속도 억제|-0.05|
|joint_vel|관절 속도 L2|-0.001|
|joint_torque|관절 torque L2|-2e-5|
|joint_acc|관절 가속도 L2|-2.5e-7|
|action_rate|action 변화량|-0.05|
|dof_pos_limits|관절 위치 limit 위반|-5.0|
|dof_vel_limits|관절 속도 soft limit 위반|-1.0|
|energy|torque×velocity proxy|-2e-5|
|flat_orientation_l2|upright orientation|-5.0|
|base_height|0.78 m 높이|-10.0|
|gait|좌우 접촉 timing|0.5|
|feet_slide|발 미끄럼|-0.2|
|feet_clearance|swing clearance|1.0|
|undesired_contacts|발 외 접촉|-1.0|
|termination|비-timeout 종료|-200.0|

팔/허리/hip roll-yaw posture regularization도 각각 별도 term으로 구성됩니다.

## Randomization

`nominal.yaml`은 friction, base mass, initial joint velocity, push, observation noise의
기본 범위를 사용합니다. `sim2real.yaml`은 COM, actuator kp/kd, joint friction,
encoder/IMU noise, latency/packet drop, payload 등을 포함하는 점진적 preset입니다.
현재 Isaac event manager에 직접 연결된 항목은 material/restitution, base
mass/COM, actuator kp/kd, joint friction, reset velocity, observation noise, push입니다.
latency/packet drop와 motor/action-scale 변동은 config contract와 deployment hook에
보존되어 있으며 실제 학습 적용 전 후속 action-buffer 연구가 필요합니다.
