# Architecture

## 실행 데이터 경로

```text
official Unitree G1 URDF
  → G1_29DOF_CFG (Isaac articulation/implicit PD)
  → ManagerBasedRLEnv (flat/rough/play)
  → RSL-RL PPO (actor 480, critic 495, action 29)
  → model_N.pt
  → TorchScript/ONNX export + SHA256SUMS
  → C++ ObservationHistory/LibTorch/ActionAdapter
  → SDK2 rt/lowcmd
  → Unitree MuJoCo or physical G1 rt/lowstate
```

`configs/robot/g1_29dof.yaml`이 관절 limit/default/kp/kd의 authoritative
source입니다. `JointContract`가 URDF revolute order, policy order, MuJoCo actuator
order와 SDK2 motor index를 검증하고 JSON/Markdown artifact를 생성합니다.

Actor의 한 frame은 angular velocity 3 + projected gravity 3 + command 3 + joint
position 29 + joint velocity 29 + previous action 29 = 96차원입니다. 가장 오래된
frame부터 5개를 연결해 480차원이 됩니다. Critic만 base linear velocity 3을
추가하여 frame당 99, 총 495차원을 사용합니다.

Task는 package import 시 Gymnasium에 등록되며 Isaac Lab core registry는 변경하지
않습니다.

- `Humanoid-G1-29DoF-Velocity-Flat-v0`
- `Humanoid-G1-29DoF-Velocity-Rough-v0`
- `Humanoid-G1-29DoF-Velocity-Play-v0`
- `Unitree-G1-29dof-Velocity` (compatibility alias)

향후 확장은 `interfaces/hooks.py`의 command provider, identity safety filter,
sensor registry를 통해 연결합니다. SeRT/SERT 코드를 직접 import하거나 수정하지
않습니다.
