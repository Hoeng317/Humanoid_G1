# Upstream usage and patches

`third_party/`의 공식 저장소는 모두 clean 상태이며 직접 수정하지 않았습니다.
연구용 변경은 `source/humanoid_g1`, `scripts`, `deploy/cpp`, `sim2sim`에 있습니다.

가져온 부분:

- `unitree_rl_lab`: G1 asset/task baseline과 locomotion MDP port
- `unitree_ros`: G1 29-DoF rev_1_0 URDF/meshes
- `unitree_mujoco`: G1 MJCF/scene 및 official SDK2 bridge simulator
- `unitree_sdk2`: DDS types, channel API, CRC/reference controller conventions
- `unitree_sdk2_python`: read/debug reference; real publisher 경로에는 사용하지 않음

Local adaptation:

- Isaac Lab 2.3.2 API에 맞춘 독립 task registration
- Actor에서 base linear velocity를 제외하고 5-frame contract 고정
- action order를 policy order로 보존하고 SDK/MuJoCo order adapter 제공
- official MuJoCo source를 수정하지 않는 overlay CMake build
- simulation domain을 1/lo로 강제
- LibTorch state machine, runtime safety gate와 real-mode guard 추가

복사된 locomotion MDP 파일은 upstream copyright/SPDX header를 유지합니다.
