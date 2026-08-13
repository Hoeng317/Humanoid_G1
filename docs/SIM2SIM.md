# MuJoCo Sim2Sim

공식 `unitree_mujoco` C++ simulator, MuJoCo 3.3.6, project-local SDK2/GLFW를
사용합니다. 설치 경로는 모두 `.local/` 아래입니다.

```bash
./humanoid_G1/scripts/setup/build_local_sdk.sh
./humanoid_G1/scripts/setup/build_mujoco.sh
./humanoid_G1/scripts/setup/build_controller.sh
```

실행기는 다음을 순서대로 수행합니다.

1. 29-DoF joint contract와 export `SHA256SUMS` 검증
2. `lo`, DDS domain 1 강제
3. official `scene_29dof.xml`로 MuJoCo 시작
4. G1 `rt/lowstate` 수신
5. C++ controller를 IDLE로 시작
6. CLI의 명시적 `--stand` 또는 `--run` 이후에만 `rt/lowcmd` publisher 생성
7. 종료 시 controller가 `STOPPING`과 damping을 송신한 뒤 simulator 종료

```bash
./humanoid_G1/g1.sh sim2sim \
  --policy workspace/artifacts/policies/<RUN_ID>/policy.pt --run
```

자동 smoke는 `--duration 10 --xvfb`를 추가합니다. 실행별 controller/MuJoCo
log와 `SIM2SIM_<timestamp>.md`가 `workspace/artifacts/reports/`에 저장됩니다.

`compare_rollout.py`는 동일 command로 기록한 Isaac/MuJoCo NPZ의 command,
base quaternion, joint position, action RMSE와 saturation을 비교합니다.
