# Safety contract

C++ controller state:

```text
DISCONNECTED → IDLE → DAMPING → MOVE_TO_DEFAULT → STAND_READY
→ POLICY_READY → POLICY_RUNNING → STOPPING
                               ↘ FAULT → damping
```

POLICY_RUNNING 전이는 명시적 operator `--run`이 있어야 합니다. LowCmd publisher
자체도 검증된 LowState와 명시적 stand/run 이전에는 생성되지 않습니다.

Runtime gate:

- LowState CRC/freshness 100 ms와 network disconnect
- finite joint position/velocity 및 URDF position/velocity limit
- quaternion norm 0.8–1.2
- roll/pitch 0.7 rad
- remote L2+B emergency stop
- finite action, `[-1,1]` clip, step delta 0.15
- target soft limit 0.05 rad
- inference 10 ms, loop work 30 ms

오류 시 FAULT로 이동하고 마지막으로 확인된 joint position에서 kp=0, kd=3의
damping command를 보냅니다. 정책 checksum, joint contract, hardware profile,
non-loopback interface/domain 0은 controller process를 시작하기 전에 Python
real guard에서 검사합니다.

Sim2Sim은 `lo/domain 1`, physical은 non-loopback/domain 0으로 고정되어 서로
교차 송신할 수 없습니다.
