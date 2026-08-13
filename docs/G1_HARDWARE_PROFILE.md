# G1 hardware profile

현재 확정된 항목:

- manufacturer/model: Unitree G1
- body variant: 29-DoF rev_1_0
- policy body joints: 29
- hand: none/unknown, body policy와 분리
- SDK message: `unitree_hg`
- simulation DDS: domain 1, `lo`
- intended physical DDS: domain 0
- physics/policy: 200/50 Hz

현재 미확정 항목:

- serial number
- firmware version
- 실제 hand option (none/Dex3/Inspire)
- 전용 network interface와 IP
- 현장 motor index 확인
- physical emergency stop 검증

미확정 값을 추측해 채우지 않습니다. 실제 로봇에서 확인 후
`configs/robot/g1_variant.yaml`만 authoritative hardware profile로 갱신합니다.
Dex3/Inspire placeholder는 interface/asset 선택지만 나타내며 가짜 hand joint를
만들지 않습니다.
