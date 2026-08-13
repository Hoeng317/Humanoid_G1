# Physical G1 commissioning

실물 G1이 없는 현재에는 물리 network로 LowCmd를 전송하지 않았습니다.
`g1_variant.yaml`의 hardware와 motor index는 모두 미확정 상태입니다.

실물 수령 후 다음 순서를 건너뛰지 마십시오.

1. 구매 모델이 G1 29-DoF rev_1_0인지 확인
2. serial, firmware, SDK2 호환 버전을 기록
3. motor index를 제조사 도구와 read-only LowState로 현장 검증
4. 전용 유선 network interface/IP와 domain 0 확인
5. 기본 motion service 중지 절차 및 리모컨 L2+B 비상 정지 확인
6. robot을 support rig에 매달고 주변을 비움
7. 별도 비상 정지 담당자 배치
8. `deploy-real`을 `--stand/--run` 없이 실행해 LowState read-only 확인
9. zero torque/damping 확인
10. 한 관절, 낮은 kp/kd, 작은 범위 시험
11. 3초 default-pose transition 시험
12. support rig standing 시험
13. 낮은 command range 정책 시험
14. 여러 회 정지/timeout/network 단절 fault 시험
15. 점진적으로 속도와 command 범위 확대

그 후에만 `configs/robot/g1_variant.yaml`의 확인된 값을 채우고 `verified` 및
`motor_index_verified`를 true로 변경합니다.

```bash
./humanoid_G1/g1.sh deploy-real \
  --interface <DEDICATED_ETHERNET> --domain-id 0 \
  --policy workspace/artifacts/policies/<RUN_ID>/policy.pt \
  --real --acknowledge-hardware-risk --stand
```

`--run`은 full commissioning을 통과한 후보 정책에만 사용합니다. controller는
스스로 POLICY_RUNNING에 진입하지 않습니다.
