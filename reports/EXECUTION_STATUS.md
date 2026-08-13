# ACCAD → G1 실행 상태

기준일: 2026-08-08  
작업 경계: 모든 신규 코드·산출물은 `/home/hoeng/IsaacLab/humanoid_G1/ACCAD` 내부

## Gate 상태

| Gate | 상태 | 결과 |
|---|---|---|
| 1. ACCAD 선별·검증 | PASS | stageii 252개 검사, 고유 motion 249개, exact duplicate 3개 제외 |
| 2. SMPL-X Human Reconstruction | PASS | B3 walk1 120 Hz FK/contact → ground-normalized 50 Hz NPZ 생성·검증 |
| 3. Human → G1 Retargeting | NEXT | G1 29-DoF correspondence와 contact-aware optimizer 구현 차례 |
| 4. Isaac Motion Tracking | WAIT | 검증된 G1 50 Hz reference 필요 |
| 5. Deployment/Prior | WAIT | tracker checkpoint 필요 |

## Gate 1 결과

- train: 215개 (`Female1`, `Male2`)
- validation: 5개 (`s001`, `s007`, `s008`, `s009`, `s011`)
- test: 29개 (`Male1`)
- subject/path/SHA-256 split 교집합: 모두 0
- SHA-256 exact alias 3개는 원본을 삭제하지 않고 학습 대상에서만 제외
- `Male2Running_c3d/C11_-_run_turn_left_90_stageii.npz`의 left-elbow
  2.733 rad/frame 변화는 review flag로 보존하고 자동 제외하지 않음

ACCAD의 큰 subject가 세 명뿐이므로 5개 validation은 smoke 용도입니다. 주 평가는
held-out `Male1` test이며, 논문 수준 평가는 major-subject 교차검증을 추가해야 합니다.

## Gate 2 결과

입력:

```text
ACCAD/Female1Walking_c3d/B3_-_walk1_stageii.npz
915 frames @ 120 Hz, neutral SMPL-X
```

출력:

```text
ACCAD/_g1_pipeline/work/human/Female1Walking_c3d/B3_-_walk1_human_50hz.npz
381 frames @ 50 Hz, 55 rotation joints, 127 named SMPL-X output joints
```

실행 순서는 문서 계약과 동일합니다.

```text
SMPL-X FK/contact @120 Hz
→ speed-gated adaptive floor baseline
→ 약 6.46 cm의 fitting floor drift 제거
→ root anti-alias + rotation SLERP
→ contact majority resampling
→ SMPL-X FK/velocity @50 Hz
```

검증 리포트:

```text
ACCAD/_g1_pipeline/work/reports/B3_-_walk1_human_50hz_validation.json
```

모든 검증 그룹이 PASS입니다.

- schema: safe no-pickle load, dtype/shape/name/time-grid 계약
- numeric: quaternion norm/sign, rotation FK, rigid bone, velocity 재계산
- contact: semantic keypoint, 120→50 Hz label replay, clearance/speed/run
- coordinate: right-handed Z-up, identity basis, fixed ground `z=0`
- provenance: source 및 licensed model SHA-256, 120 Hz FK/contact 기록
- B3 walking profile: airborne frame 0, 좌우 stance가 clip 끝까지 유지

Licensed model은 다운로드한 locked-head neutral 파일을 심볼릭 링크로 사용합니다.

```text
SMPLX_NEUTRAL.npz SHA-256:
43d8f3a1375d7c5baae207870a5d51def0f7e6b507df709b4937598b5e7d965d
```

## 현재 명령

```bash
cd /home/hoeng/IsaacLab/humanoid_G1
../_isaac_sim/python.sh ACCAD/_g1_pipeline/run.py status
PYTHONDONTWRITEBYTECODE=1 ../_isaac_sim/python.sh \
  ACCAD/_g1_pipeline/run.py all
```

현재 회귀 테스트: `17 passed`.

## 다음 구현 범위

다음은 Gate 3입니다. 상위 프로젝트의 G1 29-DoF YAML/URDF를 읽어 다음을
`ACCAD/_g1_pipeline` 안에 구현합니다.

1. human semantic joint ↔ G1 link correspondence와 startup 검증
2. G1 joint/root limit 및 policy-order 계약 로딩
3. 1–2초 sliding-window task-space retargeting
4. pelvis/feet/torso/wrist 및 temporal/contact-aware loss
5. B3 G1 50 Hz NPZ 생성과 Isaac kinematic replay 검증

Gate 3 이전 단계에서는 PPO 학습을 시작하지 않습니다.
