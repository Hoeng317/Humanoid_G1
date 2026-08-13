# Reference motions

ACCAD → G1 reconstruction, retargeting, physics tracking, deployment의 검토 반영
전체 계획은
[`docs/HUMANOID_DATASET_PRETRAINING_PIPELINE_KO.md`](../../docs/HUMANOID_DATASET_PRETRAINING_PIPELINE_KO.md)에
정리했습니다.

서기, 걷기, 회전, recovery 같은 시간축 reference motion을 둡니다. 권장 최소 필드:

```text
time_s: [T]
joint_position_policy: [T, 29]
joint_velocity_policy: [T, 29]
root_position: [T, 3]
root_quaternion_wxyz: [T, 4]
```

관절 순서는 `configs/robot/g1_29dof.yaml`의 `policy_index`를 따라야 합니다.
