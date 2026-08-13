# ACCAD 외부 데이터 배치

이 폴더에는 GitHub에 저장하는 자체 구현과, GitHub에 올리지 않는 라이선스 데이터가
함께 배치된다.

- 추적됨: `_g1_pipeline/`, 설명서, 설정, 테스트
- 추적 안 됨: AMASS ACCAD `*_stagei.npz`/`*_stageii.npz`, SMPL-X 모델,
  `vendor/`, `work/`

공식 AMASS에서 ACCAD의 **SMPL-X 형식**을 받은 뒤 category/subject 폴더가 이
README와 같은 깊이에 오도록 압축을 푼다. 공식 SMPL-X 사이트에서는
**SMPL-X with removed head bun (NPZ)**를 받는다.

준비와 검증 명령:

```bash
cd /path/to/IsaacLab/humanoid_G1
./scripts/setup/prepare_motion_data.sh \
  --smplx-model /path/to/models_lockedhead/smplx/SMPLX_NEUTRAL.npz
```

전체 재설치와 파이프라인 순서는
[`docs/REINSTALL_FROM_GITHUB_KO.md`](../docs/REINSTALL_FROM_GITHUB_KO.md)를 따른다.
