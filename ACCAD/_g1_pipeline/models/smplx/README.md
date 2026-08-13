# Licensed SMPL-X model location

공식 SMPL-X 라이선스에 동의해 직접 받은 neutral NPZ를 이 폴더에서 다음 이름으로
연결합니다. 로컬에 생성되는 심볼릭 링크와 모델 파일은 Git에 포함하지 않습니다.

공식 등록·다운로드: <https://smpl-x.is.tue.mpg.de/>

```text
SMPLX_NEUTRAL.npz
```

기존 환경에서 사용한 링크 대상 예시:

```text
../../../smplx_lockedhead_20230207/models_lockedhead/smplx/SMPLX_NEUTRAL.npz
```

ACCAD의 각 `neutral_stagei.npz`는 subject betas/marker metadata일 뿐 body model이
아니므로 이 파일을 대신할 수 없습니다. 모델 파일 자체는 Git에 포함하지 않습니다.

새 clone에서는 저장소 루트의 다음 스크립트가 올바른 파일의 SHA-256을 확인하고 이
링크와 `smplx==0.1.28` local vendor를 만든다.

```bash
./scripts/setup/prepare_motion_data.sh \
  --smplx-model /path/to/models_lockedhead/smplx/SMPLX_NEUTRAL.npz
```
