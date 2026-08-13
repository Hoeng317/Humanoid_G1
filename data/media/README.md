# 이미지와 시각화 작업 공간

- `images/`: 문서용 이미지, 로봇/환경 reference 이미지
- `videos/`: 사용자가 보존할 재생·학습 영상
- `plots/`: reward, 안정성, evaluation 그래프

자동 생성되는 검증 결과는 계속 `workspace/artifacts/reports/`, 학습 영상은 run의
`workspace/logs/rsl_rl/.../videos/`에 저장됩니다. 이 폴더는 사람이 선별해 보존하거나 후속
비전/멀티모달 연구에 사용할 자료를 두는 곳입니다.
