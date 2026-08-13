# Runtime workspace

코드와 입력 데이터가 아닌 자동 생성 결과를 한곳에 모읍니다.

- `logs/`: RSL-RL checkpoint, TensorBoard, setup audit
- `artifacts/contracts/`: joint/observation contract
- `artifacts/reports/`: doctor, stability, evaluation, Sim2Sim 보고서
- `artifacts/policies/`: export된 TorchScript/ONNX 배포 bundle

`logs/`, exported policy와 큰 영상은 Git에서 제외됩니다. contract와 작은 검증
보고서는 재현성 확인을 위해 보존할 수 있습니다.
