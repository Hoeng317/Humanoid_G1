# Changelog

## 2026-08-07

- 학습 로그와 생성 결과를 `workspace/logs/`, `workspace/artifacts/`로 통합
- 이미지·영상은 `data/media/`, MuJoCo 빌드 정의는 `deploy/sim2sim/`로 이동
- 실패한 smoke run, cache, 빈 placeholder, 중복 legacy backup 제거
- 실행 코드·설정·문서의 경로를 새 구조에 맞게 갱신

## 2026-07-30

- 기존 SKRL/demo wrapper를 `g1.sh` 중심의 독립 RSL-RL 연구 프로젝트로 교체
- 교체 전 소스/설정 백업은 2026-08-07 구조 정리에서 제거
- 공식 Unitree 5개 저장소를 `third_party/`에 commit 고정
- G1 29-DoF URDF/MuJoCo/SDK2 joint contract 확정
- flat/rough/play ManagerBasedRLEnv와 480/495 observation 구현
- PPO train/resume/play/evaluate/export 명령 구현
- TorchScript/ONNX bundle, 100 golden vector와 C++ parity 구현
- project-local SDK2/MuJoCo/GLFW build와 loopback Sim2Sim 구현
- C++ state machine/watchdog/real-mode safety gate 구현
- 긴 stability/debug는 사용자 요청에 따라 후속 단계로 보류하고 1-step smoke를 기준으로 전환
