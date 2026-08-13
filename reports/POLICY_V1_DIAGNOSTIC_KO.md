# 선행 254-D PPO 정책(v1) 실패 진단

이 문서는 최종 recovery 정책과 혼동하지 않도록, 최초 5,000-update
실험을 별도 음성 결과(negative result)로 보존한다. 모든 수치는 저장된
run/evaluation JSON을 다시 읽어 확인했다.

## 1. 완료된 학습 계약

- run: `work/training/runs/2026-08-08_22-28-06_accad_all_v27_seed42`
- run summary SHA-256:
  `b3e0b63d2142a91ce25883ee5b664c366d7f8092ff0876ae6fe45c6c963a5cfd`
- 시간: 2026-08-08 13:28:06.700598–16:57:26.171442 UTC
  (3 h 29 m 19.470844 s)
- actor/critic/action: 254/472/29
- physics/control: 200/50 Hz
- seed: 42
- 용량: 2,048 env × 32 step × 5,000 update = 327,680,000 transition
- train motion: 215/215 sampled, 총 4,764,368 episode reset,
  motion별 21,814–22,598회, 미샘플 0
- PPO: initial action noise std 0.5, entropy coefficient 0.005,
  wrapper action clip ±1
- 최종 checkpoint SHA-256:
  `608e880df410fbc9b71032f3304029dde0e50ed73bb8263bb595abd4c95c7198`
- TorchScript SHA-256:
  `e0565d6494dd010ab3ecc6726156cb6166bd26c2d65a9b529836ba849d9b87b3`
- ONNX SHA-256:
  `e6638e3af31dda63bad1633b97cca3e1c54ca3d92c166d505d70dfa2da97b68c`

두 export는 모두 254→29 실행/형식 검사를 통과했다. 즉 export 성공은
동역학 추종 성공과 별개이다.

## 2. 결정론적 validation 결과

- report:
  `work/training/evaluations/2026-08-09_01-58-46_accad_validation_model4999/suite_evaluation_summary.json`
- report SHA-256:
  `855b09c88db5e1b36e7e83b91f41d1135ca2b81d14fd29421e7c0a4e765f15e8`
- 계약: validation 5개 모션에 각 1 env, frame 0, deterministic policy,
  noise/domain randomization 없음, 첫 episode만 측정
- clean motion-end: **0/5**
- 평균 completion: 0.2470935274 (기준 ≥ 0.90)
- 종료: fall 2, tracking divergence 3, joint limit 0, nonfinite 0
- joint MAE: 0.1239499754 rad
- root position error: 0.1436317344 m
- tracking-body position error: 0.1555356987 m
- pooled contact F1: 0.8978260870
- weighted raw-action clip fraction: 0.5344827586
- stance/contact slide-speed p95: 1.1215993643 m/s

개별 completion은 EricCamper04 141/807, QkWalk1 54/128, Run1 29/91,
Sprint1 32/111, walkdog 26/815였다. 따라서 위 오차 일부가 기준을
통과한 것은 전체 clip 성공이 아니라 조기 종료 전 짧은 prefix의 국소
추종 결과이다.

## 3. checkpoint sweep

저장된 0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500,
4999 checkpoint를 같은 validation 계약으로 전부 평가했다. 모두 0/5였고
평균 completion의 최댓값은 model 500의 0.255192602였다. raw-action clip
fraction은 model 0의 0에서 model 4999의 0.534482759까지 증가했다.

따라서 최종 checkpoint 선택이나 단순 과학습만으로 설명할 수 없고,
학습 계약 자체를 복구해야 한다.

## 4. 복구 근거와 데이터 누출 방지

확정된 관측 사실은 다음과 같다.

1. v1 actor에는 기준 대비 pelvis translation/linear-velocity 오차가 없고
   해당 정보는 critic에만 있었다.
2. train은 연속 random start였지만 validation은 frame 0 full clip이었다.
3. entropy 0.005에서 action noise std가 약 0.50→1.83으로 증가했고,
   raw action 포화도 함께 악화됐다.
4. wrapper가 먼저 ±1로 잘라 action term이 원시 초과량을 볼 수 없었다.

이들은 구조적 실패의 강한 원인 가설이지만 각각의 단독 인과를
주장하지 않는다. v1은 validation에서 거부됐으므로 held-out test는
실행하지 않았다. 최종 recovery 정책이 validation을 통과하기 전 test를
소모하지 않는 것이 본 프로젝트의 데이터 누출 방지 절차이다.
