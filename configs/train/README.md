# 학습 실행 설정

이 폴더는 rollout 수집과 학습 실행 길이를 설정하고 `policy/`, `algorithm/`을
선택합니다.

- `num_steps_per_env`: 한 PPO update 전에 환경별로 수집할 step 수
- `max_iterations`: PPO update 횟수
- `save_interval`: `model_N.pt` 저장 주기
- `clip_actions`: Isaac 환경에 전달하기 전 action clip
- `policy`, `algorithm`: 각각 별도 YAML 참조

`ppo_smoke.yaml`은 코드 경로 확인 전용이며 보행을 학습할 만큼 길지 않습니다.
