# PPO 알고리즘 설정

이 폴더는 로봇 환경이나 신경망 모양이 아니라 PPO 업데이트 방식을 조정합니다.

- `learning_rate`: optimizer 학습률
- `entropy_coef`: exploration 유지 강도
- `num_learning_epochs`, `num_mini_batches`: rollout 하나당 학습량
- `gamma`, `lam`: return/GAE 시간 가중치
- `clip_param`: PPO policy ratio 제한
- `desired_kl`, `schedule`: adaptive learning-rate 기준

먼저 `ppo_custom.yaml`에서 작은 변경 하나씩 실험하고 TensorBoard 학습 곡선을
비교하십시오.
