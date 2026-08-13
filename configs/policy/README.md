# Policy network 설정

현재 정책 구현체는 설치된 RSL-RL의 `ActorCritic` MLP입니다. 이 폴더는 네트워크
크기와 activation을 바꾸는 곳입니다.

- `actor_hidden_dims`: 실기에도 export되는 actor MLP
- `critic_hidden_dims`: 학습 중에만 사용하는 critic MLP
- `activation`: `elu`, `relu`, `tanh`, `gelu` 등
- `init_noise_std`: PPO 초기 exploration 크기
- `*_obs_normalization`: observation running normalization 여부

hidden dimension이나 observation 차원을 바꾸면 기존 checkpoint는 호환되지
않습니다. 새 experiment 이름으로 처음부터 학습하고 정책을 다시 export해야 합니다.
