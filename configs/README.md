# 설정 폴더 안내

학습 실행 하나는 `experiments/*.yaml`에서 시작하며 아래 설정을 조합합니다.

|폴더|수정하는 내용|
|---|---|
|`experiments/`|실험 이름, task, command 범위, reward weight, 전체 preset 조합|
|`policy/`|Actor/Critic 은닉층, activation, observation normalization|
|`algorithm/`|PPO learning rate, entropy, epoch, mini-batch, gamma/lambda|
|`train/`|rollout step, iteration, checkpoint 저장 주기, policy/algorithm 선택|
|`robot/`|29개 관절 순서·limit·기본 자세·kp/kd와 hardware profile|
|`sim/`|physics dt, decimation, episode 길이, device|
|`terrain/`|평지/거친 지형/계단 preset|
|`randomization/`|마찰, 질량, gain, noise, push 등 domain randomization|
|`deploy/`|실기/Sim2Sim 제어 주기와 안전 제한|

처음 실험할 때는 baseline을 직접 바꾸지 말고 다음 사용자 파일을 수정합니다.

- `experiments/g1_custom_ppo.yaml`
- `policy/mlp_custom.yaml`
- `algorithm/ppo_custom.yaml`
- `train/ppo_custom.yaml`

정책 구조와 PPO 설정은 `train/*.yaml`에서 별도 YAML로 참조됩니다. 예전처럼
`policy:`와 `algorithm:`을 inline mapping으로 작성하는 형식도 계속 지원합니다.
