# Training

기본 학습기는 RSL-RL `OnPolicyRunner` + PPO입니다. Actor는 실물에서 직접 얻을
수 있는 480차원만 사용하고 critic은 privileged base linear velocity를 포함한
495차원을 사용합니다.

설정은 정책 구조 `configs/policy/`, PPO update `configs/algorithm/`, rollout 실행
`configs/train/`, 최상위 실험 `configs/experiments/`로 분리되어 있습니다. 처음
수정할 때는 `g1_custom_ppo.yaml`과 각 `*_custom.yaml`을 사용하십시오.

Preset:

- `g1_smoke.yaml`: 32 env, 2 iterations
- `g1_flat_ppo.yaml`: 평지 baseline, 기본 4096 env/1500 iterations
- `g1_rough_ppo.yaml`: mixed rough terrain, 기본 4096 env/3000 iterations
- `g1_sim2real_ppo.yaml`: rough curriculum + sim2real randomization

```bash
./humanoid_G1/g1.sh train --config configs/experiments/g1_smoke.yaml --headless
./humanoid_G1/g1.sh train --config configs/experiments/g1_flat_ppo.yaml \
  --num-envs 2048 --max-iterations 1500 --seed 42 --headless
```

RTX 4090에서도 다른 프로세스가 VRAM을 사용하면 환경 수를 2048 또는 1024로
낮춥니다. TensorBoard event, resolved config, agent config, manifest와
`model_N.pt`는 `workspace/logs/rsl_rl/<experiment>/<timestamp_run-name>/`에 저장됩니다.
RSL-RL checkpoint에는 policy, optimizer, iteration 및 normalizer state가
포함됩니다.

Resume:

```bash
./humanoid_G1/g1.sh train --config configs/experiments/g1_flat_ppo.yaml \
  --resume /absolute/path/model_100.pt --headless
```

구조적 observation/action/hidden-dimension 변경 뒤 기존 checkpoint resume는
허용하지 마십시오. 새 experiment 이름으로 시작하고 export contract도 다시
생성해야 합니다.
