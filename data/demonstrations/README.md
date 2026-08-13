# Demonstrations

Behavior cloning 또는 offline 학습용 trajectory를 둡니다. 권장 최소 필드:

```text
actor_observation: [N, 480]
action_policy: [N, 29]
episode_id: [N]
done: [N]
```

현재 online PPO 스크립트에는 demonstration loader가 연결되어 있지 않습니다.
