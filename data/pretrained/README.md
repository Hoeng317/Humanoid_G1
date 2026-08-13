# Pretrained checkpoints

동일한 G1 29-DoF joint order, actor 480, critic 495, action 29와 동일한 hidden
dimension을 가진 `model_N.pt`만 바로 resume할 수 있습니다.

```bash
./humanoid_G1/g1.sh train \
  --config configs/experiments/g1_custom_ppo.yaml \
  --resume humanoid_G1/data/pretrained/model_N.pt --headless
```
