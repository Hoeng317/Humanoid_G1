# Environment report

- Date: 2026-07-30 (Asia/Seoul)
- Host: Ubuntu 22.04.5, kernel 6.8.0-136
- CPU: Intel Core i7-14700K
- GPU: NVIDIA GeForce RTX 4090 24564 MiB
- Driver: 570.207
- Isaac Lab: 2.3.2, commit `2210934acca1a2f2401d541874163406b7ca8b53`
- Isaac Sim: 5.1.0 runtime
- Isaac Python: 3.11.13
- PyTorch/CUDA: 2.7.0+cu128
- RSL-RL: 3.1.2
- ONNX: 1.20.1
- MuJoCo: project-local C++ 3.3.6
- Unitree SDK2: project-local commit
  `21d0a3b2c46ee48c8fdf2783becb6be3beb0a59b`

Python `mujoco` package는 shared Isaac Python에 설치하지 않았습니다. Sim2Sim은
공식 C++ simulator를 `.local/`에 독립 빌드하여 사용하므로 기능상 필요하지
않습니다.
