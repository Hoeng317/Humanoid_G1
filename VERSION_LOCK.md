# Version lock — 2026-07-30

- Isaac Lab: `2210934acca1a2f2401d541874163406b7ca8b53`, local version 2.3.2
- Isaac Sim: 5.1.0 runtime
- Python: 3.11.13 (Isaac Sim interpreter)
- PyTorch: 2.7.0+cu128
- RSL-RL: 3.1.2
- ONNX: 1.20.1
- GPU: NVIDIA RTX 4090 24 GB, driver 570.207
- MuJoCo C++: 3.3.6, archive SHA256
  `049204172901afad251070385a6badf46d795ebe47403d093f8469557eeeab5a`
- GLFW: commit `7b6aead9fb88b3623e3b3725ebb42670cbe4c579`
- SMPL-X Python: 0.1.28 (pipeline-local vendor)
- SMPL-X neutral locked-head model SHA256:
  `43d8f3a1375d7c5baae207870a5d51def0f7e6b507df709b4937598b5e7d965d`
- ACCAD 252 Stage-II + 20 Stage-I relative-path/file-hash aggregate SHA256:
  `1d7a5803fc106ae5bee0d05b74fc04457418c4cd931e1d53d54beea60cc5f8f6`

Official repositories are clean shallow clones:

|repository|branch|commit|license|
|---|---|---|---|
|unitree_rl_lab|main|`4960b84732b0c2ec593dccbfe963fda1bcd7b1e3`|Apache-2.0|
|unitree_ros|master|`f3772ce54c56ef2d34c6aee8100bc768896c7d19`|BSD-3-Clause|
|unitree_mujoco|main|`ae6a8403e272733e9996ef59990880330496177f`|BSD-3-Clause|
|unitree_sdk2|main|`21d0a3b2c46ee48c8fdf2783becb6be3beb0a59b`|BSD-3-Clause|
|unitree_sdk2_python|master|`65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5`|BSD-3-Clause|

Clone date: 2026-07-30. 원본 license는 각 repository에 보존됩니다.

기계 판독용 단일 기준은 `scripts/setup/dependency_lock.env`이며, 새 clone에서는
`scripts/setup/bootstrap.sh`가 이 commit들을 자동으로 복원·검증합니다.
