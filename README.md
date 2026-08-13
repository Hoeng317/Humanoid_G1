# Humanoid G1 generated artifacts

This branch stores generated training and evaluation artifacts for the
`Hoeng317/Humanoid_G1` implementation. Large binary files are managed with
Git LFS.

Included directories:

- `training/`: PPO checkpoints, exported ONNX policies, run configs, and logs
- `reports/`: retargeting and physics-evaluation reports
- `media/`: Isaac Sim validation videos
- `tensorboard/`: compact TensorBoard summaries
- `manifests/`: dataset inventory and split metadata

The licensed ACCAD/SMPL-X corpus and directly derived motion arrays are not
distributed in this branch. The omitted directories are `human/`, `g1/`,
`dynamic/`, `quarantine/`, and `contact_v29/`.

To restore the artifacts:

```bash
git clone --single-branch --branch artifacts git@github.com:Hoeng317/Humanoid_G1.git
cd Humanoid_G1
git lfs pull
```

