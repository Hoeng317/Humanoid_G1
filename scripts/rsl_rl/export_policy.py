#!/usr/bin/env python3
"""Offline checkpoint export using captured Isaac actor observations.

Isaac Sim is intentionally not started here: policy export is deterministic
and can run on a development/CI machine after observations were captured by
``g1.sh simulate --capture-observations ...``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic

from common import add_experiment_args, latest_checkpoint, prepare_experiment
from humanoid_g1.assets.joint_contract import load_joint_contract
from humanoid_g1.utils.paths import ARTIFACTS, ROOT
from humanoid_g1.utils.reproducibility import git_revision, sha256


class ExportedActor(torch.nn.Module):
    """Deployment module with normalization embedded in the graph."""

    def __init__(self, actor: torch.nn.Module, normalizer: torch.nn.Module):
        super().__init__()
        self.normalizer = normalizer
        self.actor = actor

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor(self.normalizer(observation))

    @torch.jit.export
    def reset(self) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    add_experiment_args(parser, checkpoint=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument(
        "--observations",
        type=Path,
        default=ARTIFACTS / "contracts" / "isaac_actor_observations.npz",
    )
    args = parser.parse_args()
    exp = prepare_experiment(args)
    checkpoint = args.checkpoint or latest_checkpoint(exp.name)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not args.observations.is_file():
        raise FileNotFoundError(
            f"{args.observations} is missing. Capture real Isaac observations first with:\n"
            "./humanoid_G1/g1.sh simulate --headless --steps 100 "
            "--capture-observations workspace/artifacts/contracts/isaac_actor_observations.npz"
        )
    captured = np.load(args.observations)["actor_observation"].astype(np.float32, copy=False)
    if captured.ndim != 2 or captured.shape[1] != 480 or captured.shape[0] < args.samples:
        raise ValueError(f"captured actor observations must have shape (N>=100, 480), got {captured.shape}")
    actor_observations = captured[: args.samples]
    training = exp.data["training"]
    policy_cfg = dict(training["policy"])
    observation_template = TensorDict(
        {"policy": torch.zeros(1, 480), "critic": torch.zeros(1, 495)},
        batch_size=[1],
    )
    policy = ActorCritic(
        observation_template,
        {"policy": ["policy"], "critic": ["critic"]},
        num_actions=29,
        **policy_cfg,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    policy.load_state_dict(payload["model_state_dict"], strict=True)
    policy.eval()
    module = ExportedActor(policy.actor, policy.actor_obs_normalizer).eval()
    run_id = args.run_id or checkpoint.parent.name
    output = ARTIFACTS / "policies" / run_id
    output.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.script(module)
    scripted.save(str(output / "policy.pt"))
    example = torch.from_numpy(actor_observations[:1])
    torch.onnx.export(
        module,
        example,
        str(output / "policy.onnx"),
        input_names=["actor_observation"],
        output_names=["joint_action"],
        dynamic_axes={"actor_observation": {0: "batch"}, "joint_action": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    with torch.inference_mode():
        expected = module(torch.from_numpy(actor_observations)).numpy()
        actual = scripted(torch.from_numpy(actor_observations)).numpy()
    max_error = float(np.max(np.abs(expected - actual)))
    if max_error > 1.0e-6:
        raise RuntimeError(f"TorchScript golden-vector error {max_error} exceeds 1e-6")
    np.savez_compressed(
        output / "golden_vectors.npz",
        actor_observation=actor_observations,
        expected_action=expected,
    )
    np.savetxt(
        output / "golden_vectors.csv",
        np.concatenate((actor_observations, expected), axis=1),
        delimiter=",",
        fmt="%.9g",
    )
    contract = load_joint_contract()
    contract.write_json(output / "joint_contract.json")
    shutil.copy2(ARTIFACTS / "contracts" / "actor_observation_schema.json", output)
    shutil.copy2(ROOT / "configs" / "deploy" / "controller.yaml", output)
    metadata = {
        "schema_version": 1,
        "input_dimension": 480,
        "output_dimension": 29,
        "recurrent": False,
        "hidden_state_shape": None,
        "observation_order": json.loads(
            (ARTIFACTS / "contracts" / "actor_observation_schema.json").read_text(encoding="utf-8")
        ),
        "observation_sample_source": str(args.observations),
        "normalization": (
            "rsl_rl_normalizer_embedded_in_exported_module"
            if policy_cfg["actor_obs_normalization"]
            else "identity_embedded_in_exported_module"
        ),
        "output_action_order": contract.policy_names,
        "policy_to_sdk": contract.policy_to_sdk,
        "action_scale": contract.action_scale,
        "policy_dt": contract.policy_dt,
        "training_checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "experiment_config": str(exp.source),
        "experiment_config_sha256": sha256(exp.source),
        "isaaclab_git": git_revision(ROOT.parent),
        "unitree_rl_lab_git": git_revision(ROOT / "third_party" / "unitree_rl_lab"),
        "golden_vector_count": int(args.samples),
        "golden_vector_max_abs_error": max_error,
    }
    (output / "policy_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    normalizer_state = {
        key: value.detach().cpu().tolist()
        for key, value in policy.actor_obs_normalizer.state_dict().items()
    }
    (output / "normalization.json").write_text(
        json.dumps(
            {
                "mode": "embedded_rsl_rl" if policy_cfg["actor_obs_normalization"] else "embedded_identity",
                "actor_obs_normalization": bool(policy_cfg["actor_obs_normalization"]),
                "state_dict": normalizer_state,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    files = sorted(path for path in output.iterdir() if path.name != "SHA256SUMS")
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in files if path.is_file()), encoding="utf-8"
    )
    print(f"Exported deployment bundle: {output}")
    print(f"Golden-vector count/error: {args.samples}/{max_error:.3e}")


if __name__ == "__main__":
    main()
