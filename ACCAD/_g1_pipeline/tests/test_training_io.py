from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from accad_g1 import training_io


class _DummyNormalizer(torch.nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.register_buffer("_mean", torch.zeros(dimension))
        self.register_buffer("_var", torch.ones(dimension))
        self.register_buffer("_std", torch.ones(dimension))
        self.register_buffer("count", torch.zeros(()))


class _DummyPolicy(torch.nn.Module):
    def __init__(self, actor_dimension: int = 3, critic_dimension: int = 5):
        super().__init__()
        self.actor_obs_normalizer = _DummyNormalizer(actor_dimension)
        self.critic_obs_normalizer = _DummyNormalizer(critic_dimension)
        self.actor = torch.nn.Linear(actor_dimension, 2)
        self.critic = torch.nn.Linear(critic_dimension, 1)
        self.log_std = torch.nn.Parameter(torch.zeros(2))


def _filled_policy() -> _DummyPolicy:
    policy = _DummyPolicy()
    with torch.no_grad():
        for index, tensor in enumerate(policy.state_dict().values(), start=1):
            tensor.fill_(float(index) / 10.0)
    return policy


def _write_policy_checkpoint(
    root: Path,
    *,
    iteration: int = 12,
    policy: _DummyPolicy | None = None,
    model_state_dict: dict[str, torch.Tensor] | None = None,
) -> Path:
    checkpoint = root / "runs" / "source" / f"model_{iteration}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    state = (policy or _filled_policy()).state_dict()
    if model_state_dict is not None:
        state = model_state_dict
    torch.save(
        {
            "model_state_dict": state,
            "optimizer_state_dict": {"sentinel": torch.tensor(123.0)},
            "iter": iteration,
            "infos": None,
        },
        checkpoint,
    )
    return checkpoint


def _fresh_runner(policy: _DummyPolicy | None = None) -> SimpleNamespace:
    target = policy or _DummyPolicy()
    optimizer = torch.optim.Adam(target.parameters(), lr=1.0e-4)
    return SimpleNamespace(
        current_learning_iteration=0,
        tot_timesteps=0,
        tot_time=0.0,
        alg=SimpleNamespace(policy=target, optimizer=optimizer),
    )


def test_path_below_rejects_escape(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    inside = root / "inside.txt"
    inside.write_text("ok", encoding="utf-8")

    assert training_io.path_below(inside, root, must_exist=True) == inside.resolve()
    with pytest.raises(ValueError, match="must remain below"):
        training_io.path_below(tmp_path / "outside.txt", root)


def test_direct_motion_resolution_is_unique_sorted_and_limited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "ACCAD"
    references = root / "references"
    references.mkdir(parents=True)
    first = references / "a_g1_50hz.npz"
    second = references / "z_g1_50hz.npz"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    monkeypatch.setattr(training_io, "ACCAD_ROOT", root)

    selection = training_io.resolve_motion_inputs(
        motions=[second, first, second],
        manifests=[],
        split="train",
        reference_root=references,
        allow_missing=False,
        max_motions=1,
    )

    assert selection.paths == (first.resolve(),)
    assert selection.manifests == ()
    assert selection.metadata()["motion_count"] == 1
    assert selection.metadata()["manifest_checksum_verified_motion_count"] == 0
    assert selection.metadata()["manifest_promotion_verified"] is False


def test_manifest_resolution_merges_inputs_and_enforces_archive_checksum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "ACCAD"
    references = root / "references"
    references.mkdir(parents=True)
    motion = references / "walk_g1_50hz.npz"
    motion.write_bytes(b"retargeted-motion")
    checksum = hashlib.sha256(motion.read_bytes()).hexdigest()
    first_manifest = root / "first.json"
    second_manifest = root / "second.json"
    record = {"g1_file": motion.name, "archive_checksum_sha256": checksum}
    first_manifest.write_text(json.dumps({"motions": [record]}), encoding="utf-8")
    second_manifest.write_text(json.dumps({"motions": [record]}), encoding="utf-8")
    monkeypatch.setattr(training_io, "ACCAD_ROOT", root)

    selection = training_io.resolve_motion_inputs(
        motions=[],
        manifests=[first_manifest, second_manifest],
        split="train",
        reference_root=references,
        allow_missing=False,
        max_motions=None,
    )
    assert selection.paths == (motion.resolve(),)
    assert len(selection.manifests) == 2
    assert selection.metadata()["manifest_checksum_verified_motion_count"] == 1
    assert selection.metadata()["manifest_promotion_verified"] is True

    document = json.loads(second_manifest.read_text(encoding="utf-8"))
    document["motions"][0]["archive_checksum_sha256"] = "0" * 64
    second_manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="disagree on checksum"):
        training_io.resolve_motion_inputs(
            motions=[],
            manifests=[first_manifest, second_manifest],
            split="train",
            reference_root=references,
            allow_missing=False,
            max_motions=None,
        )


def test_artifacts_and_atomic_json_remain_under_training_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    training_root = tmp_path / "training"
    monkeypatch.setattr(training_io, "TRAINING_ROOT", training_root)
    monkeypatch.setattr(training_io, "RUNS_ROOT", training_root / "runs")
    monkeypatch.setattr(training_io, "EVALUATIONS_ROOT", training_root / "evaluations")

    timestamp = datetime(2026, 1, 2, 3, 4, 5)
    run_dir = training_io.new_artifact_dir("runs", "unsafe / run", timestamp=timestamp)
    report = training_io.write_json_atomic(run_dir / "report.json", {"status": "ok"})

    assert run_dir.name == "2026-01-02_03-04-05_unsafe_run"
    assert report.relative_to(training_root)
    assert json.loads(report.read_text(encoding="utf-8")) == {"status": "ok"}


def test_latest_checkpoint_selection_is_local_and_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    training_root = tmp_path / "training"
    runs_root = training_root / "runs"
    older = runs_root / "old" / "model_9.pt"
    newer = runs_root / "new" / "model_1.pt"
    ignored = runs_root / "new" / "policy.pt"
    for path in (older, newer, ignored):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode())
    older.touch()
    newer.touch()
    # Use explicit nanosecond times so filesystem timestamp resolution cannot
    # make the ordering flaky.
    older_ns = 1_700_000_000_000_000_000
    newer_ns = older_ns + 1_000_000
    import os

    os.utime(older, ns=(older_ns, older_ns))
    os.utime(newer, ns=(newer_ns, newer_ns))
    monkeypatch.setattr(training_io, "TRAINING_ROOT", training_root)
    monkeypatch.setattr(training_io, "RUNS_ROOT", runs_root)

    assert training_io.checkpoint_candidates() == (older.resolve(), newer.resolve())
    assert training_io.resolve_checkpoint(None, latest_if_missing=True) == newer.resolve()


def test_checkpoint_provenance_binds_completed_run_and_dynamics_contract(
    tmp_path: Path,
) -> None:
    run = tmp_path / "runs" / "v6"
    run.mkdir(parents=True)
    checkpoint = run / "model_12.pt"
    checkpoint.write_bytes(b"checkpoint")
    contract = "tracking-v6"
    objective = "objective-v2"
    runtime = {
        "tracking_policy_contract_version": contract,
        "training_objective": {"schema_version": objective},
    }
    summary = {
        "status": "completed",
        "runtime_contract": {
            "tracking_policy_contract_version": contract,
            "training_objective": {"schema_version": objective},
        },
    }
    (run / "runtime_contract.json").write_text(
        json.dumps(runtime), encoding="utf-8"
    )
    (run / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    provenance = training_io.checkpoint_training_contract_provenance(
        checkpoint,
        expected_policy_contract_version=contract,
        expected_training_objective_schema=objective,
    )
    assert provenance["verified"] is True
    assert provenance["compatible"] is True
    assert provenance["reason"] is None
    assert provenance["runtime_contract"]["policy_contract_version"] == contract
    assert provenance["runtime_contract"]["training_objective_schema"] == objective
    assert provenance["run_summary"]["status"] == "completed"

    incompatible = training_io.checkpoint_training_contract_provenance(
        checkpoint,
        expected_policy_contract_version="tracking-v7",
        expected_training_objective_schema=objective,
    )
    assert incompatible["verified"] is True
    assert incompatible["compatible"] is False
    assert incompatible["reason"] == (
        "checkpoint_trained_under_different_policy_contract"
    )

    incompatible_objective = training_io.checkpoint_training_contract_provenance(
        checkpoint,
        expected_policy_contract_version=contract,
        expected_training_objective_schema="objective-v3",
    )
    assert incompatible_objective["verified"] is True
    assert incompatible_objective["compatible"] is False
    assert incompatible_objective["reason"] == (
        "checkpoint_trained_under_different_training_objective"
    )


def test_checkpoint_provenance_reports_missing_parent_metadata(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model_3.pt"
    checkpoint.write_bytes(b"checkpoint")
    provenance = training_io.checkpoint_training_contract_provenance(
        checkpoint,
        expected_policy_contract_version="tracking-v6",
    )
    assert provenance["verified"] is False
    assert provenance["compatible"] is False
    assert provenance["reason"] == (
        "missing_parent_run_metadata:runtime_contract.json,run_summary.json"
    )


def test_training_checkpoint_cli_modes_are_mutually_exclusive():
    parser = argparse.ArgumentParser()
    training_io.add_training_checkpoint_args(parser)

    assert parser.parse_args([]).resume is False
    assert parser.parse_args(["--resume"]).resume is True
    assert parser.parse_args(["--checkpoint", "model_1.pt"]).checkpoint == "model_1.pt"
    assert (
        parser.parse_args(["--initialize-policy-from", "model_2.pt"]).initialize_policy_from
        == "model_2.pt"
    )
    for arguments in (
        ["--resume", "--checkpoint", "model_1.pt"],
        ["--resume", "--initialize-policy-from", "model_1.pt"],
        [
            "--checkpoint",
            "model_1.pt",
            "--initialize-policy-from",
            "model_2.pt",
        ],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(arguments)


def test_policy_initialization_checkpoint_must_be_accad_internal_model_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "ACCAD"
    root.mkdir()
    valid = _write_policy_checkpoint(root)
    wrong_name = root / "policy.pt"
    wrong_name.write_bytes(b"not-used")
    outside = tmp_path / "model_12.pt"
    outside.write_bytes(b"not-used")
    directory = root / "model_2.pt"
    directory.mkdir()
    monkeypatch.setattr(training_io, "ACCAD_ROOT", root)

    assert training_io.resolve_policy_initialization_checkpoint(valid) == valid.resolve()
    with pytest.raises(ValueError, match="model_<iteration>"):
        training_io.resolve_policy_initialization_checkpoint(wrong_name)
    with pytest.raises(ValueError, match="must remain below"):
        training_io.resolve_policy_initialization_checkpoint(outside)
    with pytest.raises(ValueError, match="model_<iteration>"):
        training_io.resolve_policy_initialization_checkpoint(directory)


def test_policy_only_initialization_loads_normalizers_and_keeps_runner_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "ACCAD"
    root.mkdir()
    source_policy = _filled_policy()
    checkpoint = _write_policy_checkpoint(root, policy=source_policy)
    monkeypatch.setattr(training_io, "ACCAD_ROOT", root)

    source = training_io.load_policy_initialization_source(checkpoint)
    assert source is not None
    runner = _fresh_runner()
    policy_identity = id(runner.alg.policy)
    optimizer_identity = id(runner.alg.optimizer)
    witness = training_io.initialize_runner_policy_from_source(runner, source)

    assert source.metadata()["path"] == str(checkpoint.resolve())
    assert source.metadata()["checksum_sha256"] == training_io.sha256_file(checkpoint)
    assert source.metadata()["bytes"] == checkpoint.stat().st_size
    assert source.metadata()["iter"] == 12
    assert source.metadata()["iteration"] == 12
    assert source.metadata()["filename_iteration"] == 12
    assert source.metadata()["checkpoint_optimizer_state_present"] is True
    assert witness["mode"] == "strict_model_state_dict_with_fresh_optimizer"
    assert witness["load_contract"]["strict"] is True
    assert witness["load_contract"]["optimizer_state_loaded"] is False
    assert witness["load_contract"]["missing_keys"] == []
    assert witness["load_contract"]["unexpected_keys"] == []
    assert witness["load_contract"]["all_loaded_tensors_exactly_equal"] is True
    assert witness["fresh_runner_state"]["before"]["current_learning_iteration"] == 0
    assert witness["fresh_runner_state"]["after"]["current_learning_iteration"] == 0
    assert witness["fresh_runner_state"]["before"]["tot_timesteps"] == 0
    assert witness["fresh_runner_state"]["after"]["tot_timesteps"] == 0
    assert witness["fresh_runner_state"]["after"]["optimizer_state_entry_count"] == 0
    assert id(runner.alg.policy) == policy_identity
    assert id(runner.alg.optimizer) == optimizer_identity
    assert len(runner.alg.optimizer.state) == 0
    for key, expected in source_policy.state_dict().items():
        assert torch.equal(runner.alg.policy.state_dict()[key], expected)
    for normalizer_key in training_io._NORMALIZER_STATE_KEYS:
        assert normalizer_key in witness["load_contract"]["normalizer_state_keys_loaded"]


def test_policy_initialization_rejects_checkpoint_payload_contract_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "ACCAD"
    root.mkdir()
    monkeypatch.setattr(training_io, "ACCAD_ROOT", root)

    missing_model = root / "model_3.pt"
    torch.save({"iter": 3}, missing_model)
    with pytest.raises(ValueError, match="missing model_state_dict"):
        training_io.load_policy_initialization_source(missing_model)

    missing_normalizer_state = dict(_filled_policy().state_dict())
    missing_normalizer_state.pop("actor_obs_normalizer._mean")
    missing_normalizer = _write_policy_checkpoint(
        root, iteration=4, model_state_dict=missing_normalizer_state
    )
    with pytest.raises(ValueError, match="missing actor/critic normalizer"):
        training_io.load_policy_initialization_source(missing_normalizer)

    mismatched_iteration = _write_policy_checkpoint(root, iteration=5)
    payload = torch.load(mismatched_iteration, map_location="cpu", weights_only=True)
    payload["iter"] = 6
    torch.save(payload, mismatched_iteration)
    with pytest.raises(ValueError, match="filename/payload iteration mismatch"):
        training_io.load_policy_initialization_source(mismatched_iteration)


def test_policy_initialization_rejects_architecture_mismatch_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "ACCAD"
    root.mkdir()
    checkpoint = _write_policy_checkpoint(root)
    monkeypatch.setattr(training_io, "ACCAD_ROOT", root)
    source = training_io.load_policy_initialization_source(checkpoint)
    assert source is not None
    runner = _fresh_runner(_DummyPolicy(actor_dimension=4))
    original = {
        key: value.clone() for key, value in runner.alg.policy.state_dict().items()
    }

    with pytest.raises(ValueError, match="tensor shape mismatch"):
        training_io.initialize_runner_policy_from_source(runner, source)
    for key, expected in original.items():
        assert torch.equal(runner.alg.policy.state_dict()[key], expected)
    assert len(runner.alg.optimizer.state) == 0


@pytest.mark.parametrize(
    ("iteration", "timesteps", "dirty_optimizer", "message"),
    (
        (1, 0, False, "iteration 0 before"),
        (0, 32, False, "0 timesteps before"),
        (0, 0, True, "optimizer must be fresh before"),
    ),
)
def test_policy_initialization_requires_a_fresh_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    iteration: int,
    timesteps: int,
    dirty_optimizer: bool,
    message: str,
):
    root = tmp_path / "ACCAD"
    root.mkdir()
    checkpoint = _write_policy_checkpoint(root)
    monkeypatch.setattr(training_io, "ACCAD_ROOT", root)
    source = training_io.load_policy_initialization_source(checkpoint)
    assert source is not None
    runner = _fresh_runner()
    runner.current_learning_iteration = iteration
    runner.tot_timesteps = timesteps
    if dirty_optimizer:
        parameter = next(iter(runner.alg.policy.parameters()))
        runner.alg.optimizer.state[parameter]["sentinel"] = torch.tensor(1.0)

    with pytest.raises(ValueError, match=message):
        training_io.initialize_runner_policy_from_source(runner, source)
