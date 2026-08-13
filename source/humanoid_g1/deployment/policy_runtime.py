"""Checksum-verified TorchScript policy runtime."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import time

import numpy as np
import torch


class PolicyError(RuntimeError):
    """The exported policy bundle or inference result violates its contract."""


def verify_bundle_checksums(bundle: Path) -> None:
    checksum_path = bundle / "SHA256SUMS"
    if not checksum_path.is_file():
        raise PolicyError(f"missing checksum manifest: {checksum_path}")
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected, filename = line.split(maxsplit=1)
        path = bundle / filename.strip()
        if not path.is_file():
            raise PolicyError(f"missing policy bundle file: {path.name}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise PolicyError(f"checksum mismatch: {path.name}")


class PolicyRuntime:
    def __init__(self, policy_path: Path | str, verify_checksums: bool = True):
        self.path = Path(policy_path).resolve()
        self.bundle = self.path.parent
        if verify_checksums:
            verify_bundle_checksums(self.bundle)
        metadata_path = self.bundle / "policy_metadata.json"
        if not metadata_path.is_file():
            raise PolicyError("policy_metadata.json is required")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("input_dimension") != 480 or self.metadata.get("output_dimension") != 29:
            raise PolicyError("policy dimensions do not match G1 deployment contract")
        self.module = torch.jit.load(str(self.path), map_location="cpu").eval()
        # Compile/cache kernels before the real-time loop; warm-up latency is not
        # allowed to consume the 10 ms inference budget.
        with torch.inference_mode():
            self.module(torch.zeros(1, 480, dtype=torch.float32))

    def infer(self, observation: np.ndarray) -> tuple[np.ndarray, float]:
        value = np.asarray(observation, dtype=np.float32)
        if value.shape != (480,) or not np.isfinite(value).all():
            raise PolicyError("policy observation must be finite shape (480,)")
        start = time.perf_counter()
        with torch.inference_mode():
            output = self.module(torch.from_numpy(value).unsqueeze(0)).squeeze(0).cpu().numpy()
        elapsed = time.perf_counter() - start
        if output.shape != (29,) or not np.isfinite(output).all():
            raise PolicyError("policy returned invalid action")
        return output, elapsed

    def reset(self) -> None:
        if hasattr(self.module, "reset"):
            self.module.reset()
