"""Convert policy actions into bounded Unitree SDK joint targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from humanoid_g1.assets.joint_contract import JointContract, load_joint_contract


@dataclass(frozen=True)
class ActionResult:
    raw_policy: np.ndarray
    clipped_policy: np.ndarray
    target_policy: np.ndarray
    target_sdk: np.ndarray
    saturated_count: int


class ActionAdapter:
    def __init__(
        self,
        contract: JointContract | None = None,
        action_clip: float = 1.0,
        delta_limit: float = 0.15,
        joint_margin: float = 0.05,
    ):
        self.contract = contract or load_joint_contract()
        self.action_clip = float(action_clip)
        self.delta_limit = float(delta_limit)
        self.joint_margin = float(joint_margin)
        self._previous = np.zeros(self.contract.action_dim, dtype=np.float32)

    def reset(self) -> None:
        self._previous.fill(0.0)

    def apply(self, action: Iterable[float]) -> ActionResult:
        raw = np.asarray(list(action), dtype=np.float32)
        self.contract.validate_vector(raw, "policy action")
        clipped = np.clip(raw, -self.action_clip, self.action_clip)
        clipped = np.clip(clipped, self._previous - self.delta_limit, self._previous + self.delta_limit)
        defaults = np.asarray([joint.default for joint in self.contract.joints], dtype=np.float32)
        lower = np.asarray([joint.lower + self.joint_margin for joint in self.contract.joints], dtype=np.float32)
        upper = np.asarray([joint.upper - self.joint_margin for joint in self.contract.joints], dtype=np.float32)
        target_policy = np.clip(defaults + self.contract.action_scale * clipped, lower, upper)
        target_sdk = np.asarray(self.contract.reorder_policy_to_sdk(target_policy), dtype=np.float32)
        saturated = int(np.count_nonzero(np.abs(raw) > self.action_clip))
        self._previous = clipped.copy()
        return ActionResult(raw, clipped, target_policy, target_sdk, saturated)

