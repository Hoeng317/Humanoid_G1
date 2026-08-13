"""Fail-closed v29 contact, centroidal-ZMP, and capture-point audit.

The module is an immutable downstream consumer of the promoted v27 and v28
archives.  It does not rewrite either archive.  A train invocation freezes the
algorithm and exact URDF provenance; validation (and a later explicit test
invocation) must use that policy byte-for-byte.

Only the Python standard library and NumPy are used here.  In particular, the
URDF inertial model, fixed-link transforms, quaternion rotations, convex hull,
and finite differences do not depend on SciPy or Pinocchio.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from .paths import PIPELINE_ROOT, PROJECT_ROOT, assert_output_is_local


POLICY_SCHEMA_VERSION = "accad_g1_contact_feasibility_policy_v1"
CLIP_AUDIT_SCHEMA_VERSION = "accad_g1_contact_feasibility_clip_audit_v1"
PLAN_SCHEMA_VERSION = "accad_g1_contact_feasibility_plan_v1"
REPORT_SCHEMA_VERSION = "accad_g1_contact_feasibility_report_v1"
MANIFEST_SCHEMA_VERSION = "accad_g1_contact_feasibility_manifest_v1"
CONTACT_GATE_VERSION = "g1-contact-capture-v29-centroidal-zmp-preview-v1"

V27_MANIFEST_SCHEMA_VERSION = "accad_g1_gate3_manifest_v1"
V28_MANIFEST_SCHEMA_VERSION = "accad_g1_dynamic_manifest_v1"

G1_URDF_PATH = (
    PROJECT_ROOT
    / "third_party"
    / "unitree_ros"
    / "robots"
    / "g1_description"
    / "g1_29dof_rev_1_0.urdf"
)
DEFAULT_DYNAMIC_POLICY_PATH = PIPELINE_ROOT / "work" / "dynamic" / "scale_policy.json"
DEFAULT_V27_MANIFESTS = {
    split: PIPELINE_ROOT / "work" / "manifests" / f"g1_{split}.json"
    for split in ("train", "validation", "test")
}
DEFAULT_V28_MANIFESTS = {
    split: PIPELINE_ROOT / "work" / "dynamic" / f"g1_{split}.json"
    for split in ("train", "validation", "test")
}
DEFAULT_OUTPUT_ROOT = PIPELINE_ROOT / "work" / "contact_v29"
IMPLEMENTATION_PATH = Path(__file__).resolve()

CONTACT_FIELDS = (
    "left_heel_contact",
    "left_toe_contact",
    "right_heel_contact",
    "right_toe_contact",
)
FOOT_CONTACT_FIELDS = ("left_foot_contact", "right_foot_contact")
FOOT_LINKS = ("left_ankle_roll_link", "right_ankle_roll_link")
FOOT_SUPPORT_NAMES = (
    "left_rear_outer",
    "left_rear_inner",
    "left_front_outer",
    "left_front_inner",
    "right_rear_inner",
    "right_rear_outer",
    "right_front_inner",
    "right_front_outer",
)

# Every threshold is tied to an existing asset/control contract.  The absolute
# spatial tolerance is resolved at runtime as support sphere radius plus one
# accepted semantic-contact step (0.60 m/s / 50 Hz).  The degradation tolerance
# omits the common sphere radius and permits exactly that one contact step.
POLICY_ALGORITHM: dict[str, Any] = {
    "schema_version": "accad_g1_contact_feasibility_formula_v1",
    "fps": 50.0,
    "gravity_mps2": 9.81,
    "derivative": {
        "method": "five_frame_quadratic_least_squares",
        "velocity_numerator_coefficients": [-2.0, -1.0, 0.0, 1.0, 2.0],
        "velocity_denominator_dt_multiple": 10.0,
        "acceleration_numerator_coefficients": [2.0, -1.0, -2.0, -1.0, 2.0],
        "acceleration_denominator_dt2_multiple": 7.0,
        "boundary_guard_frames": 2,
    },
    "support": {
        "contact_fields": list(CONTACT_FIELDS),
        "optimistic_gate_polygon": "all_four_sphere_centers_of_each_contacting_foot",
        "strict_diagnostic_polygon": "active_heel_or_toe_sphere_center_pairs",
        "plane_height": "median(active_support_center_z_minus_urdf_sphere_radius)",
    },
    "phase": {
        "minimum_frames": 7,
        "bad_run_frames": 3,
        "single_support": "exactly_one_foot_has_any_heel_or_toe_contact",
    },
    "thresholds": {
        "semantic_contact_speed_budget_mps": 0.60,
        "absolute_margin_tolerance": (
            "maximum_urdf_support_sphere_radius + "
            "semantic_contact_speed_budget_mps/fps"
        ),
        "retime_degradation_tolerance": "semantic_contact_speed_budget_mps/fps",
        "normal_force": "strictly_positive",
    },
    "capture_preview": {
        "capture_point": "com_xy + com_velocity_xy/sqrt(gravity/com_height)",
        "horizon_natural_time_constants": 3.0,
        "corridor": (
            "convex_hull(current_support_foot, first_opposite_foot_touchdown_"
            "within_horizon); otherwise current_support_foot"
        ),
    },
    "centroidal_zmp": {
        "force": "total_mass*(com_acceleration + [0,0,gravity])",
        "x": "com_x-(height*force_x+centroidal_momentum_rate_y)/force_z",
        "y": "com_y-(height*force_y-centroidal_momentum_rate_x)/force_z",
        "includes_urdf_link_inertia_and_fixed_child_inertials": True,
    },
    "comparison": {
        "source_coordinate": "candidate_frame/realized_uniform_scale",
        "interpolation": "linear_only_inside_equal_contact_state",
        "criteria": [
            "candidate_centroidal_zmp_absolute_margin",
            "candidate_positive_normal_force",
            "candidate_preview_capture_absolute_margin",
            "v27_to_v28_centroidal_zmp_margin_degradation",
            "v27_to_v28_preview_capture_margin_degradation",
        ],
    },
}


class ContactFeasibilityError(RuntimeError):
    """Raised when the v29 audit cannot make a safe deterministic decision."""


@dataclass(frozen=True)
class _Joint:
    kind: str
    parent: str
    child: str
    transform_parent_child: np.ndarray


@dataclass(frozen=True)
class _RawInertial:
    link: str
    mass: float
    com_in_link: np.ndarray
    rotation_link_inertial: np.ndarray
    inertia_in_inertial: np.ndarray


@dataclass(frozen=True)
class UrdfAsset:
    checksum_sha256: str
    raw_inertials: tuple[_RawInertial, ...]
    joints_by_child: Mapping[str, _Joint]
    support_centers: np.ndarray
    support_radii: np.ndarray
    total_mass_kg: float


@dataclass(frozen=True)
class ResolvedInertialModel:
    body_names: tuple[str, ...]
    anchor_indices: np.ndarray
    masses: np.ndarray
    com_offsets_in_anchor: np.ndarray
    rotations_anchor_inertial: np.ndarray
    inertias_in_inertial: np.ndarray
    source_links: tuple[str, ...]
    anchor_links: tuple[str, ...]
    support_radius_maximum_m: float
    total_mass_kg: float
    urdf_checksum_sha256: str


@dataclass(frozen=True)
class ClipSignals:
    frame_count: int
    fps: float
    contact_state: np.ndarray
    eligible_support: np.ndarray
    eligible_single_support: np.ndarray
    phase_records: tuple[dict[str, Any], ...]
    com_position: np.ndarray
    com_velocity: np.ndarray
    com_acceleration: np.ndarray
    centroidal_momentum: np.ndarray
    centroidal_momentum_rate: np.ndarray
    normal_force_z: np.ndarray
    centroidal_zmp: np.ndarray
    lipm_zmp: np.ndarray
    zmp_margin_optimistic: np.ndarray
    zmp_margin_strict: np.ndarray
    capture_point: np.ndarray
    capture_margin_optimistic: np.ndarray
    capture_preview_available: np.ndarray
    com_height: np.ndarray


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_checksum(document: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    output = assert_output_is_local(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".json", dir=output.parent, delete=False
        ) as temporary:
            temporary_name = temporary.name
            json.dump(
                document,
                temporary,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, output)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _safe_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
    except (OSError, ValueError) as exc:
        raise ContactFeasibilityError(f"cannot safely load archive {path}: {exc}") from exc
    objects = sorted(name for name, value in arrays.items() if value.dtype.hasobject)
    if objects:
        raise ContactFeasibilityError(f"object arrays are forbidden in {path}: {objects}")
    return arrays


def _scalar(arrays: Mapping[str, np.ndarray], name: str) -> Any:
    try:
        value = arrays[name]
    except KeyError as exc:
        raise ContactFeasibilityError(f"archive is missing scalar {name!r}") from exc
    if value.shape != () or value.dtype.hasobject:
        raise ContactFeasibilityError(
            f"{name} must be a safe scalar; got shape={value.shape}, dtype={value.dtype}"
        )
    return value.item()


def _xyz(text: str | None) -> np.ndarray:
    return np.asarray(
        [float(value) for value in text.split()] if text else (0.0, 0.0, 0.0),
        dtype=np.float64,
    )


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray(((1, 0, 0), (0, cr, -sr), (0, sr, cr)), dtype=np.float64)
    ry = np.asarray(((cp, 0, sp), (0, 1, 0), (-sp, 0, cp)), dtype=np.float64)
    rz = np.asarray(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)), dtype=np.float64)
    return rz @ ry @ rx


def _transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = _rpy_matrix(rpy)
    result[:3, 3] = xyz
    return result


@lru_cache(maxsize=1)
def load_urdf_asset(path: str | Path = G1_URDF_PATH) -> UrdfAsset:
    resolved = Path(path).expanduser().resolve()
    try:
        root = ET.parse(resolved).getroot()
    except (OSError, ET.ParseError) as exc:
        raise ContactFeasibilityError(f"cannot parse G1 URDF {resolved}: {exc}") from exc

    inertials: list[_RawInertial] = []
    links = {str(link.attrib["name"]): link for link in root.findall("link")}
    for name, link in links.items():
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass_element = inertial.find("mass")
        inertia_element = inertial.find("inertia")
        if mass_element is None or inertia_element is None:
            raise ContactFeasibilityError(f"incomplete inertial element for URDF link {name}")
        mass = float(mass_element.attrib["value"])
        if not math.isfinite(mass) or mass <= 0.0:
            raise ContactFeasibilityError(f"non-positive mass for URDF link {name}: {mass}")
        origin = inertial.find("origin")
        xyz = _xyz(origin.attrib.get("xyz") if origin is not None else None)
        rpy = _xyz(origin.attrib.get("rpy") if origin is not None else None)
        attributes = inertia_element.attrib
        inertia = np.asarray(
            (
                (float(attributes["ixx"]), float(attributes["ixy"]), float(attributes["ixz"])),
                (float(attributes["ixy"]), float(attributes["iyy"]), float(attributes["iyz"])),
                (float(attributes["ixz"]), float(attributes["iyz"]), float(attributes["izz"])),
            ),
            dtype=np.float64,
        )
        if not np.isfinite(inertia).all() or np.min(np.linalg.eigvalsh(inertia)) <= 0.0:
            raise ContactFeasibilityError(f"inertia tensor is not positive definite: {name}")
        inertials.append(
            _RawInertial(
                link=name,
                mass=mass,
                com_in_link=xyz,
                rotation_link_inertial=_rpy_matrix(rpy),
                inertia_in_inertial=inertia,
            )
        )

    joints_by_child: dict[str, _Joint] = {}
    for joint_element in root.findall("joint"):
        parent_element = joint_element.find("parent")
        child_element = joint_element.find("child")
        if parent_element is None or child_element is None:
            raise ContactFeasibilityError("URDF joint is missing parent or child")
        parent = str(parent_element.attrib["link"])
        child = str(child_element.attrib["link"])
        origin = joint_element.find("origin")
        xyz = _xyz(origin.attrib.get("xyz") if origin is not None else None)
        rpy = _xyz(origin.attrib.get("rpy") if origin is not None else None)
        if child in joints_by_child:
            raise ContactFeasibilityError(f"URDF link has multiple parents: {child}")
        joints_by_child[child] = _Joint(
            kind=str(joint_element.attrib["type"]),
            parent=parent,
            child=child,
            transform_parent_child=_transform(xyz, rpy),
        )

    centers_by_foot: list[np.ndarray] = []
    radii_by_foot: list[np.ndarray] = []
    for foot_link in FOOT_LINKS:
        try:
            link = links[foot_link]
        except KeyError as exc:
            raise ContactFeasibilityError(f"URDF is missing support link {foot_link}") from exc
        centers: list[np.ndarray] = []
        radii: list[float] = []
        for collision in link.findall("collision"):
            sphere = collision.find("./geometry/sphere")
            if sphere is None:
                continue
            origin = collision.find("origin")
            centers.append(_xyz(origin.attrib.get("xyz") if origin is not None else None))
            radii.append(float(sphere.attrib["radius"]))
        if len(centers) != 4 or len(radii) != 4:
            raise ContactFeasibilityError(
                f"{foot_link} must expose exactly four support spheres"
            )
        centers_by_foot.append(np.stack(centers))
        radii_by_foot.append(np.asarray(radii, dtype=np.float64))

    support_centers = np.stack(centers_by_foot)
    support_radii = np.stack(radii_by_foot)
    if not np.isfinite(support_centers).all() or not np.isfinite(support_radii).all():
        raise ContactFeasibilityError("URDF support geometry is non-finite")
    if np.any(support_radii <= 0.0):
        raise ContactFeasibilityError("URDF support sphere radius must be positive")
    return UrdfAsset(
        checksum_sha256=_sha256(resolved),
        raw_inertials=tuple(inertials),
        joints_by_child=joints_by_child,
        support_centers=support_centers,
        support_radii=support_radii,
        total_mass_kg=float(sum(item.mass for item in inertials)),
    )


def resolve_inertial_model(
    body_names: tuple[str, ...], asset: UrdfAsset | None = None
) -> ResolvedInertialModel:
    model_asset = load_urdf_asset() if asset is None else asset
    if not body_names or len(body_names) != len(set(body_names)):
        raise ContactFeasibilityError("body_names must be a non-empty unique sequence")
    body_index = {name: index for index, name in enumerate(body_names)}
    memo: dict[str, tuple[str, np.ndarray]] = {}
    visiting: set[str] = set()

    def anchor_for(link: str) -> tuple[str, np.ndarray]:
        if link in memo:
            return memo[link]
        if link in body_index:
            result = (link, np.eye(4, dtype=np.float64))
            memo[link] = result
            return result
        if link in visiting:
            raise ContactFeasibilityError(f"cycle while resolving fixed inertial {link}")
        visiting.add(link)
        try:
            joint = model_asset.joints_by_child[link]
        except KeyError as exc:
            raise ContactFeasibilityError(
                f"positive-mass link {link} has no stored body or parent joint"
            ) from exc
        if joint.kind != "fixed":
            raise ContactFeasibilityError(
                f"positive-mass movable link {link} is absent from body_names"
            )
        anchor, transform_anchor_parent = anchor_for(joint.parent)
        result = (anchor, transform_anchor_parent @ joint.transform_parent_child)
        memo[link] = result
        visiting.remove(link)
        return result

    anchor_indices: list[int] = []
    masses: list[float] = []
    offsets: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    inertias: list[np.ndarray] = []
    source_links: list[str] = []
    anchor_links: list[str] = []
    for inertial in model_asset.raw_inertials:
        anchor, transform_anchor_link = anchor_for(inertial.link)
        rotation_anchor_link = transform_anchor_link[:3, :3]
        offset_anchor = (
            transform_anchor_link[:3, 3]
            + rotation_anchor_link @ inertial.com_in_link
        )
        anchor_indices.append(body_index[anchor])
        masses.append(inertial.mass)
        offsets.append(offset_anchor)
        rotations.append(rotation_anchor_link @ inertial.rotation_link_inertial)
        inertias.append(inertial.inertia_in_inertial)
        source_links.append(inertial.link)
        anchor_links.append(anchor)

    mass_array = np.asarray(masses, dtype=np.float64)
    total_mass = float(np.sum(mass_array))
    if not math.isclose(total_mass, model_asset.total_mass_kg, rel_tol=0.0, abs_tol=1.0e-12):
        raise ContactFeasibilityError("resolved inertial mass does not cover the exact URDF")
    return ResolvedInertialModel(
        body_names=body_names,
        anchor_indices=np.asarray(anchor_indices, dtype=np.int64),
        masses=mass_array,
        com_offsets_in_anchor=np.stack(offsets),
        rotations_anchor_inertial=np.stack(rotations),
        inertias_in_inertial=np.stack(inertias),
        source_links=tuple(source_links),
        anchor_links=tuple(anchor_links),
        support_radius_maximum_m=float(np.max(model_asset.support_radii)),
        total_mass_kg=total_mass,
        urdf_checksum_sha256=model_asset.checksum_sha256,
    )


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError(f"quaternion must end in four values; got {q.shape}")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm <= 1.0e-12) or not np.isfinite(q).all():
        raise ValueError("quaternion is zero-length or non-finite")
    q = q / norm
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def five_frame_derivatives(values: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 2 or len(array) < 5:
        raise ValueError("five-frame derivatives require at least five frames")
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be finite and positive")
    velocity = np.full_like(array, np.nan, dtype=np.float64)
    acceleration = np.full_like(array, np.nan, dtype=np.float64)
    dt = 1.0 / fps
    velocity[2:-2] = (
        -2.0 * array[:-4]
        - array[1:-3]
        + array[3:-1]
        + 2.0 * array[4:]
    ) / (10.0 * dt)
    acceleration[2:-2] = (
        2.0 * array[:-4]
        - array[1:-3]
        - 2.0 * array[2:-2]
        - array[3:-1]
        + 2.0 * array[4:]
    ) / (7.0 * dt * dt)
    return velocity, acceleration


def _five_frame_velocity(values: np.ndarray, fps: float) -> np.ndarray:
    return five_frame_derivatives(values, fps)[0]


def _convex_hull(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all():
        raise ValueError("convex hull points must be finite [N,2]")
    unique = sorted(set((float(x), float(y)) for x, y in array))
    if len(unique) <= 1:
        return np.asarray(unique, dtype=np.float64).reshape(-1, 2)

    def cross(origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (
            a[1] - origin[1]
        ) * (b[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 1.0e-12:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 1.0e-12:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def signed_distance_to_support(point: np.ndarray, support_points: np.ndarray) -> float:
    """Return positive distance inside a convex hull and negative outside.

    Degenerate one/two-point supports have no interior: their result is zero on
    the point/segment and negative by Euclidean distance elsewhere.  The caller
    applies the URDF sphere-radius/timing tolerance explicitly.
    """

    query = np.asarray(point, dtype=np.float64)
    if query.shape != (2,) or not np.isfinite(query).all():
        raise ValueError("support query must be a finite 2-vector")
    hull = _convex_hull(support_points)
    if len(hull) == 0:
        raise ValueError("support set cannot be empty")
    if len(hull) == 1:
        return -float(np.linalg.norm(query - hull[0]))
    if len(hull) == 2:
        edge = hull[1] - hull[0]
        alpha = float(np.clip(np.dot(query - hull[0], edge) / np.dot(edge, edge), 0.0, 1.0))
        return -float(np.linalg.norm(query - (hull[0] + alpha * edge)))
    start = hull
    stop = np.roll(hull, -1, axis=0)
    edge = stop - start
    relative = query - start
    cross_value = edge[:, 0] * relative[:, 1] - edge[:, 1] * relative[:, 0]
    inside = bool(np.all(cross_value >= -1.0e-12))
    alpha = np.clip(np.sum(relative * edge, axis=1) / np.sum(edge * edge, axis=1), 0.0, 1.0)
    distance = float(np.min(np.linalg.norm(query - (start + alpha[:, None] * edge), axis=1)))
    return distance if inside else -distance


def _longest_true_run(mask: np.ndarray) -> int:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1:
        raise ValueError("run mask must be one-dimensional")
    transitions = np.diff(np.pad(values.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    return int(np.max(stops - starts)) if starts.size else 0


def _phase_masks(contact_state: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[dict[str, Any], ...]]:
    state = np.asarray(contact_state, dtype=np.int64)
    guard = int(POLICY_ALGORITHM["derivative"]["boundary_guard_frames"])
    minimum = int(POLICY_ALGORITHM["phase"]["minimum_frames"])
    eligible = np.zeros(len(state), dtype=bool)
    single = np.zeros(len(state), dtype=bool)
    records: list[dict[str, Any]] = []
    start = 0
    while start < len(state):
        stop = start + 1
        while stop < len(state) and state[stop] == state[start]:
            stop += 1
        code = int(state[start])
        left = bool(code & 0b0011)
        right = bool(code & 0b1100)
        stable_start = start + guard
        stable_stop = stop - guard
        stable = code != 0 and stop - start >= minimum and stable_stop > stable_start
        if stable:
            eligible[stable_start:stable_stop] = True
            if left != right:
                single[stable_start:stable_stop] = True
        records.append(
            {
                "start_frame": start,
                "stop_frame_exclusive": stop,
                "frame_count": stop - start,
                "contact_state_bits": code,
                "left_contact": left,
                "right_contact": right,
                "single_support": left != right,
                "stable_audit_start_frame": stable_start if stable else None,
                "stable_audit_stop_frame_exclusive": stable_stop if stable else None,
            }
        )
        start = stop
    return eligible, single, tuple(records)


def _contact_arrays(arrays: Mapping[str, np.ndarray], frame_count: int) -> tuple[np.ndarray, np.ndarray]:
    missing = [name for name in (*CONTACT_FIELDS, *FOOT_CONTACT_FIELDS) if name not in arrays]
    if missing:
        raise ContactFeasibilityError(f"archive is missing contact fields: {missing}")
    channels = np.stack(
        [np.asarray(arrays[name], dtype=bool) for name in CONTACT_FIELDS], axis=1
    )
    if channels.shape != (frame_count, 4):
        raise ContactFeasibilityError("detailed contact fields do not match frame_count")
    left = np.asarray(arrays["left_foot_contact"], dtype=bool)
    right = np.asarray(arrays["right_foot_contact"], dtype=bool)
    if left.shape != (frame_count,) or right.shape != (frame_count,):
        raise ContactFeasibilityError("composite contact fields do not match frame_count")
    if not np.array_equal(left, channels[:, :2].any(axis=1)) or not np.array_equal(
        right, channels[:, 2:].any(axis=1)
    ):
        raise ContactFeasibilityError("composite foot contacts disagree with heel/toe channels")
    state = channels.astype(np.int64) @ np.asarray((1, 2, 4, 8), dtype=np.int64)
    return channels, state


def _strict_support_indices(channels: np.ndarray) -> np.ndarray:
    indices: list[int] = []
    for channel in np.flatnonzero(channels):
        foot = 0 if channel < 2 else 1
        pair = (0, 1) if channel % 2 == 0 else (2, 3)
        indices.extend(foot * 4 + offset for offset in pair)
    return np.asarray(sorted(set(indices)), dtype=np.int64)


def _optimistic_support_indices(channels: np.ndarray) -> np.ndarray:
    indices: list[int] = []
    if np.any(channels[:2]):
        indices.extend(range(4))
    if np.any(channels[2:]):
        indices.extend(range(4, 8))
    return np.asarray(indices, dtype=np.int64)


def _centroidal_signals(
    arrays: Mapping[str, np.ndarray], model: ResolvedInertialModel, fps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame_count = int(_scalar(arrays, "frame_count"))
    body_position = np.asarray(arrays.get("body_pos_w"), dtype=np.float64)
    body_quaternion = np.asarray(arrays.get("body_quat_wxyz"), dtype=np.float64)
    body_linear_velocity = np.asarray(arrays.get("body_lin_vel_w"), dtype=np.float64)
    body_angular_velocity = np.asarray(arrays.get("body_ang_vel_w"), dtype=np.float64)
    expected_position_shape = (frame_count, len(model.body_names), 3)
    expected_quaternion_shape = (frame_count, len(model.body_names), 4)
    if body_position.shape != expected_position_shape:
        raise ContactFeasibilityError(
            f"body_pos_w must have shape {expected_position_shape}; got {body_position.shape}"
        )
    if body_quaternion.shape != expected_quaternion_shape:
        raise ContactFeasibilityError(
            f"body_quat_wxyz must have shape {expected_quaternion_shape}"
        )
    if body_linear_velocity.shape != expected_position_shape or body_angular_velocity.shape != expected_position_shape:
        raise ContactFeasibilityError("body velocity arrays do not match body positions")
    if not all(
        np.isfinite(value).all()
        for value in (body_position, body_quaternion, body_linear_velocity, body_angular_velocity)
    ):
        raise ContactFeasibilityError("body state arrays contain NaN or Inf")

    body_rotation = quaternion_wxyz_to_matrix(body_quaternion)
    anchor_index = model.anchor_indices
    anchor_position = body_position[:, anchor_index]
    anchor_rotation = body_rotation[:, anchor_index]
    anchor_linear_velocity = body_linear_velocity[:, anchor_index]
    angular_velocity = body_angular_velocity[:, anchor_index]
    offset_world = np.einsum(
        "teij,ej->tei", anchor_rotation, model.com_offsets_in_anchor
    )
    inertial_position = anchor_position + offset_world
    inertial_velocity = anchor_linear_velocity + np.cross(angular_velocity, offset_world)
    mass = model.masses[None, :, None]
    total_mass = model.total_mass_kg
    com_position = np.sum(mass * inertial_position, axis=1) / total_mass
    com_velocity = np.sum(mass * inertial_velocity, axis=1) / total_mass
    com_acceleration = _five_frame_velocity(com_velocity, fps)

    rotation_world_inertial = np.einsum(
        "teij,ejk->teik", anchor_rotation, model.rotations_anchor_inertial
    )
    inertia_world = np.einsum(
        "teik,ekl,tejl->teij",
        rotation_world_inertial,
        model.inertias_in_inertial,
        rotation_world_inertial,
    )
    spin_momentum = np.einsum("teij,tej->tei", inertia_world, angular_velocity)
    relative_position = inertial_position - com_position[:, None, :]
    relative_velocity = inertial_velocity - com_velocity[:, None, :]
    orbital_momentum = np.cross(relative_position, mass * relative_velocity)
    centroidal_momentum = np.sum(spin_momentum + orbital_momentum, axis=1)
    centroidal_momentum_rate = _five_frame_velocity(centroidal_momentum, fps)
    return (
        com_position,
        com_velocity,
        com_acceleration,
        centroidal_momentum,
        centroidal_momentum_rate,
    )


def centroidal_zmp_from_dynamics(
    com_position: np.ndarray,
    com_acceleration: np.ndarray,
    centroidal_momentum_rate: np.ndarray,
    *,
    support_plane_z: float,
    total_mass_kg: float,
    gravity_mps2: float = 9.81,
) -> tuple[np.ndarray, float]:
    """Evaluate the ground-plane centroidal ZMP for one finite state.

    The returned point is ``[nan, nan]`` when the implied normal force is not
    strictly positive.  Keeping that condition explicit prevents a clipped
    denominator from turning an impossible contact phase into a false pass.
    """

    com = np.asarray(com_position, dtype=np.float64)
    acceleration = np.asarray(com_acceleration, dtype=np.float64)
    momentum_rate = np.asarray(centroidal_momentum_rate, dtype=np.float64)
    if com.shape != (3,) or acceleration.shape != (3,) or momentum_rate.shape != (3,):
        raise ValueError("centroidal ZMP inputs must each be a 3-vector")
    if not all(
        math.isfinite(value)
        for value in (
            *com,
            *acceleration,
            *momentum_rate,
            support_plane_z,
            total_mass_kg,
            gravity_mps2,
        )
    ):
        raise ValueError("centroidal ZMP inputs must be finite")
    if total_mass_kg <= 0.0 or gravity_mps2 <= 0.0:
        raise ValueError("mass and gravity must be positive")
    force = total_mass_kg * (
        acceleration + np.asarray((0.0, 0.0, gravity_mps2), dtype=np.float64)
    )
    normal_force = float(force[2])
    if normal_force <= 0.0:
        return np.full(2, np.nan, dtype=np.float64), normal_force
    height = float(com[2] - support_plane_z)
    point = np.asarray(
        (
            com[0] - (height * force[0] + momentum_rate[1]) / normal_force,
            com[1] - (height * force[1] - momentum_rate[0]) / normal_force,
        ),
        dtype=np.float64,
    )
    return point, normal_force


def compute_clip_signals(
    arrays: Mapping[str, np.ndarray], *, asset: UrdfAsset | None = None
) -> ClipSignals:
    frame_count = int(_scalar(arrays, "frame_count"))
    fps = float(_scalar(arrays, "fps"))
    if frame_count < 7 or fps != float(POLICY_ALGORITHM["fps"]):
        raise ContactFeasibilityError("v29 requires a canonical >=7-frame 50 Hz archive")
    body_names_array = np.asarray(arrays.get("body_names"))
    if body_names_array.ndim != 1 or body_names_array.dtype.hasobject:
        raise ContactFeasibilityError("body_names must be a safe one-dimensional string array")
    body_names = tuple(str(value) for value in body_names_array.tolist())
    model_asset = load_urdf_asset() if asset is None else asset
    model = resolve_inertial_model(body_names, model_asset)
    archive_urdf_checksum = str(_scalar(arrays, "g1_urdf_checksum_sha256"))
    if archive_urdf_checksum != model.urdf_checksum_sha256:
        raise ContactFeasibilityError(
            "archive G1 URDF checksum does not match the exact v29 inertial asset"
        )

    channels, contact_state = _contact_arrays(arrays, frame_count)
    eligible_support, eligible_single, phase_records = _phase_masks(contact_state)
    support = np.asarray(arrays.get("foot_support_position_w"), dtype=np.float64)
    if support.shape != (frame_count, 8, 3) or not np.isfinite(support).all():
        raise ContactFeasibilityError("foot_support_position_w must be finite [frames,8,3]")
    support_names_array = np.asarray(arrays.get("foot_support_names"))
    if support_names_array.shape != (8,) or support_names_array.dtype.hasobject:
        raise ContactFeasibilityError("foot_support_names must be a safe eight-name array")
    support_names = tuple(str(value) for value in support_names_array.tolist())
    if support_names != FOOT_SUPPORT_NAMES:
        raise ContactFeasibilityError("foot support ordering differs from the G1 URDF contract")
    archive_radius = float(_scalar(arrays, "foot_collision_radius_m"))
    if not math.isclose(
        archive_radius,
        float(np.max(model_asset.support_radii)),
        rel_tol=0.0,
        abs_tol=1.0e-7,
    ):
        raise ContactFeasibilityError("archive foot collision radius differs from the G1 URDF")

    (
        com_position,
        com_velocity,
        com_acceleration,
        centroidal_momentum,
        centroidal_momentum_rate,
    ) = _centroidal_signals(arrays, model, fps)
    gravity = float(POLICY_ALGORITHM["gravity_mps2"])
    force = model.total_mass_kg * (
        com_acceleration + np.asarray((0.0, 0.0, gravity), dtype=np.float64)
    )
    normal_force = force[:, 2]
    centroidal_zmp = np.full((frame_count, 2), np.nan, dtype=np.float64)
    lipm_zmp = np.full((frame_count, 2), np.nan, dtype=np.float64)
    zmp_margin_optimistic = np.full(frame_count, np.nan, dtype=np.float64)
    zmp_margin_strict = np.full(frame_count, np.nan, dtype=np.float64)
    capture_point = np.full((frame_count, 2), np.nan, dtype=np.float64)
    capture_margin = np.full(frame_count, np.nan, dtype=np.float64)
    preview_available = np.zeros(frame_count, dtype=bool)
    com_height = np.full(frame_count, np.nan, dtype=np.float64)
    maximum_radius = model.support_radius_maximum_m
    flattened_support_radii = model_asset.support_radii.reshape(8)

    for frame in np.flatnonzero(eligible_support):
        optimistic_indices = _optimistic_support_indices(channels[frame])
        strict_indices = _strict_support_indices(channels[frame])
        if optimistic_indices.size == 0 or strict_indices.size == 0:
            raise ContactFeasibilityError("stable contact phase has an empty support set")
        support_plane = float(
            np.median(
                support[frame, strict_indices, 2]
                - flattened_support_radii[strict_indices]
            )
        )
        height = float(com_position[frame, 2] - support_plane)
        com_height[frame] = height
        if not math.isfinite(height) or height <= 0.0:
            continue
        if normal_force[frame] > 0.0:
            centroidal_zmp[frame], recomputed_normal_force = centroidal_zmp_from_dynamics(
                com_position[frame],
                com_acceleration[frame],
                centroidal_momentum_rate[frame],
                support_plane_z=support_plane,
                total_mass_kg=model.total_mass_kg,
                gravity_mps2=gravity,
            )
            if not math.isclose(
                recomputed_normal_force,
                float(normal_force[frame]),
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise ContactFeasibilityError("internal normal-force inconsistency")
            zmp_margin_optimistic[frame] = signed_distance_to_support(
                centroidal_zmp[frame], support[frame, optimistic_indices, :2]
            )
            zmp_margin_strict[frame] = signed_distance_to_support(
                centroidal_zmp[frame], support[frame, strict_indices, :2]
            )
        lipm_zmp[frame] = (
            com_position[frame, :2]
            - height / gravity * com_acceleration[frame, :2]
        )

        if not eligible_single[frame]:
            continue
        omega = math.sqrt(gravity / height)
        capture_point[frame] = com_position[frame, :2] + com_velocity[frame, :2] / omega
        left_active = bool(contact_state[frame] & 0b0011)
        current_foot = 0 if left_active else 1
        opposite_foot = 1 - current_foot
        corridor_indices = list(range(current_foot * 4, current_foot * 4 + 4))
        preview_frames = int(
            math.ceil(
                float(POLICY_ALGORITHM["capture_preview"]["horizon_natural_time_constants"])
                / (omega / fps)
            )
        )
        stop = min(frame_count, frame + preview_frames + 1)
        touchdown: int | None = None
        opposite_slice = slice(opposite_foot * 2, opposite_foot * 2 + 2)
        for future in range(frame + 1, stop):
            if np.any(channels[future, opposite_slice]):
                touchdown = future
                break
        corridor_points = support[frame, corridor_indices, :2]
        if touchdown is not None:
            opposite_indices = np.arange(opposite_foot * 4, opposite_foot * 4 + 4)
            corridor_points = np.concatenate(
                (corridor_points, support[touchdown, opposite_indices, :2]), axis=0
            )
            preview_available[frame] = True
        capture_margin[frame] = signed_distance_to_support(
            capture_point[frame], corridor_points
        )

    return ClipSignals(
        frame_count=frame_count,
        fps=fps,
        contact_state=contact_state,
        eligible_support=eligible_support,
        eligible_single_support=eligible_single,
        phase_records=phase_records,
        com_position=com_position,
        com_velocity=com_velocity,
        com_acceleration=com_acceleration,
        centroidal_momentum=centroidal_momentum,
        centroidal_momentum_rate=centroidal_momentum_rate,
        normal_force_z=normal_force,
        centroidal_zmp=centroidal_zmp,
        lipm_zmp=lipm_zmp,
        zmp_margin_optimistic=zmp_margin_optimistic,
        zmp_margin_strict=zmp_margin_strict,
        capture_point=capture_point,
        capture_margin_optimistic=capture_margin,
        capture_preview_available=preview_available,
        com_height=com_height,
    )


def _finite_stats(values: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    selected = np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    finite = selected[np.isfinite(selected)]
    if finite.size == 0:
        return {"count": 0, "minimum": None, "p01": None, "p05": None, "median": None}
    p01, p05, median = np.percentile(finite, (1.0, 5.0, 50.0))
    return {
        "count": int(finite.size),
        "minimum": float(np.min(finite)),
        "p01": float(p01),
        "p05": float(p05),
        "median": float(median),
    }


def _violation_summary(mask: np.ndarray) -> dict[str, Any]:
    values = np.asarray(mask, dtype=bool)
    indices = np.flatnonzero(values)
    return {
        "frame_count": int(indices.size),
        "longest_consecutive_frames": _longest_true_run(values),
        "first_frame": int(indices[0]) if indices.size else None,
    }


def _interpolate_same_phase(
    source_values: np.ndarray,
    source_state: np.ndarray,
    candidate_state: np.ndarray,
    source_coordinate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(source_values, dtype=np.float64)
    lower = np.floor(source_coordinate).astype(np.int64)
    upper = np.minimum(lower + 1, len(values) - 1)
    alpha = source_coordinate - lower
    result_shape = (len(source_coordinate),) + values.shape[1:]
    result = np.full(result_shape, np.nan, dtype=np.float64)
    valid = (
        (source_state[lower] == candidate_state)
        & (source_state[upper] == candidate_state)
        & np.isfinite(values[lower]).all(axis=tuple(range(1, values.ndim)))
        & np.isfinite(values[upper]).all(axis=tuple(range(1, values.ndim)))
    )
    reshape = (len(alpha),) + (1,) * (values.ndim - 1)
    interpolated = (1.0 - alpha.reshape(reshape)) * values[lower] + alpha.reshape(reshape) * values[upper]
    result[valid] = interpolated[valid]
    return result, valid


def _interpolate_scalar_same_phase(
    source_values: np.ndarray,
    source_state: np.ndarray,
    candidate_state: np.ndarray,
    source_coordinate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(source_values, dtype=np.float64)
    expanded, valid = _interpolate_same_phase(
        values[:, None], source_state, candidate_state, source_coordinate
    )
    return expanded[:, 0], valid


def audit_contact_pair(
    source_arrays: Mapping[str, np.ndarray],
    candidate_arrays: Mapping[str, np.ndarray],
    *,
    asset: UrdfAsset | None = None,
) -> dict[str, Any]:
    """Audit one immutable v27 source and its v28 candidate without writing."""

    model_asset = load_urdf_asset() if asset is None else asset
    source = compute_clip_signals(source_arrays, asset=model_asset)
    candidate = compute_clip_signals(candidate_arrays, asset=model_asset)
    if source.fps != candidate.fps:
        raise ContactFeasibilityError("source and candidate fps differ")
    realized_scale = float(_scalar(candidate_arrays, "dynamic_realized_scale"))
    count_scale = (candidate.frame_count - 1) / (source.frame_count - 1)
    if not math.isclose(realized_scale, count_scale, rel_tol=0.0, abs_tol=1.0e-10):
        raise ContactFeasibilityError(
            "candidate dynamic_realized_scale disagrees with endpoint-preserving frame counts"
        )
    source_coordinate = np.arange(candidate.frame_count, dtype=np.float64) / realized_scale
    source_coordinate[-1] = source.frame_count - 1

    source_zmp_margin, zmp_compare_valid = _interpolate_scalar_same_phase(
        source.zmp_margin_optimistic,
        source.contact_state,
        candidate.contact_state,
        source_coordinate,
    )
    source_capture_margin, capture_compare_valid = _interpolate_scalar_same_phase(
        source.capture_margin_optimistic,
        source.contact_state,
        candidate.contact_state,
        source_coordinate,
    )
    zmp_degradation = candidate.zmp_margin_optimistic - source_zmp_margin
    capture_degradation = candidate.capture_margin_optimistic - source_capture_margin

    contact_step = float(
        POLICY_ALGORITHM["thresholds"]["semantic_contact_speed_budget_mps"]
    ) / candidate.fps
    maximum_radius = float(np.max(model_asset.support_radii))
    absolute_tolerance = maximum_radius + contact_step
    degradation_tolerance = contact_step
    bad_run = int(POLICY_ALGORITHM["phase"]["bad_run_frames"])

    zmp_absolute_bad = (
        candidate.eligible_support
        & np.isfinite(candidate.zmp_margin_optimistic)
        & (candidate.zmp_margin_optimistic < -absolute_tolerance)
    )
    normal_force_bad = candidate.eligible_support & (
        ~np.isfinite(candidate.normal_force_z) | (candidate.normal_force_z <= 0.0)
    )
    height_bad = candidate.eligible_support & (
        ~np.isfinite(candidate.com_height) | (candidate.com_height <= 0.0)
    )
    capture_absolute_bad = (
        candidate.eligible_single_support
        & np.isfinite(candidate.capture_margin_optimistic)
        & (candidate.capture_margin_optimistic < -absolute_tolerance)
    )
    zmp_degradation_bad = (
        candidate.eligible_support
        & zmp_compare_valid
        & np.isfinite(zmp_degradation)
        & (zmp_degradation < -degradation_tolerance)
    )
    capture_degradation_bad = (
        candidate.eligible_single_support
        & capture_compare_valid
        & np.isfinite(capture_degradation)
        & (capture_degradation < -degradation_tolerance)
    )
    violations = {
        "centroidal_zmp_absolute": _violation_summary(zmp_absolute_bad),
        "positive_normal_force": _violation_summary(normal_force_bad),
        "positive_com_height": _violation_summary(height_bad),
        "preview_capture_absolute": _violation_summary(capture_absolute_bad),
        "centroidal_zmp_retime_degradation": _violation_summary(zmp_degradation_bad),
        "preview_capture_retime_degradation": _violation_summary(capture_degradation_bad),
    }
    quarantine_reasons = sorted(
        name
        for name, evidence in violations.items()
        if int(evidence["longest_consecutive_frames"]) >= bad_run
    )

    source_action = str(_scalar(candidate_arrays, "dynamic_source_action"))
    final_action = str(_scalar(candidate_arrays, "dynamic_final_action"))
    phase_counts = {
        "total": len(candidate.phase_records),
        "single_support": sum(bool(item["single_support"]) for item in candidate.phase_records),
        "double_support": sum(
            bool(item["left_contact"] and item["right_contact"])
            for item in candidate.phase_records
        ),
        "flight": sum(
            not bool(item["left_contact"] or item["right_contact"])
            for item in candidate.phase_records
        ),
        "stable_support_frames": int(np.count_nonzero(candidate.eligible_support)),
        "stable_single_support_frames": int(
            np.count_nonzero(candidate.eligible_single_support)
        ),
    }
    return {
        "schema_version": CLIP_AUDIT_SCHEMA_VERSION,
        "gate_version": CONTACT_GATE_VERSION,
        "passed": not quarantine_reasons,
        "final_action": "promote_v28_reference" if not quarantine_reasons else "quarantine_v29_contact_feasibility",
        "quarantine_reasons": quarantine_reasons,
        "frame_contract": {
            "source_v27_frames": source.frame_count,
            "candidate_v28_frames": candidate.frame_count,
            "fps": candidate.fps,
            "realized_uniform_scale": realized_scale,
            "source_coordinate_last": float(source_coordinate[-1]),
            "dynamic_source_action": source_action,
            "dynamic_final_action": final_action,
        },
        "asset_contract": {
            "urdf_checksum_sha256": model_asset.checksum_sha256,
            "urdf_total_mass_kg": model_asset.total_mass_kg,
            "positive_mass_link_count": len(model_asset.raw_inertials),
            "support_radius_maximum_m": maximum_radius,
        },
        "thresholds": {
            "absolute_margin_tolerance_m": absolute_tolerance,
            "retime_degradation_tolerance_m": degradation_tolerance,
            "bad_run_frames": bad_run,
            "bad_run_duration_sec": bad_run / candidate.fps,
            "phase_boundary_guard_frames": int(
                POLICY_ALGORITHM["derivative"]["boundary_guard_frames"]
            ),
            "minimum_phase_frames": int(POLICY_ALGORITHM["phase"]["minimum_frames"]),
        },
        "phase_counts": phase_counts,
        "metrics": {
            "source_v27_centroidal_zmp_margin_m": _finite_stats(
                source.zmp_margin_optimistic, source.eligible_support
            ),
            "candidate_v28_centroidal_zmp_margin_m": _finite_stats(
                candidate.zmp_margin_optimistic, candidate.eligible_support
            ),
            "candidate_v28_centroidal_zmp_strict_margin_m": _finite_stats(
                candidate.zmp_margin_strict, candidate.eligible_support
            ),
            "candidate_v28_com_height_m": _finite_stats(
                candidate.com_height, candidate.eligible_support
            ),
            "candidate_v28_normal_force_z_n": _finite_stats(
                candidate.normal_force_z, candidate.eligible_support
            ),
            "source_v27_preview_capture_margin_m": _finite_stats(
                source.capture_margin_optimistic, source.eligible_single_support
            ),
            "candidate_v28_preview_capture_margin_m": _finite_stats(
                candidate.capture_margin_optimistic,
                candidate.eligible_single_support,
            ),
            "centroidal_zmp_margin_degradation_m": _finite_stats(
                zmp_degradation, candidate.eligible_support & zmp_compare_valid
            ),
            "preview_capture_margin_degradation_m": _finite_stats(
                capture_degradation,
                candidate.eligible_single_support & capture_compare_valid,
            ),
            "zmp_same_phase_comparison_frames": int(
                np.count_nonzero(candidate.eligible_support & zmp_compare_valid)
            ),
            "capture_same_phase_comparison_frames": int(
                np.count_nonzero(candidate.eligible_single_support & capture_compare_valid)
            ),
            "capture_preview_available_frames": int(
                np.count_nonzero(
                    candidate.eligible_single_support
                    & candidate.capture_preview_available
                )
            ),
        },
        "violations": violations,
    }


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContactFeasibilityError(f"cannot read {description} {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ContactFeasibilityError(f"{description} must be a JSON object")
    return document


def _safe_member(root: Path, value: str, description: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ContactFeasibilityError(f"unsafe {description} path: {value}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ContactFeasibilityError(f"{description} escapes reference root: {value}") from exc
    return resolved


def _load_split_manifests(
    split: str, v27_path: Path, v28_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    v27 = _load_json(v27_path, "v27 manifest")
    v28 = _load_json(v28_path, "v28 manifest")
    if (
        v27.get("schema_version") != V27_MANIFEST_SCHEMA_VERSION
        or v27.get("split") != split
        or v27.get("kind") != f"split:{split}"
    ):
        raise ContactFeasibilityError("v27 manifest schema/split contract is invalid")
    if (
        v28.get("schema_version") != V28_MANIFEST_SCHEMA_VERSION
        or v28.get("split") != split
        or v28.get("kind") != f"split:{split}"
    ):
        raise ContactFeasibilityError("v28 manifest schema/split contract is invalid")
    for name, document in (("v27", v27), ("v28", v28)):
        motions = document.get("motions")
        if not isinstance(motions, list) or int(document.get("num_motions", -1)) != len(motions):
            raise ContactFeasibilityError(f"{name} manifest motion count is invalid")
    if v28.get("source_manifest_checksum_sha256") != _sha256(v27_path):
        raise ContactFeasibilityError("v28 manifest does not bind the selected v27 manifest")
    return v27, v28


def _resolved_policy_asset(asset: UrdfAsset) -> dict[str, Any]:
    return {
        "g1_urdf_path": G1_URDF_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "g1_urdf_checksum_sha256": asset.checksum_sha256,
        "urdf_total_mass_kg": asset.total_mass_kg,
        "positive_mass_link_count": len(asset.raw_inertials),
        "support_centers_local_m": asset.support_centers.tolist(),
        "support_radii_m": asset.support_radii.tolist(),
        "support_radius_maximum_m": float(np.max(asset.support_radii)),
    }


def _policy_document(
    train_v27_path: Path,
    train_v28_path: Path,
    train_v27: Mapping[str, Any],
    train_v28: Mapping[str, Any],
    asset: UrdfAsset,
) -> dict[str, Any]:
    dynamic_policy_checksum = str(train_v28.get("dynamic_policy_checksum_sha256", ""))
    if not DEFAULT_DYNAMIC_POLICY_PATH.is_file() or _sha256(DEFAULT_DYNAMIC_POLICY_PATH) != dynamic_policy_checksum:
        raise ContactFeasibilityError("v28 frozen dynamic policy file/checksum is unavailable")
    return {
        "schema_version": POLICY_SCHEMA_VERSION,
        "status": "frozen_from_train",
        "created_at_utc": _utc_now(),
        "gate_version": CONTACT_GATE_VERSION,
        "implementation": {
            "path": IMPLEMENTATION_PATH.relative_to(PIPELINE_ROOT).as_posix(),
            "checksum_sha256": _sha256(IMPLEMENTATION_PATH),
        },
        "algorithm": POLICY_ALGORITHM,
        "algorithm_checksum_sha256": _canonical_checksum(POLICY_ALGORITHM),
        "resolved_asset": _resolved_policy_asset(asset),
        "train_inputs": {
            "v27_manifest_path": train_v27_path.relative_to(PIPELINE_ROOT).as_posix(),
            "v27_manifest_checksum_sha256": _sha256(train_v27_path),
            "v27_motion_set_checksum_sha256": train_v27.get("motion_set_checksum_sha256"),
            "v28_manifest_path": train_v28_path.relative_to(PIPELINE_ROOT).as_posix(),
            "v28_manifest_checksum_sha256": _sha256(train_v28_path),
            "v28_motion_set_checksum_sha256": train_v28.get("motion_set_checksum_sha256"),
            "v28_num_motions": int(train_v28["num_motions"]),
            "v28_dynamic_policy_path": DEFAULT_DYNAMIC_POLICY_PATH.relative_to(PIPELINE_ROOT).as_posix(),
            "v28_dynamic_policy_checksum_sha256": dynamic_policy_checksum,
        },
        "split_policy": {
            "train": "allowed_and_freezes_policy",
            "validation": "allowed_only_with_existing_frozen_train_policy",
            "test": "opened_only_by_explicit_split_test_invocation_after_external_validation_acceptance",
        },
        "publication_policy": "audit_only_by_default_manifest_requires_explicit_publish_manifest",
    }


def _load_or_freeze_policy(
    *,
    split: str,
    v27_path: Path,
    v28_path: Path,
    v27: Mapping[str, Any],
    v28: Mapping[str, Any],
    output_root: Path,
    asset: UrdfAsset,
) -> tuple[dict[str, Any], Path]:
    policy_path = output_root / "policy.json"
    if split == "train":
        expected = _policy_document(v27_path, v28_path, v27, v28, asset)
        if policy_path.exists():
            existing = _load_json(policy_path, "v29 frozen policy")
            existing_compare = dict(existing)
            expected_compare = dict(expected)
            existing_compare.pop("created_at_utc", None)
            expected_compare.pop("created_at_utc", None)
            if existing_compare != expected_compare:
                raise ContactFeasibilityError(
                    "existing v29 train policy differs; use a new output root rather than overwriting it"
                )
            return existing, policy_path
        _atomic_json(policy_path, expected)
        return expected, policy_path
    if not policy_path.is_file():
        raise ContactFeasibilityError(
            f"{split} audit requires policy.json frozen by a train audit"
        )
    policy = _load_json(policy_path, "v29 frozen policy")
    if (
        policy.get("schema_version") != POLICY_SCHEMA_VERSION
        or policy.get("status") != "frozen_from_train"
        or policy.get("gate_version") != CONTACT_GATE_VERSION
        or policy.get("implementation")
        != {
            "path": IMPLEMENTATION_PATH.relative_to(PIPELINE_ROOT).as_posix(),
            "checksum_sha256": _sha256(IMPLEMENTATION_PATH),
        }
        or policy.get("algorithm") != POLICY_ALGORITHM
        or policy.get("algorithm_checksum_sha256") != _canonical_checksum(POLICY_ALGORITHM)
        or policy.get("resolved_asset") != _resolved_policy_asset(asset)
    ):
        raise ContactFeasibilityError("v29 frozen train policy is missing, altered, or unsupported")
    expected_dynamic_policy = policy["train_inputs"]["v28_dynamic_policy_checksum_sha256"]
    if v28.get("dynamic_policy_checksum_sha256") != expected_dynamic_policy:
        raise ContactFeasibilityError("selected v28 split uses a different frozen dynamic policy")
    if not DEFAULT_DYNAMIC_POLICY_PATH.is_file() or _sha256(DEFAULT_DYNAMIC_POLICY_PATH) != expected_dynamic_policy:
        raise ContactFeasibilityError("v28 frozen dynamic policy checksum changed after train freeze")
    return policy, policy_path


def build_contact_feasibility_split(
    *,
    split: str,
    v27_manifest_path: str | Path | None = None,
    v28_manifest_path: str | Path | None = None,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    force: bool = False,
    publish_manifest: bool = False,
) -> dict[str, Any]:
    """Audit exactly one split; no all-splits scan exists.

    Merely importing or invoking this function for train/validation never opens
    a test manifest or test NPZ.  Test access requires the literal
    ``split="test"`` argument.
    """

    if split not in DEFAULT_V27_MANIFESTS:
        raise ValueError("split must be exactly train, validation, or test")
    v27_path = Path(
        DEFAULT_V27_MANIFESTS[split] if v27_manifest_path is None else v27_manifest_path
    ).expanduser().resolve()
    v28_path = Path(
        DEFAULT_V28_MANIFESTS[split] if v28_manifest_path is None else v28_manifest_path
    ).expanduser().resolve()
    output = assert_output_is_local(output_root)
    plan_path = output / f"g1_{split}_plan.json"
    report_path = output / f"g1_{split}_report.json"
    manifest_path = output / f"g1_{split}.json"
    existing = [path for path in (plan_path, report_path) if path.exists()]
    if existing and not force:
        raise ContactFeasibilityError(
            "v29 audit outputs already exist; use --force to replace audit JSON: "
            + ", ".join(str(path) for path in existing)
        )
    if manifest_path.exists() and (not force or not publish_manifest):
        raise ContactFeasibilityError(
            "an opt-in published manifest exists; only --force --publish-manifest may replace it"
        )

    v27, v28 = _load_split_manifests(split, v27_path, v28_path)
    asset = load_urdf_asset()
    policy, policy_path = _load_or_freeze_policy(
        split=split,
        v27_path=v27_path,
        v28_path=v28_path,
        v27=v27,
        v28=v28,
        output_root=output,
        asset=asset,
    )
    policy_checksum = _sha256(policy_path)
    v27_root = (PIPELINE_ROOT / str(v27["reference_root"])).resolve()
    v28_root = (PIPELINE_ROOT / str(v28["reference_root"])).resolve()
    for root, label in ((v27_root, "v27"), (v28_root, "v28")):
        try:
            root.relative_to(PIPELINE_ROOT)
        except ValueError as exc:
            raise ContactFeasibilityError(f"{label} reference root escapes pipeline") from exc
        if not root.is_dir():
            raise ContactFeasibilityError(f"{label} reference root is missing: {root}")
    v27_by_file = {str(record["g1_file"]): record for record in v27["motions"]}
    if len(v27_by_file) != len(v27["motions"]):
        raise ContactFeasibilityError("v27 manifest contains duplicate g1_file records")

    records: list[dict[str, Any]] = []
    promoted: list[dict[str, Any]] = []
    started = _utc_now()
    for candidate_record in v28["motions"]:
        relative = str(candidate_record["g1_file"])
        if relative not in v27_by_file:
            raise ContactFeasibilityError(f"v28 motion has no v27 source record: {relative}")
        source_record = v27_by_file[relative]
        source_path = _safe_member(v27_root, relative, "v27 archive")
        candidate_path = _safe_member(v28_root, relative, "v28 archive")
        for path, expected_checksum, expected_size, label in (
            (
                source_path,
                source_record.get("g1_checksum_sha256"),
                source_record.get("g1_size_bytes"),
                "v27",
            ),
            (
                candidate_path,
                candidate_record.get("g1_checksum_sha256"),
                candidate_record.get("g1_size_bytes"),
                "v28",
            ),
        ):
            if not path.is_file():
                raise ContactFeasibilityError(f"{label} archive is missing: {path}")
            actual_checksum = _sha256(path)
            if actual_checksum != expected_checksum:
                raise ContactFeasibilityError(f"{label} archive checksum mismatch: {relative}")
            if path.stat().st_size != int(expected_size):
                raise ContactFeasibilityError(f"{label} archive size mismatch: {relative}")
        if candidate_record.get("source_v27_g1_checksum_sha256") != source_record.get(
            "g1_checksum_sha256"
        ):
            raise ContactFeasibilityError(f"v28 record does not bind its v27 source: {relative}")
        source_arrays = _safe_npz(source_path)
        candidate_arrays = _safe_npz(candidate_path)
        if str(_scalar(candidate_arrays, "dynamic_split")) != split:
            raise ContactFeasibilityError(
                f"v28 archive split provenance differs from selected split: {relative}"
            )
        if str(_scalar(candidate_arrays, "dynamic_source_g1_checksum_sha256")) != _sha256(source_path):
            raise ContactFeasibilityError(f"v28 archive provenance does not bind v27: {relative}")
        if str(
            _scalar(candidate_arrays, "dynamic_source_manifest_checksum_sha256")
        ) != _sha256(v27_path):
            raise ContactFeasibilityError(
                f"v28 archive provenance does not bind selected v27 manifest: {relative}"
            )
        audit = audit_contact_pair(source_arrays, candidate_arrays, asset=asset)
        record = {
            "g1_file": relative,
            "source_v27_checksum_sha256": source_record["g1_checksum_sha256"],
            "candidate_v28_checksum_sha256": candidate_record["g1_checksum_sha256"],
            "audit": audit,
            "status": "eligible" if audit["passed"] else "quarantined",
            "final_action": audit["final_action"],
        }
        records.append(record)
        if audit["passed"]:
            promoted.append(
                {
                    **candidate_record,
                    "contact_v29_gate_complete": True,
                    "contact_v29_gate_version": CONTACT_GATE_VERSION,
                    "contact_v29_policy_checksum_sha256": policy_checksum,
                    "contact_v29_audit_checksum_sha256": _canonical_checksum(audit),
                }
            )

    records.sort(key=lambda item: item["g1_file"])
    promoted.sort(key=lambda item: item["g1_file"])
    reason_counts: dict[str, int] = {}
    for record in records:
        for reason in record["audit"]["quarantine_reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    counts = {
        "source_v28": len(records),
        "eligible": len(promoted),
        "quarantined": len(records) - len(promoted),
        "quarantine_reason_counts": dict(sorted(reason_counts.items())),
    }
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "completed",
        "mode": "audit_and_publish" if publish_manifest else "audit_only",
        "split": split,
        "created_at_utc": _utc_now(),
        "gate_version": CONTACT_GATE_VERSION,
        "inputs": {
            "v27_manifest_path": v27_path.relative_to(PIPELINE_ROOT).as_posix(),
            "v27_manifest_checksum_sha256": _sha256(v27_path),
            "v28_manifest_path": v28_path.relative_to(PIPELINE_ROOT).as_posix(),
            "v28_manifest_checksum_sha256": _sha256(v28_path),
            "v28_dynamic_policy_checksum_sha256": v28.get("dynamic_policy_checksum_sha256"),
        },
        "frozen_policy": {
            "path": policy_path.relative_to(PIPELINE_ROOT).as_posix(),
            "checksum_sha256": policy_checksum,
            "algorithm_checksum_sha256": policy["algorithm_checksum_sha256"],
        },
        "counts": counts,
        "records": records,
    }
    _atomic_json(plan_path, plan)

    manifest_document: dict[str, Any] | None = None
    publication_status = "not_requested"
    if publish_manifest:
        if not promoted:
            publication_status = "refused_empty_eligible_set"
        else:
            motion_set_checksum = hashlib.sha256(
                json.dumps(
                    [
                        [split, record["g1_file"], record["g1_checksum_sha256"]]
                        for record in promoted
                    ],
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            manifest_document = {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "dataset": "ACCAD",
                "kind": f"split:{split}",
                "split": split,
                "gate": "gate3_kinematic_complete+dynamic_v28+contact_v29",
                "reference_root": v28["reference_root"],
                "retarget_version": v28["retarget_version"],
                "contact_feasibility_gate_version": CONTACT_GATE_VERSION,
                "motion_set_checksum_sha256": motion_set_checksum,
                "source_v28_motion_set_checksum_sha256": v28.get(
                    "motion_set_checksum_sha256"
                ),
                "source_v28_manifest_checksum_sha256": _sha256(v28_path),
                "contact_v29_policy_checksum_sha256": policy_checksum,
                "contact_v29_plan_checksum_sha256": _sha256(plan_path),
                "num_source_motions": len(records),
                "num_motions": len(promoted),
                "num_quarantined": len(records) - len(promoted),
                "num_frames": sum(int(record["frame_count"]) for record in promoted),
                "duration_sec": sum(float(record["duration_sec"]) for record in promoted),
                "motions": promoted,
            }
            _atomic_json(manifest_path, manifest_document)
            publication_status = "published"

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": (
            "completed"
            if publication_status != "refused_empty_eligible_set"
            else "completed_audit_publication_refused"
        ),
        "mode": "audit_and_publish" if publish_manifest else "audit_only",
        "split": split,
        "started_at_utc": started,
        "finished_at_utc": _utc_now(),
        "counts": counts,
        "publication": {
            "requested": bool(publish_manifest),
            "status": publication_status,
            "manifest": (
                None
                if manifest_document is None
                else {
                    "path": manifest_path.relative_to(PIPELINE_ROOT).as_posix(),
                    "checksum_sha256": _sha256(manifest_path),
                }
            ),
        },
        "artifacts": {
            "policy": {
                "path": policy_path.relative_to(PIPELINE_ROOT).as_posix(),
                "checksum_sha256": policy_checksum,
            },
            "plan": {
                "path": plan_path.relative_to(PIPELINE_ROOT).as_posix(),
                "checksum_sha256": _sha256(plan_path),
            },
        },
        "sealed_test_contract": (
            "test manifest and NPZ contents are opened only by a literal split=test invocation"
        ),
        "test_opened_by_this_invocation": split == "test",
    }
    _atomic_json(report_path, report)
    return report


__all__ = [
    "CLIP_AUDIT_SCHEMA_VERSION",
    "CONTACT_GATE_VERSION",
    "ContactFeasibilityError",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_V27_MANIFESTS",
    "DEFAULT_V28_MANIFESTS",
    "MANIFEST_SCHEMA_VERSION",
    "PLAN_SCHEMA_VERSION",
    "POLICY_ALGORITHM",
    "POLICY_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "audit_contact_pair",
    "build_contact_feasibility_split",
    "centroidal_zmp_from_dynamics",
    "compute_clip_signals",
    "five_frame_derivatives",
    "load_urdf_asset",
    "quaternion_wxyz_to_matrix",
    "resolve_inertial_model",
    "signed_distance_to_support",
]
