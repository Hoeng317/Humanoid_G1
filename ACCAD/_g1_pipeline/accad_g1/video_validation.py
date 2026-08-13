"""CPU-side integrity inspection for recorded Isaac evaluation videos."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


class VideoValidationError(ValueError):
    """Raised when an MP4 cannot be decoded into useful visual evidence."""


def inspect_video(path: str | Path) -> dict[str, Any]:
    """Decode every frame and return a JSON-safe visual integrity record.

    The Gymnasium recorder can contain one renderer warm-up frame that is
    completely black.  Therefore the contract rejects an *entirely* black
    video, rather than rejecting a valid clip for one warm-up frame.
    """

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - Isaac environment guard
        raise VideoValidationError(f"OpenCV is unavailable: {exc}") from exc

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise VideoValidationError(f"video is missing: {resolved}")
    capture = cv2.VideoCapture(str(resolved))
    if not capture.isOpened():
        capture.release()
        raise VideoValidationError(f"video cannot be opened: {resolved}")

    declared_frames = int(round(float(capture.get(cv2.CAP_PROP_FRAME_COUNT))))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(round(float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))))
    height = int(round(float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    decoded_frames = 0
    all_black_frames = 0
    near_black_frames = 0
    global_min = 255
    global_max = 0
    previous_frame = None
    frame_pair_count = 0
    changed_frame_pair_count = 0
    temporal_change_sum = 0.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.size == 0:
                raise VideoValidationError(
                    f"decoder returned an empty frame at index {decoded_frames}: {resolved}"
                )
            decoded_frames += 1
            frame_min = int(frame.min())
            frame_max = int(frame.max())
            frame_channel_means = cv2.mean(frame)[:3]
            frame_mean = float(sum(frame_channel_means) / len(frame_channel_means))
            global_min = min(global_min, frame_min)
            global_max = max(global_max, frame_max)
            all_black_frames += int(frame_max == 0)
            near_black_frames += int(frame_mean <= 2.0)
            if previous_frame is not None:
                difference = cv2.absdiff(frame, previous_frame)
                channel_means = cv2.mean(difference)[:3]
                mean_absolute_change = float(sum(channel_means) / len(channel_means))
                temporal_change_sum += mean_absolute_change
                frame_pair_count += 1
                changed_frame_pair_count += int(mean_absolute_change >= 0.25)
            previous_frame = frame
    finally:
        capture.release()

    if decoded_frames <= 0:
        raise VideoValidationError(f"video contains no decodable frames: {resolved}")
    if width <= 0 or height <= 0:
        raise VideoValidationError(f"video has invalid dimensions: {width}x{height}")
    if not math.isfinite(fps) or fps <= 0.0:
        raise VideoValidationError(f"video has invalid FPS: {fps!r}")
    if declared_frames > 0 and declared_frames != decoded_frames:
        raise VideoValidationError(
            f"video frame-count metadata disagrees with decoding: "
            f"declared={declared_frames}, decoded={decoded_frames}"
        )
    if all_black_frames == decoded_frames:
        raise VideoValidationError(f"every decoded frame is black: {resolved}")

    return {
        "decoder": "opencv",
        "decoded_frame_count": decoded_frames,
        "container_declared_frame_count": declared_frames,
        "fps": fps,
        "width_px": width,
        "height_px": height,
        "duration_s": decoded_frames / fps,
        "all_black_frame_count": all_black_frames,
        "all_black_frame_fraction": all_black_frames / decoded_frames,
        "near_black_frame_count": near_black_frames,
        "visually_nonblack_frame_fraction": 1.0
        - near_black_frames / decoded_frames,
        "all_frames_black": False,
        "frame_pair_count": frame_pair_count,
        "changed_frame_pair_count": changed_frame_pair_count,
        "changed_frame_pair_fraction": changed_frame_pair_count
        / max(frame_pair_count, 1),
        "temporal_mean_absolute_pixel_change": temporal_change_sum
        / max(frame_pair_count, 1),
        "decoded_pixel_min": global_min,
        "decoded_pixel_max": global_max,
    }


__all__ = ["VideoValidationError", "inspect_video"]
