"""Shared pytest fixtures for pipeline-local write tests."""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest


PIPELINE_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def pipeline_tmp_path() -> Iterator[Path]:
    """Provide an isolated temporary directory inside the writable pipeline root.

    Production batch code deliberately rejects every output outside
    ``ACCAD/_g1_pipeline``.  Tests that exercise real batch path validation must
    therefore use a local sandbox rather than pytest's default system ``/tmp``.
    The sandbox is unique per test and removed automatically afterwards.
    """

    sandbox_root = PIPELINE_ROOT / "work" / "pytest_local"
    sandbox_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="case-", dir=sandbox_root) as path:
        yield Path(path).resolve()
