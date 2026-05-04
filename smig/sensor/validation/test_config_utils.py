"""
smig/sensor/validation/test_config_utils.py
=============================================
Contract tests for smig.config.schemas and smig.config.utils.

These tests verify the config schema boundary (validation rules, hash
stability, YAML loader robustness) independently of the sensor pipeline.

Run from the project root:
    python -m pytest smig/sensor/validation/test_config_utils.py -v
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from smig.config.schemas import DetectorConfig, GeometryConfig, IPCConfig, NonlinearityConfig
from smig.config.utils import get_config_sha256, load_detector_config


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def test_ipc_sca_id_in_range():
    """IPCConfig must reject sca_id=0 (valid range is 1–18)."""
    with pytest.raises(ValidationError):
        IPCConfig(sca_id=0)


def test_ipc_sca_id_upper_bound():
    """IPCConfig must reject sca_id=19 (valid range is 1–18)."""
    with pytest.raises(ValidationError):
        IPCConfig(sca_id=19)


def test_ipc_sca_id_valid_range():
    """IPCConfig must accept all valid sca_id values 1–18."""
    for sca_id in (1, 9, 18):
        cfg = IPCConfig(sca_id=sca_id)
        assert cfg.sca_id == sca_id


# ---------------------------------------------------------------------------
# YAML loader robustness
# ---------------------------------------------------------------------------

def test_load_detector_config_missing_file():
    """load_detector_config must raise FileNotFoundError for a nonexistent path."""
    with pytest.raises(FileNotFoundError):
        load_detector_config("/nonexistent/path/does_not_exist.yaml")


def test_load_detector_config_empty_yaml():
    """load_detector_config must raise ValueError for an empty YAML file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("")  # completely empty
        tmp_path = Path(f.name)
    try:
        with pytest.raises(ValueError, match="empty"):
            load_detector_config(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def test_load_detector_config_comment_only_yaml():
    """load_detector_config must raise ValueError for a comments-only YAML file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("# This file contains only a comment\n")
        tmp_path = Path(f.name)
    try:
        with pytest.raises(ValueError, match="empty"):
            load_detector_config(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def test_load_detector_config_list_yaml():
    """load_detector_config must raise ValueError when top-level YAML is a list."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("- item1\n- item2\n")
        tmp_path = Path(f.name)
    try:
        with pytest.raises(ValueError, match="mapping"):
            load_detector_config(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# SHA-256 hash stability
# ---------------------------------------------------------------------------

# CANARY: This is the expected SHA-256 of the canonical JSON serialization of
# a default DetectorConfig().  If this hash changes, a field default or Pydantic
# serialization behavior has changed.  Do NOT auto-update this value.  Report the
# dependency/version delta and update deliberately after code review.
#
# Updated in phase-A (2026-04-10): Added NoiseConfig sub-model with six default
# fields to DetectorConfig; changed ipc_kernel_path from Path|None to str|None.
# Old hash: 0bf688dd1f69bc8d08d4814463463b7bce88829ebf0f26d3440235a83dc2def1
#
# Updated in phase-B (2026-04-15): Python 3.13 changed the JSON serialization
# of floating-point values (e.g. trailing zeros, exponent notation) relative to
# 3.11, causing a hash shift with no schema changes.  DetectorConfig fields and
# defaults are UNCHANGED; only the CPython JSON encoder output differs.
# Old hash (Python 3.11): a2ce8d9319461c4de9e802cd0e3b4db1862bd34f5af3d6ea8fc869654bfe76eb
_CANARY_HASH = "1f71193213d695079cc063f37be6b5c6f554e9370ce49acff0d9e74ebec62ea7"


def test_config_sha256_stability_canary():
    """SHA-256 of the default DetectorConfig must match the pinned canary value.

    If this test fails, a field default, Pydantic serialization rule, or
    dependency version has changed.  Do NOT update the canary without explicit
    review — the hash change is the signal that provenance fingerprints from
    prior runs are no longer comparable to new runs.
    """
    cfg = DetectorConfig()
    actual = get_config_sha256(cfg)
    assert actual == _CANARY_HASH, (
        f"Config SHA-256 changed!\n"
        f"  expected: {_CANARY_HASH}\n"
        f"  actual:   {actual}\n"
        "Report which field default or dependency version changed before "
        "updating this canary."
    )


def test_config_sha256_list_vs_tuple_normalization():
    """NonlinearityConfig.coefficients accepts list or tuple; both hash identically.

    Pydantic v2 coerces a list to tuple[float, ...] during validation, so the
    canonical JSON representation is identical regardless of the Python type
    used at construction time.
    """
    cfg_list = DetectorConfig(
        nonlinearity=NonlinearityConfig(coefficients=[1.0, -2.7e-6, 7.8e-12])
    )
    cfg_tuple = DetectorConfig(
        nonlinearity=NonlinearityConfig(coefficients=(1.0, -2.7e-6, 7.8e-12))
    )
    assert get_config_sha256(cfg_list) == get_config_sha256(cfg_tuple), (
        "Configs built with list vs. tuple coefficients must produce the same hash"
    )
