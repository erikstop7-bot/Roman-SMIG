"""Backend version pinning tests.

Checks:
- get_primary_backend() returns ("VBBinaryLensing", "3.7.0").
- _PINNED_VERSION in backends.py matches pyproject.toml [project.dependencies].
- Module-level assertion fires on import (already tested by importing backends).
"""
from __future__ import annotations

import importlib.metadata
import sys
import tomllib
from pathlib import Path

import pytest

from smig.microlensing.backends import _BACKEND_NAME, _PINNED_DIST, _PINNED_VERSION, get_primary_backend

# Probe once at collection time; used only to skip the version-match test below.
try:
    importlib.metadata.version(_PINNED_DIST)
    _vbbl_metadata_present = True
except importlib.metadata.PackageNotFoundError:
    _vbbl_metadata_present = False

_PYPROJECT = Path(__file__).parent.parent.parent.parent / "pyproject.toml"


class TestGetPrimaryBackend:
    def test_returns_tuple(self):
        result = get_primary_backend()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_backend_name_is_exact(self):
        name, _ = get_primary_backend()
        assert name == "VBBinaryLensing", (
            "Logical backend name must be exactly 'VBBinaryLensing'"
        )

    def test_version_is_non_empty(self):
        _, version = get_primary_backend()
        assert version and version != "N/A"

    @pytest.mark.skipif(
        not _vbbl_metadata_present,
        reason="VBBinaryLensing not installed; cannot compare get_primary_backend() version to package metadata",
    )
    def test_version_matches_installed(self):
        _, version = get_primary_backend()
        installed = importlib.metadata.version(_PINNED_DIST)
        assert version == installed, (
            f"get_primary_backend version {version!r} != installed {installed!r}"
        )

    @pytest.mark.skipif(
        _vbbl_metadata_present,
        reason="VBBinaryLensing is installed; this test only applies when it is absent",
    )
    def test_require_installed_raises_when_absent(self):
        """C.2 regression: require_installed=True raises ImportError when VBBL missing."""
        with pytest.raises(ImportError, match="VBBinaryLensing"):
            get_primary_backend(require_installed=True)


class TestPyprojectPin:
    def test_pinned_version_in_pyproject_dependencies(self):
        with _PYPROJECT.open("rb") as f:
            cfg = tomllib.load(f)
        deps = cfg.get("project", {}).get("dependencies", [])
        pinned_entry = f"VBBinaryLensing=={_PINNED_VERSION}"
        assert any(pinned_entry in dep for dep in deps), (
            f"{pinned_entry!r} not found in [project.dependencies]. "
            f"Got: {deps}"
        )

    def test_phase3_test_optional_group_exists(self):
        with _PYPROJECT.open("rb") as f:
            cfg = tomllib.load(f)
        opt_deps = cfg.get("project", {}).get("optional-dependencies", {})
        assert "phase3-test" in opt_deps, (
            "[project.optional-dependencies].phase3-test not found in pyproject.toml"
        )
        assert any("MulensModel" in d for d in opt_deps["phase3-test"]), (
            "MulensModel not listed in [project.optional-dependencies].phase3-test"
        )
