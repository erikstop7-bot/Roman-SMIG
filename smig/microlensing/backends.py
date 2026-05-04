"""VBBinaryLensing backend version pinning and verification.

Single source of truth: _PINNED_VERSION here.
test_backend_pin.py asserts this matches the "VBBinaryLensing==<version>" entry
in pyproject.toml [project.dependencies].
"""
from __future__ import annotations

import importlib.metadata

# PyPI distribution name — used with importlib.metadata.version()
_PINNED_DIST = "VBBinaryLensing"

# Logical backend identifier — written to MicrolensingEvent.backend for binary events.
# Intentionally kept as a separate constant from _PINNED_DIST: they happen to share
# the same string today, but the logical name and the PyPI distribution name are
# distinct concepts and must be independently updateable.
_BACKEND_NAME = "VBBinaryLensing"

# Pinned exact version — single source of truth; test enforces match to pyproject.toml.
_PINNED_VERSION = "3.7.0"

# Probe installed version without asserting at module import time so that
# PSPL/FSPL import paths remain usable in envs without VBBL.
# Binary events call get_primary_backend(require_installed=True) to fail loudly.
try:
    _installed = importlib.metadata.version(_PINNED_DIST)
    _version_ok: bool = _installed == _PINNED_VERSION
except importlib.metadata.PackageNotFoundError:
    _installed = None
    _version_ok = True  # absent is not a version mismatch


def get_primary_backend(require_installed: bool = False) -> tuple[str, str]:
    """Return (logical_backend_name, exact_version) for the pinned 2L1S backend.

    Parameters
    ----------
    require_installed:
        If True, raise ImportError when VBBL is absent or the installed version
        does not match _PINNED_VERSION.  Call this before constructing VBBL
        objects so that binary-event failures are loud and informative.
    """
    if require_installed:
        if _installed is None:
            raise ImportError(
                f"VBBinaryLensing is required for binary lens events but is not "
                f"installed. Install the pinned version: "
                f"pip install VBBinaryLensing=={_PINNED_VERSION}"
            )
        if not _version_ok:
            raise ImportError(
                f"VBBinaryLensing version mismatch: installed {_installed!r} != "
                f"pinned {_PINNED_VERSION!r}. "
                f"Update pyproject.toml and _PINNED_VERSION together."
            )
    return (_BACKEND_NAME, _PINNED_VERSION)
