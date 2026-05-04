"""
smig/optics/validation/test_filter_alias.py
============================================
Tests for _resolve_stpsf_filter_name — the W146→F146 alias used in the
WebbPSF/STPSF code path.  No galsim dependency; runs unconditionally.
"""
from __future__ import annotations

import pytest

_WEBBPSF_AVAILABLE = False
try:
    import webbpsf  # noqa: F401
    _WEBBPSF_AVAILABLE = True
except ImportError:
    pass


def test_filter_alias_w146_maps_to_f146() -> None:
    """_resolve_stpsf_filter_name must map W146 to F146 (STPSF convention)."""
    from smig.optics.psf import _resolve_stpsf_filter_name
    assert _resolve_stpsf_filter_name("W146") == "F146"


def test_filter_alias_passthrough_for_valid_names() -> None:
    """Already-valid STPSF filter names must pass through unchanged."""
    from smig.optics.psf import _resolve_stpsf_filter_name
    for name in ("F062", "F087", "F106", "F129", "F158", "F184", "F146", "F213"):
        assert _resolve_stpsf_filter_name(name) == name


def test_filter_alias_unknown_passthrough() -> None:
    """An unknown filter name must pass through unchanged (no silent corruption)."""
    from smig.optics.psf import _resolve_stpsf_filter_name
    assert _resolve_stpsf_filter_name("CUSTOM") == "CUSTOM"


@pytest.mark.skipif(
    not _WEBBPSF_AVAILABLE,
    reason="webbpsf not installed; cannot inspect WFI.filter_list",
)
def test_w146_resolved_name_in_wfi_filter_list() -> None:
    """The resolved name for W146 must appear in webbpsf.roman.WFI().filter_list."""
    import webbpsf
    from smig.optics.psf import _resolve_stpsf_filter_name
    wfi = webbpsf.roman.WFI()
    resolved = _resolve_stpsf_filter_name("W146")
    assert resolved in wfi.filter_list, (
        f"Resolved filter '{resolved}' not in WFI.filter_list: {wfi.filter_list}. "
        "Update _STPSF_FILTER_ALIAS in smig/optics/psf.py to match."
    )
