"""Binary lens (2L1S) regression tests against Suzuki et al. (2016) reference events.

Fast tests: load suzuki2016_sample.json and verify VBBL 3.7.0 reproduces the
committed peak_magnification_vbbl within 1e-3 relative error.

Slow tests (MulensModel cross-check): marked @pytest.mark.slow.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from smig.microlensing.binary import magnification_2l1s
from smig.microlensing.errors import MicrolensingComputationError


def _vbbl_importable() -> bool:
    """Return True iff VBBinaryLensing can be imported (catches OSError on .so load)."""
    try:
        import VBBinaryLensing  # noqa: F401
        return True
    except (ImportError, OSError):
        return False


_requires_vbbl = pytest.mark.skipif(
    not _vbbl_importable(),
    reason="VBBinaryLensing not installed or shared library failed to load",
)

_SUZUKI_JSON = (
    Path(__file__).parent.parent.parent.parent
    / "data" / "microlensing" / "reference_events" / "suzuki2016_sample.json"
)


def _load_suzuki():
    with _SUZUKI_JSON.open() as f:
        return json.load(f)


@_requires_vbbl
class TestSuzukiRegressions:
    """Verify VBBL reproduces committed peak magnifications for all 5 events."""

    @pytest.fixture(scope="class")
    def events(self):
        return _load_suzuki()

    def test_exactly_five_events(self, events):
        assert len(events) == 5

    @pytest.mark.parametrize("idx", range(5))
    def test_peak_magnification_regression(self, events, idx):
        ev = events[idx]
        t0 = ev["t0_mjd"]
        tE = ev["tE_days"]
        u0 = ev["u0"]
        rho = ev["rho"]
        alpha = ev["alpha_rad"]
        q = ev["q"]
        s = ev["s"]
        peak_t = ev["peak_time_mjd"]
        A_ref = ev["peak_magnification_vbbl"]

        A_computed = magnification_2l1s(
            np.array([peak_t]), t0, tE, u0, rho, alpha, q, s
        )[0]
        rel_err = abs(A_computed - A_ref) / A_ref
        assert rel_err < 1e-3, (
            f"{ev['event_name']}: VBBL A={A_computed:.6f} vs. ref={A_ref:.6f}, "
            f"rel_err={rel_err:.2e}"
        )


@_requires_vbbl
class TestMicrolensingComputationError:
    def test_raises_on_unphysical_magnification(self):
        from unittest.mock import patch

        with patch("VBBinaryLensing.VBBinaryLensing") as mock_cls:
            mock_cls.return_value.BinaryMag2.return_value = 0.5  # A < 1.0
            with pytest.raises(MicrolensingComputationError):
                magnification_2l1s(np.array([0.0]), 0.0, 20.0, 0.1, 1e-3, 0.0, 1e-3, 1.0)

    def test_raises_on_nan_magnification(self):
        from unittest.mock import patch

        with patch("VBBinaryLensing.VBBinaryLensing") as mock_cls:
            mock_cls.return_value.BinaryMag2.return_value = float("nan")
            with pytest.raises(MicrolensingComputationError):
                magnification_2l1s(np.array([0.0]), 0.0, 20.0, 0.1, 1e-3, 0.0, 1e-3, 1.0)


@pytest.mark.slow
class TestMulensModelCrossCheck:
    """Cross-check two non-caustic-crossing events against MulensModel."""

    def test_mulensmodel_available(self):
        mm = pytest.importorskip("MulensModel")
        assert mm is not None

    def test_crosscheck_event_0(self):
        mm = pytest.importorskip("MulensModel")
        events = _load_suzuki()
        ev = events[0]  # MOA-2007-BLG-400Lb
        t_test = np.array([ev["t0_mjd"] - 2.0 * ev["tE_days"]])  # far from peak

        A_vbbl = magnification_2l1s(
            t_test, ev["t0_mjd"], ev["tE_days"], ev["u0"],
            ev["rho"], ev["alpha_rad"], ev["q"], ev["s"]
        )[0]

        model = mm.Model(dict(
            t_0=ev["t0_mjd"], u_0=ev["u0"], t_E=ev["tE_days"],
            rho=ev["rho"], alpha=math.degrees(ev["alpha_rad"]),
            s=ev["s"], q=ev["q"],
        ))
        model.set_magnification_methods([
            ev["t0_mjd"] - 3 * ev["tE_days"], "VBBL",
            ev["t0_mjd"] + 3 * ev["tE_days"],
        ])
        A_mm = model.get_magnification(t_test)[0]
        assert abs(A_vbbl - A_mm) / A_mm < 1e-2, (
            f"VBBL={A_vbbl:.6f} vs MulensModel={A_mm:.6f}, rel_err={abs(A_vbbl-A_mm)/A_mm:.2e}"
        )

    def test_crosscheck_event_1(self):
        mm = pytest.importorskip("MulensModel")
        events = _load_suzuki()
        ev = events[1]  # MOA-2008-BLG-310Lb
        t_test = np.array([ev["t0_mjd"] - 1.5 * ev["tE_days"]])

        A_vbbl = magnification_2l1s(
            t_test, ev["t0_mjd"], ev["tE_days"], ev["u0"],
            ev["rho"], ev["alpha_rad"], ev["q"], ev["s"]
        )[0]

        model = mm.Model(dict(
            t_0=ev["t0_mjd"], u_0=ev["u0"], t_E=ev["tE_days"],
            rho=ev["rho"], alpha=math.degrees(ev["alpha_rad"]),
            s=ev["s"], q=ev["q"],
        ))
        model.set_magnification_methods([
            ev["t0_mjd"] - 3 * ev["tE_days"], "VBBL",
            ev["t0_mjd"] + 3 * ev["tE_days"],
        ])
        A_mm = model.get_magnification(t_test)[0]
        assert abs(A_vbbl - A_mm) / A_mm < 1e-2
