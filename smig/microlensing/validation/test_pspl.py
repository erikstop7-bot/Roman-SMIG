"""PSPL magnification accuracy tests.

Checks:
- Peak magnification at u0=1e-3 within 0.1% of 1/u0 (high-mag limit A ≈ 1/u).
- Baseline asymptote |t-t0|=10·tE within 1e-4 of 1.0.
- Time symmetry around t0 within 1e-12 relative error.
"""
from __future__ import annotations

import numpy as np
import pytest

from smig.microlensing.pspl import magnification_pspl


class TestPSPLAccuracy:
    def test_peak_near_high_mag_limit(self):
        u0 = 1e-3
        t0, tE = 0.0, 20.0
        t_peak = np.array([t0])
        A = magnification_pspl(t_peak, t0, tE, u0)[0]
        A_expected = (u0**2 + 2) / (u0 * np.sqrt(u0**2 + 4))
        assert abs(A - A_expected) / A_expected < 1e-10, "Exact formula mismatch"
        # High-mag limit: A ≈ 1/u0 for u0 << 1
        A_high_mag = 1.0 / u0
        assert abs(A - A_high_mag) / A_high_mag < 1e-3, (
            f"Peak magnification {A:.6f} deviates from 1/u0={A_high_mag:.6f} by more than 0.1%"
        )

    def test_baseline_asymptote(self):
        # At |t-t0| = 20*tE: tau=20, u = sqrt(u0^2 + 400) >> 1.
        # A - 1 ~ 2/u^4 ~ 2/160000 ~ 1.25e-5 << 1e-4.
        u0 = 0.5
        t0, tE = 0.0, 10.0
        t_far = np.array([t0 + 20.0 * tE, t0 - 20.0 * tE])
        A = magnification_pspl(t_far, t0, tE, u0)
        for a in A:
            assert abs(a - 1.0) < 1e-4, (
                f"Baseline magnification {a:.8f} deviates from 1.0 by more than 1e-4"
            )

    def test_time_symmetry(self):
        u0 = 0.3
        t0, tE = 100.0, 15.0
        dt = np.array([1.0, 5.0, 9.9])
        A_plus  = magnification_pspl(t0 + dt, t0, tE, u0)
        A_minus = magnification_pspl(t0 - dt, t0, tE, u0)
        np.testing.assert_allclose(A_plus, A_minus, rtol=1e-12,
                                   err_msg="PSPL must be symmetric around t0")

    def test_vectorised_output_shape(self):
        t = np.linspace(-50, 50, 200)
        A = magnification_pspl(t, 0.0, 20.0, 0.2)
        assert A.shape == t.shape

    def test_always_greater_than_one(self):
        t = np.linspace(-100, 100, 500)
        A = magnification_pspl(t, 0.0, 20.0, 0.01)
        assert np.all(A >= 1.0), "PSPL magnification must be ≥ 1 everywhere"
