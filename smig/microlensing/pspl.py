"""Exact analytic Paczyński (PSPL) magnification.

Band-agnostic. Does not import any other smig.microlensing module.
"""
from __future__ import annotations

import numpy as np


def magnification_pspl(
    t_mjd: np.ndarray,
    t0: float,
    tE: float,
    u0: float,
) -> np.ndarray:
    """PSPL magnification A(t) = (u²+2)/(u·√(u²+4)).

    Args:
        t_mjd: Time array (MJD).
        t0:    Time of closest approach (MJD).
        tE:    Einstein ring crossing time (days).
        u0:    Impact parameter at t0 (dimensionless, units of θ_E).

    Returns:
        Magnification array with same shape as t_mjd.
    """
    tau = (t_mjd - t0) / tE
    u = np.sqrt(u0**2 + tau**2)
    return (u**2 + 2.0) / (u * np.sqrt(u**2 + 4.0))
