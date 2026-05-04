"""Finite-source point-lens (FSPL) magnification via direct numerical integration.

Uses the classical Claret linear LD law:
    S(r) = 1 - a_linear * (1 - sqrt(1 - r^2))
where r = r_phys / rho is the normalised radial coordinate (0 to 1).

Magnification formula (derived in docs/phase3_2_design.md §6.5):
    A_fs = [6 / (3 - a)] * integral_0^1 r * <A_ps(r*rho, u)>_phi * S(r) dr

The azimuthal average <A_ps>_phi uses a 32-point Gauss-Legendre quadrature.
The radial integral uses scipy.integrate.quad.
For z = u/rho > 10, the PSPL result is returned directly (error < 0.1%).
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad

from smig.microlensing.event import SourceProperties
from smig.microlensing import limb_darkening

# Pre-compute GL nodes and weights for azimuthal average over [0, pi]
_GL_N = 32
_GL_X, _GL_W = leggauss(_GL_N)
_GL_PHI = np.pi * (1.0 + _GL_X) / 2.0    # map [-1,1] -> [0, pi]
_GL_PHI_W = np.pi / 2.0 * _GL_W           # Jacobian
_GL_COS_PHI = np.cos(_GL_PHI)
_GL_SIN_PHI = np.sin(_GL_PHI)

_PSPL_FALLBACK_Z = 10.0   # use PSPL for z = u/rho > this threshold

# Roman F146 is not in the Claret 2000 grid; H-band is the designated proxy
# (see docs/phase3_2_design.md §6.2 and priors.py _LD_BAND).
_LD_BAND_MAP: dict[str, str] = {"F146": "H"}


def _pspl_A(u: float) -> float:
    return (u * u + 2.0) / (u * np.sqrt(u * u + 4.0))


def _azimuthal_avg_A(r_phys: float, u_center: float) -> float:
    """Average PSPL magnification over the azimuthal ring at radius r_phys."""
    u_locs = np.sqrt(
        (u_center + r_phys * _GL_COS_PHI) ** 2 + (r_phys * _GL_SIN_PHI) ** 2
    )
    u_locs = np.maximum(u_locs, 1e-10)
    A_vals = (u_locs**2 + 2.0) / (u_locs * np.sqrt(u_locs**2 + 4.0))
    return float(np.sum(A_vals * _GL_PHI_W) / np.pi)


def _fspl_scalar(u_center: float, rho: float, a_linear: float) -> float:
    """FSPL magnification for a single (u_center, rho, a_linear) triplet."""
    z = u_center / rho if rho > 0 else np.inf
    if z > _PSPL_FALLBACK_Z:
        return _pspl_A(u_center)

    def integrand(r_norm: float) -> float:
        r_phys = r_norm * rho
        A_ring = _azimuthal_avg_A(r_phys, u_center)
        mu = np.sqrt(max(0.0, 1.0 - r_norm * r_norm))
        S = 1.0 - a_linear * (1.0 - mu)
        return r_norm * A_ring * S

    num, _ = quad(integrand, 0.0, 1.0, limit=100, epsabs=1e-6, epsrel=1e-6)
    # Denominator = (3 - a_linear) / 6 (exact analytic result; see design doc)
    den = (3.0 - a_linear) / 6.0
    return num / den


def magnification_fspl(
    t_mjd: np.ndarray,
    t0: float,
    tE: float,
    u0: float,
    rho: float,
    source_props: SourceProperties,
    band: str,
) -> np.ndarray:
    """FSPL magnification lightcurve with Claret linear limb darkening.

    LD coefficient is looked up from the Claret 2000 H-band grid (F146 proxy).
    Uses strict=False so that a fallback coefficient is returned when source
    parameters are near but outside the grid (ld_fallback_used flag was already
    set on the frozen event at construction time).
    """
    ld_band = _LD_BAND_MAP.get(band, band)
    a_linear, _ = limb_darkening.get_coefficient(source_props, ld_band, strict=False)
    tau = (t_mjd - t0) / tE
    u_centers = np.sqrt(u0**2 + tau**2)
    return np.array([_fspl_scalar(u, rho, a_linear) for u in u_centers])
