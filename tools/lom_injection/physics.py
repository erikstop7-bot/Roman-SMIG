"""LOM-aware 2L1S magnification wrapper for the SMIG Phase 3 LOM ablation sprint.

Linear LOM Approximation (phenomenological)
-------------------------------------------
Lens orbital motion is approximated as a first-order Taylor expansion around t0:

    s(t_i)     = s0     + ds_dt     [θ_E / yr] * Δt_i [yr]
    alpha(t_i) = alpha0 + dalpha_dt [rad  / yr] * Δt_i [yr]

where Δt_i = (t_i - t0) / 365.25 converts the epoch offset from days to years.

Units
-----
ds_dt :
    Dimensionless θ_E per year.  Because s is the *projected binary separation
    in units of the Einstein ring radius* (not an angular or physical distance),
    ds_dt is the change in that dimensionless ratio per year.  A value of
    0.05 yr⁻¹ means the Einstein-normalised separation drifts by 0.05 per year.
    Sprint bracket values: 0.02, 0.05, 0.10 yr⁻¹ ("weak / typical / aggressive").

dalpha_dt :
    Radians per year.  A value of 0.05 rad yr⁻¹ ≈ 2.9° yr⁻¹, comparable to
    an orbital period of t_E × 2π / |dalpha_dt| ≈ 7 yr for t_E = 60 d.
    Sprint bracket values mirror ds_dt: 0.02, 0.05, 0.10 rad yr⁻¹.

Source Trajectory per Epoch
---------------------------
The rectilinear source trajectory (no parallax) at epoch i is::

    tau_i = (t_i - t0) / tE
    y1_i  =  tau_i * cos(alpha_i) - u0 * sin(alpha_i)   # along binary axis
    y2_i  =  tau_i * sin(alpha_i) + u0 * cos(alpha_i)   # perpendicular

These coordinates are identical to those in smig.microlensing.binary, evaluated
at the epoch-specific (s_i, alpha_i) instead of the static (s0, alpha0).

Alpha Continuity — No Wrapping
-------------------------------
alpha(t) is computed as a continuous real-valued function.  **It is never wrapped
to [0, 2π) or [−π, +π).**  Wrapping would introduce a discontinuity at the wrap
boundary that has no physical justification and would inject an impulse into the
source-trajectory co-ordinates (y1, y2), corrupting the magnification across
that epoch.  Callers that supply large |dalpha_dt| * baseline values must
accept that alpha may fall far outside [−π, +π]; this is correct.  The VBBL
BinaryMag2 call is invariant to the branch choice because the trajectory
coordinates (y1, y2) absorb the full alpha value via cos/sin, which are
themselves 2π-periodic.

VBBL Feasibility Note
---------------------
The pinned VBBinaryLensing C extension (bound via the Python shim installed at
Phase 3.2) exposes ``BinaryMag2(s, q, y1, y2, rho)`` as a scalar-in /
scalar-out function.  Time-varying s and alpha are fully accommodated by calling
``BinaryMag2`` **once per epoch** with the epoch-specific scalar pair
``(s_i, alpha_i)`` — no change to the installed VBBL API or binary is needed.
The per-epoch loop here mirrors the loop already present in
``smig.microlensing.binary.magnification_2l1s``.  Tolerance settings
(``vb.Tol = 1e-3``, ``vb.RelTol = 1e-3``) are copied from that module to
ensure comparable numerical fidelity.

Separation Bounds
-----------------
Large |ds_dt| values over long baselines (|t_i - t0| ≫ 1 yr) can drive s_i
below zero.  A negative or zero separation is unphysical and will crash
BinaryMag2.  ``magnification_2l1s_lom`` raises ``LOMComputationError`` as soon
as s_i ≤ 0 is detected at any epoch, before the VBBL call.  Callers should
choose (ds_dt, s0) pairs such that s0 + ds_dt * Δt_max > 0 over the event
baseline.

Single-Lens Events
------------------
LOM is physically undefined for PSPL and FSPL_STAR events (no binary lens
axis to rotate, no separation to evolve).  Callers are responsible for checking
``event_class`` before dispatching (runner.py enforces this gate).  This
function raises ``ValueError`` if ``q <= 0`` or ``s0 <= 0`` to make misuse
visible at the call site rather than inside the VBBL C extension.

References
----------
Dominik (1998), An. d. Phys., 7, 522 — binary-lens source-trajectory geometry.
An et al. (2002), ApJ, 572, 521 — linear LOM in microlensing light-curve fits.
"""
from __future__ import annotations

import numpy as np

# Mirror smig.microlensing.binary tolerances exactly.
_TOL: float = 1e-3
_REL_TOL: float = 1e-3

_DAYS_PER_YR: float = 365.25


class LOMComputationError(RuntimeError):
    """Raised when VBBinaryLensing returns an unphysical value under LOM, or
    when a per-epoch separation s_i ≤ 0 is detected before the VBBL call.

    Attributes
    ----------
    epoch_index : int
        Zero-based index into the t_mjd array at which the failure occurred.
    params : dict
        Snapshot of all scalar inputs at that epoch, plus the returned A value
        when it is numerically defined but unphysical (A < 1 or NaN/Inf).
    """

    def __init__(self, *, epoch_index: int, params: dict) -> None:
        self.epoch_index = epoch_index
        self.params = params
        super().__init__(
            f"LOM magnification unphysical at epoch {epoch_index}: {params}"
        )


def magnification_2l1s_lom(
    t_mjd: np.ndarray,
    t0: float,
    tE: float,
    u0: float,
    rho: float,
    alpha0: float,
    q: float,
    s0: float,
    ds_dt: float,
    dalpha_dt: float,
) -> np.ndarray:
    """2L1S finite-source magnification with linear lens orbital motion per epoch.

    Parameters
    ----------
    t_mjd :
        Epoch times in Modified Julian Date (MJD). 1-D array, length n_epochs.
    t0 :
        Time of closest approach of the source to the binary centre of mass
        (MJD).  Reconstructed from ``label__t0_mjd_normalized`` via
        ``t0 = obs_start + t0_norm * (n_epochs - 1) * cadence_days``.
    tE :
        Einstein ring crossing time (days).  Must be > 0.
    u0 :
        Impact parameter at t0 (θ_E units).
    rho :
        Source angular radius normalised to the Einstein ring (θ_E units).
    alpha0 :
        Angle of the source trajectory relative to the binary axis **at t0**
        (radians).  Taken directly from ``label__alpha_rad``.
        alpha(t) is evaluated on a **continuous real branch** — it is never
        wrapped to [0, 2π) or [−π, +π).  See module docstring.
    q :
        Binary mass ratio m2/m1.  Must be > 0; callers must guard against the
        −99 sentinel before calling.
    s0 :
        Projected binary separation at t0 (θ_E units, dimensionless ratio of
        physical separation to Einstein radius).  Must be > 0.
    ds_dt :
        Rate of change of binary separation in θ_E per year.  Positive means
        the separation increases over time.  ds_dt = 0 reduces to static 2L1S.
    dalpha_dt :
        Rate of change of the binary axis angle in radians per year.  Positive
        means counter-clockwise rotation.  dalpha_dt = 0 reduces to static 2L1S.

    Returns
    -------
    np.ndarray
        Magnification array, same shape as ``t_mjd``.  Every element ≥ 1.0.

    Raises
    ------
    ValueError
        If ``tE <= 0`` (division by zero guard).
    ValueError
        If ``q <= 0`` or ``s0 <= 0`` (single-lens / unphysical separation guard).
    LOMComputationError
        If s_i ≤ 0 at any epoch (linear drift has driven separation unphysical),
        or if VBBinaryLensing returns NaN, Inf, or A < 1.0.
    ImportError
        If VBBinaryLensing is not installed in the current Python environment.
    """
    if tE <= 0.0:
        raise ValueError(
            f"tE must be > 0 to avoid division by zero in tau = (t - t0) / tE; got tE={tE}."
        )
    if q <= 0.0:
        raise ValueError(
            f"q must be > 0 for binary-lens LOM; got q={q}. "
            "Check for the −99 sentinel (single-lens event) before calling this function."
        )
    if s0 <= 0.0:
        raise ValueError(
            f"s0 must be > 0 for binary-lens LOM; got s0={s0}."
        )

    t = np.asarray(t_mjd, dtype=float)
    dt_days = t - t0
    dt_yr = dt_days / _DAYS_PER_YR

    # Epoch-varying separation (θ_E / yr) and axis angle (rad / yr).
    # alpha is evaluated on a continuous real branch — NOT wrapped mod 2π.
    s_arr = s0 + ds_dt * dt_yr          # shape (n_epochs,)
    alpha_arr = alpha0 + dalpha_dt * dt_yr  # shape (n_epochs,), unwrapped

    # Pre-check: verify separation is positive at every epoch BEFORE importing
    # VBBinaryLensing.  Vectorised scan is O(n_epochs) and catches unphysical
    # drift early, making the guard testable in environments without VBBL.
    negative_mask = s_arr <= 0.0
    if negative_mask.any():
        first_bad = int(np.flatnonzero(negative_mask)[0])
        raise LOMComputationError(
            epoch_index=first_bad,
            params={
                "s_i": float(s_arr[first_bad]),
                "s0": s0,
                "ds_dt": ds_dt,
                "dt_yr": float(dt_yr[first_bad]),
                "t_mjd": float(t[first_bad]),
                "t0": t0,
                "reason": "s_i <= 0: linear LOM has driven separation unphysical",
            },
        )

    tau = dt_days / tE
    cos_a = np.cos(alpha_arr)
    sin_a = np.sin(alpha_arr)
    y1_arr = tau * cos_a - u0 * sin_a
    y2_arr = tau * sin_a + u0 * cos_a

    # Deferred import mirrors smig.microlensing.binary discipline — VBBL is
    # only imported when this function is actually called, so the module can
    # be imported and inspected even in environments where VBBL is absent.
    import VBBinaryLensing  # noqa: PLC0415

    vb = VBBinaryLensing.VBBinaryLensing()
    vb.Tol = _TOL
    vb.RelTol = _REL_TOL

    result = np.empty(len(t), dtype=float)
    for i in range(len(t)):
        s_i = float(s_arr[i])
        # Defense-in-depth: per-epoch guard in case of floating-point edge cases
        # not caught by the vectorised pre-check above (e.g. exactly-zero).
        if s_i <= 0.0:
            raise LOMComputationError(
                epoch_index=i,
                params={
                    "s_i": s_i,
                    "s0": s0,
                    "ds_dt": ds_dt,
                    "dt_yr": float(dt_yr[i]),
                    "t_mjd": float(t[i]),
                    "t0": t0,
                    "reason": "s_i <= 0: per-epoch guard (post-vectorised check)",
                },
            )

        y1_i = float(y1_arr[i])
        y2_i = float(y2_arr[i])
        epoch_params = {
            "s_i": s_i,
            "q": q,
            "y1": y1_i,
            "y2": y2_i,
            "rho": rho,
            "t_mjd": float(t[i]),
            "s0": s0,
            "alpha0": alpha0,
            "ds_dt": ds_dt,
            "dalpha_dt": dalpha_dt,
        }
        try:
            A = vb.BinaryMag2(s_i, q, y1_i, y2_i, rho)
        except Exception as exc:
            raise LOMComputationError(epoch_index=i, params=epoch_params) from exc
        if np.isnan(A) or np.isinf(A) or A < 1.0:
            raise LOMComputationError(
                epoch_index=i, params={**epoch_params, "A": A}
            )
        result[i] = A
    return result


def verify_static_equivalence(
    t_mjd: np.ndarray,
    t0: float,
    tE: float,
    u0: float,
    rho: float,
    alpha: float,
    q: float,
    s: float,
    rtol: float = 1e-5,
) -> None:
    """Assert that LOM with ds_dt=0, dalpha_dt=0 matches static magnification_2l1s.

    Calling this function with representative binary-event parameters once at
    runner startup immediately catches sign-convention mismatches, tau/beta
    confusions, or VBBL version regressions — all of which would produce a
    systematic bias invisible in the physics unit tests alone

    Parameters
    ----------
    t_mjd :
        Epoch times to evaluate at (MJD). Should span a realistic event window.
    t0, tE, u0, rho, alpha, q, s :
        Standard 2L1S parameters.  Must describe a binary event (q > 0, s > 0).
    rtol :
        Relative tolerance for ``np.allclose``.  Default 1e-5 (tighter than the
        1e-3 VBBL numerical tolerance, to catch sign errors rather than roundoff).

    Raises
    ------
    AssertionError
        If the static and LOM-zero outputs disagree beyond rtol.
    ImportError
        If VBBinaryLensing is not installed.
    """
    from smig.microlensing.binary import magnification_2l1s  # read-only import

    A_static = magnification_2l1s(t_mjd, t0, tE, u0, rho, alpha, q, s)
    A_lom_zero = magnification_2l1s_lom(
        t_mjd, t0, tE, u0, rho, alpha, q, s, ds_dt=0.0, dalpha_dt=0.0
    )

    if not np.allclose(A_static, A_lom_zero, rtol=rtol, atol=0.0):
        denom = np.maximum(np.abs(A_static), 1e-12)
        max_rel_err = float(np.max(np.abs(A_static - A_lom_zero) / denom))
        raise AssertionError(
            f"Static equivalence check failed: max relative error = {max_rel_err:.3e} "
            f"(rtol={rtol}). magnification_2l1s_lom(ds_dt=0, dalpha_dt=0) must be "
            "bit-equivalent to magnification_2l1s for the same parameters. "
            "Check sign conventions in tau, y1, y2."
        )
