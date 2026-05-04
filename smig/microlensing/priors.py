"""Microlensing event sampler: priors on lens mass, distance, and geometry.

SI constants are defined once here. All derived quantities flow from these.
See docs/phase3_2_design.md §3 for the full derivation chain.

Galactic model:
  - Lens mass: Kroupa (2001) 3-piece IMF, capped at 2 M_sun.
  - Lens distance: exponential disk (h_z=0.3 kpc, h_R=2.5 kpc) + COBE E2 bulge.
  - Relative proper motion: isotropic Gaussian, sigma=5 mas/yr.

RNG ownership: The caller constructs the Generator from:
    rng = np.random.default_rng(derive_stage_seed(event_seed, "microlensing_sample"))
and passes it in. sample_event never constructs a Generator internally.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from smig.catalogs.base import StarRecord
from smig.microlensing.backends import get_primary_backend
from smig.microlensing.errors import ClaretGridError
from smig.microlensing.event import EventClass, MicrolensingEvent, SourceProperties
from smig.microlensing import limb_darkening

# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------
_G_SI = 6.67430e-11       # m³ kg⁻¹ s⁻²  (NIST 2018)
_M_SUN_KG = 1.98847e30   # kg             (IAU 2015)
_KPC_M = 3.08568e19       # m / kpc
_C_MS = 2.99792458e8      # m / s          (exact)
_MAS_RAD = math.pi / (180.0 * 3600.0 * 1000.0)  # rad / mas

# ---------------------------------------------------------------------------
# LD band for source radius + LD coefficient computation
# ---------------------------------------------------------------------------
_LD_BAND = "H"  # H-band proxy for Roman F146; see docs/phase3_2_design.md §6.2

# ---------------------------------------------------------------------------
# Kroupa (2001) IMF
# ---------------------------------------------------------------------------
_IMF_BREAKS = (0.08, 0.5)           # M_sun
_IMF_ALPHAS = (0.3, 1.3, 2.3)       # power-law indices per segment
_IMF_M_MIN = 0.08                   # M_sun (lower integration limit)
_IMF_M_MAX = 2.0                    # M_sun (cap: avoid BH regime in Phase 3)

# ---------------------------------------------------------------------------
# Galactic density parameters
# ---------------------------------------------------------------------------
_DISK_H_Z = 0.3    # kpc — exponential disk scale height
_DISK_H_R = 2.5    # kpc — exponential disk scale length
_DISK_R_SUN = 8.0  # kpc — Sun's Galactocentric radius (Gravity Collab. 2019)
_MU_REL_SIGMA_MAS_YR = 5.0  # mas/yr — relative proper-motion dispersion

# Bulge weight relative to disk along the Roman bulge sightline (l,b)≈(1.5°,−1.5°)
# Approximate ratio from mass density models (Cao et al. 2013).
_P_BULGE = 0.35   # fraction of lenses from the COBE E2 bulge


def _imf_unnormalized(m: float) -> float:
    """Unnormalized Kroupa (2001) IMF: dN/dm ∝ m^(-α(m))."""
    if m < _IMF_BREAKS[0]:
        return m ** (-_IMF_ALPHAS[0])
    if m < _IMF_BREAKS[1]:
        return m ** (-_IMF_ALPHAS[1])
    return m ** (-_IMF_ALPHAS[2])


# Supremum of m * _imf_unnormalized(m) over [_IMF_M_MIN, _IMF_M_MAX].
# Each piecewise segment is a monotone power law, so the maximum within each
# segment is at one of its endpoints.  Evaluate at all breakpoints and endpoints
# using the actual piecewise function (not cross-product terms) to ensure
# correctness for any parameter configuration.
_IMF_PROPOSAL_MAX: float = max(
    _IMF_M_MIN * _imf_unnormalized(_IMF_M_MIN),
    _IMF_BREAKS[0] * _imf_unnormalized(_IMF_BREAKS[0]),
    _IMF_BREAKS[1] * _imf_unnormalized(_IMF_BREAKS[1]),
    _IMF_M_MAX * _imf_unnormalized(_IMF_M_MAX),
)


def _sample_kroupa_mass(rng: np.random.Generator) -> float:
    """Sample lens mass from Kroupa (2001) IMF via rejection sampling.

    Proposal is uniform in log m over [log(_IMF_M_MIN), log(_IMF_M_MAX)].
    For mass-element sampling (dN/dm), the acceptance ratio is m^(1-α(m))
    normalised by its global supremum ``_IMF_PROPOSAL_MAX``.
    """
    while True:
        log_m = rng.uniform(math.log(_IMF_M_MIN), math.log(_IMF_M_MAX))
        m = math.exp(log_m)
        accept = (m * _imf_unnormalized(m)) / _IMF_PROPOSAL_MAX
        if rng.uniform() < accept:
            return m


def _disk_weight(d: float) -> float:
    """Disk rejection weight at lens distance d: d² · exp(−d / h_R)."""
    return d ** 2 * math.exp(-d / _DISK_H_R)


# Global maximum of _disk_weight on (0, ∞): attained at D_L = 2 * h_R.
_DISK_WEIGHT_MAX: float = _disk_weight(2.0 * _DISK_H_R)


def _sample_lens_distance(rng: np.random.Generator, D_S_kpc: float) -> float:
    """Sample lens distance D_L from a Galactic disk + bulge prior.

    The PDF is proportional to rho_lens(D_L) × D_L² (volume factor).
    Uses rejection sampling against a uniform envelope.

    Disk envelope: _disk_weight(D_L) = D_L² · exp(-D_L / h_R) attains its
    global maximum at D_L = 2·h_R (_DISK_WEIGHT_MAX). When D_S < 2·h_R the
    function is monotone-increasing on [0, D_S], so the tighter cap D_S is used
    to avoid unnecessary rejections.
    """
    # Supremum of _disk_weight on [0, D_S]: at 2*h_R if reachable, else at D_S.
    if D_S_kpc >= 2.0 * _DISK_H_R:
        max_disk_weight = _DISK_WEIGHT_MAX
    else:
        max_disk_weight = _disk_weight(D_S_kpc)
    if max_disk_weight <= 0.0:
        max_disk_weight = 1e-30  # defensive — never used in practice (D_S>0)

    while True:
        if rng.uniform() < _P_BULGE:
            # Bulge: uniform in [0.1, D_S) as a rough approximation to COBE E2.
            D_L = rng.uniform(0.1, 0.99 * D_S_kpc)
        else:
            # Exponential disk: rejection-sample against uniform proposal,
            # weighting by D_L^2 * exp(-D_L / h_R).
            D_L = rng.uniform(0.0, D_S_kpc)
            if rng.uniform() > _disk_weight(D_L) / max_disk_weight:
                continue
        if 0.0 < D_L < D_S_kpc:
            return D_L


def _derive_theta_E_mas(M_L_msun: float, D_L_kpc: float, D_S_kpc: float) -> float:
    """Compute Einstein ring angular radius in mas (see design doc §3.2).

    Correct formula: θ_E² = (4GM/c²) × D_LS / (D_L × D_S)
    where D_LS = D_S - D_L (all in metres).
    """
    D_L_m = D_L_kpc * _KPC_M
    D_S_m = D_S_kpc * _KPC_M
    D_LS_m = D_S_m - D_L_m
    if D_LS_m <= 0.0:
        return 0.0
    # θ_E² = 4GM/c² × D_LS / (D_L × D_S)
    theta_E_sq_rad = (4.0 * _G_SI * M_L_msun * _M_SUN_KG * D_LS_m
                      / (_C_MS**2 * D_L_m * D_S_m))
    return math.sqrt(theta_E_sq_rad) / _MAS_RAD


def _derive_rho(source_star: StarRecord, theta_E_mas: float) -> float:
    """Compute source angular radius / θ_E using strict SI chain (design doc §3.3).

    Raises:
        ValueError: if log_g is outside the physically meaningful range [0.5, 6.0].
    """
    log_g = source_star.log_g
    if not (0.5 <= log_g <= 6.0):
        raise ValueError(
            f"log_g={log_g!r} is outside the valid range [0.5, 6.0]. "
            "Use only StarRecord objects from validated catalog providers."
        )
    g_si = (10.0 ** log_g) / 100.0                    # cgs → m/s²
    M_star_kg = source_star.mass_msun * _M_SUN_KG
    R_star_m = math.sqrt(_G_SI * M_star_kg / g_si)    # do NOT use Stefan-Boltzmann
    D_S_m = source_star.distance_kpc * _KPC_M
    theta_star_rad = R_star_m / D_S_m
    theta_star_mas = theta_star_rad / _MAS_RAD
    if theta_E_mas <= 0.0:
        return 0.0
    return theta_star_mas / theta_E_mas


def _theta_star_mas(source_star: StarRecord) -> float:
    """Angular radius of the source star in mas, using the same SI chain as _derive_rho."""
    g_si = (10.0 ** source_star.log_g) / 100.0
    R_star_m = math.sqrt(_G_SI * source_star.mass_msun * _M_SUN_KG / g_si)
    return (R_star_m / (source_star.distance_kpc * _KPC_M)) / _MAS_RAD


def _classify(q: float, u0: float, rho: float) -> EventClass:
    """Assign EventClass per the precedence rules in docs/phase3_2_design.md §5.3."""
    if q == 0.0 and rho < 1e-3:
        return EventClass.PSPL
    if q == 0.0:
        return EventClass.FSPL_STAR
    if u0 < 0.05:                   # HIGH_MAGNIFICATION_CUSP fires FIRST for binary
        return EventClass.HIGH_MAGNIFICATION_CUSP
    if q < 0.03:
        return EventClass.PLANETARY_CAUSTIC
    return EventClass.STELLAR_BINARY


_MAX_RETRIES = 1000


def sample_event(
    rng: np.random.Generator,
    source_star: StarRecord,
    event_id: str,
    event_class_target: Optional[EventClass] = None,
    strict_ld_grid: bool = True,
) -> MicrolensingEvent:
    """Sample a MicrolensingEvent from Galactic microlensing priors.

    The caller is responsible for constructing rng:
        event_seed  = derive_event_seed(master_seed, event_id)
        stage_seed  = derive_stage_seed(event_seed, "microlensing_sample")
        rng         = np.random.default_rng(stage_seed)

    Args:
        rng:                Pre-constructed NumPy Generator (caller owns seeding).
        source_star:        Frozen StarRecord for the background source.
        event_id:           Stable string identifier written to MicrolensingEvent.event_id.
        event_class_target: If given, restrict sampling to events of this class.
        strict_ld_grid:     If True, ClaretGridError is raised for out-of-grid sources.
                            If False, nearest-neighbour fallback is used.

    Returns:
        A fully populated frozen MicrolensingEvent.

    Raises:
        ValueError:    If source_star.log_g is outside [0.5, 6.0].
        ClaretGridError: If strict_ld_grid=True and source is outside LD grid.
        RuntimeError:  If event_class_target cannot be achieved in _MAX_RETRIES tries.
    """
    # Validate log_g before any computation (fail-fast; no clamping)
    if not (0.5 <= source_star.log_g <= 6.0):
        raise ValueError(
            f"log_g={source_star.log_g!r} is outside the valid range [0.5, 6.0]. "
            "Use only StarRecord objects from validated catalog providers."
        )

    source_props = SourceProperties(
        teff_K=source_star.teff_K,
        log_g=source_star.log_g,
        metallicity_feh=source_star.metallicity_feh,
        distance_kpc=source_star.distance_kpc,
        mass_msun=source_star.mass_msun,
    )

    backend_name, backend_version = get_primary_backend()

    # Fast feasibility check: determine whether the target class is physically
    # achievable for this source star before spending _MAX_RETRIES attempts.
    # PSPL requires rho < 1e-3. We bound rho from below using the largest
    # Einstein ring producible within the prior (M_L = cap, D_L = 0.01 kpc).
    # If even that configuration yields rho >= 1e-3, no lens draw can produce PSPL.
    if event_class_target == EventClass.PSPL:
        _theta_E_bound = _derive_theta_E_mas(_IMF_M_MAX, 0.01, source_star.distance_kpc)
        if _theta_E_bound > 0.0:
            _rho_bound = _theta_star_mas(source_star) / _theta_E_bound
        else:
            _rho_bound = math.inf
        if _rho_bound >= 1e-3:
            raise RuntimeError(
                f"PSPL is not achievable for source {source_star.source_id!r} "
                f"(Teff={source_star.teff_K:.0f} K, log_g={source_star.log_g:.2f}, "
                f"D_S={source_star.distance_kpc:.2f} kpc): "
                f"minimum achievable rho \u2248 {_rho_bound:.4f} \u2265 1e-3. "
                "Source angular radius exceeds the Einstein ring for all prior-space lenses. "
                "Use FSPL_STAR target for giant or nearby sources."
            )

    for attempt in range(_MAX_RETRIES):
        # --- lens mass ---
        M_L = _sample_kroupa_mass(rng)

        # --- lens distance ---
        D_L = _sample_lens_distance(rng, source_star.distance_kpc)
        if D_L >= source_star.distance_kpc:
            continue

        # --- θ_E (single source of truth) ---
        theta_E_mas = _derive_theta_E_mas(M_L, D_L, source_star.distance_kpc)
        if theta_E_mas <= 0.0:
            continue

        # --- ρ (strict SI chain) ---
        rho = _derive_rho(source_star, theta_E_mas)

        if event_class_target == EventClass.PSPL and rho >= 1e-3:
            continue
        if event_class_target == EventClass.FSPL_STAR and rho < 1e-3:
            continue

        # --- relative proper motion → tE ---
        mu_rel_mas_yr = abs(rng.normal(0.0, _MU_REL_SIGMA_MAS_YR))
        if mu_rel_mas_yr < 0.1:
            continue
        mu_rel_mas_day = mu_rel_mas_yr / 365.25  # explicit day/year conversion
        tE_days = theta_E_mas / mu_rel_mas_day

        # --- binary / single-lens geometry ---
        if event_class_target in (EventClass.PSPL, EventClass.FSPL_STAR):
            # Single-lens targets always have q=s=0.
            q, s, alpha = 0.0, 0.0, 0.0
            u0 = rng.uniform(0.0, 1.5)
        elif event_class_target == EventClass.HIGH_MAGNIFICATION_CUSP:
            u0 = rng.uniform(0.0, 0.05)            # conditional draw
            q = 10.0 ** rng.uniform(-5.0, 0.0)
            s = 10.0 ** rng.uniform(math.log10(0.3), math.log10(3.0))
            alpha = rng.uniform(0.0, 2.0 * math.pi)
        elif event_class_target == EventClass.PLANETARY_CAUSTIC:
            u0 = rng.uniform(0.05, 1.5)            # avoid HMC
            q = 10.0 ** rng.uniform(-5.0, math.log10(0.03))
            s = 10.0 ** rng.uniform(math.log10(0.3), math.log10(3.0))
            alpha = rng.uniform(0.0, 2.0 * math.pi)
        elif event_class_target == EventClass.STELLAR_BINARY:
            u0 = rng.uniform(0.05, 1.5)
            q = 10.0 ** rng.uniform(math.log10(0.03), 0.0)
            s = 10.0 ** rng.uniform(math.log10(0.3), math.log10(3.0))
            alpha = rng.uniform(0.0, 2.0 * math.pi)
        else:
            # event_class_target is None — full prior: ~40% single-lens, ~60% binary.
            if rng.uniform() < 0.4:
                q, s, alpha = 0.0, 0.0, 0.0
            else:
                q = 10.0 ** rng.uniform(-5.0, 0.0)
                s = 10.0 ** rng.uniform(math.log10(0.3), math.log10(3.0))
                alpha = rng.uniform(0.0, 2.0 * math.pi)
            u0 = rng.uniform(0.0, 1.5)

        # --- classify ---
        event_class = _classify(q, u0, rho)

        # --- accept / reject ---
        if event_class_target is not None and event_class != event_class_target:
            continue

        # --- LD coefficient: only needed for FSPL_STAR ---
        if event_class == EventClass.FSPL_STAR:
            _, ld_fallback_used = limb_darkening.get_coefficient(
                source_props, _LD_BAND, strict=strict_ld_grid
            )
        else:
            ld_fallback_used = False

        # --- assign backend provenance ---
        if event_class in (EventClass.PSPL, EventClass.FSPL_STAR):
            ev_backend = "analytic"
            ev_backend_version = "N/A"
        else:
            ev_backend = backend_name
            ev_backend_version = backend_version

        return MicrolensingEvent(
            event_id=event_id,
            t0_mjd=0.0,       # caller sets absolute time if needed; default 0.0
            tE_days=tE_days,
            u0=u0,
            rho=rho,
            alpha_rad=alpha,
            q=q,
            s=s,
            pi_E_N=0.0,
            pi_E_E=0.0,
            theta_E_mas=theta_E_mas,
            lens_mass_msun=M_L,
            lens_distance_kpc=D_L,
            event_class=event_class,
            backend=ev_backend,
            backend_version=ev_backend_version,
            ld_fallback_used=ld_fallback_used,
        )

    raise RuntimeError(
        f"Could not sample event of class {event_class_target!r} "
        f"in {_MAX_RETRIES} attempts. Check prior parameters."
    )
