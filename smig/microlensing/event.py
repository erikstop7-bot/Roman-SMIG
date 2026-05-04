"""Frozen MicrolensingEvent, SourceProperties, and EventClass for SMIG v2 Phase 3.2.

INTERFACE FROZEN: The MicrolensingEvent field list, SourceProperties field list, and
.magnification(t_mjd, band, source_props) signature are LOCKED as of Phase 3.2.
Any future change is a versioned migration requiring explicit sign-off.

Signal chain note: .magnification() dispatches on event_class to pspl / fspl / binary
modules. It is a pure read-only function — it never mutates the frozen event.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class EventClass(str, Enum):
    PSPL = "PSPL"
    FSPL_STAR = "FSPL_STAR"
    PLANETARY_CAUSTIC = "PLANETARY_CAUSTIC"
    STELLAR_BINARY = "STELLAR_BINARY"
    HIGH_MAGNIFICATION_CUSP = "HIGH_MAGNIFICATION_CUSP"


@dataclass(frozen=True)
class SourceProperties:
    """Stellar source parameters consumed by magnification calculators."""

    teff_K: float
    log_g: float
    metallicity_feh: float
    distance_kpc: float
    mass_msun: float


@dataclass(frozen=True)
class MicrolensingEvent:
    """Frozen, immutable record of one microlensing event.

    All fields are computed once at construction via priors.sample_event.
    Consumers read; they never re-derive theta_E_mas or rho.
    """

    # --- identity ---
    event_id: str

    # --- physics ---
    t0_mjd: float
    tE_days: float
    u0: float
    rho: float
    alpha_rad: float
    q: float = 0.0
    s: float = 0.0
    pi_E_N: float = 0.0
    pi_E_E: float = 0.0
    theta_E_mas: float = 0.0

    # --- lens properties (Phase 3.5 additive; defaults preserve backwards compat) ---
    lens_mass_msun: float = 0.0
    lens_distance_kpc: float = 0.0

    # --- classification ---
    event_class: EventClass = EventClass.PSPL

    # --- provenance ---
    backend: str = "analytic"
    backend_version: str = "N/A"
    ld_fallback_used: bool = False

    def magnification(
        self,
        t_mjd: np.ndarray,
        band: str,
        source_props: SourceProperties,
    ) -> np.ndarray:
        """Compute dimensionless magnification A(t) for this event.

        Pure function — does not mutate self. The returned array has the same
        shape as t_mjd.

        Args:
            t_mjd: Time array in MJD.
            band: Photometric band string (consumed by FSPL for LD; PSPL/binary ignore).
            source_props: Source stellar parameters (consumed by FSPL).
        """
        t_mjd = np.asarray(t_mjd, dtype=float)
        if self.event_class == EventClass.PSPL:
            from smig.microlensing import pspl
            return pspl.magnification_pspl(t_mjd, self.t0_mjd, self.tE_days, self.u0)
        if self.event_class == EventClass.FSPL_STAR:
            from smig.microlensing import fspl
            return fspl.magnification_fspl(
                t_mjd, self.t0_mjd, self.tE_days, self.u0, self.rho, source_props, band
            )
        from smig.microlensing import binary
        return binary.magnification_2l1s(
            t_mjd,
            self.t0_mjd,
            self.tE_days,
            self.u0,
            self.rho,
            self.alpha_rad,
            self.q,
            self.s,
        )
