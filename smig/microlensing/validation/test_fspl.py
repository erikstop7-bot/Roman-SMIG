"""FSPL magnification tests.

Checks:
- z >> 1 (large impact parameter): FSPL → PSPL within 0.1%.
- z << 1 (source centred on lens): FSPL approaches the uniform-disk plateau 2/rho.
- Out-of-grid source under strict_ld_grid=True raises ClaretGridError.
- Out-of-grid source under strict_ld_grid=False sets ld_fallback_used=True on the
  returned frozen event.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from smig.microlensing.errors import ClaretGridError
from smig.microlensing.event import EventClass, MicrolensingEvent, SourceProperties
from smig.microlensing.fspl import _fspl_scalar, magnification_fspl
from smig.microlensing.limb_darkening import get_coefficient
from smig.microlensing.pspl import magnification_pspl


def _source(teff=5000.0, logg=4.5, feh=0.0):
    return SourceProperties(
        teff_K=teff, log_g=logg, metallicity_feh=feh,
        distance_kpc=8.0, mass_msun=1.0,
    )


class TestFSPLLimits:
    def test_large_z_approaches_pspl(self):
        # z=15 > _PSPL_FALLBACK_Z=10, so the code returns PSPL directly.
        # This tests that the fallback path gives exact PSPL.
        rho = 0.01
        u = 15.0 * rho    # z = 15 > threshold
        A_fs = _fspl_scalar(u, rho, a_linear=0.0)
        A_ps = (u**2 + 2) / (u * math.sqrt(u**2 + 4))
        assert abs(A_fs - A_ps) / A_ps < 1e-10, (
            f"FSPL(z=15) fallback should return exact PSPL: got {A_fs:.8f} vs {A_ps:.8f}"
        )

    def test_small_z_plateau_uniform_disk(self):
        rho = 0.05
        u = 0.001 * rho   # z = 0.001, essentially source centred on lens
        A_fs = _fspl_scalar(u, rho, a_linear=0.0)
        A_plateau = 2.0 / rho   # analytic result for uniform disk (see design doc)
        assert abs(A_fs - A_plateau) / A_plateau < 1e-3, (
            f"FSPL plateau {A_fs:.4f} deviates from 2/rho={A_plateau:.4f} by > 0.1%"
        )

    def test_always_greater_than_one(self):
        rho = 0.02
        u_arr = np.linspace(1e-4, 0.5, 20)
        for u in u_arr:
            A = _fspl_scalar(u, rho, a_linear=0.4)
            assert A >= 1.0, f"FSPL gave A={A} < 1 at u={u}"


class TestLDPolicyStrict:
    def test_strict_out_of_grid_raises(self):
        sp_bad = SourceProperties(
            teff_K=1000.0,   # below grid minimum (3500 K)
            log_g=4.5, metallicity_feh=0.0,
            distance_kpc=8.0, mass_msun=1.0,
        )
        with pytest.raises(ClaretGridError):
            get_coefficient(sp_bad, band="H", strict=True)

    def test_non_strict_out_of_grid_fallback(self):
        sp_bad = SourceProperties(
            teff_K=1000.0,   # below grid minimum
            log_g=4.5, metallicity_feh=0.0,
            distance_kpc=8.0, mass_msun=1.0,
        )
        a, was_fallback = get_coefficient(sp_bad, band="H", strict=False)
        assert was_fallback is True
        assert 0.0 < a < 1.0


class TestLDFallbackOnEvent:
    def test_strict_ld_grid_raises_via_priors(self):
        from smig.microlensing.priors import sample_event
        from smig.catalogs.base import StarRecord

        sp_bad = StarRecord(
            galactic_l_deg=1.0, galactic_b_deg=-1.5,
            distance_kpc=8.0, mass_msun=1.0,
            teff_K=1000.0,     # out of grid
            log_g=4.5, metallicity_feh=0.0,
            mag_F146_ab=18.0, source_id="bad", catalog_tile_id="t0",
        )
        rng = np.random.default_rng(42)
        with pytest.raises(ClaretGridError):
            sample_event(
                rng,
                sp_bad,
                event_id="test-bad",
                event_class_target=EventClass.FSPL_STAR,
                strict_ld_grid=True,
            )

    def test_non_strict_ld_sets_fallback_flag(self):
        from smig.microlensing.priors import sample_event
        from smig.catalogs.base import StarRecord

        sp_bad = StarRecord(
            galactic_l_deg=1.0, galactic_b_deg=-1.5,
            distance_kpc=8.0, mass_msun=1.0,
            teff_K=1000.0,     # out of grid
            log_g=4.5, metallicity_feh=0.0,
            mag_F146_ab=18.0, source_id="bad", catalog_tile_id="t0",
        )
        rng = np.random.default_rng(42)
        ev = sample_event(
            rng,
            sp_bad,
            event_id="test-bad",
            event_class_target=EventClass.FSPL_STAR,
            strict_ld_grid=False,
        )
        assert ev.ld_fallback_used is True
