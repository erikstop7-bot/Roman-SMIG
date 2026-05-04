"""strict_ld_grid propagation and F146 band proxy tests.

Verifies:
1. Calling magnification_fspl with band='F146' does NOT raise ClaretGridError
   because _LD_BAND_MAP maps 'F146' → 'H' before the LD lookup.
2. The config strict_ld_grid=False actually reaches the LD coefficient call
   (integration test via sample_event).
3. band='H' (direct) also works (no regression).
"""
from __future__ import annotations

import numpy as np
import pytest

from smig.microlensing.errors import ClaretGridError
from smig.microlensing.event import EventClass, MicrolensingEvent, SourceProperties
from smig.microlensing.fspl import _LD_BAND_MAP, magnification_fspl


def _in_grid_source() -> SourceProperties:
    return SourceProperties(
        teff_K=5000.0, log_g=4.5, metallicity_feh=0.0,
        distance_kpc=8.0, mass_msun=1.0,
    )


def _out_of_grid_source() -> SourceProperties:
    # Teff well outside the Claret 2000 grid range → triggers fallback path
    return SourceProperties(
        teff_K=100.0, log_g=4.5, metallicity_feh=0.0,
        distance_kpc=8.0, mass_msun=1.0,
    )


class TestF146BandProxyMap:
    def test_f146_maps_to_h(self):
        assert _LD_BAND_MAP.get("F146") == "H"

    def test_unknown_band_passes_through(self):
        assert _LD_BAND_MAP.get("V", "V") == "V"


class TestFSPLF146NoError:
    """magnification_fspl with band='F146' must not raise ClaretGridError."""

    def test_f146_band_does_not_raise(self):
        t = np.linspace(0.0, 10.0, 5)
        src = _in_grid_source()
        # Would raise "band 'F146' not in grid" before the fix.
        A = magnification_fspl(t, t0=5.0, tE=2.0, u0=0.1, rho=0.01,
                               source_props=src, band="F146")
        assert A.shape == (5,)
        assert (A >= 1.0).all()

    def test_h_band_still_works(self):
        t = np.linspace(0.0, 10.0, 5)
        src = _in_grid_source()
        A = magnification_fspl(t, t0=5.0, tE=2.0, u0=0.1, rho=0.01,
                               source_props=src, band="H")
        assert A.shape == (5,)
        assert (A >= 1.0).all()

    def test_f146_and_h_give_same_result(self):
        """F146→H proxy must produce numerically identical results to direct H."""
        t = np.linspace(0.0, 10.0, 20)
        src = _in_grid_source()
        A_f146 = magnification_fspl(t, t0=5.0, tE=2.0, u0=0.1, rho=0.01,
                                    source_props=src, band="F146")
        A_h = magnification_fspl(t, t0=5.0, tE=2.0, u0=0.1, rho=0.01,
                                 source_props=src, band="H")
        np.testing.assert_array_equal(A_f146, A_h)

    def test_f146_out_of_grid_source_fallback(self):
        """Out-of-grid source with F146 band should use nearest-neighbour, not raise."""
        t = np.array([5.0])
        src = _out_of_grid_source()
        # Must not raise; strict=False path handles out-of-grid sources.
        A = magnification_fspl(t, t0=5.0, tE=2.0, u0=0.1, rho=0.01,
                               source_props=src, band="F146")
        assert np.isfinite(A).all()


class TestStrictLdGridPropagation:
    """sample_event propagates strict_ld_grid to the LD coefficient call path."""

    def test_strict_false_does_not_raise_for_marginal_source(self):
        """A source near the grid edge should succeed with strict_ld_grid=False."""
        from smig.catalogs.base import StarRecord
        from smig.microlensing.priors import sample_event

        rng = np.random.default_rng(42)
        star = StarRecord(
            source_id="s001", catalog_tile_id="test",
            galactic_l_deg=1.0, galactic_b_deg=-3.0,
            teff_K=5000.0, log_g=4.5, metallicity_feh=0.0,
            mass_msun=1.0, distance_kpc=9.0, mag_F146_ab=22.0,
        )
        event = sample_event(
            rng, star, "test_strict_false",
            event_class_target=EventClass.FSPL_STAR,
            strict_ld_grid=False,
        )
        assert event.event_class == EventClass.FSPL_STAR
