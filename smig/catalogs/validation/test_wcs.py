"""Tests for smig.catalogs.wcs — Galactic → SCA pixel projection."""
from __future__ import annotations

import math

import pytest

astropy = pytest.importorskip("astropy", reason="astropy not installed")
import astropy.coordinates as acoords
import astropy.units as u

from smig.catalogs.wcs import (
    _PLATE_SCALE_ARCSEC_PER_PIX,
    _STAMP_CENTER_PIX,
    galactic_to_sca_pixel,
)


class TestFieldCenterMapping:
    def test_field_center_maps_to_stamp_center(self):
        """Field centre (l, b) must map to exactly (128.0, 128.0)."""
        l0, b0 = 1.0, -3.0
        x, y = galactic_to_sca_pixel(l0, b0, sca_id=1, field_center_l_deg=l0, field_center_b_deg=b0)
        assert x == pytest.approx(128.0, abs=1e-8)
        assert y == pytest.approx(128.0, abs=1e-8)

    def test_stamp_center_constant(self):
        assert _STAMP_CENTER_PIX == pytest.approx(128.0)

    def test_plate_scale_constant(self):
        assert _PLATE_SCALE_ARCSEC_PER_PIX == pytest.approx(0.11, rel=1e-6)

    def test_various_field_centers(self):
        """Field centre must map to (128, 128) for any (l, b) value."""
        for l0, b0 in [(0.0, 0.0), (1.0, -3.0), (350.0, -5.0), (180.0, 5.0)]:
            x, y = galactic_to_sca_pixel(l0, b0, 1, l0, b0)
            assert x == pytest.approx(128.0, abs=1e-7), f"l0={l0}, b0={b0}: x={x}"
            assert y == pytest.approx(128.0, abs=1e-7), f"l0={l0}, b0={b0}: y={y}"


class TestLinearScaling:
    """Total pixel displacement scales linearly with angular offset.

    The Galactic and ICRS frames are rotated relative to each other, so a
    Galactic (l, b) offset projects onto BOTH RA and Dec in ICRS.  The
    individual pixel components (dx, dy) are frame-dependent, but the
    TOTAL pixel distance sqrt(dx²+dy²) must equal
    angular_separation / plate_scale to good approximation for the gnomonic
    projection.
    """

    def _total_pix_offset(self, dl_deg: float = 0.0, db_deg: float = 0.0):
        l0, b0 = 0.0, 0.0
        x, y = galactic_to_sca_pixel(
            l0 + dl_deg, b0 + db_deg,
            sca_id=1,
            field_center_l_deg=l0,
            field_center_b_deg=b0,
        )
        return math.hypot(x - 128.0, y - 128.0)

    def _angular_sep_arcsec(self, dl_deg: float = 0.0, db_deg: float = 0.0):
        """True angular separation using astropy."""
        l0, b0 = 0.0, 0.0
        center = acoords.SkyCoord(l=l0 * u.deg, b=b0 * u.deg, frame="galactic")
        star = acoords.SkyCoord(l=(l0 + dl_deg) * u.deg, b=(b0 + db_deg) * u.deg, frame="galactic")
        return float(center.separation(star).arcsec)

    def test_one_plate_scale_north_total_dist(self):
        """0.11 arcsec offset in Galactic b → total pixel distance ≈ 1 px."""
        pix = self._total_pix_offset(db_deg=0.11 / 3600.0)
        assert pix == pytest.approx(1.0, abs=0.01)

    def test_one_plate_scale_east_total_dist(self):
        """0.11 arcsec offset in Galactic l → total pixel distance ≈ 1 px."""
        pix = self._total_pix_offset(dl_deg=0.11 / 3600.0)
        assert pix == pytest.approx(1.0, abs=0.05)

    def test_linear_grid_up_to_one_degree(self):
        """Total pixel distance = angular_sep / plate_scale for offsets up to 1 deg."""
        test_cases = [
            (0.0, 0.001),
            (0.0, 0.01),
            (0.0, 0.1),
            (0.0, 0.5),
            (0.0, 1.0),
            (0.001, 0.0),
            (0.1, 0.0),
            (0.001, 0.001),
        ]
        for dl_deg, db_deg in test_cases:
            ang_arcsec = self._angular_sep_arcsec(dl_deg, db_deg)
            pix = self._total_pix_offset(dl_deg, db_deg)
            expected_pix = ang_arcsec / _PLATE_SCALE_ARCSEC_PER_PIX
            tol = max(0.01 * expected_pix, 0.01)
            assert abs(pix - expected_pix) < tol, (
                f"dl={dl_deg}, db={db_deg}: pix={pix:.4f}, expected {expected_pix:.4f}"
            )
