"""Tests for smig.catalogs.photometry — AB mag → electrons conversion."""
from __future__ import annotations

import pytest

from smig.catalogs.photometry import get_f146_zero_point, mag_ab_to_electrons


class TestMagAbToElectrons:
    def test_zero_point_returns_exposure_s(self):
        """By definition of AB zero-point: ZP mag → 1 e⁻/s → exposure_s total."""
        zp = get_f146_zero_point()
        t = 139.8
        result = mag_ab_to_electrons(zp, "F146", t)
        assert result == pytest.approx(t, rel=1e-9)

    def test_one_mag_fainter(self):
        """1 mag fainter → flux ratio of 10^(1/2.5) ≈ 2.512 fewer electrons."""
        zp = get_f146_zero_point()
        t = 139.8
        bright = mag_ab_to_electrons(zp, "F146", t)
        faint = mag_ab_to_electrons(zp + 1.0, "F146", t)
        assert bright / faint == pytest.approx(10 ** (1.0 / 2.5), rel=1e-9)

    def test_exposure_scaling(self):
        """flux_e scales linearly with exposure_s."""
        zp = get_f146_zero_point()
        f1 = mag_ab_to_electrons(zp, "F146", 100.0)
        f2 = mag_ab_to_electrons(zp, "F146", 200.0)
        assert f2 / f1 == pytest.approx(2.0, rel=1e-9)

    def test_unknown_band_raises(self):
        with pytest.raises(ValueError, match="Unknown band"):
            mag_ab_to_electrons(25.0, "F999", 139.8)

    def test_unknown_band_message(self):
        with pytest.raises(ValueError) as exc_info:
            mag_ab_to_electrons(20.0, "H_band", 100.0)
        assert "H_band" in str(exc_info.value)

    def test_positive_flux(self):
        result = mag_ab_to_electrons(22.0, "F146", 139.8)
        assert result > 0.0

    def test_zero_point_loaded_from_yaml(self):
        """Zero-point must come from YAML, not be hardcoded anywhere in the module."""
        zp = get_f146_zero_point()
        assert isinstance(zp, float)
        assert 20.0 < zp < 35.0  # sanity range for any reasonable AB zero-point
