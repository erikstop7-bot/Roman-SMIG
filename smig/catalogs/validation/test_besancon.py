"""Tests for smig.catalogs.besancon — Besançon catalog ingestor."""
from __future__ import annotations

from pathlib import Path

import pytest

from smig.catalogs.base import MissingColumnError, StarRecord
from smig.catalogs.besancon import BesanconProvider, _REQUIRED_COLS

_FIXTURE_DIR = Path(__file__).parent / "fixtures"
_SMOKE_CSV = _FIXTURE_DIR / "besancon_smoke.csv"


class TestBesanconCSVLoad:
    def test_loads_smoke_fixture(self):
        prov = BesanconProvider(_SMOKE_CSV)
        assert prov is not None

    def test_returns_star_records(self):
        prov = BesanconProvider(_SMOKE_CSV)
        rng = __import__("numpy").random.default_rng(0)
        stars = prov.sample_field(l_deg=1.0, b_deg=-3.0, fov_deg=1.0, rng=rng)
        assert len(stars) > 0
        assert all(isinstance(s, StarRecord) for s in stars)

    def test_star_fields_populated(self):
        prov = BesanconProvider(_SMOKE_CSV)
        rng = __import__("numpy").random.default_rng(0)
        stars = prov.sample_field(l_deg=1.0, b_deg=-3.0, fov_deg=1.0, rng=rng)
        star = stars[0]
        assert isinstance(star.galactic_l_deg, float)
        assert isinstance(star.galactic_b_deg, float)
        assert isinstance(star.distance_kpc, float)
        assert isinstance(star.mass_msun, float)
        assert isinstance(star.teff_K, float)
        assert isinstance(star.log_g, float)
        assert isinstance(star.metallicity_feh, float)
        assert isinstance(star.mag_F146_ab, float)
        assert isinstance(star.source_id, str)
        assert isinstance(star.catalog_tile_id, str)

    def test_fov_filter(self):
        """Stars outside FOV must be excluded."""
        prov = BesanconProvider(_SMOKE_CSV)
        rng = __import__("numpy").random.default_rng(0)
        # Very small FOV: should return only stars within ±0.001 deg.
        stars = prov.sample_field(l_deg=1.0, b_deg=-3.0, fov_deg=0.001, rng=rng)
        for s in stars:
            assert abs(s.galactic_l_deg - 1.0) <= 0.0005
            assert abs(s.galactic_b_deg - (-3.0)) <= 0.0005

    def test_list_bands(self):
        prov = BesanconProvider(_SMOKE_CSV)
        assert prov.list_bands() == ("F146",)

    def test_missing_column_raises(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("l_deg,b_deg\n1.0,-3.0\n")
        with pytest.raises(MissingColumnError) as exc_info:
            BesanconProvider(bad_csv)
        assert len(exc_info.value.missing) > 0

    def test_unsupported_extension_raises(self, tmp_path):
        f = tmp_path / "bad.txt"
        f.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported"):
            BesanconProvider(f)
