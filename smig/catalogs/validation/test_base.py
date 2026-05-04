"""Tests for smig.catalogs.base — StarRecord frozenness and defaults."""
from __future__ import annotations

import dataclasses
from types import MappingProxyType

import pytest

from smig.catalogs.base import MissingColumnError, StarRecord


def _make_star(**overrides) -> StarRecord:
    defaults = dict(
        galactic_l_deg=1.0,
        galactic_b_deg=-3.0,
        distance_kpc=8.0,
        mass_msun=1.0,
        teff_K=5800.0,
        log_g=4.4,
        metallicity_feh=0.0,
        mag_F146_ab=18.0,
        source_id="x",
        catalog_tile_id="t",
    )
    defaults.update(overrides)
    return StarRecord(**defaults)


class TestStarRecordFrozenness:
    def test_mutation_raises(self):
        star = _make_star()
        with pytest.raises(dataclasses.FrozenInstanceError):
            star.mag_F146_ab = 99.0  # type: ignore[misc]

    def test_mutation_mass_raises(self):
        star = _make_star()
        with pytest.raises(dataclasses.FrozenInstanceError):
            star.mass_msun = 2.0  # type: ignore[misc]

    def test_mutation_source_id_raises(self):
        star = _make_star()
        with pytest.raises(dataclasses.FrozenInstanceError):
            star.source_id = "y"  # type: ignore[misc]


class TestStarRecordDefaults:
    def test_mag_other_ab_is_mappingproxy(self):
        star = _make_star()
        assert type(star.mag_other_ab).__name__ == "mappingproxy"

    def test_mag_other_ab_is_empty(self):
        star = _make_star()
        assert dict(star.mag_other_ab) == {}

    def test_mag_other_ab_immutable(self):
        star = _make_star()
        with pytest.raises((TypeError, AttributeError)):
            star.mag_other_ab["F106"] = 19.0  # type: ignore[index]

    def test_mag_other_ab_populated(self):
        star = _make_star(mag_other_ab=MappingProxyType({"F106": 19.5}))
        assert star.mag_other_ab["F106"] == pytest.approx(19.5)
        assert type(star.mag_other_ab).__name__ == "mappingproxy"

    def test_import_smoke(self):
        from smig.catalogs import CatalogProvider, StarRecord, BesanconProvider, RomanBulgeProvider, SyntheticCatalogProvider
        from types import MappingProxyType
        s = StarRecord(
            galactic_l_deg=0, galactic_b_deg=0, distance_kpc=8, mass_msun=1,
            teff_K=5800, log_g=4.4, metallicity_feh=0, mag_F146_ab=18,
            source_id='x', catalog_tile_id='t'
        )
        assert type(s.mag_other_ab).__name__ == "mappingproxy"


class TestMissingColumnError:
    def test_message_contains_missing(self):
        err = MissingColumnError(["col_a", "col_b"])
        assert "col_a" in str(err)
        assert "col_b" in str(err)

    def test_missing_attribute(self):
        err = MissingColumnError(["x"])
        assert err.missing == ["x"]
