"""Smoke tests for SyntheticCatalogProvider → adapter pipeline."""
from __future__ import annotations

import numpy as np
import pytest

pandas = pytest.importorskip("pandas", reason="pandas not installed")
astropy = pytest.importorskip("astropy", reason="astropy not installed")

from smig.catalogs.adapter import project_to_sca_dataframe
from smig.catalogs.base import StarRecord
from smig.catalogs.synthetic import SyntheticCatalogProvider
from smig.rendering.crowding import _REQUIRED_COLUMNS as _RENDERER_COLS


class TestSyntheticProviderSmoke:
    def test_generates_50_stars(self):
        provider = SyntheticCatalogProvider(n_stars=50)
        rng = np.random.default_rng(7)
        stars = provider.sample_field(l_deg=1.0, b_deg=-3.0, fov_deg=0.28, rng=rng)
        assert len(stars) == 50

    def test_all_star_records(self):
        provider = SyntheticCatalogProvider(n_stars=50)
        rng = np.random.default_rng(7)
        stars = provider.sample_field(l_deg=1.0, b_deg=-3.0, fov_deg=0.28, rng=rng)
        assert all(isinstance(s, StarRecord) for s in stars)

    def test_stars_within_fov(self):
        l0, b0, fov = 1.0, -3.0, 0.28
        provider = SyntheticCatalogProvider(n_stars=50)
        rng = np.random.default_rng(7)
        stars = provider.sample_field(l0, b0, fov, rng)
        half = fov / 2.0
        for s in stars:
            assert abs(s.galactic_l_deg - l0) <= half + 1e-9
            assert abs(s.galactic_b_deg - b0) <= half + 1e-9

    def test_project_50_stars_to_dataframe(self):
        provider = SyntheticCatalogProvider(n_stars=50)
        rng = np.random.default_rng(7)
        stars = provider.sample_field(l_deg=1.0, b_deg=-3.0, fov_deg=0.28, rng=rng)
        df = project_to_sca_dataframe(
            stars, sca_id=1, field_center_l_deg=1.0,
            field_center_b_deg=-3.0, exposure_s=139.8
        )
        # DataFrame validates against renderer requirements.
        assert tuple(df.columns) == _RENDERER_COLS
        assert len(df) == 50
        assert (df["flux_e"] > 0).all()

    def test_list_bands(self):
        provider = SyntheticCatalogProvider()
        assert provider.list_bands() == ("F146",)

    def test_deterministic_with_same_seed(self):
        provider = SyntheticCatalogProvider(n_stars=10)
        s1 = provider.sample_field(1.0, -3.0, 0.28, np.random.default_rng(42))
        s2 = provider.sample_field(1.0, -3.0, 0.28, np.random.default_rng(42))
        for a, b in zip(s1, s2):
            assert a.mag_F146_ab == b.mag_F146_ab
