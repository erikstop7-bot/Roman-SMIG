"""Tests for smig.catalogs.adapter — ProjectedStarTable DataFrame emitter."""
from __future__ import annotations

import numpy as np
import pytest

pandas = pytest.importorskip("pandas", reason="pandas not installed")
astropy = pytest.importorskip("astropy", reason="astropy not installed")

from smig.catalogs.adapter import project_to_sca_dataframe
from smig.catalogs.base import StarRecord
from smig.catalogs.synthetic import SyntheticCatalogProvider

# Import the authoritative column tuple from the renderer (not hardcoded here).
from smig.rendering.crowding import _REQUIRED_COLUMNS as _RENDERER_COLS


def _make_stars(n: int = 5) -> list[StarRecord]:
    rng = np.random.default_rng(42)
    provider = SyntheticCatalogProvider(n_stars=n)
    return provider.sample_field(l_deg=1.0, b_deg=-3.0, fov_deg=0.28, rng=rng)


class TestAdapterColumns:
    def test_columns_match_renderer_required(self):
        """Output DataFrame columns must equal _REQUIRED_COLUMNS exactly."""
        stars = _make_stars()
        df = project_to_sca_dataframe(stars, sca_id=1, field_center_l_deg=1.0,
                                      field_center_b_deg=-3.0, exposure_s=139.8)
        assert tuple(df.columns) == _RENDERER_COLS

    def test_columns_are_not_hardcoded_subset(self):
        """Verify the columns we check come from the renderer import, not a literal."""
        assert "x_pix" in _RENDERER_COLS
        assert "y_pix" in _RENDERER_COLS
        assert "flux_e" in _RENDERER_COLS
        assert "mag_w146" in _RENDERER_COLS

    def test_empty_stars_valid_dataframe(self):
        df = project_to_sca_dataframe([], sca_id=1, field_center_l_deg=1.0,
                                      field_center_b_deg=-3.0, exposure_s=139.8)
        assert tuple(df.columns) == _RENDERER_COLS
        assert len(df) == 0


class TestAdapterDtypes:
    def test_all_float64(self):
        stars = _make_stars()
        df = project_to_sca_dataframe(stars, sca_id=1, field_center_l_deg=1.0,
                                      field_center_b_deg=-3.0, exposure_s=139.8)
        for col in _RENDERER_COLS:
            assert df[col].dtype == np.float64, f"Column {col!r} is not float64"


class TestAdapterValues:
    def test_flux_e_positive(self):
        stars = _make_stars(20)
        df = project_to_sca_dataframe(stars, sca_id=1, field_center_l_deg=1.0,
                                      field_center_b_deg=-3.0, exposure_s=139.8)
        assert (df["flux_e"] > 0.0).all()

    def test_mag_w146_matches_star_mag(self):
        """mag_w146 must equal StarRecord.mag_F146_ab."""
        rng = np.random.default_rng(0)
        provider = SyntheticCatalogProvider(n_stars=10)
        stars = provider.sample_field(1.0, -3.0, 0.28, rng)
        df = project_to_sca_dataframe(stars, sca_id=1, field_center_l_deg=1.0,
                                      field_center_b_deg=-3.0, exposure_s=139.8)
        for i, star in enumerate(stars):
            assert df["mag_w146"].iloc[i] == pytest.approx(star.mag_F146_ab)

    def test_row_count_matches_input(self):
        stars = _make_stars(15)
        df = project_to_sca_dataframe(stars, sca_id=1, field_center_l_deg=1.0,
                                      field_center_b_deg=-3.0, exposure_s=139.8)
        assert len(df) == 15


class TestProjectionAnchor:
    """Regression: a star used as its own projection anchor must land at stamp centre.

    The crowding frame must be centred on the source star so that the microlensed
    source (drawn at centroid_offset_pix=(0,0) = stamp centre) aligns with its
    true catalog position.  This is the geometric contract enforced by the fix to
    worker.py (A.1): use source_star coords as the projection anchor instead of
    the field centre.
    """

    def _make_star_at(self, l: float, b: float) -> StarRecord:
        return StarRecord(
            galactic_l_deg=l,
            galactic_b_deg=b,
            distance_kpc=8.0,
            mass_msun=1.0,
            teff_K=5000.0,
            log_g=4.5,
            metallicity_feh=0.0,
            mag_F146_ab=18.0,
            source_id="anchor_star",
            catalog_tile_id="tile_0",
        )

    def test_star_at_anchor_projects_to_stamp_centre(self):
        """A star whose coords are used as the anchor must project to x=128, y=128."""
        l_s, b_s = 1.5, -2.0
        star = self._make_star_at(l_s, b_s)
        df = project_to_sca_dataframe(
            [star],
            sca_id=1,
            field_center_l_deg=l_s,   # source star IS the anchor
            field_center_b_deg=b_s,
            exposure_s=100.0,
        )
        assert df["x_pix"].iloc[0] == pytest.approx(128.0, abs=1e-6)
        assert df["y_pix"].iloc[0] == pytest.approx(128.0, abs=1e-6)

    def test_neighbor_at_field_center_is_offset_when_anchor_is_source_star(self):
        """When anchor = source star, a neighbor at the original field centre is off-centre.

        This confirms that rebinding the anchor to the source star (instead of the
        field centre) produces a non-trivial, correct pixel offset for other stars.
        """
        # Source star at l=1.5, b=-2.0; field centre at l=1.0, b=-3.0 (different).
        l_s, b_s = 1.5, -2.0
        l_field, b_field = 1.0, -3.0
        field_centre_star = self._make_star_at(l_field, b_field)
        df = project_to_sca_dataframe(
            [field_centre_star],
            sca_id=1,
            field_center_l_deg=l_s,   # anchor is source star, not field centre
            field_center_b_deg=b_s,
            exposure_s=100.0,
        )
        # The field-centre star is NOT at the anchor → must not be at (128, 128).
        assert not (
            abs(df["x_pix"].iloc[0] - 128.0) < 0.1
            and abs(df["y_pix"].iloc[0] - 128.0) < 0.1
        ), "Field-centre star should NOT project to stamp centre when anchor is the source star."
