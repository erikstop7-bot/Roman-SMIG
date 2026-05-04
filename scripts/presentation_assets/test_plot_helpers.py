"""test_plot_helpers.py

Unit tests for pure helper functions in plot_h5_lightcurves.py.

Run from the SMIG project root::

    python -m pytest scripts/presentation_assets/test_plot_helpers.py -v
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import helpers directly from the script under test.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from plot_h5_lightcurves import (  # noqa: E402
    _aperture_sum,
    _rank_top_per_class,
    _resolve_cadence,
)


# ---------------------------------------------------------------------------
# _aperture_sum
# ---------------------------------------------------------------------------

class TestApertureSum:
    def test_shape(self):
        stamps = np.ones((10, 64, 64), dtype=np.float32)
        result = _aperture_sum(stamps, half_width=3)
        assert result.shape == (10,)

    def test_zero_stamps_returns_zeros(self):
        stamps = np.zeros((5, 64, 64), dtype=np.float32)
        result = _aperture_sum(stamps, half_width=3)
        np.testing.assert_array_equal(result, 0.0)

    def test_uniform_stamps_sum_equals_aperture_area(self):
        stamps = np.ones((3, 64, 64), dtype=np.float32)
        hw = 3
        result = _aperture_sum(stamps, half_width=hw)
        expected_area = (2 * hw + 1) ** 2
        np.testing.assert_allclose(result, float(expected_area))

    def test_saturation_mask_zeros_masked_pixels(self):
        stamps = np.ones((1, 64, 64), dtype=np.float32)
        sat = np.zeros((1, 64, 64), dtype=bool)
        # Mask entire aperture.
        cy, cx = 32, 32
        hw = 3
        sat[0, cy - hw : cy + hw + 1, cx - hw : cx + hw + 1] = True
        result = _aperture_sum(stamps, half_width=hw, sat_mask=sat)
        np.testing.assert_array_equal(result, 0.0)

    def test_negative_values_preserved(self):
        stamps = np.full((4, 64, 64), -5.0, dtype=np.float32)
        result = _aperture_sum(stamps, half_width=2)
        assert (result < 0).all()


# ---------------------------------------------------------------------------
# _rank_top_per_class — absolute residual ranking
# ---------------------------------------------------------------------------

def _make_data(science: np.ndarray, event_class: np.ndarray, event_ids: list[str]) -> dict:
    return {
        "science": science.astype(np.float32),
        "event_class": event_class.astype(np.uint8),
        "event_ids": event_ids,
        "saturation": None,
    }


class TestRankTopPerClass:
    def test_prefers_large_negative_over_small_positive(self):
        """An event with large negative peak should rank above one with small positive peak."""
        n_epochs, h, w = 10, 64, 64
        # Event 0 (class 0): small positive residual at centre.
        s0 = np.zeros((n_epochs, h, w), dtype=np.float32)
        s0[:, 32, 32] = 10.0
        # Event 1 (class 0): large negative residual at centre.
        s1 = np.zeros((n_epochs, h, w), dtype=np.float32)
        s1[:, 32, 32] = -200.0

        science = np.stack([s0, s1])
        event_class = np.array([0, 0], dtype=np.uint8)
        event_ids = ["event_a", "event_b"]

        data = _make_data(science, event_class, event_ids)
        rng = np.random.default_rng(0)
        winners = _rank_top_per_class(data, aperture_radius=3, rng=rng)
        # Event 1 has higher abs peak (200 > 10).
        assert winners[0] == 1

    def test_positive_event_wins_when_larger_abs(self):
        n_epochs, h, w = 10, 64, 64
        s0 = np.zeros((n_epochs, h, w), dtype=np.float32)
        s0[:, 32, 32] = 500.0
        s1 = np.zeros((n_epochs, h, w), dtype=np.float32)
        s1[:, 32, 32] = -100.0

        science = np.stack([s0, s1])
        event_class = np.array([0, 0], dtype=np.uint8)
        event_ids = ["ev_0", "ev_1"]

        data = _make_data(science, event_class, event_ids)
        rng = np.random.default_rng(0)
        winners = _rank_top_per_class(data, aperture_radius=3, rng=rng)
        assert winners[0] == 0

    def test_tie_broken_by_event_id(self):
        """Equal abs peak: lexicographically smaller event_id wins."""
        n_epochs, h, w = 5, 64, 64
        val = 42.0
        s0 = np.full((n_epochs, h, w), val / (7 * 7), dtype=np.float32)
        s1 = np.full((n_epochs, h, w), val / (7 * 7), dtype=np.float32)

        science = np.stack([s0, s1])
        event_class = np.array([0, 0], dtype=np.uint8)
        event_ids = ["zz_event", "aa_event"]

        data = _make_data(science, event_class, event_ids)
        rng = np.random.default_rng(0)
        winners = _rank_top_per_class(data, aperture_radius=3, rng=rng)
        # "aa_event" (row 1) should win tie-break over "zz_event" (row 0).
        assert winners[0] == 1

    def test_multiple_classes_returned(self):
        n_epochs, h, w = 5, 64, 64
        science = np.zeros((3, n_epochs, h, w), dtype=np.float32)
        science[0, :, 32, 32] = 10.0
        science[1, :, 32, 32] = 20.0
        science[2, :, 32, 32] = 30.0

        event_class = np.array([0, 1, 2], dtype=np.uint8)
        event_ids = ["e0", "e1", "e2"]

        data = _make_data(science, event_class, event_ids)
        rng = np.random.default_rng(0)
        winners = _rank_top_per_class(data, aperture_radius=3, rng=rng)
        assert set(winners.keys()) == {0, 1, 2}
        assert winners[0] == 0
        assert winners[1] == 1
        assert winners[2] == 2


# ---------------------------------------------------------------------------
# _resolve_cadence
# ---------------------------------------------------------------------------

def _ns(**kwargs) -> argparse.Namespace:
    defaults = {"cadence_days": None, "phase3_config": None}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestResolveCadence:
    def test_smoke_default_when_nothing_supplied(self):
        ns = _ns()
        cadence = _resolve_cadence(ns)
        assert cadence == pytest.approx(0.0104)

    def test_explicit_cadence_days_used(self):
        ns = _ns(cadence_days=2.5)
        cadence = _resolve_cadence(ns)
        assert cadence == pytest.approx(2.5)

    def test_phase3_config_overrides_cadence_days(self, tmp_path):
        config_file = tmp_path / "cfg.yaml"
        config_file.write_text("cadence_days: 2.5\nn_science_epochs: 30\n")
        ns = _ns(cadence_days=99.0, phase3_config=config_file)
        cadence = _resolve_cadence(ns)
        # Config wins over explicit --cadence-days.
        assert cadence == pytest.approx(2.5)

    def test_phase3_config_without_cadence_days_arg(self, tmp_path):
        config_file = tmp_path / "cfg.yaml"
        config_file.write_text("cadence_days: 2.5\n")
        ns = _ns(phase3_config=config_file)
        cadence = _resolve_cadence(ns)
        assert cadence == pytest.approx(2.5)

    def test_sprint_static_config_reads_correctly(self):
        sprint_cfg = Path(__file__).parent.parent.parent / "configs" / "phase3_sprint_static.yaml"
        if not sprint_cfg.is_file():
            pytest.skip("configs/phase3_sprint_static.yaml not present")
        ns = _ns(phase3_config=sprint_cfg)
        cadence = _resolve_cadence(ns)
        assert cadence == pytest.approx(2.5)
