"""
smig/sensor/validation/test_integration.py
============================================
End-to-end smoke test for the SMIG v2 Phase 1 detector chain.

Verifies the full path: process_event → ProvenanceTracker → JSON sidecar →
ProvenanceRecord.model_validate round-trip.

Run from the project root:
    python -m pytest smig/sensor/validation/test_integration.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from smig.config.schemas import DetectorConfig, GeometryConfig
from smig.config.utils import load_detector_config
from smig.provenance.schema import ProvenanceRecord
from smig.provenance.tracker import ProvenanceTracker
from smig.sensor.detector import H4RG10Detector


# ---------------------------------------------------------------------------
# Config loading smoke test (uses the real reference YAML)
# ---------------------------------------------------------------------------

def test_load_detector_config_from_yaml():
    """load_detector_config correctly parses smig/config/roman_wfi.yaml."""
    # parents[0] = validation/, parents[1] = sensor/, parents[2] = smig/
    yaml_path = Path(__file__).resolve().parents[2] / "config" / "roman_wfi.yaml"
    cfg = load_detector_config(yaml_path)
    assert isinstance(cfg, DetectorConfig)
    assert cfg.schema_version == "2.0"
    assert cfg.geometry.nx == 4096
    assert cfg.geometry.ny == 4096


# ---------------------------------------------------------------------------
# Full chain integration test
# ---------------------------------------------------------------------------

def test_full_chain_integration():
    """process_event → ProvenanceTracker → sidecar round-trips as valid JSON."""
    # Use a tiny geometry so shape validation passes without a 4096×4096 cube.
    cfg = DetectorConfig(geometry=GeometryConfig(nx=8, ny=8))
    n_epochs = 5

    rng = np.random.default_rng(seed=12345)
    ideal_cube = rng.uniform(100.0, 50_000.0, size=(n_epochs, 8, 8))
    timestamps = np.linspace(60_000.0, 60_004.0, n_epochs)

    detector = H4RG10Detector(cfg, np.random.default_rng(seed=42))
    event_id = "integration_test_001"
    tracker = ProvenanceTracker(event_id=event_id)

    event_output = detector.process_event(event_id, ideal_cube, timestamps)

    for record in event_output.provenance_records:
        tracker.append_record(record)

    with tempfile.TemporaryDirectory() as tmpdir:
        sidecar_path = tracker.write_sidecar(Path(tmpdir))
        sidecar_text = sidecar_path.read_text(encoding="utf-8")

    # Must parse as valid JSON without raising
    sidecar = json.loads(sidecar_text)

    # Required top-level keys
    for key in ("event_id", "git_commit", "container_digest", "epoch_count", "records"):
        assert key in sidecar, f"Missing top-level key: {key!r}"

    assert sidecar["event_id"] == event_id
    assert sidecar["epoch_count"] == n_epochs
    assert len(sidecar["records"]) == n_epochs

    # Epoch indices must be sequential starting at 0
    epoch_indices = [r["epoch_index"] for r in sidecar["records"]]
    assert epoch_indices == list(range(n_epochs))

    # Every record must round-trip through ProvenanceRecord validation
    for raw_record in sidecar["records"]:
        parsed = ProvenanceRecord.model_validate(raw_record)
        assert parsed.event_id == event_id


# ---------------------------------------------------------------------------
# Physics integration (placeholder — not yet implemented)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Physics not implemented")
def test_physics_integration_128x128():
    """Full physics integration smoke test at 128×128 with all models active.

    When physics is implemented, this test should exercise the full chain
    (charge diffusion → IPC → persistence → MULTIACCUM → noise) and verify
    that output statistics (mean, std, saturation fraction) fall within
    physically expected bounds for a 128×128 detector with a realistic scene.
    """
    pass
