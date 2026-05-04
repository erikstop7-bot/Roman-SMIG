"""
smig/config/validation/test_yaml_equivalence.py
================================================
Drift guard: ``simulation.yaml`` and ``simulation_science.yaml`` must remain
functionally identical so production builds (which historically referenced
``simulation.yaml``) and the explicitly-named science profile produce
identical SimulationConfig fingerprints.

If this test fires, somebody edited one of the two files without mirroring
the change to the other.  Decide whether the divergence is intentional:

* If yes — the two files have a real reason to differ — delete this test and
  document the difference in both file headers.
* If no — restore the equivalence (or, preferred, point production configs at
  ``simulation_science.yaml`` and stop maintaining the duplicate).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from smig.config.utils import (
    get_simulation_config_sha256,
    load_simulation_config,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SIMULATION_YAML = _REPO_ROOT / "smig" / "config" / "simulation.yaml"
_SIMULATION_SCIENCE_YAML = _REPO_ROOT / "smig" / "config" / "simulation_science.yaml"


@pytest.mark.parametrize(
    "path",
    [_SIMULATION_YAML, _SIMULATION_SCIENCE_YAML],
)
def test_config_file_present(path: Path) -> None:
    assert path.exists(), f"Required config not found: {path}"


def test_simulation_and_science_configs_have_identical_sha256() -> None:
    """Both legacy simulation.yaml and explicit simulation_science.yaml must hash equal."""
    sim_cfg = load_simulation_config(_SIMULATION_YAML)
    sci_cfg = load_simulation_config(_SIMULATION_SCIENCE_YAML)
    sim_sha = get_simulation_config_sha256(sim_cfg)
    sci_sha = get_simulation_config_sha256(sci_cfg)
    assert sim_sha == sci_sha, (
        "simulation.yaml and simulation_science.yaml have drifted apart.\n"
        f"  simulation.yaml         SHA-256: {sim_sha}\n"
        f"  simulation_science.yaml SHA-256: {sci_sha}\n"
        "These two files are documented as 'functionally equivalent'.  "
        "If the divergence is intentional, delete this test and update both file "
        "headers.  Otherwise, restore equivalence — production phase3 configs and "
        "release-gate tests both depend on it."
    )


def test_science_profile_invariants() -> None:
    """simulation_science.yaml must hold the science-profile contract values."""
    cfg = load_simulation_config(_SIMULATION_SCIENCE_YAML)
    assert cfg.dia.subtraction_method == "identity"
    assert cfg.dia.reference_processing == "detector_matched"
    assert cfg.dia.reference_jitter_mode == "per_epoch"
    assert cfg.psf.fov_native_pixels == 256
    assert cfg.psf.oversample == 4
    assert cfg.psf.n_wavelengths == 10
