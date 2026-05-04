"""
smig/config/validation/test_seed.py
=====================================
Contract tests for smig.config.seed and Phase 2 config round-trips.

Covers
------
1. Seed range: all outputs strictly in (0, 2**31).
2. Seed determinism: same inputs → same seed, independent of invocation order.
3. Seed independence: different inputs → different seeds.
4. Domain separation: derive_event_seed and derive_stage_seed don't collide
   even when given numerically identical arguments.
5. SimulationConfig geometry validator: rejects detector array smaller than
   dia.context_stamp_size.
6. SimulationConfig round-trip: load_simulation_config → model_dump_json →
   SHA-256 is stable across two independent load calls.
7. PSFConfig wavelength_range_um validator.
8. Phase 1 canary is unaffected (import check only; the actual canary lives in
   test_config_utils.py and must continue to pass unmodified).

Run from the SMIG project root:
    python -m pytest smig/config/validation/test_seed.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from smig.config.optics_schemas import (
    CrowdedFieldConfig,
    DIAConfig,
    PSFConfig,
    RenderingConfig,
    SimulationConfig,
)
from smig.config.schemas import DetectorConfig, GeometryConfig
from smig.config.seed import derive_event_seed, derive_stage_seed
from smig.config.utils import get_simulation_config_sha256, load_simulation_config

# Path to the canonical simulation YAML (sibling of this package's parent).
_SIMULATION_YAML = Path(__file__).parent.parent / "simulation.yaml"

_SEED_MAX = 2**31  # exclusive upper bound for GalSim-safe seeds


# ---------------------------------------------------------------------------
# Seed range — all outputs must be in (0, 2**31)
# ---------------------------------------------------------------------------

def test_derive_event_seed_in_range():
    seed = derive_event_seed(42, "OB240123")
    assert 0 < seed < _SEED_MAX, (
        f"derive_event_seed returned {seed}, expected 0 < seed < {_SEED_MAX}"
    )


def test_derive_stage_seed_in_range():
    event_seed = derive_event_seed(42, "OB240123")
    stage_seed = derive_stage_seed(event_seed, "psf_rendering")
    assert 0 < stage_seed < _SEED_MAX, (
        f"derive_stage_seed returned {stage_seed}, expected 0 < seed < {_SEED_MAX}"
    )


def test_derive_event_seed_never_zero():
    """Enumerate a range of inputs and assert no seed is 0."""
    for i in range(200):
        seed = derive_event_seed(i, f"event_{i:04d}")
        assert seed != 0, f"derive_event_seed({i}, ...) returned 0 (fatal for GalSim)"


def test_derive_stage_seed_never_zero():
    """Enumerate a range of stage names and assert no seed is 0."""
    stages = ["psf", "noise", "source_injection", "subtraction", "ramp_readout"]
    event_seed = derive_event_seed(1, "OB000001")
    for stage in stages:
        seed = derive_stage_seed(event_seed, stage)
        assert seed != 0, f"derive_stage_seed(..., {stage!r}) returned 0"


# ---------------------------------------------------------------------------
# Seed determinism — same inputs always produce same seed
# ---------------------------------------------------------------------------

def test_derive_event_seed_deterministic():
    s1 = derive_event_seed(42, "OB240123")
    s2 = derive_event_seed(42, "OB240123")
    assert s1 == s2, "derive_event_seed must be deterministic for identical inputs"


def test_derive_stage_seed_deterministic():
    event_seed = derive_event_seed(42, "OB240123")
    s1 = derive_stage_seed(event_seed, "psf_rendering")
    s2 = derive_stage_seed(event_seed, "psf_rendering")
    assert s1 == s2, "derive_stage_seed must be deterministic for identical inputs"


def test_derive_event_seed_deterministic_across_custom_namespace():
    """Custom namespace must also be stable across calls."""
    s1 = derive_event_seed(7, "ev_007", namespace="test/ns")
    s2 = derive_event_seed(7, "ev_007", namespace="test/ns")
    assert s1 == s2


# ---------------------------------------------------------------------------
# Seed independence — different inputs produce different seeds
# ---------------------------------------------------------------------------

def test_derive_event_seed_different_event_ids():
    s1 = derive_event_seed(42, "OB240123")
    s2 = derive_event_seed(42, "OB240124")
    assert s1 != s2, "Different event_ids must yield different seeds"


def test_derive_event_seed_different_master_seeds():
    s1 = derive_event_seed(42, "OB240123")
    s2 = derive_event_seed(99, "OB240123")
    assert s1 != s2, "Different master_seeds must yield different event seeds"


def test_derive_stage_seed_different_stage_names():
    event_seed = derive_event_seed(42, "OB240123")
    s_psf = derive_stage_seed(event_seed, "psf_rendering")
    s_noise = derive_stage_seed(event_seed, "noise")
    s_inject = derive_stage_seed(event_seed, "source_injection")
    assert s_psf != s_noise
    assert s_psf != s_inject
    assert s_noise != s_inject


def test_derive_stage_seed_different_event_seeds():
    es1 = derive_event_seed(1, "OB000001")
    es2 = derive_event_seed(2, "OB000001")
    assert derive_stage_seed(es1, "noise") != derive_stage_seed(es2, "noise"), (
        "Same stage_name with different event_seeds must yield different stage seeds"
    )


# ---------------------------------------------------------------------------
# Domain separation — event and stage namespaces must not collide
# ---------------------------------------------------------------------------

def test_domain_separation_event_vs_stage_default_namespaces():
    """derive_event_seed and derive_stage_seed must not collide on same inputs.

    Both functions receive the integer 42 and the string "OB240123" as their
    two positional arguments, but their different default namespaces must
    produce distinct outputs.
    """
    s_event = derive_event_seed(42, "OB240123")
    # Pass the same numeric values to derive_stage_seed to test isolation.
    s_stage = derive_stage_seed(42, "OB240123")
    assert s_event != s_stage, (
        "derive_event_seed and derive_stage_seed must not collide on identical "
        "numeric inputs — domain-separation namespaces must differ"
    )


# ---------------------------------------------------------------------------
# SimulationConfig geometry validator
# ---------------------------------------------------------------------------

def test_simulation_config_rejects_nx_smaller_than_context_stamp():
    """Detector nx < dia.context_stamp_size must raise ValidationError."""
    with pytest.raises(ValidationError, match="nx"):
        SimulationConfig(
            detector=DetectorConfig(geometry=GeometryConfig(nx=64, ny=256)),
            dia=DIAConfig(context_stamp_size=256),
        )


def test_simulation_config_rejects_ny_smaller_than_context_stamp():
    """Detector ny < dia.context_stamp_size must raise ValidationError."""
    with pytest.raises(ValidationError, match="ny"):
        SimulationConfig(
            detector=DetectorConfig(geometry=GeometryConfig(nx=256, ny=128)),
            dia=DIAConfig(context_stamp_size=256),
        )


def test_simulation_config_accepts_geometry_equal_to_context_stamp():
    """nx == ny == context_stamp_size (boundary value) must be accepted."""
    cfg = SimulationConfig(
        detector=DetectorConfig(geometry=GeometryConfig(nx=256, ny=256)),
        dia=DIAConfig(context_stamp_size=256),
    )
    assert cfg.detector.geometry.nx == 256
    assert cfg.detector.geometry.ny == 256


def test_simulation_config_accepts_geometry_larger_than_context_stamp():
    """nx, ny >> context_stamp_size must be accepted."""
    cfg = SimulationConfig(
        detector=DetectorConfig(geometry=GeometryConfig(nx=4096, ny=4096)),
        dia=DIAConfig(context_stamp_size=256),
    )
    assert cfg.detector.geometry.nx == 4096


# ---------------------------------------------------------------------------
# PSFConfig wavelength_range_um validator
# ---------------------------------------------------------------------------

def test_psf_config_rejects_inverted_wavelength_range():
    """wavelength_range_um with lower >= upper must raise ValidationError."""
    with pytest.raises(ValidationError):
        PSFConfig(wavelength_range_um=(2.00, 0.93))


def test_psf_config_rejects_equal_wavelength_bounds():
    """wavelength_range_um with lower == upper must raise ValidationError."""
    with pytest.raises(ValidationError):
        PSFConfig(wavelength_range_um=(1.50, 1.50))


def test_psf_config_accepts_valid_wavelength_range():
    cfg = PSFConfig(wavelength_range_um=(0.93, 2.00))
    assert cfg.wavelength_range_um == (0.93, 2.00)


# ---------------------------------------------------------------------------
# SimulationConfig frozen / extra-forbid contract
# ---------------------------------------------------------------------------

def test_simulation_config_is_frozen():
    """Assigning a field on a frozen SimulationConfig must raise an error.

    Pydantic v2 raises ValidationError (frozen_instance type) on __setattr__,
    whereas plain dataclasses raise AttributeError.  Accept both so the test
    remains correct across Pydantic minor releases.
    """
    from pydantic import ValidationError as PydanticValidationError

    cfg = SimulationConfig(
        detector=DetectorConfig(geometry=GeometryConfig(nx=256, ny=256)),
    )
    with pytest.raises((AttributeError, TypeError, PydanticValidationError)):
        cfg.psf = PSFConfig()  # type: ignore[misc]


def test_simulation_config_forbids_extra_fields():
    """Extra keys in the YAML mapping must raise ValidationError."""
    with pytest.raises(ValidationError):
        SimulationConfig.model_validate(
            {
                "detector": {},
                "unexpected_key": 99,
            }
        )


# ---------------------------------------------------------------------------
# SimulationConfig round-trip hash stability
# ---------------------------------------------------------------------------

def test_simulation_yaml_loads_cleanly():
    """simulation.yaml must load without errors."""
    assert _SIMULATION_YAML.exists(), (
        f"simulation.yaml not found at {_SIMULATION_YAML}"
    )
    cfg = load_simulation_config(_SIMULATION_YAML)
    assert isinstance(cfg, SimulationConfig)


def test_simulation_config_round_trip_hash_stable():
    """Two independent loads of simulation.yaml must produce identical SHA-256."""
    cfg1 = load_simulation_config(_SIMULATION_YAML)
    cfg2 = load_simulation_config(_SIMULATION_YAML)
    h1 = get_simulation_config_sha256(cfg1)
    h2 = get_simulation_config_sha256(cfg2)
    assert h1 == h2, (
        "SHA-256 of SimulationConfig must be identical across independent loads of "
        "the same YAML file."
    )
    assert len(h1) == 64, "SHA-256 digest must be a 64-character hex string"


def test_simulation_config_hash_changes_on_field_mutation():
    """Configs that differ in one field must produce different hashes."""
    cfg_default = load_simulation_config(_SIMULATION_YAML)
    # Rebuild with a different PSF filter name.
    cfg_modified = SimulationConfig(
        detector=cfg_default.detector,
        psf=PSFConfig(filter_name="F087"),
        rendering=cfg_default.rendering,
        crowded_field=cfg_default.crowded_field,
        dia=cfg_default.dia,
    )
    h_default = get_simulation_config_sha256(cfg_default)
    h_modified = get_simulation_config_sha256(cfg_modified)
    assert h_default != h_modified, (
        "Changing psf.filter_name must produce a different SimulationConfig hash"
    )


# ---------------------------------------------------------------------------
# Phase 1 import guard — DetectorConfig must still import cleanly
# ---------------------------------------------------------------------------

def test_detector_config_imports_unaffected():
    """DetectorConfig must still import and instantiate with defaults unchanged.

    This does NOT re-check the canary hash (that lives in test_config_utils.py),
    but ensures that importing optics_schemas.py has not broken the Phase 1
    import chain.
    """
    cfg = DetectorConfig()
    assert cfg.geometry.nx == 4096
    assert cfg.geometry.ny == 4096
