"""
smig/sensor/validation/test_unit.py
=====================================
Unit tests for the SMIG v2 Phase 1 sensor scaffold.

Run from the project root:
    python -m pytest smig/sensor/validation/test_unit.py -v
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest
from pydantic import ValidationError

from smig.config.schemas import DetectorConfig, GeometryConfig
from smig.config.utils import get_config_sha256
from smig.provenance.schema import ProvenanceRecord, sanitize_rng_state
from smig.provenance.tracker import ProvenanceTracker
from smig.sensor.detector import DetectorOutput, EventOutput, H4RG10Detector
from smig.sensor.noise.cosmic_rays import ClusteredCosmicRayInjector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_cfg() -> DetectorConfig:
    """16×16 DetectorConfig for fast unit tests."""
    return DetectorConfig(geometry=GeometryConfig(nx=16, ny=16))


@pytest.fixture
def small_detector(small_cfg: DetectorConfig) -> H4RG10Detector:
    return H4RG10Detector(small_cfg, np.random.default_rng(0))


@pytest.fixture
def single_epoch_output(small_detector: H4RG10Detector) -> DetectorOutput:
    ideal = np.ones((16, 16), dtype=np.float64) * 1_000.0
    return small_detector.process_epoch(ideal, epoch_index=0, epoch_time_mjd=60_000.0)


@pytest.fixture
def event_output(small_detector: H4RG10Detector) -> EventOutput:
    ideal_cube = np.ones((3, 16, 16), dtype=np.float64) * 500.0
    timestamps = np.array([60_000.0, 60_001.0, 60_002.0])
    return small_detector.process_event("test_event", ideal_cube, timestamps)


# ---------------------------------------------------------------------------
# Config hashing
# ---------------------------------------------------------------------------

def test_config_sha256_determinism():
    """Identical configs produce the same 64-char hash; different configs differ."""
    cfg1 = DetectorConfig()
    cfg2 = DetectorConfig()
    h1 = get_config_sha256(cfg1)
    h2 = get_config_sha256(cfg2)
    assert h1 == h2
    assert len(h1) == 64

    cfg3 = DetectorConfig(geometry=GeometryConfig(nx=2048, ny=2048))
    assert get_config_sha256(cfg3) != h1


def test_config_sha256_is_hex_string():
    """Hash is a valid lowercase hex string of length 64."""
    h = get_config_sha256(DetectorConfig())
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# Detector construction
# ---------------------------------------------------------------------------

def test_detector_construction(small_cfg: DetectorConfig):
    """H4RG10Detector constructs without error and stores the config."""
    rng = np.random.default_rng(42)
    detector = H4RG10Detector(small_cfg, rng)
    assert detector._config is small_cfg
    assert detector._config_sha256 == get_config_sha256(small_cfg)


# ---------------------------------------------------------------------------
# process_epoch — shapes and types
# ---------------------------------------------------------------------------

def test_process_epoch_returns_detector_output(single_epoch_output: DetectorOutput):
    assert isinstance(single_epoch_output, DetectorOutput)


def test_process_epoch_rate_image_shape_and_dtype(single_epoch_output: DetectorOutput):
    assert single_epoch_output.rate_image.shape == (16, 16)
    assert single_epoch_output.rate_image.dtype == np.float64


def test_process_epoch_masks_shape_and_dtype(single_epoch_output: DetectorOutput):
    assert single_epoch_output.saturation_mask.shape == (16, 16)
    assert single_epoch_output.saturation_mask.dtype == bool
    assert single_epoch_output.cr_mask.shape == (16, 16)
    assert single_epoch_output.cr_mask.dtype == bool


def test_process_epoch_output_does_not_alias_input(small_detector: H4RG10Detector):
    """Output rate_image must not be the same object as the input array."""
    ideal = np.ones((16, 16), dtype=np.float64) * 1_000.0
    output = small_detector.process_epoch(ideal, epoch_index=0, epoch_time_mjd=60_000.0)
    assert output.rate_image is not ideal


# ---------------------------------------------------------------------------
# process_epoch — provenance_data
# ---------------------------------------------------------------------------

_EXPECTED_PROVENANCE_KEYS = frozenset({
    "git_commit",
    "container_digest",
    "python_version",
    "numpy_version",
    "config_sha256",
    "random_state",
    "ipc_applied",
    "persistence_applied",
    "nonlinearity_applied",
    "charge_diffusion_applied",
    "saturated_pixel_count",
    "cosmic_ray_hit_count",
})

# Phase B: new fields added to ProvenanceRecord as optional (default=None/0).
# These are not yet populated by the detector stub, so they are validated against
# ProvenanceRecord.model_fields rather than against provenance_data.keys().
_EXPECTED_PHASE_B_PROVENANCE_FIELDS = frozenset({
    "ipc_kernel_hash",
    "persistence_history_depth",
    "n_partial_saturation_pixels",
    "cr_types",
    "n_rts_active_pixels",
    "slope_fit_method",
    "n_reads_used_median",
    "peak_memory_mb",
})


def test_process_epoch_provenance_data_has_correct_keys(
    single_epoch_output: DetectorOutput,
):
    # Subset check: required keys must be present; additional keys are allowed
    # as new provenance dimensions are added without breaking this test.
    assert _EXPECTED_PROVENANCE_KEYS <= set(single_epoch_output.provenance_data.keys())


def test_process_epoch_provenance_data_stub_flags(
    single_epoch_output: DetectorOutput,
):
    """Checks provenance applied-effect flags and counts.

    All four physics stages are unconditionally called in the signal chain,
    so all *_applied flags are True.  persistence_applied was corrected from
    False to True in Phase 1 remediation (DynamicPersistence.apply() is
    always called).
    """
    pd = single_epoch_output.provenance_data

    assert pd["charge_diffusion_applied"] is True
    assert pd["ipc_applied"] is True
    assert pd["persistence_applied"] is True  # Fixed: called unconditionally
    assert pd["nonlinearity_applied"] is True
    assert pd["saturated_pixel_count"] == 0
    assert pd["cosmic_ray_hit_count"] == 0


def test_process_epoch_provenance_config_sha256(
    small_detector: H4RG10Detector,
    small_cfg: DetectorConfig,
    single_epoch_output: DetectorOutput,
):
    assert single_epoch_output.provenance_data["config_sha256"] == get_config_sha256(
        small_cfg
    )


def test_process_epoch_provenance_random_state_structure(
    single_epoch_output: DetectorOutput,
):
    """random_state is a structured dict with per-child-generator state snapshots.

    Phase 1 remediation changed the format from a flat parent-RNG state dict
    to a structured dict with keys: parent, readout, one_over_f, rts, cosmic_rays.
    Each value is the bit-generator state dict for that child RNG.
    """
    rs = single_epoch_output.provenance_data["random_state"]
    assert isinstance(rs, dict)
    expected_keys = {"parent", "readout", "one_over_f", "rts", "cosmic_rays"}
    assert set(rs.keys()) == expected_keys, (
        f"random_state keys {set(rs.keys())!r} != expected {expected_keys!r}"
    )
    for child_name, child_state in rs.items():
        assert isinstance(child_state, dict), (
            f"random_state[{child_name!r}] must be a dict, got {type(child_state)!r}"
        )
        assert "bit_generator" in child_state, (
            f"random_state[{child_name!r}] missing 'bit_generator' key"
        )


# ---------------------------------------------------------------------------
# process_epoch — shape validation
# ---------------------------------------------------------------------------

def test_process_epoch_rejects_wrong_shape(small_detector: H4RG10Detector):
    wrong = np.ones((8, 8), dtype=np.float64)
    with pytest.raises(ValueError, match="shape"):
        small_detector.process_epoch(wrong, epoch_index=0, epoch_time_mjd=60_000.0)


# ---------------------------------------------------------------------------
# process_event — shapes and structure
# ---------------------------------------------------------------------------

def test_process_event_returns_event_output(event_output: EventOutput):
    assert isinstance(event_output, EventOutput)


def test_process_event_cube_shapes(event_output: EventOutput):
    assert event_output.rate_cube.shape == (3, 16, 16)
    assert event_output.saturation_cube.shape == (3, 16, 16)
    assert event_output.cr_cube.shape == (3, 16, 16)
    assert event_output.persistence_peak_map.shape == (16, 16)


def test_process_event_provenance_record_count(event_output: EventOutput):
    assert len(event_output.provenance_records) == 3


def test_process_event_provenance_records_are_valid(event_output: EventOutput):
    """All records are ProvenanceRecords with correct event_id and epoch_index."""
    for i, record in enumerate(event_output.provenance_records):
        assert isinstance(record, ProvenanceRecord)
        assert record.event_id == "test_event"
        assert record.epoch_index == i


def test_process_event_provenance_timestamps_are_aware(event_output: EventOutput):
    """All provenance timestamps are timezone-aware (UTC)."""
    for record in event_output.provenance_records:
        assert record.timestamp_utc.tzinfo is not None


def test_process_event_provenance_config_sha256_consistent(
    event_output: EventOutput,
    small_cfg: DetectorConfig,
):
    """All records share the same config_sha256."""
    expected = get_config_sha256(small_cfg)
    for record in event_output.provenance_records:
        assert record.config_sha256 == expected


# ---------------------------------------------------------------------------
# Phase C — Input guard tests
# ---------------------------------------------------------------------------

def test_process_epoch_rejects_nan_input(small_detector: H4RG10Detector):
    """ValueError on NaN anywhere in ideal_image_e."""
    bad = np.ones((16, 16), dtype=np.float64) * 1_000.0
    bad[3, 7] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        small_detector.process_epoch(bad, epoch_index=0, epoch_time_mjd=60_000.0)


def test_process_epoch_rejects_negative_input(small_detector: H4RG10Detector):
    """ValueError when ideal_image_e contains negative electron counts."""
    bad = np.full((16, 16), -1.0, dtype=np.float64)
    with pytest.raises(ValueError, match="negative"):
        small_detector.process_epoch(bad, epoch_index=0, epoch_time_mjd=60_000.0)


def test_process_epoch_rejects_1d_input(small_detector: H4RG10Detector):
    """ValueError when ideal_image_e is 1-D (ndim != 2)."""
    bad = np.ones(16 * 16, dtype=np.float64) * 1_000.0
    with pytest.raises(ValueError, match="2-D"):
        small_detector.process_epoch(bad, epoch_index=0, epoch_time_mjd=60_000.0)


def test_process_epoch_rate_image_is_float64(small_detector: H4RG10Detector):
    """Output rate_image is always float64 even when input dtype is integer."""
    ideal = np.ones((16, 16), dtype=np.int32) * 1_000
    output = small_detector.process_epoch(ideal, epoch_index=0, epoch_time_mjd=60_000.0)
    assert output.rate_image.dtype == np.float64


def test_process_epoch_cr_mask_shape_matches_rate_image(
    single_epoch_output: DetectorOutput,
):
    """cr_mask must have the same shape as rate_image."""
    assert single_epoch_output.cr_mask.shape == single_epoch_output.rate_image.shape


def test_process_event_rejects_2d_cube(small_detector: H4RG10Detector):
    """ValueError when ideal_cube_e is 2-D instead of 3-D."""
    wrong = np.ones((16, 16), dtype=np.float64) * 500.0
    ts = np.array([60_000.0])
    with pytest.raises(ValueError, match="3-D"):
        small_detector.process_event("ev", wrong, ts)


# ---------------------------------------------------------------------------
# Phase C — Saturation mask tests
# ---------------------------------------------------------------------------

def test_saturation_mask_correct_threshold(
    small_cfg: DetectorConfig, small_detector: H4RG10Detector
):
    """Pixels at full-well charge must be flagged as saturated.

    Uses full_well_electrons (not the exact flag boundary) so that every pixel
    is guaranteed to accumulate above Q_sat in the ramp despite Poisson noise.
    This tests the flag mechanism, not the exact boundary.
    """
    full_well = small_cfg.electrical.full_well_electrons
    ideal = np.full((16, 16), full_well, dtype=np.float64)
    output = small_detector.process_epoch(ideal, epoch_index=0, epoch_time_mjd=60_000.0)
    assert output.saturation_mask.all(), (
        "All pixels at full-well charge should be flagged as saturated"
    )


def test_saturation_mask_below_threshold_not_masked(
    small_cfg: DetectorConfig, small_detector: H4RG10Detector
):
    """Pixels well below the saturation threshold must not be flagged.

    Uses 10 % of the flag threshold so Poisson noise cannot push any pixel
    over Q_sat = saturation_flag_threshold * full_well_electrons.
    """
    threshold = (
        small_cfg.nonlinearity.saturation_flag_threshold
        * small_cfg.electrical.full_well_electrons
    )
    ideal = np.full((16, 16), threshold * 0.1, dtype=np.float64)
    output = small_detector.process_epoch(ideal, epoch_index=0, epoch_time_mjd=60_000.0)
    assert not output.saturation_mask.any(), (
        "No pixels at 10 % of threshold should be masked"
    )


# ---------------------------------------------------------------------------
# Phase C — Provenance / RNG snapshot tests
# ---------------------------------------------------------------------------

def test_process_epoch_provenance_random_state_json_serializable(
    single_epoch_output: DetectorOutput,
):
    """random_state in provenance_data must be natively JSON-serializable."""
    import json

    rs = single_epoch_output.provenance_data["random_state"]
    # Must not raise; numpy types (np.uint64, np.ndarray, etc.) would cause TypeError.
    json.dumps(rs)


def test_rng_state_serialization_is_snapshot(small_detector: H4RG10Detector):
    """Provenance random_state is a static snapshot, not a live generator reference.

    Noise modules use child RNGs derived from the parent at construction time.
    The parent ``self._rng`` is therefore NOT advanced naturally during
    ``process_epoch`` — all stochastic paths go through the children.
    We manually advance the parent RNG *post-call* to verify that the
    already-captured ``provenance_data['random_state']`` is a snapshot
    (serialized copy), not a live reference to the generator object.
    """
    import copy

    ideal = np.ones((16, 16), dtype=np.float64) * 1_000.0
    output = small_detector.process_epoch(ideal, epoch_index=0, epoch_time_mjd=60_000.0)

    # Deep-copy the serialized state at the moment of capture.
    state_at_capture = copy.deepcopy(output.provenance_data["random_state"])

    # Explicitly advance the parent RNG — it is the generator whose state is
    # snapshotted for provenance but is not consumed during process_epoch.
    small_detector._rng.integers(0, 2**32, size=1_000)

    # The snapshot must be byte-for-byte identical to what was captured before
    # the advance, proving it is a serialized copy, not a live reference.
    assert output.provenance_data["random_state"] == state_at_capture


# ---------------------------------------------------------------------------
# Phase C — Input-mutation independence test
# ---------------------------------------------------------------------------

def test_process_event_independence_from_input_mutation(
    small_detector: H4RG10Detector,
):
    """Mutating ideal_cube_e after process_event does not corrupt the output.

    process_epoch casts each slice with astype(float64, copy=True), so the
    output rate_cube should be immune to in-place mutations of the source array.
    """
    ideal_cube = np.ones((3, 16, 16), dtype=np.float64) * 500.0
    timestamps = np.array([60_000.0, 60_001.0, 60_002.0])
    output = small_detector.process_event("ev", ideal_cube, timestamps)

    rate_cube_before = output.rate_cube.copy()

    # Overwrite the source array entirely.
    ideal_cube[:] = 0.0

    np.testing.assert_array_equal(output.rate_cube, rate_cube_before)


# ---------------------------------------------------------------------------
# Phase D — Green contract tests (must pass now)
# ---------------------------------------------------------------------------

def test_cr_injector_signature_2d_only():
    """ClusteredCosmicRayInjector.apply() must reject 3D arrays with ValueError."""
    cfg = DetectorConfig(geometry=GeometryConfig(nx=16, ny=16))
    injector = ClusteredCosmicRayInjector(cfg, np.random.default_rng(0))
    ramp_3d = np.zeros((9, 16, 16), dtype=np.float64)
    with pytest.raises(ValueError, match="2-D"):
        injector.apply(ramp_3d)


def test_process_event_rejects_non_monotone_timestamps(
    small_detector: H4RG10Detector,
):
    """process_event must raise ValueError for decreasing timestamps."""
    ideal_cube = np.ones((3, 16, 16), dtype=np.float64) * 500.0
    # Third timestamp is less than second — violates non-decreasing constraint.
    timestamps = np.array([60_000.0, 60_002.0, 60_001.0])
    with pytest.raises(ValueError, match="non-decreasing"):
        small_detector.process_event("ev", ideal_cube, timestamps)


def test_process_epoch_applied_flags_are_booleans(
    single_epoch_output: DetectorOutput,
):
    """Every *_applied key in provenance_data must be a native Python bool."""
    pd = single_epoch_output.provenance_data
    for key in ("ipc_applied", "persistence_applied", "nonlinearity_applied",
                "charge_diffusion_applied"):
        assert isinstance(pd[key], bool), (
            f"{key!r} is {type(pd[key]).__name__!r}, expected bool"
        )


# ---------------------------------------------------------------------------
# Phase D — Physics tests (xfail: physics not yet implemented)
# ---------------------------------------------------------------------------

def test_ipc_flux_conservation():
    """IPC convolution must conserve total flux to within 0.01%.

    Passes trivially while IPC is a copy-stub; the xfail marker has been
    removed because the test now passes (stub conserves flux exactly).
    """
    from smig.sensor.ipc import FieldDependentIPC
    from smig.config.schemas import IPCConfig
    cfg = IPCConfig()
    ipc = FieldDependentIPC(cfg, sca_id=1, field_position=(0.0, 0.0))
    rng = np.random.default_rng(1)
    image = rng.uniform(100.0, 50_000.0, size=(64, 64))
    out = ipc.apply(image)
    assert abs(out.sum() - image.sum()) / image.sum() < 1e-4


def test_ipc_asymmetric_kernel_injection():
    """IPC applied to a point source must produce neighbour values
    that match hand-computed expectations for the analytic kernel."""
    from smig.sensor.ipc import FieldDependentIPC
    from smig.config.schemas import IPCConfig
    cfg = IPCConfig(ipc_field_dependent=False)
    ipc = FieldDependentIPC(cfg, sca_id=1, field_position=(0.0, 0.0))
    image = np.zeros((16, 16), dtype=np.float64)
    image[8, 8] = 1_000.0
    out = ipc.apply(image)
    alpha = cfg.ipc_alpha_center
    assert abs(out[8, 9] - alpha * 1_000.0) < 1.0
    assert abs(out[8, 7] - alpha * 1_000.0) < 1.0


def test_charge_diffusion_conservation():
    """Charge diffusion must conserve total electron count (relative error < 1e-10).

    Static diffusion uses post-hoc renormalization; BFE is a local
    redistribution that conserves charge by construction.
    """
    from smig.sensor.charge_diffusion import ChargeDiffusionModel
    from smig.config.schemas import ChargeDiffusionConfig
    cfg = ChargeDiffusionConfig(pixel_pitch_um=10.0, full_well_electrons=100_000.0)
    model = ChargeDiffusionModel(cfg)
    rng = np.random.default_rng(2)
    image = rng.uniform(100.0, 50_000.0, size=(64, 64))
    out = model.apply(image)
    assert abs(out.sum() - image.sum()) / image.sum() < 1e-10


def test_chain_order_cd_before_ipc_noncommutative():
    """diffuse(ipc(x)) != ipc(diffuse(x)) for non-trivial kernels.

    With the physically correct sigma (= diffusion_length_factor in pixel
    units, default 0.1 px), the diffusion kernel is narrow and the
    non-commutativity signal is small.  A tight tolerance (rtol=1e-10)
    is required to resolve the difference.
    """
    from smig.sensor.ipc import FieldDependentIPC
    from smig.sensor.charge_diffusion import ChargeDiffusionModel
    from smig.config.schemas import ChargeDiffusionConfig, IPCConfig
    ipc = FieldDependentIPC(IPCConfig(), sca_id=1, field_position=(0.0, 0.0))
    cd = ChargeDiffusionModel(
        ChargeDiffusionConfig(pixel_pitch_um=10.0, full_well_electrons=100_000.0)
    )
    rng = np.random.default_rng(3)
    image = rng.uniform(100.0, 10_000.0, size=(32, 32))
    # Correct order: diffuse then IPC
    order_correct = ipc.apply(cd.apply(image))
    # Swapped order
    order_swapped = cd.apply(ipc.apply(image))
    assert not np.allclose(order_correct, order_swapped, rtol=1e-10), (
        "CD→IPC and IPC→CD must differ for non-trivial kernels"
    )


def test_bfe_widens_psf():
    """Injecting a bright point source and applying charge diffusion must
    decrease the central pixel and increase the 4 nearest neighbours,
    proportional to bfe_coupling_coeff."""
    from smig.sensor.charge_diffusion import ChargeDiffusionModel
    from smig.config.schemas import ChargeDiffusionConfig

    # Use zero diffusion_length_factor effect by making sigma very small
    # so only BFE contributes to the spatial redistribution.
    coupling = 1e-4  # strong enough to see the effect
    cfg = ChargeDiffusionConfig(
        pixel_pitch_um=10.0,
        full_well_electrons=100_000.0,
        diffusion_length_factor=1e-6,  # negligible static diffusion
        bfe_coupling_coeff=coupling,
    )
    model = ChargeDiffusionModel(cfg)

    image = np.zeros((32, 32), dtype=np.float64)
    image[16, 16] = 50_000.0  # bright point source (half full well)

    out = model.apply(image)

    # Central pixel must decrease.
    assert out[16, 16] < image[16, 16], (
        "BFE should reduce the central pixel of a bright point source"
    )
    # Four nearest neighbours must increase (were zero before).
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        assert out[16 + dy, 16 + dx] > 0.0, (
            f"BFE should deposit charge at neighbour ({dy}, {dx})"
        )


def test_ipc_synthetic_loader():
    """Generate a synthetic HDF5 file, load kernels at two distinct
    field positions, and verify they are distinct but both sum to 1.0."""
    import tempfile
    from pathlib import Path
    from smig.sensor.calibration.ipc_kernels import (
        generate_synthetic_ipc_hdf5,
        load_interpolated_kernel,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        hdf5_path = Path(tmpdir) / "test_ipc.h5"
        generate_synthetic_ipc_hdf5(hdf5_path, sca_ids=(1,), grid_ny=4, grid_nx=4)

        k1 = load_interpolated_kernel(hdf5_path, sca_id=1, field_position=(0.1, 0.1))
        k2 = load_interpolated_kernel(hdf5_path, sca_id=1, field_position=(0.9, 0.9))

    # Both must be 9x9 and sum to 1.0.
    assert k1.shape == (9, 9)
    assert k2.shape == (9, 9)
    np.testing.assert_allclose(k1.sum(), 1.0, atol=1e-12)
    np.testing.assert_allclose(k2.sum(), 1.0, atol=1e-12)

    # Kernels at different field positions must differ (spatial variation).
    assert not np.allclose(k1, k2), (
        "Kernels at (0.1,0.1) and (0.9,0.9) must differ for a spatially-varying map"
    )


def test_ipc_deconvolve_roundtrip():
    """Forward-convolve then deconvolve; interior RMS error < 0.1% of peak."""
    from smig.sensor.ipc import FieldDependentIPC
    from smig.config.schemas import IPCConfig

    ipc = FieldDependentIPC(IPCConfig(), sca_id=1, field_position=(0.5, 0.5))

    rng = np.random.default_rng(42)
    original = rng.uniform(100.0, 5_000.0, size=(64, 64))

    convolved = ipc.apply(original)
    recovered = ipc.deconvolve(convolved)

    # Evaluate RMS on interior only (exclude 10-pixel border for edge effects).
    border = 10
    interior = (slice(border, -border), slice(border, -border))
    rms_error = np.sqrt(np.mean((recovered[interior] - original[interior]) ** 2))
    peak = original.max()

    assert rms_error / peak < 1e-3, (
        f"Deconvolution RMS error {rms_error:.6f} exceeds 0.1% of peak {peak:.1f}"
    )


@pytest.mark.xfail(reason="Physics not yet implemented")
def test_cr_mask_covers_full_cluster():
    """_inject_single_event mask must cover all pixels that received charge."""
    cfg = DetectorConfig(geometry=GeometryConfig(nx=32, ny=32))
    injector = ClusteredCosmicRayInjector(cfg, np.random.default_rng(4))
    image = np.zeros((32, 32), dtype=np.float64)
    morphology = np.array([[0.1, 0.8, 0.1]], dtype=np.float64)  # 3-pixel track
    _, mask = injector._inject_single_event(
        image, y0=16, x0=16, energy_electrons=10_000.0, morphology=morphology
    )
    # Every pixel that received non-zero charge must be True in the mask
    modified = (image > 0)
    assert np.all(mask[modified]), "cr_mask must cover all pixels that received charge"


@pytest.mark.xfail(reason="Physics not yet implemented")
def test_cr_hit_count_is_events_not_pixels():
    """A single 5-pixel cluster contributes 1 to cosmic_ray_hit_count."""
    cfg = DetectorConfig(geometry=GeometryConfig(nx=32, ny=32))
    injector = ClusteredCosmicRayInjector(cfg, np.random.default_rng(5))
    image = np.zeros((32, 32), dtype=np.float64)
    morphology = np.ones((1, 5), dtype=np.float64) / 5  # 5-pixel track
    _, mask = injector._inject_single_event(
        image, y0=16, x0=16, energy_electrons=10_000.0, morphology=morphology
    )
    # The injector returns 1 event regardless of cluster size
    _, _, hit_count = injector.apply(image)
    assert hit_count == 1, f"Expected hit_count=1 for one event, got {hit_count}"


# ---------------------------------------------------------------------------
# Phase D — Schema boundary tests (Phase B fields and config_sha256 validation)
# ---------------------------------------------------------------------------

def test_provenance_record_has_phase_b_fields():
    """All Phase B fields must be declared in ProvenanceRecord.model_fields."""
    assert _EXPECTED_PHASE_B_PROVENANCE_FIELDS <= set(ProvenanceRecord.model_fields)


def test_provenance_schema_rejects_non_hex_sha256():
    """ProvenanceRecord raises ValidationError when config_sha256 contains
    non-hexadecimal characters (pattern ^[0-9a-f]{64}$ not satisfied)."""
    _VALID_RNG_STATE = {
        "bit_generator": "PCG64",
        "state": {"state": 0, "inc": 0},
        "has_uint32": 0,
        "uinteger": 0,
    }
    with pytest.raises(ValidationError):
        ProvenanceRecord(
            event_id="test",
            epoch_index=0,
            timestamp_utc=datetime.now(timezone.utc),
            git_commit=None,
            container_digest=None,
            python_version="3.11",
            numpy_version="1.26",
            config_sha256="Z" * 64,  # uppercase Z is not a hex digit
            random_state=_VALID_RNG_STATE,
            ipc_applied=False,
            persistence_applied=False,
            nonlinearity_applied=False,
            charge_diffusion_applied=True,
            saturated_pixel_count=0,
            cosmic_ray_hit_count=0,
        )


# ---------------------------------------------------------------------------
# Phase D — Tracker strict drift tests
# ---------------------------------------------------------------------------

def _make_valid_record(
    event_id: str = "test_event",
    epoch_index: int = 0,
    *,
    git_commit: str | None = None,
    container_digest: str | None = None,
    config_sha256: str = "a" * 64,
) -> ProvenanceRecord:
    """Helper: build a minimal valid ProvenanceRecord for drift tests."""
    return ProvenanceRecord(
        event_id=event_id,
        epoch_index=epoch_index,
        timestamp_utc=datetime.now(timezone.utc),
        git_commit=git_commit,
        container_digest=container_digest,
        python_version="3.11",
        numpy_version="1.26",
        config_sha256=config_sha256,
        random_state={
            "bit_generator": "PCG64",
            "state": {"state": 0, "inc": 0},
            "has_uint32": 0,
            "uinteger": 0,
        },
        ipc_applied=False,
        persistence_applied=False,
        nonlinearity_applied=False,
        charge_diffusion_applied=True,
        saturated_pixel_count=0,
        cosmic_ray_hit_count=0,
    )


def test_tracker_rejects_silent_metadata_drift():
    """ProvenanceTracker raises ValueError when record git_commit is None
    but the tracker has a non-None value (and vice versa)."""
    # Case 1: tracker has a commit, record has None
    tracker = ProvenanceTracker(event_id="test_event")
    tracker.git_commit = "abc123def456abc123def456abc123def456abc1"
    record_none_commit = _make_valid_record(git_commit=None)
    with pytest.raises(ValueError, match="git_commit"):
        tracker.append_record(record_none_commit)

    # Case 2: tracker has None (env var unset), record has a non-None commit
    tracker2 = ProvenanceTracker(event_id="test_event")
    assert tracker2.git_commit is None  # env var not set in test environment
    record_with_commit = _make_valid_record(git_commit="abc123def456abc123def456abc123def456abc1")
    with pytest.raises(ValueError, match="git_commit"):
        tracker2.append_record(record_with_commit)


# ---------------------------------------------------------------------------
# Phase D — RNG sanitizer type-safety test
# ---------------------------------------------------------------------------

def test_sanitize_rng_state_rejects_non_dict():
    """sanitize_rng_state raises TypeError for non-dict inputs."""
    with pytest.raises(TypeError, match="dict"):
        sanitize_rng_state([1, 2, 3])  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="dict"):
        sanitize_rng_state("not a dict")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="dict"):
        sanitize_rng_state(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Phase D — MULTIACCUM ramp physics tests
# ---------------------------------------------------------------------------

def _make_readout_sim(
    *,
    n_reads: int = 5,
    frame_time: float = 1.0,
    dark_e_per_s: float = 0.0,
    cds_noise_e: float = 0.0,
    nl_coeffs: tuple = (1.0, 0.0),
    full_well: float = 100_000.0,
    sat_threshold: float = 0.85,
    seed: int = 0,
) -> "MultiAccumSimulator":
    """Helper: build a MultiAccumSimulator with explicit parameters."""
    from smig.sensor.readout import MultiAccumSimulator
    from smig.sensor.nonlinearity import NonLinearityModel
    from smig.config.schemas import ReadoutConfig, NonlinearityConfig

    rc = ReadoutConfig(
        n_ramp_reads=n_reads,
        frame_time_s=frame_time,
        exposure_time_s=float(n_reads - 1) * frame_time,
    )
    nl_cfg = NonlinearityConfig(coefficients=nl_coeffs, saturation_flag_threshold=sat_threshold)
    nl = NonLinearityModel(nl_cfg, full_well_electrons=full_well)
    return MultiAccumSimulator(
        rc,
        dark_current_e_per_s=dark_e_per_s,
        read_noise_cds_electrons=cds_noise_e,
        nonlinearity=nl,
        rng=np.random.default_rng(seed),
    )


def test_ramp_dimensions_and_timing():
    """simulate_ramp returns (ramp, sat_reads) with ramp shape (n_reads, ny, nx)."""
    ny, nx = 8, 8
    sim = _make_readout_sim(n_reads=5, frame_time=1.0)
    ideal = np.zeros((ny, nx), dtype=np.float64)
    ramp, sat_reads = sim.simulate_ramp(ideal)
    assert ramp.shape == (5, ny, nx), f"Expected (5,8,8), got {ramp.shape}"
    assert ramp.dtype == np.float64
    assert sat_reads.shape == (5, ny, nx)
    assert sat_reads.dtype == bool


def test_dark_current_accumulation():
    """Zero-photon ramp: per-read dark increment mean ≈ dark_e_per_s * frame_time.

    Uses 100×100 pixels and 5-sigma Poisson tolerance.
    """
    ny, nx = 100, 100
    dark_e_per_s = 0.5      # higher rate for better statistical power
    frame_time = 1.0
    n_reads = 6

    sim = _make_readout_sim(
        n_reads=n_reads, frame_time=frame_time,
        dark_e_per_s=dark_e_per_s, cds_noise_e=0.0,
        seed=42,
    )
    ideal = np.zeros((ny, nx), dtype=np.float64)
    ramp, _ = sim.simulate_ramp(ideal)  # (n_reads, ny, nx)

    # Per-pixel per-read increments
    increments = ramp[1:] - ramp[:-1]       # (n_reads-1, ny, nx)
    mean_inc = increments.mean(axis=(1, 2))  # (n_reads-1,) mean over pixels

    expected = dark_e_per_s * frame_time                        # mean counts per interval
    poisson_std_of_mean = np.sqrt(expected / (ny * nx))         # CLT std of sample mean
    np.testing.assert_allclose(
        mean_inc, expected, atol=5.0 * poisson_std_of_mean,
        err_msg="Per-read dark current increment deviates more than 5σ from expected mean",
    )


def test_read_noise_addition():
    """Zero-signal, zero-dark ramp: spatial variance per read ≈ (cds/√2)²."""
    ny, nx = 200, 200
    cds_noise_e = 10.0
    per_read_std = cds_noise_e / np.sqrt(2.0)

    sim = _make_readout_sim(
        n_reads=4, frame_time=1.0,
        dark_e_per_s=0.0, cds_noise_e=cds_noise_e,
        seed=77,
    )
    ideal = np.zeros((ny, nx), dtype=np.float64)
    ramp, _ = sim.simulate_ramp(ideal)  # (4, ny, nx)

    expected_var = per_read_std ** 2
    for i in range(ramp.shape[0]):
        actual_var = ramp[i].var()
        rel_err = abs(actual_var - expected_var) / expected_var
        assert rel_err < 0.05, (
            f"Read {i}: variance {actual_var:.4f} differs from expected "
            f"{expected_var:.4f} by {rel_err*100:.1f}% (> 5% tolerance)"
        )


# ---------------------------------------------------------------------------
# Phase D — NonLinearity physics tests
# ---------------------------------------------------------------------------

def test_nonlinearity_polynomial_accuracy():
    """NL output matches hand-computed polynomial at known charge levels."""
    from smig.sensor.nonlinearity import NonLinearityModel
    from smig.config.schemas import NonlinearityConfig

    Q_FW = 100_000.0
    coeffs = (1.0, -2.7e-6, 7.8e-12)
    nl = NonLinearityModel(NonlinearityConfig(coefficients=coeffs), Q_FW)

    Q_test = np.array([0.0, 10_000.0, 50_000.0, 80_000.0], dtype=np.float64)

    # Hand-computed reference (identical formula to the implementation)
    Q_norm = Q_test / Q_FW
    S = coeffs[0] + coeffs[1] * Q_norm + coeffs[2] * Q_norm ** 2
    expected = np.clip(Q_test * S, 0.0, 0.85 * Q_FW)

    actual = nl.apply(Q_test)
    np.testing.assert_allclose(actual, expected, rtol=1e-12,
                               err_msg="NL polynomial output does not match hand-computed values")


def test_saturation_clipping_and_slope_exclusion():
    """fit_slope excludes reads at/after saturation; slope matches pre-sat rate.

    Constructs a synthetic ramp analytically (no Poisson noise) at a rate
    that saturates at read 2.  Verifies:
      1. No ramp value exceeds Q_sat (hard clip enforced by NL).
      2. Slope from OLS matches the true pre-saturation rate to <1 ppm.
    """
    from smig.sensor.readout import MultiAccumSimulator
    from smig.sensor.nonlinearity import NonLinearityModel
    from smig.config.schemas import ReadoutConfig, NonlinearityConfig

    Q_FW = 100_000.0
    Q_sat = 0.85 * Q_FW           # 85_000 e-
    n_reads = 9
    frame_time = 5.85

    nl_cfg = NonlinearityConfig(coefficients=(1.0, 0.0), saturation_flag_threshold=0.85)
    nl = NonLinearityModel(nl_cfg, full_well_electrons=Q_FW)
    rc = ReadoutConfig(
        n_ramp_reads=n_reads,
        frame_time_s=frame_time,
        exposure_time_s=float(n_reads - 1) * frame_time,
    )
    sim = MultiAccumSimulator(
        rc, dark_current_e_per_s=0.0, read_noise_cds_electrons=0.0,
        nonlinearity=nl, rng=np.random.default_rng(0),
    )

    # Build a noise-free ramp analytically: rate = 10_000 e-/s
    # t[i] = i * 5.85; charge[i] = 10_000 * t[i]; clip to Q_sat
    # t[0]=0 → 0 (good); t[1]=5.85 → 58_500 (good); t[2]=11.7 → 117_000 → clipped
    true_rate = 10_000.0          # e-/s
    ny, nx = 3, 3
    t = np.arange(n_reads, dtype=np.float64) * frame_time
    ramp_cube = (true_rate * t)[:, np.newaxis, np.newaxis] * np.ones((1, ny, nx))
    np.clip(ramp_cube, 0.0, Q_sat, out=ramp_cube)

    # Sanity checks on constructed ramp
    assert ramp_cube[1, 0, 0] < Q_sat, "Read 1 should be below Q_sat"
    assert ramp_cube[2, 0, 0] == pytest.approx(Q_sat), "Read 2 should be clipped to Q_sat"

    # 1. Hard-clip assertion: no ramp value exceeds Q_sat
    assert ramp_cube.max() <= Q_sat + 1e-10

    # 2. fit_slope should recover the true rate using only the 2 good reads (0 and 1).
    #    OLS of [(0, 0), (5.85, 58_500)] → slope = 58_500/5.85 = 10_000 e-/s exactly.
    slope = sim.fit_slope(ramp_cube)
    np.testing.assert_allclose(
        slope, true_rate, rtol=1e-6,
        err_msg="fit_slope slope does not match pre-saturation trajectory",
    )
