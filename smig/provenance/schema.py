"""
smig/provenance/schema.py
=========================
Pydantic v2 model representing a single epoch's provenance record.

One ProvenanceRecord is created after each detector-simulation epoch
(i.e. after each simulated exposure in a multi-epoch microlensing event).
A list of these records is accumulated by ProvenanceTracker and serialised
as a JSON sidecar alongside the output FITS/HDF5 products.

Pydantic v2 is required (>= 2.0).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator


def sanitize_rng_state(state: dict[str, Any]) -> dict[str, Any]:
    """Recursively convert numpy types in an RNG state dict to native Python.

    Converts ``np.ndarray`` -> list and ``np.generic`` -> Python scalar,
    traversing nested dicts and sequences.  Safe to call even when numpy is
    not installed (returns the input unchanged in that case).

    Parameters
    ----------
    state:
        A dict as returned by ``numpy.random.Generator.bit_generator.state``.
        Must be a ``dict``; passing any other type raises ``TypeError``.

    Returns
    -------
    dict[str, Any]
        The same structure with all numpy types replaced by native Python types,
        making the result safe to pass to ``json.dumps``.

    Raises
    ------
    TypeError
        If ``state`` is not a ``dict``.
    """
    if not isinstance(state, dict):
        raise TypeError(
            f"sanitize_rng_state expects a dict, got {type(state).__name__!r}.  "
            "Pass the dict returned by "
            "numpy.random.Generator.bit_generator.state."
        )

    try:
        import numpy as np
    except ImportError:
        return state

    def _convert(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, dict):
            return {k: _convert(val) for k, val in sorted(obj.items())}
        if isinstance(obj, set):
            return sorted([_convert(item) for item in obj])
        if isinstance(obj, (list, tuple)):
            converted = (_convert(item) for item in obj)
            return type(obj)(converted)
        return obj

    return _convert(state)


class ProvenanceRecord(BaseModel):
    """Immutable audit record for one simulated epoch of one microlensing event.

    All fields are required at construction time (no defaults) to force the
    caller to populate every provenance dimension explicitly.

    Timezone requirement
    --------------------
    ``timestamp_utc`` uses Pydantic's ``AwareDatetime`` type, which rejects
    naive (timezone-unaware) datetime objects at validation time.  Callers
    must supply a timezone-aware value, e.g.::

        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc)

    NumPy random-state snapshot
    ---------------------------
    The ``random_state`` field holds the dict returned by
    ``numpy.random.Generator.bit_generator.state``, e.g.::

        rng = numpy.random.default_rng(seed=42)
        state = rng.bit_generator.state   # {"bit_generator": "PCG64", ...}

    A field validator (``mode="before"``) recursively sanitizes the dict,
    converting ``np.ndarray`` to lists and ``np.generic`` scalars to native
    Python types so the output is cleanly JSON-serializable.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # ------------------------------------------------------------------
    # Event / epoch identity
    # ------------------------------------------------------------------

    event_id: str = Field(
        description=(
            "Unique string identifier for the microlensing event, e.g. "
            "'ob230001' or a UUID.  All epochs belonging to the same event "
            "share this identifier."
        ),
    )
    epoch_index: int = Field(
        ge=0,
        description=(
            "Zero-based index of this epoch within the event's observation "
            "sequence.  Together with event_id this forms a unique primary key."
        ),
    )

    # ------------------------------------------------------------------
    # Wall-clock / audit metadata
    # ------------------------------------------------------------------

    timestamp_utc: AwareDatetime = Field(
        description=(
            "UTC timestamp at which this epoch's simulation completed.  "
            "Must be timezone-aware (naive datetimes are rejected at "
            "validation time).  Serialised to ISO-8601 string."
        ),
    )

    # ------------------------------------------------------------------
    # Software / environment fingerprints
    # ------------------------------------------------------------------

    git_commit: str | None = Field(
        description=(
            "Git commit identifier (SHA-1 or SHA-256 hex string) of the "
            "source code used to produce this epoch, or None if the "
            "environment variable GIT_COMMIT_SHA was not set (e.g. in a "
            "local development run outside CI)."
        ),
    )
    container_digest: str | None = Field(
        description=(
            "OCI container image digest (e.g. 'sha256:abc123...') of the "
            "execution environment, or None if the environment variable "
            "IMAGE_DIGEST was not set."
        ),
    )
    python_version: str = Field(
        description=(
            "Python interpreter version string (e.g. '3.11.7'), captured "
            "via ``platform.python_version()`` at simulation time."
        ),
    )
    numpy_version: str = Field(
        description=(
            "NumPy package version string (e.g. '1.26.3'), captured via "
            "``numpy.__version__`` at simulation time."
        ),
    )

    # ------------------------------------------------------------------
    # Configuration fingerprint
    # ------------------------------------------------------------------

    config_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description=(
            "SHA-256 hex digest of the canonical JSON serialisation of the "
            "DetectorConfig used for this epoch.  Allows downstream users to "
            "verify that two epochs used identical detector settings even if "
            "the YAML source file has changed.  "
            "Must be exactly 64 lowercase hexadecimal characters."
        ),
    )

    # ------------------------------------------------------------------
    # Reproducibility: PRNG state
    # ------------------------------------------------------------------

    # BREAKING CHANGE (Phase 1 remediation): random_state now accepts both
    # ``dict[str, Any]`` (new structured format) and ``str`` (legacy sidecar
    # format).  The new dict format captures per-child-generator states with
    # the keys ``parent``, ``readout``, ``one_over_f``, ``rts``,
    # ``cosmic_rays`` — each value being the bit-generator state dict for that
    # child RNG — so any epoch can be perfectly replayed from this snapshot.
    # Old sidecar readers that stored the flat parent-RNG state dict (keys
    # ``bit_generator``, ``state``) will continue to round-trip as ``dict``.
    # Readers that serialised the state as a plain string pass through unchanged.
    random_state: dict[str, Any] | str = Field(
        description=(
            "Structured RNG state snapshot captured immediately *before* each "
            "epoch's stochastic stages, allowing perfect replay.  "
            "New format: dict with keys ``parent``, ``readout``, "
            "``one_over_f``, ``rts``, ``cosmic_rays`` — each containing the "
            "bit-generator state dict for that child generator.  "
            "Legacy format (flat dict with ``bit_generator`` / ``state`` keys "
            "or a plain string) is accepted for backward compatibility with "
            "existing sidecar files.  "
            "NumPy array/scalar types are sanitized to native Python before storage."
        ),
    )

    @field_validator("random_state", mode="before")
    @classmethod
    def _sanitize_numpy_types(cls, v: dict[str, Any] | str) -> dict[str, Any] | str:
        """Sanitize numpy types in dict-form state; pass strings through unchanged."""
        if isinstance(v, str):
            # Legacy sidecar format: plain string representation — accept as-is.
            return v
        return sanitize_rng_state(v)

    # ------------------------------------------------------------------
    # Applied-effect flags
    # ------------------------------------------------------------------

    ipc_applied: bool = Field(
        description=(
            "True if inter-pixel capacitance convolution was applied to this "
            "epoch's charge image."
        ),
    )
    persistence_applied: bool = Field(
        description=(
            "True if residual-image (persistence) charge was injected into "
            "this epoch."
        ),
    )
    nonlinearity_applied: bool = Field(
        description=(
            "True if the polynomial detector nonlinearity model was applied "
            "to convert accumulated charge to measured signal."
        ),
    )
    charge_diffusion_applied: bool = Field(
        description=(
            "True if the charge diffusion and brighter-fatter effect model "
            "was applied to this epoch's charge image."
        ),
    )

    # ------------------------------------------------------------------
    # Per-epoch pixel-level statistics
    # ------------------------------------------------------------------

    saturated_pixel_count: int = Field(
        ge=0,
        description=(
            "Number of pixels that reached or exceeded the saturation flag "
            "threshold (nonlinearity.saturation_flag_threshold x full_well) "
            "in this epoch."
        ),
    )
    cosmic_ray_hit_count: int = Field(
        ge=0,
        description=(
            "Number of distinct CR events injected, not the number of affected "
            "pixels.  A single event with a 5-pixel morphology contributes 1 to "
            "this count, regardless of cluster size.  "
            "Stub: always 0 until injection is implemented."
        ),
    )

    # ------------------------------------------------------------------
    # Phase 1 extended provenance fields
    # ------------------------------------------------------------------

    ipc_kernel_hash: str | None = Field(
        default=None,
        description=(
            "SHA-256 hex digest of the IPC kernel array loaded from the HDF5 "
            "calibration file, or None if the uniform fallback kernel was used.  "
            "Allows verification that the same calibration file was used across epochs."
        ),
    )
    persistence_history_depth: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of prior epochs whose accumulated trap state was available "
            "when computing the persistence signal for this epoch.  "
            "0 = no persistence history (first epoch or persistence disabled)."
        ),
    )
    n_partial_saturation_pixels: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of pixels that crossed the saturation flag threshold "
            "(nonlinearity.saturation_flag_threshold) but did not reach the "
            "hard-clip ceiling (full_well_electrons) in this epoch.  "
            "Distinct from saturated_pixel_count, which counts all flagged pixels."
        ),
    )
    cr_types: list[str] | None = Field(
        default=None,
        description=(
            "List of cosmic ray event morphology labels injected in this epoch "
            "(e.g. ['point', 'track', 'cluster']), or None if CR injection was "
            "disabled or not yet implemented."
        ),
    )
    n_rts_active_pixels: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of pixels in the active RTS pixel set that switched state "
            "during this epoch's ramp.  "
            "0 until RTS noise injection is implemented."
        ),
    )
    slope_fit_method: str | None = Field(
        default=None,
        description=(
            "Algorithm used to reduce the MULTIACCUM ramp to a slope image "
            "(e.g. 'least_squares', 'optimal_weighting', 'cds').  "
            "None until ramp-fitting is implemented."
        ),
    )
    n_reads_used_median: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Median number of non-destructive reads actually used in the slope "
            "fit after cosmic-ray and saturation masking, across all pixels.  "
            "None until ramp-fitting is implemented."
        ),
    )
    peak_memory_mb: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Peak resident memory consumed during this epoch's simulation in "
            "megabytes, as measured by the memory profiler.  "
            "None if memory profiling was not available."
        ),
    )

    # ------------------------------------------------------------------
    # Phase 2 extended provenance fields
    # ------------------------------------------------------------------

    psf_config_hash: str | None = Field(
        default=None,
        description=(
            "SHA-256 hex digest of the canonical JSON serialisation of the "
            "PSFConfig used for this epoch, or None if PSF provenance was not "
            "captured (e.g. Phase 1 records without rendering)."
        ),
    )
    n_neighbors_rendered: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of neighbour stars rendered into the crowded-field stamp "
            "after applying the brightness-cap filter.  "
            "0 = no crowded-field renderer was used, or all neighbours were "
            "excluded by the brightness cap."
        ),
    )
    dia_method: Literal["identity", "alard_lupton", "sfft"] | None = Field(
        default=None,
        description=(
            "Image subtraction algorithm used to produce the difference stamp "
            "for this epoch.  "
            "'identity': direct ``science - reference`` (matched-reference simulations).  "
            "'alard_lupton': Alard & Lupton (1998) Gaussian kernel basis.  "
            "'sfft': Saccadic Fast Fourier Transform (Hu et al. 2022).  "
            "None if DIA was not applied (Phase 1 records)."
        ),
    )
    reference_n_epochs: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of pre-event reference epochs co-added to build the DIA "
            "template image.  "
            "0 if DIA was not applied (Phase 1 records)."
        ),
    )
    reference_processing: Literal["detector_matched", "ideal_gaussian"] | None = Field(
        default=None,
        description=(
            "DIA reference-template construction fidelity for this epoch.  "
            "'detector_matched': each reference epoch was rendered through the full "
            "H4RG10Detector signal chain and rate-coadded (the science default).  "
            "'ideal_gaussian': reference epochs skipped the detector chain "
            "(development-only fast path).  None if DIA was not applied."
        ),
    )
    reference_jitter_mode: Literal["per_epoch", "shared"] | None = Field(
        default=None,
        description=(
            "Jitter realization policy used for the reference stack.  "
            "'per_epoch': each reference epoch drew an independent jitter seed "
            "(the science default).  "
            "'shared': every reference epoch reused a single jitter seed "
            "(development-only).  None if DIA was not applied."
        ),
    )
