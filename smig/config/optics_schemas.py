"""
smig/config/optics_schemas.py
=============================
Immutable, validated Pydantic v2 configuration models for Phase 2 modules:
PSF rendering, crowded-field image generation, difference image analysis (DIA),
and the top-level SimulationConfig that composes all sub-configs.

Design rules
------------
* Every model uses ``ConfigDict(frozen=True, extra="forbid")``.
* SimulationConfig wraps DetectorConfig by composition — DetectorConfig itself
  is never modified (its SHA-256 canary in test_config_utils.py must stay intact).
* A model_validator on SimulationConfig enforces that the detector geometry is
  at least as large as the DIA context stamp, preventing silent mismatches
  between the sensor array size and the stamp dimensions.

Pydantic v2 is required (>= 2.0).
"""

from __future__ import annotations

from typing import Literal

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from smig.config.schemas import DetectorConfig


# ---------------------------------------------------------------------------
# PSFConfig
# ---------------------------------------------------------------------------

class PSFConfig(BaseModel):
    """Configuration for the WebbPSF-based point-spread function model.

    Controls the filter bandpass, wavelength sampling, oversampling factor,
    line-of-sight jitter, and an optional on-disk PSF cache directory.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    # W146 is SMIG's user-facing label for the wide band; STPSF expects F146.
    # smig/optics/psf._resolve_stpsf_filter_name handles the alias translation.
    filter_name: str = Field(
        default="W146",
        description=(
            "Roman WFI filter identifier (e.g. 'W146', 'F062', 'F087').  "
            "Passed to WebbPSF's RomanWFI instrument object (W146 is aliased to F146)."
        ),
    )
    oversample: int = Field(
        default=4,
        ge=1,
        description=(
            "PSF oversampling factor relative to the native pixel scale.  "
            "The PSF array is computed at oversample × native resolution and "
            "then down-sampled for convolution.  Must be >= 1."
        ),
    )
    fov_native_pixels: int = Field(
        default=128,
        ge=16,
        description=(
            "Side length of the PSF stamp in native WFI pixels (before oversampling).  "
            "The oversampled PSF grid has shape "
            "``(fov_native_pixels * oversample, fov_native_pixels * oversample)`` "
            "and is passed as WebbPSF ``fov_pixels``.  Must be >= 16.  Set equal to "
            "``dia.context_stamp_size`` (typically 256) so the PSF footprint covers "
            "the full rendering context — GalSim's InterpolatedImage returns hard "
            "zero outside the array boundary (not a taper), so clipping PSF wings "
            "produces rectangular artefacts in rendered stamps.  Enforced "
            "<= detector.geometry.nx/ny by SimulationConfig.  "
            "Migration note: this field participates in PSFConfig's SHA-256 hash via "
            "``model_dump_json``; both the in-process LRU cache and the on-disk HDF5 "
            "PSF cache are keyed on that hash, so changing fov_native_pixels "
            "invalidates any existing cached PSF arrays for this provider — an "
            "expected one-time recompute, not a bug."
        ),
    )
    n_wavelengths: int = Field(
        default=10,
        ge=2,
        description=(
            "Number of monochromatic wavelength samples used to build the "
            "polychromatic PSF by quadrature summation over the filter bandpass.  "
            "Must be >= 2 to span the wavelength range."
        ),
    )
    jitter_rms_mas: float = Field(
        default=5.0,
        ge=0.0,
        description=(
            "Line-of-sight pointing jitter RMS in milli-arcseconds (mas) to add "
            "in quadrature to the WebbPSF diffraction model.  "
            "5.0 mas is the Roman pointing requirement."
        ),
    )
    cache_dir: str | None = Field(
        default=None,
        description=(
            "Optional path to a directory where computed PSF arrays are cached "
            "to avoid redundant WebbPSF calls across simulation runs.  "
            "If None, caching is disabled."
        ),
    )
    wavelength_range_um: tuple[float, float] = Field(
        default=(0.93, 2.00),
        description=(
            "Inclusive wavelength range in micrometres (lower, upper) over which "
            "the polychromatic PSF is integrated.  "
            "Default (0.93, 2.00) spans the W146 wide-band filter."
            "Must satisfy wavelength_range_um[0] < wavelength_range_um[1]."
        ),
    )

    psf_edge_taper_pixels: float | None = Field(
        default=None,
        description=(
            "Optional width in pixels of a cosine edge taper applied to the raw PSF "
            "array (on both backends, before jitter) to reduce Gibbs-style ringing from "
            "the hard rectangular array boundary.  The taper rolls from 1 (interior) to "
            "0 (array edge) over this many pixels on all four sides.  None (default) "
            "disables the taper and preserves exact pre-feature behavior.  Must be >= 0.0."
        ),
    )

    @field_validator("psf_edge_taper_pixels")
    @classmethod
    def _check_taper_pixels_nonneg(cls, v: float | None) -> float | None:
        if v is not None and v < 0.0:
            raise ValueError(
                f"psf_edge_taper_pixels must be >= 0.0, got {v!r}."
            )
        return v

    @field_validator("wavelength_range_um", mode="before")
    @classmethod
    def _coerce_wavelength_range(cls, v: Any) -> Any:
        if isinstance(v, (list, tuple)):
            return tuple(v)
        return v

    @field_validator("wavelength_range_um")
    @classmethod
    def _check_wavelength_order(
        cls, v: tuple[float, float]
    ) -> tuple[float, float]:
        if v[0] >= v[1]:
            raise ValueError(
                f"wavelength_range_um lower bound ({v[0]}) must be strictly less "
                f"than upper bound ({v[1]})."
            )
        return v


# ---------------------------------------------------------------------------
# RenderingConfig
# ---------------------------------------------------------------------------

class RenderingConfig(BaseModel):
    """Placeholder configuration for the CrowdedFieldRenderer rendering pipeline.

    Fields are TBD — to be defined during Phase 2 rendering implementation
    (GalSim draw mode, sky-background handling, bandpass flux normalisation, etc.).
    This class exists now so that SimulationConfig can embed it and YAML round-trips
    remain stable when Phase 2 fills in the fields.
    """

    # Fields are TBD for Phase 2 CrowdedFieldRenderer implementation.
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


# ---------------------------------------------------------------------------
# CrowdedFieldConfig
# ---------------------------------------------------------------------------

class CrowdedFieldConfig(BaseModel):
    """Configuration for the crowded-field stamp-based image renderer.

    Controls the science-output stamp size, pixel scale, neighbour inclusion
    radius, and optional source brightness cap.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    stamp_size: int = Field(
        default=64,
        gt=0,
        description=(
            "Side length of the square science output stamp in pixels.  "
            "Must be > 0.  Typical Phase 2 value: 64 px."
        ),
    )
    pixel_scale_arcsec: float = Field(
        default=0.11,
        gt=0.0,
        description=(
            "On-sky plate scale of the rendered stamp in arcseconds per pixel.  "
            "Should match detector.geometry.pixel_scale_arcsec for native-resolution "
            "simulations."
        ),
    )
    brightness_cap_mag: float | None = Field(
        default=None,
        description=(
            "Optional AB magnitude cap below which sources are excluded from the "
            "rendered stamp (i.e. sources brighter than this value are omitted).  "
            "Useful to skip saturated stars during training data generation.  "
            "None disables the cap."
        ),
    )
    neighbor_mag_limit: float = Field(
        default=26.0,
        ge=-5.0,
        le=40.0,
        description=(
            "Faintest AB magnitude of neighbour sources drawn into the crowded-field "
            "stamp.  Sources fainter than this limit are excluded.  "
            "Default 26.0 AB roughly matches the Roman WFI 5σ point-source depth "
            "for a ~100 s exposure.  AB magnitudes can be negative for very bright "
            "sources, so the valid range is [-5.0, 40.0]."
        ),
    )


# ---------------------------------------------------------------------------
# DIAConfig
# ---------------------------------------------------------------------------

class DIAConfig(BaseModel):
    """Configuration for the Difference Image Analysis (DIA) pipeline.

    Controls the reference-stack depth, context- and science-stamp dimensions,
    and the image subtraction algorithm.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    n_reference_epochs: int = Field(
        default=30,
        gt=0,
        description=(
            "Number of pre-event reference epochs co-added to build the template "
            "image.  Must be > 0."
        ),
    )
    context_stamp_size: int = Field(
        default=256,
        gt=0,
        description=(
            "Side length of the square context (PSF-fitting / template) stamp in "
            "pixels.  Must be <= detector.geometry.nx and detector.geometry.ny "
            "(enforced by SimulationConfig validator).  Typical Phase 2 value: 256 px."
        ),
    )
    science_stamp_size: int = Field(
        default=64,
        gt=0,
        description=(
            "Side length of the square science-output (difference-image) stamp in "
            "pixels.  Typically matches crowded_field.stamp_size.  Typical: 64 px."
        ),
    )
    subtraction_method: Literal["identity", "alard_lupton", "sfft"] = Field(
        default="identity",
        description=(
            "Image subtraction algorithm.  "
            "'identity' (default for matched-reference simulated Phase 2 runs): "
            "direct ``science - reference`` subtraction in rate space.  Appropriate "
            "only when reference and science share the same PSF and detector chain "
            "— current QA shows AL introduces repeated source-core null bias and "
            "suppresses lensed signal, while identity preserves both null and "
            "lensed structure cleanly.  Do NOT use for real DIA workflows where "
            "the reference and science PSFs differ.  "
            "'alard_lupton': classic Alard & Lupton (1998) kernel-based subtraction. "
            "Retained for PSF-mismatch experiments and real-DIA paths.  "
            "'sfft': Saccadic Fast Fourier Transform method (Hu et al. 2022)."
        ),
    )
    reference_processing: Literal["detector_matched", "ideal_gaussian"] = Field(
        default="detector_matched",
        description=(
            "Reference-template construction fidelity selector for SceneSimulator.  "
            "'detector_matched' (science default): each reference epoch is rendered "
            "through the same H4RG10Detector signal chain as science epochs, then "
            "the rate images are averaged.  This matches IPC, charge diffusion, and "
            "Poisson statistics between reference and science — required for trusted "
            "DIA residuals.  "
            "'ideal_gaussian' (development only): reference epochs skip the detector "
            "chain and are coadded by DIAPipeline.build_reference using a scalar "
            "per-epoch variance (read noise + dark + sky background).  Faster, but "
            "the reference no longer matches the per-pixel detector physics of the "
            "science epochs and AC-4-style null/RMS guarantees are weaker.  Use only "
            "in fast_dev profiles, never as the default release gate."
        ),
    )
    reference_jitter_mode: Literal["per_epoch", "shared"] = Field(
        default="per_epoch",
        description=(
            "Jitter realization policy for reference epochs.  "
            "'per_epoch' (science default): each reference epoch draws its own PSF "
            "jitter seed, so reference-stack averaging samples the jitter ensemble.  "
            "'shared' (development only): every reference epoch reuses a single "
            "jitter seed.  This dramatically reduces PSF cache misses (one PSF "
            "compute instead of n_reference_epochs) but suppresses the reference's "
            "natural jitter averaging — only safe when the development workflow does "
            "not depend on realistic jitter statistics in the template."
        ),
    )


# ---------------------------------------------------------------------------
# SimulationConfig  (top-level composite)
# ---------------------------------------------------------------------------

class SimulationConfig(BaseModel):
    """Top-level immutable configuration for a Phase 2 SMIG simulation run.

    Composes DetectorConfig (sensor physics) with optics and pipeline sub-configs.
    DetectorConfig is embedded by composition — its schema and SHA-256 canary
    are never modified.

    Geometry constraints
    --------------------
    The detector array must be at least as large as the DIA context stamp:
        detector.geometry.nx >= dia.context_stamp_size
        detector.geometry.ny >= dia.context_stamp_size

    The PSF stamp must not exceed the detector dimensions:
        psf.fov_native_pixels <= detector.geometry.nx
        psf.fov_native_pixels <= detector.geometry.ny

    Phase 2 stamp-based runs should override the geometry defaults to 256×256
    (matching the DIA context stamp) in simulation.yaml, rather than using the
    full 4096×4096 H4RG-10 sensor dimensions.

    Example
    -------
    >>> from pathlib import Path
    >>> import yaml
    >>> cfg = SimulationConfig.model_validate(
    ...     yaml.safe_load(Path("smig/config/simulation.yaml").read_text())
    ... )
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    detector: DetectorConfig = Field(
        default_factory=DetectorConfig,
        description="Sensor-physics configuration (H4RG-10 detector model).",
    )
    psf: PSFConfig = Field(
        default_factory=PSFConfig,
        description="PSF rendering configuration.",
    )
    rendering: RenderingConfig = Field(
        default_factory=RenderingConfig,
        description="CrowdedFieldRenderer rendering pipeline configuration (TBD).",
    )
    crowded_field: CrowdedFieldConfig = Field(
        default_factory=CrowdedFieldConfig,
        description="Crowded-field stamp-based image renderer configuration.",
    )
    dia: DIAConfig = Field(
        default_factory=DIAConfig,
        description="Difference Image Analysis pipeline configuration.",
    )

    @model_validator(mode="after")
    def _check_geometry_vs_context_stamp(self) -> SimulationConfig:
        """Reject configs where the detector array is smaller than the DIA stamp.

        The DIA context stamp is extracted from the rendered detector array, so the
        detector dimensions must be at least as large as ``dia.context_stamp_size``
        on both axes.
        """
        ctx = self.dia.context_stamp_size
        nx = self.detector.geometry.nx
        ny = self.detector.geometry.ny
        if nx < ctx:
            raise ValueError(
                f"detector.geometry.nx ({nx}) must be >= dia.context_stamp_size "
                f"({ctx}).  Override detector geometry in simulation.yaml to at "
                "least match the context stamp size."
            )
        if ny < ctx:
            raise ValueError(
                f"detector.geometry.ny ({ny}) must be >= dia.context_stamp_size "
                f"({ctx}).  Override detector geometry in simulation.yaml to at "
                "least match the context stamp size."
            )
        return self

    @model_validator(mode="after")
    def _check_psf_fov_vs_geometry(self) -> SimulationConfig:
        """Reject configs where psf.fov_native_pixels exceeds the detector dimensions.

        GalSim's InterpolatedImage returns hard zero outside its array bounds, so a PSF
        stamp larger than the detector array would be silently clipped.
        """
        fov = self.psf.fov_native_pixels
        nx = self.detector.geometry.nx
        ny = self.detector.geometry.ny
        if fov > nx:
            raise ValueError(
                f"psf.fov_native_pixels ({fov}) must be <= detector.geometry.nx ({nx})."
            )
        if fov > ny:
            raise ValueError(
                f"psf.fov_native_pixels ({fov}) must be <= detector.geometry.ny ({ny})."
            )
        return self
