"""
smig/rendering/pipeline.py
==========================
Phase 2 top-level simulation orchestrator for SMIG v2.

:class:`SceneSimulator` wires together:

    STPSFProvider → CrowdedFieldRenderer → FiniteSourceRenderer
    → H4RG10Detector → DIAPipeline

into a single per-event simulation call that returns a fully-processed
:class:`EventSceneOutput` containing 64×64 difference stamps, saturation
masks, cosmic-ray masks, and updated Phase 2 provenance records.

RNG wiring
----------
All randomness is derived deterministically from ``(master_seed, event_id)``
via SHA-256 — never via Python's ``hash()`` (whose output varies across
processes and Python versions).  Per-stage child generators are spawned with
independent seeds so that inserting or removing a stage never perturbs other
stages' random streams.

Memory contract
---------------
The full ``ideal_cube_e`` array (shape: ``n_science_epochs × ny × nx``) is
held in memory only while ``H4RG10Detector.process_event`` runs.  As soon as
``process_event`` returns, ``ideal_cube_e`` is explicitly ``del``-eted and
``gc.collect()`` is called before DIA begins.

Architecture boundary
---------------------
This module may import ``H4RG10Detector`` (the sealed sensor-layer API) and
``DetectorConfig`` (the configuration type).  It must NOT import internal
sensor-physics leaf modules from ``smig.sensor.*`` (e.g. ``charge_diffusion``,
``ipc``, ``noise.*``).
"""
from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from smig.config.optics_schemas import SimulationConfig
from smig.config.schemas import DetectorConfig  # noqa: F401 — allowed import per spec
from smig.config.seed import derive_event_seed, derive_stage_seed
from smig.optics.psf import STPSFProvider
from smig.provenance.schema import ProvenanceRecord
from smig.rendering.crowding import CrowdedFieldRenderer
from smig.rendering.dia import DIAPipeline
from smig.rendering.source import FiniteSourceRenderer
from smig.sensor.detector import H4RG10Detector

if TYPE_CHECKING:
    import pandas as _pd_type

# ---------------------------------------------------------------------------
# Optional runtime dependencies (galsim, pandas)
# ---------------------------------------------------------------------------

_GALSIM_AVAILABLE: bool = False
_galsim: Any = None
try:
    import galsim as _galsim  # type: ignore[assignment]
    _GALSIM_AVAILABLE = True
except ImportError:
    pass

_PANDAS_AVAILABLE: bool = False
_pd: Any = None
try:
    import pandas as _pd  # type: ignore[assignment]
    _PANDAS_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class EventSceneOutput:
    """Output of a single simulated microlensing event.

    All stamp arrays are cropped to ``science_stamp_size × science_stamp_size``
    to minimise the final memory footprint.  By default, the full context-sized
    pre-detector images are not stored; pass ``retain_ideal_cube=True`` to
    :meth:`~smig.rendering.pipeline.SceneSimulator.simulate_event` to include
    them (callers accept the memory cost of the full cube).

    Attributes
    ----------
    difference_stamps:
        Shape ``(n_science_epochs, science_stamp_size, science_stamp_size)``,
        dtype ``float64``.  DIA difference image crops, one per science epoch.
    saturation_stamps:
        Shape ``(n_science_epochs, science_stamp_size, science_stamp_size)``,
        dtype ``bool``.  Saturation-flag crops matching the difference stamp
        footprint.
    cr_stamps:
        Shape ``(n_science_epochs, science_stamp_size, science_stamp_size)``,
        dtype ``bool``.  Cosmic-ray-flag crops matching the difference stamp
        footprint.
    provenance:
        One :class:`~smig.provenance.schema.ProvenanceRecord` per science
        epoch, updated with Phase 2 fields (``psf_config_hash``,
        ``n_neighbors_rendered``, ``dia_method``, ``reference_n_epochs``).
        Length equals ``n_science_epochs``.
    ideal_cube_e:
        Shape ``(n_science_epochs, ny, nx)``, dtype ``float64``.  Ideal
        electron count per pixel *before* detector physics (charge diffusion,
        IPC, readout, noise).  ``None`` unless ``retain_ideal_cube=True`` was
        passed to :meth:`~smig.rendering.pipeline.SceneSimulator.simulate_event`.
    """

    difference_stamps: np.ndarray
    saturation_stamps: np.ndarray
    cr_stamps: np.ndarray
    provenance: list[ProvenanceRecord]
    ideal_cube_e: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Synthetic neighbor catalog generator
# ---------------------------------------------------------------------------

def _generate_catalog(
    rng: np.random.Generator,
    n_stars: int,
    ctx: int,
) -> "_pd_type.DataFrame":
    """Generate a synthetic uniform-random neighbor catalog for one event.

    Star positions are drawn uniformly within the context stamp footprint.
    Magnitudes are drawn uniformly in [20, 26] AB mag; fluxes are derived
    from a simple zero-point calibration so that ``flux_e`` is authoritative.

    This is a Phase 2 placeholder.  File-based catalogs from Galaxia
    population-synthesis grids are a Phase 3 deliverable.

    Parameters
    ----------
    rng:
        Crowding-stage RNG.  Consumed ONCE per event; caller must not reuse
        the same generator for other stages.
    n_stars:
        Number of synthetic neighbor stars to generate.
    ctx:
        Side length of the context stamp in pixels; positions are drawn in
        ``[0, ctx)``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``x_pix`` (float64), ``y_pix`` (float64), ``flux_e``
        (float64), ``mag_w146`` (float64).
    """
    mag_w146 = rng.uniform(20.0, 26.0, size=n_stars)
    # Simple zero-point: mag 25 → 1 e⁻/s × t_exp; scale by 2.5-mag steps.
    flux_e = 10.0 ** ((25.0 - mag_w146) / 2.5)
    return _pd.DataFrame(
        {
            "x_pix": rng.uniform(0.0, float(ctx), size=n_stars).astype(np.float64),
            "y_pix": rng.uniform(0.0, float(ctx), size=n_stars).astype(np.float64),
            "flux_e": flux_e.astype(np.float64),
            "mag_w146": mag_w146.astype(np.float64),
        }
    )


# ---------------------------------------------------------------------------
# SceneSimulator
# ---------------------------------------------------------------------------

class SceneSimulator:
    """Phase 2 top-level orchestrator.

    Wraps :class:`~smig.sensor.detector.H4RG10Detector` — does NOT replace it.
    Owns a :class:`~smig.optics.psf.STPSFProvider` shared across events (its
    in-process LRU cache amortises the per-PSF WebbPSF cost); all other
    per-event objects (detector, DIA pipeline, crowded-field renderer) are
    freshly instantiated per :meth:`simulate_event` call with deterministic,
    event-specific RNG seeds.

    Parameters
    ----------
    config:
        Top-level simulation configuration.  Must satisfy the geometry
        constraint: ``detector.geometry.nx == detector.geometry.ny ==
        dia.context_stamp_size``.
    master_seed:
        Master RNG seed.  Combined with ``event_id`` via SHA-256 to derive all
        per-event child seeds deterministically.

    Raises
    ------
    ImportError
        If ``galsim`` or ``pandas`` is not installed.
    """

    #: Number of synthetic neighbor stars generated per event.
    _N_CATALOG_STARS: int = 20

    def __init__(self, config: SimulationConfig, master_seed: int) -> None:
        if not _GALSIM_AVAILABLE:
            raise ImportError(
                "galsim is required for SceneSimulator. "
                "Install Phase 2 extras: pip install -e '.[phase2]'"
            )
        if not _PANDAS_AVAILABLE:
            raise ImportError(
                "pandas is required for SceneSimulator. "
                "Install Phase 2 extras: pip install -e '.[phase2]'"
            )
        self._config = config
        self._master_seed = master_seed
        # Shared across events: PSF cache amortises WebbPSF computation cost.
        self._psf_provider = STPSFProvider(config.psf)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate_event(
        self,
        event_id: str,
        source_params_sequence: list[dict],
        timestamps_mjd: np.ndarray,
        backgrounds_e_per_s: list[float],
        neighbor_catalog: "_pd_type.DataFrame | None" = None,
        baseline_source_flux_e: float | None = None,
        retain_ideal_cube: bool = False,
    ) -> EventSceneOutput:
        """Simulate a complete microlensing event end-to-end.

        Builds a DIA reference from ``config.dia.n_reference_epochs`` static
        frames, renders ``len(timestamps_mjd)`` science epochs (static
        background + source), processes the full epoch cube through
        :class:`~smig.sensor.detector.H4RG10Detector`, subtracts the
        reference with :class:`~smig.rendering.dia.DIAPipeline`, crops all
        outputs to ``science_stamp_size``, and returns an
        :class:`EventSceneOutput` with updated Phase 2 provenance.

        Parameters
        ----------
        event_id:
            Unique string identifier for this event.  Used as input to the
            SHA-256 seed derivation — never passed to Python's ``hash()``.
        source_params_sequence:
            Per-epoch source parameters, one ``dict`` per science epoch.
            Required key:

            * ``flux_e`` (``float``) — total source flux in electrons.

            Optional keys (fall back to point-source defaults if absent):

            * ``centroid_offset_pix`` (``tuple[float, float]``) — ``(dx, dy)``
              offset from stamp centre in pixels.  Default ``(0.0, 0.0)``.
            * ``rho_star_arcsec`` (``float``) — projected stellar radius in
              arcseconds.  Default ``0.0`` (point source).
            * ``limb_darkening_coeffs`` (``tuple[float, float] | None``) —
              Quadratic LD coefficients ``(u1, u2)``, or ``None`` for uniform
              disk / point source.

        timestamps_mjd:
            1-D array of Modified Julian Dates for science epochs.  Length
            must equal ``len(source_params_sequence)``.  Must be
            non-decreasing.
        backgrounds_e_per_s:
            Sky-background rates in e⁻/s, one per science epoch.  Length must
            equal ``len(timestamps_mjd)``.  The first entry is also used for
            all reference-epoch backgrounds.
        baseline_source_flux_e:
            Unlensed (A=1) source flux in electrons, used to render the
            source star into every reference epoch.  When ``None`` (default),
            the reference contains only the crowded-field neighbors — the
            legacy behavior, suitable only for tests that deliberately inject
            a source into science while leaving it absent from the reference.

            **Production use:** always supply this value.  In real DIA the
            reference template is built from pre-event baseline epochs that
            contain the unlensed source at flux ``F₀``.  Without it, the
            difference image retains the full source signal (not just the
            excess ``(A−1)·F₀``), which corrupts null-event labels and
            inflates residuals for all event classes.

            Derive from the source star's AB magnitude::

                from smig.catalogs.photometry import mag_ab_to_electrons
                baseline_source_flux_e = mag_ab_to_electrons(
                    star.mag_F146_ab, "F146", exposure_s
                )

        retain_ideal_cube:
            When ``True``, the pre-detector ideal electron cube
            (shape ``(n_science_epochs, ny, nx)``, dtype ``float64``) is
            stored in :attr:`EventSceneOutput.ideal_cube_e` instead of being
            freed before DIA.  Callers accept the additional memory cost
            (~2.6 MB per 256×256 × 5-epoch event at float64).  Default
            ``False`` preserves the original memory contract.

        Returns
        -------
        EventSceneOutput
            Contains ``(n_science_epochs, science_stamp_size, science_stamp_size)``
            arrays for differences, saturation flags, and CR flags, plus one
            updated :class:`~smig.provenance.schema.ProvenanceRecord` per epoch.

        Raises
        ------
        ValueError
            If sequence lengths are inconsistent.
        """
        n_science = len(timestamps_mjd)
        if len(source_params_sequence) != n_science:
            raise ValueError(
                f"len(source_params_sequence)={len(source_params_sequence)} "
                f"must equal len(timestamps_mjd)={n_science}."
            )
        if len(backgrounds_e_per_s) != n_science:
            raise ValueError(
                f"len(backgrounds_e_per_s)={len(backgrounds_e_per_s)} "
                f"must equal len(timestamps_mjd)={n_science}."
            )

        cfg = self._config
        n_ref = cfg.dia.n_reference_epochs
        ctx = cfg.dia.context_stamp_size
        sci_sz = cfg.dia.science_stamp_size
        pixel_scale = cfg.crowded_field.pixel_scale_arcsec
        sca_id = cfg.detector.ipc.sca_id

        # --- 1. Deterministic seed derivation (canonical smig.config.seed) ---
        event_seed = derive_event_seed(self._master_seed, event_id)

        detector_rng = np.random.default_rng(
            derive_stage_seed(event_seed, "detector")
        )
        crowding_rng = np.random.default_rng(
            derive_stage_seed(event_seed, "crowding")
        )

        # Draw one integer seed per science epoch from the psf_jitter generator.
        psf_jitter_rng = np.random.default_rng(
            derive_stage_seed(event_seed, "psf_jitter")
        )
        science_jitter_seeds: np.ndarray = psf_jitter_rng.integers(
            1, 2**31 - 1, size=n_science
        )

        dia_ref_rng = np.random.default_rng(
            derive_stage_seed(event_seed, "dia_reference")
        )
        ref_detector_rng = np.random.default_rng(
            derive_stage_seed(event_seed, "ref_detector")
        )

        # Draw one integer seed per reference epoch from the dia_jitter generator.
        # In "shared" mode every reference epoch reuses the first seed so the PSF
        # cache hits after the first lookup (development-only optimization).
        dia_jitter_rng = np.random.default_rng(
            derive_stage_seed(event_seed, "dia_jitter")
        )
        ref_jitter_seeds: np.ndarray = dia_jitter_rng.integers(
            1, 2**31 - 1, size=n_ref
        )
        if cfg.dia.reference_jitter_mode == "shared":
            ref_jitter_seeds = np.full(n_ref, int(ref_jitter_seeds[0]), dtype=ref_jitter_seeds.dtype)

        # --- 2. Per-event object instantiation ---
        detector = H4RG10Detector(cfg.detector, detector_rng)
        # ref_detector is only needed for the detector_matched reference path;
        # constructing the H4RG10Detector eagerly keeps the per-event RNG state
        # ordering identical across both reference modes.
        ref_detector = H4RG10Detector(cfg.detector, ref_detector_rng)
        dia = DIAPipeline(cfg.dia, cfg.detector, dia_ref_rng)
        source_renderer = FiniteSourceRenderer()

        # Sample catalog ONCE per event; use injected catalog if provided (Phase 3.5).
        if neighbor_catalog is None:
            catalog = _generate_catalog(crowding_rng, self._N_CATALOG_STARS, ctx)
        else:
            catalog = neighbor_catalog
        crowded_renderer = CrowdedFieldRenderer(
            catalog,
            stamp_size=ctx,
            pixel_scale=pixel_scale,
            brightness_cap_mag=cfg.crowded_field.brightness_cap_mag,
        )

        # PSF config hash captured once — config is frozen, so hash is stable.
        psf_config_hash: str = self._psf_provider.psf_config_hash

        # Stamp centre in absolute detector pixel coordinates (centre of stamp).
        stamp_center: tuple[float, float] = (float(ctx) / 2.0, float(ctx) / 2.0)

        # --- 3. Build DIA reference from static-field-only reference epochs ---
        # Field position is pinned to the SCA centre for both reference and
        # science PSFs (Phase 2 MVP simplification — PSF spatial variation
        # across the SCA is not modelled).  Lifting this requires deriving
        # (x_sca, y_sca) from the source-star projection and threading it
        # through the per-epoch get_psf calls.
        ref_field_pos = (0.5, 0.5)
        sci_field_pos = (0.5, 0.5)

        # Reference construction has two profile-selectable branches:
        #   detector_matched (science): each ref epoch goes through the full
        #     ref_detector.process_epoch chain; rate images are then mean-coadded.
        #     This is the only path that matches IPC, charge diffusion, and
        #     Poisson statistics between reference and science.
        #   ideal_gaussian (fast_dev): ref epochs skip the detector chain; the
        #     ideal-electron arrays are passed to dia.build_reference, which
        #     applies a scalar per-epoch variance and inverse-variance coadds.
        ref_base_mjd = timestamps_mjd[0] - float(n_ref)
        ref_processing = cfg.dia.reference_processing

        ideal_ref_epochs: list[np.ndarray] = []
        for j in range(n_ref):
            ref_jitter_seed = int(ref_jitter_seeds[j])
            ref_psf = self._psf_provider.get_psf(
                sca_id=sca_id,
                field_position=ref_field_pos,
                jitter_seed=ref_jitter_seed,
            )
            static_field = crowded_renderer.render_static_field(
                ref_psf,
                stamp_center,
                psf_fingerprint=(psf_config_hash, sca_id, ref_field_pos, ref_jitter_seed),
            )
            if baseline_source_flux_e is not None:
                # Render unlensed source (A=1) into each reference epoch so the
                # template contains F₀ and the difference carries only (A−1)·F₀.
                galsim_ref_stamp = _galsim.Image(
                    static_field.copy(), scale=pixel_scale
                )
                source_renderer.render_source(
                    flux_e=float(baseline_source_flux_e),
                    centroid_offset_pix=(0.0, 0.0),
                    rho_star_arcsec=0.0,
                    limb_darkening_coeffs=None,
                    psf=ref_psf,
                    stamp=galsim_ref_stamp,
                )
                ideal_ref_e = np.clip(galsim_ref_stamp.array, 0.0, None)
            else:
                ideal_ref_e = np.clip(static_field, 0.0, None)
            ideal_ref_epochs.append(ideal_ref_e)

        if ref_processing == "detector_matched":
            ref_rate_images: list[np.ndarray] = []
            for j, ideal_ref_e in enumerate(ideal_ref_epochs):
                # Process through the same detector chain as science epochs so
                # that IPC, charge diffusion, and Poisson noise are consistent
                # between reference and science rate images.  A dedicated
                # ref_detector keeps reference-epoch RNG state independent of
                # the science detector.
                ref_mjd = ref_base_mjd + float(j)
                prev_ref_mjd = None if j == 0 else ref_base_mjd + float(j - 1)
                ref_epoch_out = ref_detector.process_epoch(
                    ideal_image_e=ideal_ref_e,
                    epoch_index=j,
                    epoch_time_mjd=ref_mjd,
                    prev_epoch_time_mjd=prev_ref_mjd,
                )
                ref_rate_images.append(ref_epoch_out.rate_image)
                del ref_epoch_out
            reference = np.mean(np.stack(ref_rate_images), axis=0)
            del ref_rate_images
        elif ref_processing == "ideal_gaussian":
            # Cheap path: scalar-variance Gaussian noise model in DIAPipeline.
            # No detector chain, no IPC/diffusion match — fast_dev only.
            reference = dia.build_reference(
                ideal_ref_epochs,
                [float(backgrounds_e_per_s[0])] * n_ref,
            )
        else:
            # Pydantic Literal already constrains the value; this guard documents
            # the contract and prevents silent fall-through if the schema changes.
            raise ValueError(
                f"Unknown dia.reference_processing={ref_processing!r}; "
                "expected 'detector_matched' or 'ideal_gaussian'."
            )
        del ideal_ref_epochs
        gc.collect()

        # --- 4. Render science epochs → ideal_cube_e (n_science, ny, nx) ---
        ny = cfg.detector.geometry.ny
        nx = cfg.detector.geometry.nx
        ideal_cube_e = np.zeros((n_science, ny, nx), dtype=np.float64)

        for i, params in enumerate(source_params_sequence):
            sci_jitter_seed = int(science_jitter_seeds[i])
            sci_psf = self._psf_provider.get_psf(
                sca_id=sca_id,
                field_position=sci_field_pos,
                jitter_seed=sci_jitter_seed,
            )
            # Per-epoch static field: PSF jitter differs each epoch, so the
            # cache key changes; render_static_field re-renders correctly.
            static_field = crowded_renderer.render_static_field(
                sci_psf,
                stamp_center,
                psf_fingerprint=(psf_config_hash, sca_id, sci_field_pos, sci_jitter_seed),
            )

            # Initialise GalSim stamp with the static field, then accumulate source.
            galsim_stamp = _galsim.Image(
                static_field.copy(),
                scale=pixel_scale,
            )
            del static_field
            source_renderer.render_source(
                flux_e=float(params["flux_e"]),
                centroid_offset_pix=params.get("centroid_offset_pix", (0.0, 0.0)),
                rho_star_arcsec=float(params.get("rho_star_arcsec", 0.0)),
                limb_darkening_coeffs=params.get("limb_darkening_coeffs", None),
                psf=sci_psf,
                stamp=galsim_stamp,
            )
            # Clip to non-negative: sub-pixel PSF ringing can produce tiny negatives.
            ideal_cube_e[i] = np.clip(galsim_stamp.array, 0.0, None)

        # --- 5. Detector processing (hold ideal_cube_e until process_event returns) ---
        np.clip(ideal_cube_e, 0.0, None, out=ideal_cube_e)
        # Opt-in: keep a reference for debug/golden outputs. When False, del below
        # frees the allocation before DIA; when True, caller accepts the memory cost.
        _retained_ideal: np.ndarray | None = ideal_cube_e if retain_ideal_cube else None
        event_output = detector.process_event(event_id, ideal_cube_e, timestamps_mjd)
        # Memory contract: remove the local name. When retain_ideal_cube=False this
        # releases the allocation before DIA allocates its buffers.
        del ideal_cube_e
        gc.collect()

        # --- 6. DIA subtraction + central crop to science_stamp_size ---
        half = sci_sz // 2
        center = ctx // 2
        row_start = center - half
        row_stop = center + half
        col_start = center - half
        col_stop = center + half

        diff_stamps_list: list[np.ndarray] = []
        sat_stamps_list: list[np.ndarray] = []
        cr_stamps_list: list[np.ndarray] = []

        for i in range(n_science):
            diff_full = dia.subtract(event_output.rate_cube[i], reference)
            diff_crop = dia.extract_stamp(diff_full)
            del diff_full

            sat_crop = (
                event_output.saturation_cube[i][row_start:row_stop, col_start:col_stop]
                .copy()
            )
            cr_crop = (
                event_output.cr_cube[i][row_start:row_stop, col_start:col_stop]
                .copy()
            )

            diff_stamps_list.append(diff_crop)
            sat_stamps_list.append(sat_crop)
            cr_stamps_list.append(cr_crop)

        # --- 7. Provenance update with Phase 2 fields ---
        n_neighbors_rendered = crowded_renderer.count_neighbors_rendered()

        updated_records: list[ProvenanceRecord] = []
        for record in event_output.provenance_records:
            updated = record.model_copy(
                update={
                    "psf_config_hash": psf_config_hash,
                    "n_neighbors_rendered": n_neighbors_rendered,
                    "dia_method": cfg.dia.subtraction_method,
                    "reference_n_epochs": n_ref,
                    "reference_processing": cfg.dia.reference_processing,
                    "reference_jitter_mode": cfg.dia.reference_jitter_mode,
                }
            )
            updated_records.append(updated)

        # --- 8. Assemble output ---
        return EventSceneOutput(
            difference_stamps=np.stack(diff_stamps_list),
            saturation_stamps=np.stack(sat_stamps_list),
            cr_stamps=np.stack(cr_stamps_list),
            provenance=updated_records,
            ideal_cube_e=_retained_ideal,
        )
