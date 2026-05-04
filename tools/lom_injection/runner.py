"""Orchestrator for the SMIG Phase 3 LOM ablation sprint.

This module connects io.py (reading phase3-contract-v1 shards and writing
phase3-lom-ablation-v0 shards) with physics.py (per-epoch VBBL magnification
under linear lens orbital motion).

Per-Event Dispatch
------------------
Events are dispatched on ``event_class``:

PSPL / FSPL_STAR (event_class 0, 1) — single-lens passthrough
    LOM is physically undefined for single-lens events.  The original Phase 3
    science stamps are passed through verbatim, unchanged.  The LOM datasets
    (``lom__ds_dt_per_yr``, ``lom__dalpha_dt_rad_per_yr``) are written as 0.0
    to mark these as static null-class entries.  No SceneSimulator call is made.
    These events serve as leakage controls: any non-zero reconstructed residual
    between static and "LOM" stamps in post-processing indicates a bug.

Binary (PLANETARY_CAUSTIC=2, STELLAR_BINARY=3, HIGH_MAGNIFICATION_CUSP=4) — LOM path
    1. Reconstruct physical params via ``io.denormalize``.
    2. Compute per-epoch A(t) via ``physics.magnification_2l1s_lom``.
    3. Compute baseline flux F0 = mag_ab_to_electrons(source_mag_F146_ab, exposure_s).
    4. Build source_params_sequence: [{"flux_e": A(t_i) * F0}] per epoch.
    5. Call SceneSimulator.simulate_event with neighbor_catalog=None.
    6. Cast difference_stamps to float32 and write to LomShardWriter.

Failure Handling and Resize Discipline
---------------------------------------
Per-event failures (LOMComputationError from s_i ≤ 0 or VBBL crash,
SceneSimulator exceptions, stamp validation errors) are caught, logged, and
counted.  The failed event is skipped; the LomShardWriter is NOT written for
that event.  After all events in a shard are processed:

    writer.resize(n_written, rows_failed=n_failed)

is called **before** the ``with LomShardWriter(...)`` block exits.  This
ensures the HDF5 file is correctly trimmed before the atomic os.replace, and
the partial-failure metadata (rows_expected / rows_written / rows_failed) is
recorded in the root attrs.

Phase 3.5 Prerequisite Guard
------------------------------
``check_simulator_prerequisite`` inspects the installed SceneSimulator for the
``neighbor_catalog`` parameter introduced in Phase 3.5.  If absent, the tool
raises ``RuntimeError`` before processing any events, rather than silently
calling simulate_event without crowded-field reconstruction support.

Static Equivalence Verification
---------------------------------
``run_static_equivalence_check`` calls ``physics.verify_static_equivalence``
with a representative planetary-caustic parameter set once at startup.  This
catches sign-convention mismatches or VBBL version regressions before
processing thousands of events.
"""
from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from smig.catalogs.photometry import mag_ab_to_electrons  # read-only import
from smig.config.seed import derive_event_seed, derive_stage_seed
from smig.config.utils import load_simulation_config       # read-only import

from tools.lom_injection.io import (
    LomShardWriter,
    Phase3EventRecord,
    SurveyConfig,
    iter_shard,
    load_survey_config,
)
from tools.lom_injection.physics import LOMComputationError, magnification_2l1s_lom

log = logging.getLogger(__name__)

# event_class constants (mirrors smig.datasets.labels.EventClass without importing it)
_SINGLE_LENS_CLASSES: frozenset[int] = frozenset({0, 1})  # PSPL=0, FSPL_STAR=1
_BINARY_CLASSES: frozenset[int] = frozenset({2, 3, 4})    # PC=2, SB=3, HMC=4


# ---------------------------------------------------------------------------
# Prerequisite guards
# ---------------------------------------------------------------------------

def check_simulator_prerequisite() -> None:
    """Verify that SceneSimulator.simulate_event accepts neighbor_catalog.

    Phase 3.5 added the ``neighbor_catalog`` keyword to
    ``SceneSimulator.simulate_event``.  If it is absent the installed smig
    version pre-dates Phase 3.5 and this tool cannot safely call the simulator.

    Raises
    ------
    RuntimeError
        If SceneSimulator cannot be imported or the parameter is missing.
    """
    try:
        from smig.rendering.pipeline import SceneSimulator  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import SceneSimulator from smig.rendering.pipeline. "
            "Ensure smig is installed (pip install -e '.[phase2]')."
        ) from exc

    sig = inspect.signature(SceneSimulator.simulate_event)
    if "neighbor_catalog" not in sig.parameters:
        raise RuntimeError(
            "SceneSimulator.simulate_event is missing the 'neighbor_catalog' "
            "keyword argument.  Phase 3.5 is a prerequisite for this tool.  "
            "Update smig to a version that includes Phase 3.5."
        )
    log.debug("Prerequisite check passed: neighbor_catalog present in simulate_event.")


def run_static_equivalence_check() -> None:
    """Run physics.verify_static_equivalence with a representative event.

    Uses a planetary-caustic parameter set that exercises the caustic-crossing
    regime where trajectory-coordinate bugs are most visible.  Skipped silently
    if VBBinaryLensing is not installed (i.e. in pure I/O test environments).

    Raises
    ------
    AssertionError
        If LOM(ds_dt=0, dalpha_dt=0) disagrees with the static magnification.
    """
    from tools.lom_injection.physics import verify_static_equivalence  # noqa: PLC0415

    # Representative planetary-caustic parameters (t_E=30d, q=1e-3, s=1.1).
    t0 = 60015.0
    t_mjd = np.linspace(t0 - 30.0, t0 + 30.0, 30)
    try:
        verify_static_equivalence(
            t_mjd=t_mjd,
            t0=t0,
            tE=30.0,
            u0=0.05,
            rho=0.01,
            alpha=1.2,
            q=1e-3,
            s=1.1,
            rtol=1e-5,
        )
        log.debug("Static equivalence check passed.")
    except ImportError:
        log.warning(
            "VBBinaryLensing not installed; skipping static equivalence check."
        )


# ---------------------------------------------------------------------------
# Shard-level processing
# ---------------------------------------------------------------------------

@dataclass
class ShardResult:
    """Summary statistics from processing one input shard."""
    input_shard: Path
    output_shard: Path
    n_events_read: int
    n_written: int
    n_failed: int
    n_single_lens_passthrough: int


def _reconstruct_neighbor_catalog(
    *,
    source_master_seed: int,
    source_event_id: str,
    sim_cfg: Any,
    phase3_cfg: dict[str, Any],
    exposure_s: float,
) -> Any:
    """Reproduce the Phase 3 worker's neighbor catalog for one source event.

    Phase 3 source shards do not persist the neighbor catalog itself, only
    starfield_seed.  Because the worker's catalog draw is fully deterministic
    in (master_seed, event_id) via SHA-256 seed derivation, we can reconstruct
    the same crowded scene here as long as the LOM injection is given the
    same master_seed used at generation time (read from the YAML config).

    This mirrors smig.datasets.worker._generate_event_impl steps 2–3 + 8.
    """
    from smig.datasets.worker import _build_provider, _exclude_source_star  # noqa: PLC0415
    from smig.catalogs.adapter import project_to_sca_dataframe                # noqa: PLC0415

    ctx: int = sim_cfg.dia.context_stamp_size
    pixel_scale_arcsec: float = sim_cfg.crowded_field.pixel_scale_arcsec
    fov_deg: float = ctx * pixel_scale_arcsec / 3600.0
    sca_id: int = sim_cfg.detector.ipc.sca_id

    l_deg: float = float(phase3_cfg["field_center_l_deg"])
    b_deg: float = float(phase3_cfg["field_center_b_deg"])
    provider_name: str = str(phase3_cfg["catalog_provider"])
    catalog_path = phase3_cfg.get("catalog_path")
    catalog_path_str = str(catalog_path) if catalog_path else None

    event_seed = derive_event_seed(source_master_seed, source_event_id)
    catalog_rng = np.random.default_rng(derive_stage_seed(event_seed, "catalog_sample"))

    provider = _build_provider(provider_name, catalog_path_str)
    stars = provider.sample_field(l_deg, b_deg, fov_deg, catalog_rng)
    if not stars:
        raise RuntimeError(
            f"Catalog provider {provider_name!r} returned 0 stars when "
            f"reconstructing neighbors for source event {source_event_id!r}"
        )
    source_idx = int(catalog_rng.integers(0, len(stars)))
    source_star = stars[source_idx]
    neighbors = _exclude_source_star(stars, source_star)
    return project_to_sca_dataframe(neighbors, sca_id, l_deg, b_deg, exposure_s)


def process_shard(
    input_shard: Path,
    output_shard: Path,
    survey: SurveyConfig,
    simulator: "SceneSimulator",  # type: ignore[name-defined]  # noqa: F821
    ds_dt: float,
    dalpha_dt: float,
    shard_id: int,
    source_master_seed: int,
    sim_cfg: Any,
    phase3_cfg: dict[str, Any],
) -> ShardResult:
    """Process one Phase 3 input shard into one LOM ablation output shard.

    The output shard is written only after all events are processed; it is
    fully absent until the atomic rename succeeds.  Rows are written to the
    pre-allocated HDF5 datasets streaming-style (one event at a time) so the
    runner's peak memory is one record + one stamp cube, not the whole shard.

    Parameters
    ----------
    input_shard :
        Path to a phase3-contract-v1 HDF5 file.
    output_shard :
        Target path for the phase3-lom-ablation-v0 HDF5 output.  Must not
        overlap with input_shard (write isolation enforced by caller).
    survey :
        Survey config (observation window, cadence, exposure, background).
    simulator :
        Instantiated SceneSimulator (constructor: SceneSimulator(sim_cfg, master_seed)).
    ds_dt :
        ds/dt to apply to binary events (θ_E / yr).
    dalpha_dt :
        dα/dt to apply to binary events (rad / yr).
    shard_id :
        Integer shard identifier written into the output HDF5 root attrs.
    source_master_seed :
        Master seed used at Phase 3 generation time.  Required so that the
        SceneSimulator's per-event seed derivation here matches the source
        corpus, and so the synthetic neighbor catalog can be reconstructed
        deterministically (no false pixel-level deltas from crowding noise).
    sim_cfg :
        Loaded SimulationConfig (used to reconstruct the neighbor catalog).
    phase3_cfg :
        Raw Phase 3 YAML dict (field centre, provider, catalog_path).

    Returns
    -------
    ShardResult
    """
    output_shard.parent.mkdir(parents=True, exist_ok=True)

    # Probe the shard for n_total without materialising stamps — we need it
    # to pre-allocate the writer datasets but want to stream the actual rows.
    import h5py  # noqa: PLC0415
    with h5py.File(str(input_shard), "r") as _f:
        n_total = int(_f["science_stamps"].shape[0])

    epoch_times = survey.epoch_times_mjd
    backgrounds = [survey.sky_background_e_per_s] * survey.n_science_epochs

    n_written = 0
    n_failed = 0
    n_passthrough = 0

    with LomShardWriter(
        path=output_shard,
        shard_id=shard_id,
        expected_n=n_total,
        n_epochs=survey.n_science_epochs,
        ds_dt=ds_dt,
        dalpha_dt=dalpha_dt,
    ) as writer:
        for record, params in iter_shard(input_shard, survey):
            if record.event_class in _SINGLE_LENS_CLASSES:
                # Passthrough: LOM undefined; use original stamps, record
                # zero LOM params.  Single-lens passthroughs must be
                # byte-equivalent to the source — leakage check.
                writer.write_row(
                    lom_row_index=n_written,
                    record=record,
                    new_stamps=record.stamps.copy(),
                    ds_dt=0.0,
                    dalpha_dt=0.0,
                )
                n_written += 1
                n_passthrough += 1
                continue

            if record.event_class not in _BINARY_CLASSES:
                log.warning(
                    "Unknown event_class=%d for event %r; skipping.",
                    record.event_class, record.event_id,
                )
                n_failed += 1
                continue

            # Binary event — full LOM path.
            try:
                A_lom = magnification_2l1s_lom(
                    t_mjd=epoch_times,
                    t0=params.t0_mjd,           # type: ignore[union-attr]
                    tE=params.tE_days,          # type: ignore[union-attr]
                    u0=params.u0,               # type: ignore[union-attr]
                    rho=params.rho,             # type: ignore[union-attr]
                    alpha0=params.alpha_rad,    # type: ignore[union-attr]
                    q=params.q,                 # type: ignore[union-attr]
                    s0=params.s,                # type: ignore[union-attr]
                    ds_dt=ds_dt,
                    dalpha_dt=dalpha_dt,
                )
            except LOMComputationError as exc:
                log.warning(
                    "LOM physics failed for event %r (epoch %d): %s; skipping.",
                    record.event_id, exc.epoch_index, exc,
                )
                n_failed += 1
                continue
            except Exception as exc:
                log.warning(
                    "Unexpected LOM error for event %r: %s; skipping.",
                    record.event_id, exc,
                )
                n_failed += 1
                continue

            F0_e = mag_ab_to_electrons(
                params.source_mag_F146_ab,  # type: ignore[union-attr]
                "F146",
                survey.exposure_s,
            )
            source_params_seq = [
                {"flux_e": float(A_lom[i] * F0_e)}
                for i in range(survey.n_science_epochs)
            ]

            # Reconstruct the same neighbor catalog the source shard saw, so
            # the only pixel-level delta between LOM and source is the
            # source-flux modulation.  Failure to reconstruct (e.g. provider
            # returns no stars) is treated as a per-event skip.
            try:
                neighbor_df = _reconstruct_neighbor_catalog(
                    source_master_seed=source_master_seed,
                    source_event_id=record.event_id,
                    sim_cfg=sim_cfg,
                    phase3_cfg=phase3_cfg,
                    exposure_s=survey.exposure_s,
                )
            except Exception as exc:
                log.warning(
                    "Neighbor catalog reconstruction failed for event %r: %s; skipping.",
                    record.event_id, exc,
                )
                n_failed += 1
                continue

            try:
                # Pass record.event_id (NOT the "lom_" prefixed form) so that
                # SceneSimulator derives the same per-event seeds as the
                # source corpus — PSF jitter, detector noise, and reference
                # epochs are then bit-for-bit aligned.
                output = simulator.simulate_event(
                    event_id=record.event_id,
                    source_params_sequence=source_params_seq,
                    timestamps_mjd=epoch_times,
                    backgrounds_e_per_s=backgrounds,
                    neighbor_catalog=neighbor_df,
                )
            except Exception as exc:
                log.warning(
                    "SceneSimulator failed for event %r: %s; skipping.",
                    record.event_id, exc,
                )
                n_failed += 1
                continue

            new_stamps = output.difference_stamps.astype(np.float32, copy=True)
            writer.write_row(
                lom_row_index=n_written,
                record=record,
                new_stamps=new_stamps,
                ds_dt=ds_dt,
                dalpha_dt=dalpha_dt,
            )
            n_written += 1

        # Resize trims the pre-allocated arrays and records rows_failed.
        writer.resize(actual_n=n_written, rows_failed=n_failed)

    log.info(
        "Shard %s → %s: %d read, %d written (%d passthrough), %d failed.",
        input_shard.name, output_shard.name,
        n_total, n_written, n_passthrough, n_failed,
    )

    return ShardResult(
        input_shard=input_shard,
        output_shard=output_shard,
        n_events_read=n_total,
        n_written=n_written,
        n_failed=n_failed,
        n_single_lens_passthrough=n_passthrough,
    )


# ---------------------------------------------------------------------------
# Top-level sprint runner
# ---------------------------------------------------------------------------

def run_sprint(
    input_dir: Path,
    output_dir: Path,
    config_path: Path,
    ds_dt: float,
    dalpha_dt: float,
    master_seed: int | None = None,
    glob_pattern: str = "*.h5",
    skip_prerequisite_check: bool = False,
    skip_equivalence_check: bool = False,
) -> list[ShardResult]:
    """Process all Phase 3 shards in input_dir and write LOM ablation shards to output_dir.

    Parameters
    ----------
    input_dir :
        Directory containing phase3-contract-v1 HDF5 shards.
    output_dir :
        Output directory for phase3-lom-ablation-v0 shards.  Created if absent.
    config_path :
        Path to the Phase 3 generation YAML (e.g. configs/phase3_smoke.yaml).
    ds_dt :
        ds/dt value applied to all binary events (θ_E / yr).
    dalpha_dt :
        dα/dt value applied to all binary events (rad / yr).
    master_seed :
        Master RNG seed for SceneSimulator.  ``None`` (the default) means
        "read from the YAML config's ``master_seed`` field", which is the
        only setting that produces stochastic state matching the source
        corpus (and therefore meaningful pixel-level residuals).  Pass an
        explicit integer only when you want to deliberately diverge from
        the source corpus.
    glob_pattern :
        Glob pattern for selecting input shards.
    skip_prerequisite_check :
        Skip the SceneSimulator.simulate_event introspection check.  Use only
        in unit tests where no rendering is performed.
    skip_equivalence_check :
        Skip the static-equivalence cross-check.  Use only in environments
        where VBBinaryLensing is absent.

    Returns
    -------
    list[ShardResult]
        One result per processed shard.
    """
    if not skip_prerequisite_check:
        check_simulator_prerequisite()

    if not skip_equivalence_check:
        run_static_equivalence_check()

    survey = load_survey_config(config_path)

    phase3_cfg: dict[str, Any] = yaml.safe_load(
        config_path.read_text(encoding="utf-8")
    )
    config_master_seed = int(phase3_cfg["master_seed"])
    if master_seed is None:
        master_seed = config_master_seed
    elif master_seed != config_master_seed:
        log.warning(
            "master_seed=%d overrides YAML master_seed=%d. Pixel-level "
            "comparisons against the source corpus will diverge — use only "
            "for deliberate seed-divergence experiments.",
            master_seed, config_master_seed,
        )

    sim_cfg_path = Path(phase3_cfg["simulation_config_path"])
    if not sim_cfg_path.is_absolute():
        candidate = config_path.parent / sim_cfg_path
        sim_cfg_path = candidate if candidate.exists() else Path.cwd() / sim_cfg_path
    sim_cfg = load_simulation_config(sim_cfg_path)

    from smig.rendering.pipeline import SceneSimulator  # noqa: PLC0415
    simulator = SceneSimulator(sim_cfg, master_seed)

    input_shards = sorted(input_dir.glob(glob_pattern))
    if not input_shards:
        raise FileNotFoundError(
            f"No shards matching {glob_pattern!r} found in {input_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[ShardResult] = []

    for shard_id, input_shard in enumerate(input_shards):
        output_shard = output_dir / f"lom_{input_shard.name}"
        result = process_shard(
            input_shard=input_shard,
            output_shard=output_shard,
            survey=survey,
            simulator=simulator,
            ds_dt=ds_dt,
            dalpha_dt=dalpha_dt,
            shard_id=shard_id,
            source_master_seed=master_seed,
            sim_cfg=sim_cfg,
            phase3_cfg=phase3_cfg,
        )
        results.append(result)

    total_read = sum(r.n_events_read for r in results)
    total_written = sum(r.n_written for r in results)
    total_failed = sum(r.n_failed for r in results)
    log.info(
        "Sprint complete: %d shards, %d events read, %d written, %d failed.",
        len(results), total_read, total_written, total_failed,
    )
    return results
