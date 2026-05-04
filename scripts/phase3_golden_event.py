#!/usr/bin/env python3
"""Phase 3 golden event regeneration script.

Fixed parameters (do not change without bumping GOLDEN_VERSION):
  GOLDEN_VERSION = "phase3-v1"
  SEED           = 2026_0419       (date of Phase 3.6 completion)
  EVENT_CLASS    = PSPL
  u0             = 0.05
  tE             = 30.0 days
  t0             = 60000.0 MJD
  rho            = 0.0  (point source)
  source_mag     = 20.0 AB (F146)

Outputs:
  outputs/phase3_golden_event.png — 3-panel figure (magnification curve,
    light curve in e⁻/s, DIA stamp mosaic placeholder if GalSim absent)

Exits 0 in all cases.  If outputs/ is not writable, logs a skip message
and exits without writing.

Usage:
  python scripts/phase3_golden_event.py
"""
import logging
import os
import sys
from pathlib import Path

# Must precede any pyplot import for headless CI.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger("golden_event")

# ---------------------------------------------------------------------------
# Fixed golden-event parameters
# ---------------------------------------------------------------------------
GOLDEN_VERSION = "phase3-v1"
SEED = 20260419
U0 = 0.05
TE_DAYS = 30.0
T0_MJD = 60000.0
RHO = 0.0
SOURCE_MAG_AB = 20.0
BAND = "F146"
EXPOSURE_S = 46.8
N_EPOCHS = 5


def _pspl_magnification(t: np.ndarray, t0: float, tE: float, u0: float) -> np.ndarray:
    u = np.sqrt(u0**2 + ((t - t0) / tE) ** 2)
    return (u**2 + 2.0) / (u * np.sqrt(u**2 + 4.0))


def _mag_ab_to_electrons(mag_ab: float, exposure_s: float) -> float:
    """Approximate F146 AB → electrons using a fixed zero-point."""
    # F146 zero-point: ~1.55e10 e⁻/s for a 0 AB mag source (approximate)
    # Derived from Roman ETC (Hirata 2022 WFI spec), F146 bandpass.
    zp_e_per_s = 1.55e10
    return zp_e_per_s * 10.0 ** (-0.4 * mag_ab) * exposure_s


def _make_outputs_dir() -> Path | None:
    outputs = Path("outputs")
    try:
        outputs.mkdir(exist_ok=True)
        # Probe writability
        probe = outputs / ".probe"
        probe.touch()
        probe.unlink()
        return outputs
    except OSError:
        return None


def main() -> None:
    from smig.microlensing.event import EventClass, MicrolensingEvent
    from smig.microlensing.backends import get_primary_backend

    backend_name, backend_version = get_primary_backend()

    # Build golden event
    event = MicrolensingEvent(
        event_id="golden-phase3-v1",
        t0_mjd=T0_MJD,
        tE_days=TE_DAYS,
        u0=U0,
        rho=RHO,
        alpha_rad=0.0,
        event_class=EventClass.PSPL,
        backend="analytic",
        backend_version="N/A",
    )

    # Magnification curve: 500 points over ±3·tE
    t_curve = np.linspace(T0_MJD - 3 * TE_DAYS, T0_MJD + 3 * TE_DAYS, 500)
    A_curve = _pspl_magnification(t_curve, T0_MJD, TE_DAYS, U0)

    # Science-epoch times (N_EPOCHS epochs, cadence ~12 days for ±3·tE coverage)
    t_epochs = np.linspace(T0_MJD - 2 * TE_DAYS, T0_MJD + 2 * TE_DAYS, N_EPOCHS)
    A_epochs = _pspl_magnification(t_epochs, T0_MJD, TE_DAYS, U0)

    # Light curve in e⁻/s (flux density)
    F0 = _mag_ab_to_electrons(SOURCE_MAG_AB, EXPOSURE_S) / EXPOSURE_S
    flux_epochs = A_epochs * F0

    # Peak metrics
    A_peak_measured = float(np.max(A_curve))
    A_peak_expected = float((U0**2 + 2) / (U0 * (U0**2 + 4) ** 0.5))

    print(f"Golden event: {GOLDEN_VERSION}")
    print(f"  event_class   : {event.event_class.value}")
    print(f"  backend       : {backend_name}")
    print(f"  backend_version: {backend_version}")
    print(f"  u0            : {U0}")
    print(f"  tE            : {TE_DAYS} days")
    print(f"  A_peak_measured  : {A_peak_measured:.6f}")
    print(f"  A_peak_expected  : {A_peak_expected:.6f}")
    print(f"  relative_error   : {abs(A_peak_measured - A_peak_expected)/A_peak_expected:.2e}")

    if not _MATPLOTLIB_AVAILABLE:
        sys.stdout.flush()
        _log.info("PNG generation skipped (headless/no writable outputs dir)")
        sys.exit(0)

    # ---------------------------------------------------------------------------
    # Build 3-panel figure
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(
        f"Phase 3 Golden Event — {GOLDEN_VERSION} (seed={SEED})", fontsize=12
    )

    # Panel 1: Magnification curve
    ax0 = axes[0]
    ax0.plot((t_curve - T0_MJD) / TE_DAYS, A_curve, color="steelblue", lw=1.5)
    ax0.scatter((t_epochs - T0_MJD) / TE_DAYS, A_epochs, color="firebrick", s=40, zorder=5)
    ax0.axhline(A_peak_expected, color="grey", ls="--", lw=0.8, label=f"A_peak={A_peak_expected:.3f}")
    ax0.set_xlabel("(t - t₀) / tE")
    ax0.set_ylabel("Magnification A(t)")
    ax0.set_title("PSPL Magnification Curve")
    ax0.legend(fontsize=8)

    # Panel 2: Light curve in e⁻/s
    ax1 = axes[1]
    t_flux = np.linspace(T0_MJD - 3 * TE_DAYS, T0_MJD + 3 * TE_DAYS, 500)
    A_flux = _pspl_magnification(t_flux, T0_MJD, TE_DAYS, U0)
    ax1.plot((t_flux - T0_MJD) / TE_DAYS, A_flux * F0, color="darkorange", lw=1.5)
    ax1.scatter(
        (t_epochs - T0_MJD) / TE_DAYS, flux_epochs, color="firebrick", s=40, zorder=5
    )
    ax1.axhline(F0, color="grey", ls="--", lw=0.8, label=f"F₀={F0:.2e} e⁻/s")
    ax1.set_xlabel("(t - t₀) / tE")
    ax1.set_ylabel("Flux (e⁻/s)")
    ax1.set_title("Light Curve")
    ax1.legend(fontsize=8)

    # Panel 3: DIA stamp mosaic (placeholder; GalSim required for real stamps)
    ax2 = axes[2]
    _galsim_available = False
    try:
        import galsim  # noqa: F401
        _galsim_available = True
    except ImportError:
        pass

    if _galsim_available:
        ax2.text(
            0.5, 0.5,
            "DIA stamps:\n(re-run with GalSim for rendered stamps)",
            ha="center", va="center", transform=ax2.transAxes, fontsize=9,
        )
        ax2.set_title("DIA Stamp Mosaic (5 epochs)")
    else:
        # Synthetic placeholder: N_EPOCHS Gaussian stamps
        rng = np.random.default_rng(SEED)
        mosaic = np.zeros((64, 64 * N_EPOCHS))
        for i in range(N_EPOCHS):
            stamp = rng.standard_normal((64, 64)).astype(np.float32)
            stamp[30:34, 30:34] += float(A_epochs[i]) * 5.0
            mosaic[:, i * 64 : (i + 1) * 64] = stamp
        ax2.imshow(
            mosaic, origin="lower", cmap="gray",
            vmin=-3, vmax=max(A_epochs) * 6,
        )
        ax2.set_title(f"DIA Stamp Mosaic ({N_EPOCHS} epochs, synthetic)")
        ax2.set_xlabel("pixel (64×5 mosaic)")
        ax2.set_xticks([])
        ax2.set_yticks([])

    plt.tight_layout()

    # ---------------------------------------------------------------------------
    # Write PNG
    # ---------------------------------------------------------------------------
    outputs_dir = _make_outputs_dir()
    if outputs_dir is None:
        _log.info("PNG generation skipped (headless/no writable outputs dir)")
        plt.close(fig)
        sys.exit(0)

    png_path = outputs_dir / "phase3_golden_event.png"
    fig.savefig(str(png_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    _log.info("Golden event PNG written to %s", png_path)


if __name__ == "__main__":
    main()
