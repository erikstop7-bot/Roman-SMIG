#!/usr/bin/env python3
"""Phase 3 golden event — real rendered science, animated GIF.

What each panel shows
---------------------
Left top    — PSPL magnification curve A(t).  Animated white dot shows the
              current epoch.

Left bottom — Source light curve in e⁻/s (= A(t) × F₀).  Same animated dot.

Right       — 64×64 DIA difference stamp.  This IS DIA-subtracted: the
              reference template contains the unlensed source (baseline_source_flux_e
              supplied), so the difference carries only the lensing excess
              ≈ (A(t) − 1) × F₀ integrated over the pixel.  At baseline epochs
              (A ≈ 1) the stamp is near-zero / noise; at peak (A ≈ 20) a bright
              point appears at the source centre.  Colormap is magma on a fixed
              0→99th-percentile scale, matching animate_dia_epochs.py.

              The 256×256 ideal_cube_e (pre-detector, NOT DIA-subtracted) is
              also written as FITS for inspection but is NOT shown in the GIF.

Fixed parameters (do not change without bumping GOLDEN_VERSION):
  GOLDEN_VERSION = "phase3-v1"
  SEED           = 20260419
  u0             = 0.05
  tE             = 30.0 days
  t0             = 60000.0 MJD
  rho            = 0.0  (point source)
  source_mag     = 20.0 AB (F146)
  N_EPOCHS       = 5  (np.linspace(t0 − 2·tE, t0 + 2·tE, 5))

Requirements
------------
    pip install -e '.[phase2]'          # galsim, pandas, astropy

    STPSF environment variable (one of):
        STPSF_PATH    → path to the STPSF data directory
        WEBBPSF_PATH  → legacy alias (also accepted)

    If STPSF_PATH is unset, STPSFProvider will raise at PSF fetch time.
    Set it before running:
        export STPSF_PATH=/path/to/stpsf-data

Outputs (written to outputs/)
------------------------------
    rendered/epoch_{i}_ideal_e_256x256.fits   — pre-detector ideal electron map
    rendered/epoch_{i}_dia_64x64.fits         — post-chain DIA difference stamp
    phase3_golden_event_rendered.gif          — animated figure (3 panels)

Usage
-----
    python scripts/phase3_golden_event_rendered.py
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import animation
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger("golden_rendered")

# ---------------------------------------------------------------------------
# Fixed golden-event parameters (mirror phase3_golden_event.py)
# ---------------------------------------------------------------------------
GOLDEN_VERSION = "phase3-v1"
SEED = 20260419
U0 = 0.05
TE_DAYS = 30.0
T0_MJD = 60000.0
RHO = 0.0
SOURCE_MAG_AB = 20.0
BAND = "F146"
N_EPOCHS = 5
GIF_FPS = 2  # 2 fps → 0.5 s per epoch; full loop ≈ 2.5 s

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _REPO_ROOT / "smig" / "config" / "simulation.yaml"


# ---------------------------------------------------------------------------
# PSPL helper (identical to analytic golden script for continuity)
# ---------------------------------------------------------------------------

def _pspl_magnification(t: np.ndarray, t0: float, tE: float, u0: float) -> np.ndarray:
    u = np.sqrt(u0**2 + ((t - t0) / tE) ** 2)
    return (u**2 + 2.0) / (u * np.sqrt(u**2 + 4.0))


# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------

def _require_galsim() -> None:
    try:
        import galsim  # noqa: F401
    except ImportError:
        _log.error(
            "GalSim is not installed. Install Phase 2 extras:\n"
            "    pip install -e '.[phase2]'"
        )
        sys.exit(1)


def _check_stpsf_env() -> None:
    if not (os.environ.get("STPSF_PATH") or os.environ.get("WEBBPSF_PATH")):
        _log.warning(
            "Neither STPSF_PATH nor WEBBPSF_PATH is set.\n"
            "STPSFProvider will raise when it tries to load the PSF data.\n"
            "Set one before running:\n"
            "    export STPSF_PATH=/path/to/stpsf-data"
        )


def _require_astropy() -> None:
    try:
        from astropy.io import fits  # noqa: F401
    except ImportError:
        _log.error(
            "astropy is not installed (needed to write FITS).\n"
            "    pip install astropy"
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

def _make_output_dir() -> Path | None:
    out = _REPO_ROOT / "outputs" / "rendered"
    try:
        out.mkdir(parents=True, exist_ok=True)
        probe = out / ".probe"
        probe.touch()
        probe.unlink()
        return out
    except OSError:
        return None


# ---------------------------------------------------------------------------
# GIF builder
# ---------------------------------------------------------------------------

def _robust_vlims(cube: np.ndarray) -> tuple[float, float]:
    """Percentile colour limits robust to cosmic-ray spikes."""
    finite = cube[np.isfinite(cube)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(finite, 1.0))
    hi = float(np.percentile(finite, 99.0))
    return lo, (hi if hi > lo else lo + 1.0)


def _build_animation(
    t_curve: np.ndarray,
    A_curve: np.ndarray,
    t_epochs: np.ndarray,
    A_epochs: np.ndarray,
    F0_e_per_s: float,
    diff_stamps: np.ndarray,     # (N_EPOCHS, 64, 64) float64 — DIA subtracted
) -> animation.FuncAnimation:
    """Return a FuncAnimation: left=analytic curves (animated dot), right=DIA stamp."""
    n = diff_stamps.shape[0]
    t_norm = (t_epochs - T0_MJD) / TE_DAYS   # x-axis units for scatter dots

    # Fixed colour scale across all epochs — consistent with animate_dia_epochs.py
    vlo, vhi = _robust_vlims(diff_stamps)

    # -----------------------------------------------------------------------
    # Layout: 2-col gridspec.  Left col: 2 stacked rows (mag + lc).
    #         Right col: single stamp panel spanning both rows.
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(10, 4.8))
    fig.patch.set_facecolor("#1a1a2e")   # dark background matching magma aesthetic

    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[1.5, 1],
        hspace=0.45,
        wspace=0.35,
        left=0.08, right=0.97,
        top=0.88, bottom=0.12,
    )

    ax_mag = fig.add_subplot(gs[0, 0])
    ax_lc  = fig.add_subplot(gs[1, 0])
    ax_stamp = fig.add_subplot(gs[:, 1])   # spans both rows

    curve_x = (t_curve - T0_MJD) / TE_DAYS
    A_peak_expected = float((U0**2 + 2) / (U0 * (U0**2 + 4) ** 0.5))

    _text_kw = dict(color="#d0d0d0")
    _spine_color = "#555555"

    def _style(ax: plt.Axes) -> None:
        ax.set_facecolor("#111128")
        for spine in ax.spines.values():
            spine.set_edgecolor(_spine_color)
        ax.tick_params(colors="#aaaaaa", labelsize=7)
        ax.xaxis.label.set_color("#aaaaaa")
        ax.yaxis.label.set_color("#aaaaaa")
        ax.title.set_color("#dddddd")

    # --- magnification curve ---
    ax_mag.plot(curve_x, A_curve, color="#5b9bd5", lw=1.4)
    ax_mag.axhline(A_peak_expected, color="#888888", ls="--", lw=0.7,
                   label=f"$A_\\mathrm{{pk}}$={A_peak_expected:.1f}")
    ax_mag.scatter(t_norm, A_epochs, color="#aaaaaa", s=20, zorder=4, alpha=0.6)
    ax_mag.set_ylabel("$A(t)$", fontsize=8)
    ax_mag.set_title("PSPL Magnification", fontsize=8)
    ax_mag.legend(fontsize=7, facecolor="#1a1a2e", edgecolor="#555555",
                  labelcolor="#cccccc")
    _style(ax_mag)

    # --- light curve ---
    t_lc_x = curve_x
    ax_lc.plot(t_lc_x, A_curve * F0_e_per_s, color="#e8923a", lw=1.4)
    ax_lc.axhline(F0_e_per_s, color="#888888", ls="--", lw=0.7,
                  label=f"$F_0$={F0_e_per_s:.1e} e⁻/s")
    ax_lc.scatter(t_norm, A_epochs * F0_e_per_s, color="#aaaaaa",
                  s=20, zorder=4, alpha=0.6)
    ax_lc.set_xlabel("$(t-t_0)\\ /\\ t_E$", fontsize=8)
    ax_lc.set_ylabel("Flux (e⁻/s)", fontsize=8)
    ax_lc.set_title("Source Light Curve", fontsize=8)
    ax_lc.legend(fontsize=7, facecolor="#1a1a2e", edgecolor="#555555",
                 labelcolor="#cccccc")
    _style(ax_lc)

    # --- DIA stamp (animated) ---
    im = ax_stamp.imshow(
        diff_stamps[0], origin="lower", cmap="magma",
        vmin=vlo, vmax=vhi, interpolation="nearest",
    )
    ax_stamp.set_title("DIA difference stamp\n(64×64 px, post-detector)", fontsize=8)
    ax_stamp.set_xticks([])
    ax_stamp.set_yticks([])
    _style(ax_stamp)

    cbar = fig.colorbar(im, ax=ax_stamp, fraction=0.046, pad=0.04)
    cbar.set_label("≈(A−1)·F₀  [e⁻]", color="#aaaaaa", fontsize=7)
    cbar.ax.tick_params(colors="#aaaaaa", labelsize=6)

    # Animated dot on each curve — white ring with filled centre
    dot_mag, = ax_mag.plot([], [], "o", color="white", ms=7, zorder=10,
                           markeredgecolor="#333333", markeredgewidth=0.5)
    dot_lc,  = ax_lc.plot([], [], "o", color="white", ms=7, zorder=10,
                           markeredgecolor="#333333", markeredgewidth=0.5)

    # Epoch caption inside the stamp panel
    caption = ax_stamp.text(
        0.04, 0.97, "",
        transform=ax_stamp.transAxes,
        color="white", fontsize=7.5, va="top",
        bbox=dict(facecolor="black", alpha=0.55, pad=2.0),
    )

    # Global title (static)
    fig.suptitle(
        f"SMIG v2 — Golden Event ({GOLDEN_VERSION})   u₀={U0}  tE={TE_DAYS} d  "
        f"mag={SOURCE_MAG_AB} AB (F146)   seed={SEED}",
        fontsize=8.5, color="#eeeeee", y=0.97,
    )

    def update(frame: int):
        # Move dot to current epoch on both curve panels.
        x = float(t_norm[frame])
        dot_mag.set_data([x], [float(A_epochs[frame])])
        dot_lc.set_data([x], [float(A_epochs[frame] * F0_e_per_s)])

        # Update stamp image.
        im.set_data(diff_stamps[frame])

        # Update caption.
        caption.set_text(
            f"epoch {frame + 1}/{n}\n"
            f"t={t_norm[frame]:+.1f} tE\n"
            f"A(t)={A_epochs[frame]:.2f}"
        )
        return [dot_mag, dot_lc, im, caption]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=n,
        interval=int(round(1000 / max(1, GIF_FPS))),
        blit=False,
    )
    return anim


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _require_galsim()
    _check_stpsf_env()
    _require_astropy()

    from astropy.io import fits
    from smig.catalogs.photometry import mag_ab_to_electrons, _F146_AB_ZERO_POINT
    from smig.config.utils import load_simulation_config
    from smig.rendering.pipeline import SceneSimulator

    if not _CONFIG_PATH.exists():
        _log.error("Config not found: %s", _CONFIG_PATH)
        sys.exit(1)

    config = load_simulation_config(_CONFIG_PATH)
    exposure_s: float = config.detector.readout.exposure_time_s

    # -----------------------------------------------------------------------
    # Analytic PSPL (mirrors analytic golden script for continuity)
    # -----------------------------------------------------------------------
    t_curve = np.linspace(T0_MJD - 3 * TE_DAYS, T0_MJD + 3 * TE_DAYS, 500)
    A_curve = _pspl_magnification(t_curve, T0_MJD, TE_DAYS, U0)

    t_epochs = np.linspace(T0_MJD - 2 * TE_DAYS, T0_MJD + 2 * TE_DAYS, N_EPOCHS)
    A_epochs = _pspl_magnification(t_epochs, T0_MJD, TE_DAYS, U0)

    baseline_flux_e: float = mag_ab_to_electrons(SOURCE_MAG_AB, BAND, exposure_s)
    F0_e_per_s: float = baseline_flux_e / exposure_s
    epoch_flux_e = A_epochs * baseline_flux_e

    # -----------------------------------------------------------------------
    # Peak magnification diagnostics (same as analytic script)
    # -----------------------------------------------------------------------
    A_peak_measured = float(np.max(A_curve))
    A_peak_expected = float((U0**2 + 2) / (U0 * (U0**2 + 4) ** 0.5))

    print(f"Golden rendered event: {GOLDEN_VERSION}")
    print(f"  u0               : {U0}")
    print(f"  tE               : {TE_DAYS} days")
    print(f"  exposure_s       : {exposure_s:.1f} s (from config)")
    print(f"  baseline_flux_e  : {baseline_flux_e:.4e} e⁻")
    print(f"  A_peak_measured  : {A_peak_measured:.6f}")
    print(f"  A_peak_expected  : {A_peak_expected:.6f}")
    print(f"  relative_error   : {abs(A_peak_measured - A_peak_expected) / A_peak_expected:.2e}")

    # -----------------------------------------------------------------------
    # Build source-params sequence
    # -----------------------------------------------------------------------
    source_params = [
        {
            "flux_e": float(epoch_flux_e[i]),
            "centroid_offset_pix": (0.0, 0.0),
            "rho_star_arcsec": float(RHO),
            "limb_darkening_coeffs": None,
        }
        for i in range(N_EPOCHS)
    ]

    # Sky background: convert surface brightness → e⁻/s/pixel
    sky_sb = config.detector.environment.sky_background_mag_per_arcsec2
    pixel_area = config.crowded_field.pixel_scale_arcsec ** 2
    sky_e_per_s_per_pix = 10.0 ** ((_F146_AB_ZERO_POINT - sky_sb) / 2.5) * pixel_area
    backgrounds_e_per_s = [sky_e_per_s_per_pix] * N_EPOCHS

    # -----------------------------------------------------------------------
    # Run full pipeline
    # -----------------------------------------------------------------------
    _log.info("Initialising SceneSimulator (STPSF cache warm-up may take a moment)...")
    simulator = SceneSimulator(config, master_seed=SEED)

    _log.info("Simulating %d science epochs...", N_EPOCHS)
    output = simulator.simulate_event(
        event_id=f"golden-rendered-{GOLDEN_VERSION}",
        source_params_sequence=source_params,
        timestamps_mjd=t_epochs,
        backgrounds_e_per_s=backgrounds_e_per_s,
        baseline_source_flux_e=baseline_flux_e,
        retain_ideal_cube=True,
    )

    # 256×256 pre-detector ideal electrons (NOT DIA-subtracted, NOT shown in GIF)
    ideal_cube = output.ideal_cube_e
    # 64×64 post-detector DIA difference stamps (IS DIA-subtracted, shown in GIF)
    diff_stamps = output.difference_stamps

    # -----------------------------------------------------------------------
    # Write FITS
    # -----------------------------------------------------------------------
    out_dir = _make_output_dir()
    if out_dir is None:
        _log.error("Cannot write to outputs/rendered/ — check permissions.")
        sys.exit(1)

    for i in range(N_EPOCHS):
        hdr = fits.Header()
        hdr["BUNIT"] = ("electrons", "Pre-detector ideal electrons, NOT DIA-subtracted")
        hdr["EPOCH"] = (i, "Science epoch index (0-based)")
        hdr["T_MJD"] = (float(t_epochs[i]), "Epoch MJD")
        hdr["AMAG"] = (float(A_epochs[i]), "PSPL magnification A(t)")
        hdr["FLUX_E"] = (float(epoch_flux_e[i]), "Lensed source flux [e-]")
        hdr["F0_E"] = (baseline_flux_e, "Unlensed baseline flux [e-]")
        hdr["GLDNVER"] = GOLDEN_VERSION
        fits.writeto(
            str(out_dir / f"epoch_{i}_ideal_e_256x256.fits"),
            ideal_cube[i].astype(np.float64), hdr, overwrite=True,
        )

        hdr_dia = fits.Header()
        hdr_dia["BUNIT"] = ("electrons", "DIA difference ~(A-1)*F0, post-detector")
        hdr_dia["EPOCH"] = (i, "Science epoch index (0-based)")
        hdr_dia["T_MJD"] = (float(t_epochs[i]), "Epoch MJD")
        hdr_dia["AMAG"] = (float(A_epochs[i]), "PSPL magnification A(t)")
        hdr_dia["GLDNVER"] = GOLDEN_VERSION
        fits.writeto(
            str(out_dir / f"epoch_{i}_dia_64x64.fits"),
            diff_stamps[i].astype(np.float64), hdr_dia, overwrite=True,
        )

    _log.info("FITS files written to %s", out_dir)

    # -----------------------------------------------------------------------
    # Build and save animated GIF
    # -----------------------------------------------------------------------
    _log.info("Building animated GIF (%d fps)...", GIF_FPS)
    anim = _build_animation(t_curve, A_curve, t_epochs, A_epochs, F0_e_per_s, diff_stamps)

    gif_path = _REPO_ROOT / "outputs" / "phase3_golden_event_rendered.gif"
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(
        str(gif_path),
        writer=animation.PillowWriter(fps=GIF_FPS),
        dpi=130,
    )
    plt.close("all")
    _log.info("GIF written to %s", gif_path)


if __name__ == "__main__":
    main()
