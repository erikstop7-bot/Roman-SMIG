"""
smig.optics — Phase 2 optical rendering sub-package.

Provides PSF generation and optical diffraction modelling for the Roman WFI
crowded-field scene builder (CrowdedFieldRenderer).

NOTE: Phase 2 imports (galsim, webbpsf, poppy, synphot) must be guarded with
try/except ImportError blocks so that `import smig.optics` succeeds under a
base-only install.  Do NOT add bare top-level Phase 2 imports to this file.
"""

# STPSFProvider requires galsim (Phase 2 extra).  Guard the import so that
# ``import smig.optics`` succeeds on a base-only install.  Accessing
# ``STPSFProvider`` on a base install returns None; attempting to instantiate
# it will raise ImportError from within __init__.
STPSFProvider = None
try:
    from smig.optics.psf import STPSFProvider  # type: ignore[assignment]
except ImportError:
    pass

__all__ = ["STPSFProvider"]
