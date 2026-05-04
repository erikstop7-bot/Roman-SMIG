"""
smig.rendering — Phase 2 scene rendering sub-package.

Provides the CrowdedFieldRenderer and related utilities that assemble
synthetic sky scenes and pass plain np.ndarray photon maps to the detector
chain.

NOTE: Phase 2 imports (galsim, webbpsf, synphot) must be guarded with
try/except ImportError blocks so that `import smig.rendering` succeeds under
a base-only install.  Do NOT add bare top-level Phase 2 imports to this file.
"""
