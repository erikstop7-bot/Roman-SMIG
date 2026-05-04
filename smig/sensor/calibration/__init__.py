"""
smig/sensor/calibration/__init__.py
=====================================
Calibration data loaders for the Roman WFI H4RG-10 detector.

This package will contain:
  - HDF5 IPC kernel loader (field-dependent 9×9 kernel map per SCA)
  - Nonlinearity calibration curve loader (per-pixel polynomial coefficients)

Leaf module: must not import any sibling sensor module.
"""
