"""Tests for MicrolensingEvent and SourceProperties frozen contract.

Checks:
- Both dataclasses are frozen (mutations raise FrozenInstanceError).
- .magnification() signature matches the frozen interface exactly.
- event.py module docstring contains the mandatory freeze notice.
"""
from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pytest

from smig.microlensing.event import EventClass, MicrolensingEvent, SourceProperties


def _minimal_event(**overrides) -> MicrolensingEvent:
    defaults = dict(
        event_id="test-event-001",
        t0_mjd=0.0,
        tE_days=20.0,
        u0=0.5,
        rho=1e-4,
        alpha_rad=0.0,
    )
    defaults.update(overrides)
    return MicrolensingEvent(**defaults)


def _minimal_source() -> SourceProperties:
    return SourceProperties(
        teff_K=5000.0,
        log_g=4.5,
        metallicity_feh=0.0,
        distance_kpc=8.0,
        mass_msun=1.0,
    )


class TestFrozenness:
    def test_microlensing_event_is_frozen(self):
        ev = _minimal_event()
        with pytest.raises(dataclasses.FrozenInstanceError):
            ev.u0 = 0.1  # type: ignore[misc]

    def test_microlensing_event_field_frozen(self):
        ev = _minimal_event()
        with pytest.raises(dataclasses.FrozenInstanceError):
            ev.event_class = EventClass.FSPL_STAR  # type: ignore[misc]

    def test_source_properties_is_frozen(self):
        sp = _minimal_source()
        with pytest.raises(dataclasses.FrozenInstanceError):
            sp.teff_K = 6000.0  # type: ignore[misc]


class TestMagnificationSignature:
    def test_signature_matches_frozen_interface(self):
        sig = inspect.signature(MicrolensingEvent.magnification)
        params = list(sig.parameters.keys())
        assert params == ["self", "t_mjd", "band", "source_props"]

    def test_return_type_annotation(self):
        import typing
        hints = typing.get_type_hints(MicrolensingEvent.magnification)
        assert hints.get("return") is np.ndarray

    def test_magnification_dispatches_pspl(self):
        ev = _minimal_event(q=0.0, rho=1e-5, event_class=EventClass.PSPL)
        t = np.array([0.0])
        A = ev.magnification(t, "H", _minimal_source())
        assert A.shape == (1,)
        assert A[0] > 1.0

    def test_magnification_returns_ndarray(self):
        ev = _minimal_event(q=0.0, rho=1e-5, event_class=EventClass.PSPL)
        t = np.linspace(-10, 10, 5)
        A = ev.magnification(t, "H", _minimal_source())
        assert isinstance(A, np.ndarray)
        assert A.shape == t.shape


class TestFreezeNotice:
    def test_freeze_notice_in_module_docstring(self):
        import smig.microlensing.event as ev_module
        doc = ev_module.__doc__ or ""
        assert "INTERFACE FROZEN" in doc, (
            "event.py module docstring must contain the literal freeze notice"
        )
