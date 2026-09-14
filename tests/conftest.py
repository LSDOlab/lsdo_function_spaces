"""Pytest session fixtures for lsdo_function_spaces tests."""

import os
# Ensure JAX uses CPU when GPU is not present or in headless CI
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest
import pyvista as pv
import numpy as np
import csdl_alpha as csdl

# NumPy 2.x compatibility patch for CSDL_alpha SetVarIndex
try:
    from csdl_alpha.src.operations.set_get.setindex import SetVarIndex

    _orig_compute_inline = SetVarIndex.compute_inline

    def _safe_compute_inline(self, x, y, *slice_args):
        x_updated = x.copy()
        if getattr(y, "size", None) == 1:
            x_updated[self.slice.evaluate(*slice_args)] = y.item() if isinstance(y, np.ndarray) else y
        else:
            x_updated[self.slice.evaluate(*slice_args)] = y
        return x_updated

    SetVarIndex.compute_inline = _safe_compute_inline
except Exception:
    pass


@pytest.fixture(autouse=True, scope="session")
def setup_headless_pyvista():
    """Ensure PyVista runs off-screen during tests."""
    pv.OFF_SCREEN = True


@pytest.fixture(autouse=True)
def active_csdl_recorder():
    """Ensure an inline CSDL recorder is active for every test."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    yield recorder
    try:
        recorder.stop()
    except Exception:
        pass
