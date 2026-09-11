"""Pytest session fixtures for lsdo_function_spaces tests."""

import os
# Ensure JAX uses CPU when GPU is not present or in headless CI
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest
import pyvista as pv
import csdl_alpha as csdl


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
