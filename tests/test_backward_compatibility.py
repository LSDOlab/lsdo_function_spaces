"""Unit tests verifying backward compatibility aliases and deprecation warnings."""

import os
import pytest
import warnings
import lsdo_function_spaces as lfs

SAMPLE_STP = os.path.join(
    os.path.dirname(__file__), "..", "examples", "import_files_for_examples", "rectangular_wing.stp"
)


def test_b_spline_space_new_deprecation_warning():
    """Verify that instantiating BSplineSpaceNew emits a DeprecationWarning."""
    with pytest.deprecated_call():
        space = lfs.BSplineSpaceNew(num_parametric_dimensions=2, degree=(2, 2), coefficients_shape=(4, 4))
    assert isinstance(space, lfs.BSplineSpace)


def test_import_file_patched_deprecation_warning():
    """Verify that calling import_file_patched emits a DeprecationWarning."""
    if not os.path.exists(SAMPLE_STP):
        pytest.skip("Sample STP file not found.")

    with pytest.deprecated_call():
        fset = lfs.import_file_patched(SAMPLE_STP, parallelize=False)
    assert isinstance(fset, lfs.FunctionSet)
