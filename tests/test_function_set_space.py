"""Unit tests for FunctionSetSpace class."""

import pytest
import numpy as np
import scipy.sparse as sps
import csdl_alpha as csdl
import lsdo_function_spaces as lfs


def test_function_set_space_init_and_grid():
    """Test FunctionSetSpace initialization, grid generation, and initialize_function."""
    s1 = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(2, 2), coefficients_shape=(3, 3))
    s2 = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(2, 2), coefficients_shape=(4, 4))

    fset_space = lfs.FunctionSetSpace(
        num_parametric_dimensions={0: 2, 1: 2},
        spaces=[s1, s2],
    )
    assert len(fset_space.spaces) == 2

    # Parametric grid
    grid = fset_space.generate_parametric_grid(grid_resolution=(5, 5))
    assert len(grid) == 25 + 25
    assert grid[0][0] == 0
    assert grid[25][0] == 1

    # initialize_function
    coeff_var, fset = fset_space.initialize_function(num_physical_dimensions=3, value=1.5)
    assert isinstance(coeff_var, csdl.Variable)
    assert coeff_var.shape == (3*3 + 4*4, 3)
    assert len(fset.functions) == 2
    np.testing.assert_allclose(coeff_var.value, 1.5)


def test_function_set_space_compute_basis_matrix():
    """Test FunctionSetSpace compute_basis_matrix."""
    s1 = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(3,))
    s2 = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(3,))

    fset_space = lfs.FunctionSetSpace(
        num_parametric_dimensions={0: 1, 1: 1},
        spaces={0: s1, 1: s2},
    )

    coords = [(0, np.array([[0.5]])), (1, np.array([[0.2]]))]
    mat = fset_space.compute_basis_matrix(coords, parametric_derivative_orders=[None, None])
    assert mat.shape == (2, 3)


def test_function_set_space_fit():
    """Test FunctionSetSpace fit and fit_function_set."""
    s1 = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(2,))
    s2 = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(2,))

    fset_space = lfs.FunctionSetSpace(
        num_parametric_dimensions={0: 1, 1: 1},
        spaces={0: s1, 1: s2},
    )

    coords = [
        (0, np.array([0.0])),
        (0, np.array([1.0])),
        (1, np.array([0.0])),
        (1, np.array([1.0])),
    ]
    values = np.array([
        [0.0, 1.0],
        [2.0, 3.0],
        [10.0, 10.0],
        [20.0, 20.0],
    ])

    coeffs = fset_space.fit(values=values, parametric_coordinates=coords)
    assert 0 in coeffs
    assert 1 in coeffs
    val0 = coeffs[0].value if hasattr(coeffs[0], "value") else coeffs[0]
    val1 = coeffs[1].value if hasattr(coeffs[1], "value") else coeffs[1]
    np.testing.assert_allclose(val0, [[0.0, 1.0], [2.0, 3.0]], atol=1e-10)
    np.testing.assert_allclose(val1, [[10.0, 10.0], [20.0, 20.0]], atol=1e-10)

    # fit_monolithic
    coeffs_mono = fset_space.fit_monolithic(values=values, parametric_coordinates=coords)
    assert 0 in coeffs_mono

    # fit_function_set
    fset_fitted = fset_space.fit_function_set(values=values, parametric_coordinates=coords)
    assert isinstance(fset_fitted, lfs.FunctionSet)
    assert len(fset_fitted.functions) == 2
