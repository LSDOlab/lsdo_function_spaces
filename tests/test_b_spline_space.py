"""Unit tests for BSplineSpace representation, basis evaluation, and properties."""

import pytest
import numpy as np
import csdl_alpha as csdl
import lsdo_function_spaces as lfs


def test_b_spline_space_initialization():
    """Test standard initialization of 1D, 2D, and 3D B-spline spaces."""
    # 1D Curve
    s1 = lfs.BSplineSpace(num_parametric_dimensions=1, degree=3, coefficients_shape=(10,))
    assert s1.degree == (3,)
    assert s1.coefficients_shape == (10,)
    assert len(s1.knots) == 1
    assert len(s1.knots[0]) == 10 + 3 + 1

    # 2D Surface
    s2 = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(2, 3), coefficients_shape=(8, 12))
    assert s2.degree == (2, 3)
    assert s2.coefficients_shape == (8, 12)
    assert len(s2.knots) == 2
    assert len(s2.knots[0]) == 8 + 2 + 1
    assert len(s2.knots[1]) == 12 + 3 + 1

    # 3D Volume
    s3 = lfs.BSplineSpace(num_parametric_dimensions=3, degree=2, coefficients_shape=(5, 6, 7))
    assert s3.degree == (2, 2, 2)
    assert s3.coefficients_shape == (5, 6, 7)
    assert len(s3.knots) == 3


def test_b_spline_space_validation_errors():
    """Test validation errors for invalid degree or coefficient shape."""
    # Negative degree
    with pytest.raises(ValueError, match="must be non-negative"):
        lfs.BSplineSpace(num_parametric_dimensions=1, degree=-1, coefficients_shape=(5,))

    # Degree >= number of coefficients
    with pytest.raises(ValueError, match="must be less than the number of coefficients"):
        lfs.BSplineSpace(num_parametric_dimensions=1, degree=5, coefficients_shape=(5,))

    # Incompatible input type to compute_basis_matrix
    space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=2, coefficients_shape=(6,))
    with pytest.raises(TypeError, match="must be a numpy array or a CSDL variable"):
        space.compute_basis_matrix("invalid_coords")


def test_b_spline_partition_of_unity():
    """Verify partition of unity: sum of basis functions equals 1 across the domain."""
    space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(3, 3), coefficients_shape=(10, 10))
    u = np.linspace(0.0, 1.0, 25)
    v = np.linspace(0.0, 1.0, 25)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    para_coords = np.stack((uu.ravel(), vv.ravel()), axis=-1)

    basis_mat = space.compute_basis_matrix(para_coords)
    # Basis matrix has shape (num_points, num_coefficients)
    row_sums = np.array(basis_mat.sum(axis=1)).ravel()
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)


def test_b_spline_1d_linear_precision():
    """Verify linear precision: linear function f(u) = 3*u + 2 is reproduced exactly."""
    space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=2, coefficients_shape=(5,))
    # For open uniform knots on [0, 1], Greville abscissae for degree p:
    # xi_i = sum_{j=1}^p knot[i+j] / p
    knots = space.knots[0]
    p = 2
    greville = np.array([np.sum(knots[i + 1 : i + 1 + p]) / p for i in range(5)])
    control_points = 3.0 * greville + 2.0

    eval_u = np.linspace(0.0, 1.0, 30).reshape(-1, 1)
    basis_mat = space.compute_basis_matrix(eval_u)
    eval_vals = np.array(basis_mat @ control_points).ravel()
    expected = 3.0 * eval_u.ravel() + 2.0
    np.testing.assert_allclose(eval_vals, expected, atol=1e-12)


def test_b_spline_derivatives():
    """Verify first-derivative evaluation against central finite differences."""
    space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=3, coefficients_shape=(8,))
    np.random.seed(42)
    coeffs = np.random.randn(8)

    u_test = 0.45
    eps = 1e-6
    u_eval = np.array([[u_test - eps], [u_test + eps]])

    basis_eval = space.compute_basis_matrix(u_eval)
    f_vals = np.array(basis_eval @ coeffs).ravel()
    fd_deriv = (f_vals[1] - f_vals[0]) / (2 * eps)

    basis_deriv = space.compute_basis_matrix(np.array([[u_test]]), parametric_derivative_orders=(1,))
    exact_deriv = float(np.array(basis_deriv @ coeffs).ravel()[0])

    np.testing.assert_allclose(exact_deriv, fd_deriv, rtol=1e-5)


def test_b_spline_evaluate_csdl_symbolic():
    """Verify evaluation within a CSDL recorder generates symbolic graph node."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(2, 2), coefficients_shape=(4, 4))
    ctrl = np.ones((4, 4, 3))
    coeffs_var = csdl.Variable(value=ctrl)

    para = np.array([[0.25, 0.75], [0.5, 0.5]])
    res = space._evaluate(coeffs_var, para)

    assert isinstance(res, csdl.Variable)
    np.testing.assert_allclose(res.value, np.ones((2, 3)), atol=1e-12)


def test_b_spline_stitch():
    """Verify stitching indices calculation along adjacent faces."""
    s1 = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(2, 2), coefficients_shape=(4, 5))
    s2 = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(2, 2), coefficients_shape=(4, 5))
    c1 = np.zeros((4, 5, 3))
    c2 = np.zeros((4, 5, 3))

    inds1, inds2 = s1.stitch(self_face=2, self_coeffs=c1, other=s2, other_face=4, other_coeffs=c2)
    assert len(inds1) == 5
    assert len(inds2) == 5


def test_b_spline_compute_distance_bounds():
    """Verify computation of distance bounds to bounding boxes."""
    space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))
    ctrl = np.array([
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
    ])
    fn = lfs.Function(space=space, coefficients=ctrl)

    # Point outside box: (2, 2, 0)
    pt = np.array([2.0, 2.0, 0.0])
    d = space._compute_distance_bounds(pt, fn, direction=None)
    # Distance from (1, 1, 0) to (2, 2, 0) is sqrt(1^2 + 1^2) = sqrt(2)
    np.testing.assert_allclose(d, np.sqrt(2.0), atol=1e-10)

    # Point with direction along x-axis
    d_dir = space._compute_distance_bounds(pt, fn, direction=np.array([1.0, 0.0, 0.0]))
    assert d_dir >= 0.0


def test_b_spline_custom_ops_csdl():
    """Verify CSDL custom ops: BasisMatrixCustomOp and BSplineEvalCustomOp."""
    space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(2, 2), coefficients_shape=(3, 3))
    ctrl = np.ones((3, 3, 2))
    coeffs_var = csdl.Variable(value=ctrl)

    # 1. BasisMatrixCustomOp via csdl.Variable parametric_coordinates
    para_var = csdl.Variable(value=np.array([[0.5, 0.5]]))
    bm_custom = space.compute_basis_matrix(para_var)
    assert isinstance(bm_custom, csdl.Variable)
    assert bm_custom.shape == (1, 9)

    # 2. BSplineEvalCustomOp via csdl.Variable parametric_coordinates
    eval_custom = space._evaluate(coeffs_var, para_var)
    assert isinstance(eval_custom, csdl.Variable)
    np.testing.assert_allclose(eval_custom.value, [[1.0, 1.0]], atol=1e-10)

    # 3. Test VJP execution for both custom ops
    d_eval = csdl.derivative(csdl.sum(eval_custom), [coeffs_var, para_var])
    assert d_eval is not None
    d_bm = csdl.derivative(csdl.sum(bm_custom), para_var)
    assert d_bm is not None


def test_b_spline_space_project_internal():
    """Test BSplineSpace._project method directly."""
    space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))
    ctrl = np.array([
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
    ])
    pts = np.array([[0.3, 0.4, 0.0], [0.8, 0.2, 0.0]])
    para = space._project(pts, ctrl, grid_search_density=10)
    np.testing.assert_allclose(para, [[0.3, 0.4], [0.8, 0.2]], atol=1e-3)


def test_b_spline_second_derivatives_and_numpy_eval():
    """Test 2nd derivatives, single der_order expansion, and direct numpy evaluation."""
    from lsdo_function_spaces.core.spaces.non_cython_bsplines.compute_basis_matrix_numpy import (
        compute_basis_matrix_numpy,
        evaluate_b_spline_numpy,
    )

    degrees = (3, 3)
    knots = (
        np.array([0, 0, 0, 0, 0.5, 1, 1, 1, 1], dtype=float),
        np.array([0, 0, 0, 0, 0.5, 1, 1, 1, 1], dtype=float),
    )
    coords = np.array([[0.3, 0.7]])
    # 2nd order derivative along dim 0
    B_d2 = compute_basis_matrix_numpy(coords, degrees, knots, der_orders=(2, 0))
    assert B_d2.shape == (1, 25)

    # single element list expansion
    B_single = compute_basis_matrix_numpy(coords, degrees, knots, der_orders=[0])
    assert B_single.shape == (1, 25)

    # Invalid der_orders length raises ValueError
    with pytest.raises(ValueError, match="der_orders must be"):
        compute_basis_matrix_numpy(coords, degrees, knots, der_orders=(1, 1, 1))

    # evaluate_b_spline_numpy
    coeffs = np.ones((5, 5, 3))
    val = evaluate_b_spline_numpy(coords, degrees, knots, coeffs)
    assert val.shape == (1, 3)
    np.testing.assert_allclose(val, [[1.0, 1.0, 1.0]], atol=1e-10)


