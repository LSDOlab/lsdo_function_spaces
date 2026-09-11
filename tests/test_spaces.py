"""Unit tests for non-spline function spaces: IDW, RBF, Polynomial, Triangulation, Constant, Conditional."""

import pytest
import numpy as np
import scipy.sparse as sps
import csdl_alpha as csdl
import lsdo_function_spaces as lfs


def test_idw_space():
    """Test Inverse Distance Weighting function space initialization, basis matrix, and evaluation."""
    idw = lfs.IDWFunctionSpace(num_parametric_dimensions=2, order=2.0, grid_size=(6, 6), conserve=False)
    assert idw.order == 2.0
    assert idw.conserve is False

    pts = np.array([[0.2, 0.3], [0.5, 0.5], [0.8, 0.9]])
    basis = idw.compute_basis_matrix(pts)
    assert basis.shape == (3, 36)
    # Weights should sum to 1 across basis points for each evaluation coordinate
    row_sums = np.array(basis.sum(axis=1)).ravel()
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)


def test_idw_with_neighbors():
    """Test IDW with nearest neighbors restriction."""
    idw = lfs.IDWFunctionSpace(
        num_parametric_dimensions=2, order=3.0, grid_size=(8, 8), n_neighbors=5, conserve=False
    )
    pts = np.array([[0.3, 0.4]])
    basis = idw.compute_basis_matrix(pts)
    assert isinstance(basis, sps.spmatrix)
    # At most n_neighbors non-zero entries per evaluation point
    assert basis.count_nonzero() <= 5


def test_rbf_kernels():
    """Test RBF function space across all supported radial basis kernels."""
    kernels = ["gaussian", "polyharmonic_spline", "inverse_quadratic", "inverse_multiquadric", "bump"]
    eval_pts = np.array([[0.25, 0.75], [0.5, 0.5]])

    for k in kernels:
        rbf = lfs.RBFFunctionSpace(
            num_parametric_dimensions=2,
            radial_function=k,
            grid_size=(5, 5),
            epsilon=1.5,
            k=2,
        )
        basis = rbf.compute_basis_matrix(eval_pts)
        assert basis.shape == (2, 25)
        assert np.all(np.isfinite(basis))


def test_rbf_invalid_kernel():
    """Test that an invalid radial kernel raises a ValueError."""
    rbf = lfs.RBFFunctionSpace(num_parametric_dimensions=2, radial_function="invalid_kernel")
    with pytest.raises(ValueError, match="is not supported"):
        rbf.compute_basis_matrix(np.array([[0.5, 0.5]]))


def test_polynomial_space():
    """Test polynomial function space basis matrix and size."""
    poly = lfs.PolynomialSpace(num_parametric_dimensions=2, order=(3, 2))
    # Number of basis terms = (3+1) * (2+1) = 12
    assert poly.coefficients_shape == (12,)

    coords = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    basis = poly.compute_basis_matrix(coords)
    assert basis.shape == (3, 12)

    # At (0, 0), only the constant term (u^0 * v^0) should be 1, rest 0
    np.testing.assert_allclose(basis[0, 0], 1.0)
    np.testing.assert_allclose(basis[0, 1:], 0.0)


def test_linear_triangulation_space():
    """Test LinearTriangulationSpace grid generation and barycentric basis evaluation."""
    tri = lfs.LinearTriangulationSpace(grid_size=(5, 5))
    assert tri.nodes.shape == (25, 2)
    assert len(tri.elements) > 0

    pts = np.array([[0.2, 0.2], [0.6, 0.8]])
    basis = tri.compute_basis_matrix(pts)
    assert basis.shape == (2, 25)

    # In simplex interpolation, weights inside domain sum to 1
    row_sums = np.array(basis.sum(axis=1)).ravel()
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)


def test_constant_space():
    """Test ConstantSpace basis matrix computation."""
    const = lfs.ConstantSpace(num_parametric_dimensions=2)
    assert const.coefficients_shape == (1,)

    pts = np.array([[0.1, 0.2], [0.5, 0.8], [0.9, 0.9]])
    basis = const.compute_basis_matrix(pts)
    np.testing.assert_allclose(basis, np.ones((3, 1)))

    with pytest.raises(NotImplementedError):
        const.compute_basis_matrix(pts, parametric_derivative_orders=(1, 0))


def test_conditional_space():
    """Test ConditionalSpace evaluation with a predicate."""
    cond = lfs.ConditionalSpace(num_parametric_dimensions=1, condition=lambda u: u > 0.5)
    pts = np.array([[0.2], [0.8]])
    basis = cond.compute_basis_matrix(pts)
    assert basis.shape == (2, 1)


def test_fit_and_evaluate_all_spaces():
    """Comprehensive test: fit data from analytical surface and verify reconstruction accuracy."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    from scipy.stats.qmc import LatinHypercube

    # Analytical test surface: z = sin(2*pi*u) + 0.5*cos(2*pi*v)
    num_points = 800
    coords = LatinHypercube(d=2, seed=42).random(num_points)
    height = (np.sin(2 * np.pi * coords[:, 0]) + 0.5 * np.cos(2 * np.pi * coords[:, 1])).reshape(-1, 1)
    data = np.hstack((coords * 10.0, height))

    test_configs = [
        {"space": lfs.BSplineSpace(num_parametric_dimensions=2, degree=(3, 3), coefficients_shape=(8, 8)), "tol": 0.05},
        {"space": lfs.IDWFunctionSpace(num_parametric_dimensions=2, order=3.0, grid_size=(10, 10), conserve=False), "tol": 0.05},
        {"space": lfs.RBFFunctionSpace(num_parametric_dimensions=2, radial_function="gaussian", grid_size=(8, 8), epsilon=2.0), "tol": 0.05},
        {"space": lfs.PolynomialSpace(num_parametric_dimensions=2, order=6), "tol": 0.05},
    ]

    for cfg in test_configs:
        space = cfg["space"]
        fn = space.fit_function(data, coords)
        eval_res = fn.evaluate(coords)
        eval_val = eval_res.value if hasattr(eval_res, "value") else eval_res
        err = np.linalg.norm(eval_val - data) / num_points
        assert err < cfg["tol"], f"Space {space.__class__.__name__} failed fitting with error {err}"


def test_idw_stitch_and_sparse_fit():
    """Verify IDW space face stitching and sparse nearest-neighbor fitting."""
    s1 = lfs.IDWFunctionSpace(num_parametric_dimensions=2, order=2.0, grid_size=4)
    s2 = lfs.IDWFunctionSpace(num_parametric_dimensions=2, order=2.0, grid_size=4)

    c1 = csdl.Variable(value=np.ones((16, 3)))
    c2 = csdl.Variable(value=np.ones((16, 3)) * 2.0)

    # Stitch face 2 of s1 (x=1) to face 4 of s2 (x=0)
    new_c1, new_c2 = s1.stitch(self_face=2, self_coeffs=c1, other=s2, other_face=4, other_coeffs=c2)
    assert new_c1 is not None
    assert new_c2 is not None

    # Sparse fit with n_neighbors
    sparse_idw = lfs.IDWFunctionSpace(num_parametric_dimensions=2, order=2.0, grid_size=4, conserve=False, n_neighbors=3)
    pts = np.random.rand(20, 2)
    data = np.random.rand(20, 1)
    fn = sparse_idw.fit_function(data, pts)
    eval_val = fn.evaluate(pts).value
    assert eval_val.size == 20


def test_tri_space_derivatives_and_distance_bounds():
    """Verify LinearTriangulationSpace derivatives and distance bounds."""
    tri = lfs.LinearTriangulationSpace(grid_size=(4, 4))
    
    # Compute shape function gradients with pdo=(1, 0) and pdo=(0, 1)
    elems = np.array([0, 1])
    grad_u = tri.compute_shape_function_gradients(elems, np.zeros((2, 2)), pdo=(1, 0))
    grad_v = tri.compute_shape_function_gradients(elems, np.zeros((2, 2)), pdo=(0, 1))
    assert grad_u.shape == (2, 3)
    assert grad_v.shape == (2, 3)

    # Distance bounds with and without direction
    c = np.zeros((16, 3))
    fn = lfs.Function(space=tri, coefficients=c)
    pt = np.array([2.0, 2.0, 0.0])
    d1 = tri._compute_distance_bounds(pt, fn, direction=None)
    assert d1 >= 0.0

    d2 = tri._compute_distance_bounds(pt, fn, direction=np.array([1.0, 0.0, 0.0]))
    assert d2 >= 0.0

