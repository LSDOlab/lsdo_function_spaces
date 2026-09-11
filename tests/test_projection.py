"""Unit tests for spatial point projection algorithms onto B-splines."""

import pytest
import numpy as np
import csdl_alpha as csdl
import lsdo_function_spaces as lfs


def test_surface_point_projection_exact():
    """Verify that points evaluated on a B-spline surface project back to identical parametric coordinates."""
    space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(2, 2), coefficients_shape=(4, 4))
    np.random.seed(42)
    ctrl = np.random.uniform(-5.0, 5.0, (4, 4, 3))
    fn = lfs.Function(space=space, coefficients=ctrl, name="test_surface")

    # Sample true coordinates
    true_u = np.array([
        [0.2, 0.3],
        [0.5, 0.5],
        [0.7, 0.8],
    ])
    eval_pts = fn.evaluate(true_u).value

    # Project evaluated points onto the surface
    proj_u = space._project(eval_pts, ctrl, grid_search_density=20)
    np.testing.assert_allclose(proj_u, true_u, atol=1e-3)


def test_surface_projection_boundary_clamping():
    """Verify points far outside the surface project to boundary coordinates in [0, 1]^2."""
    space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(2, 2), coefficients_shape=(3, 3))
    # Flat unit square on z=0 plane
    x = np.linspace(0.0, 10.0, 3)
    y = np.linspace(0.0, 10.0, 3)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    ctrl = np.stack((xx, yy, np.zeros_like(xx)), axis=-1)

    # Point far off the positive x and y boundary
    outside_pt = np.array([[25.0, 25.0, 5.0], [-10.0, -10.0, 0.0]])
    proj_u = space._project(outside_pt, ctrl, grid_search_density=20)

    # Coords must be clamped inside [0, 1]
    assert np.all(proj_u >= 0.0)
    assert np.all(proj_u <= 1.0)
    # The first point should project near (1, 1) and second near (0, 0)
    np.testing.assert_allclose(proj_u[0], [1.0, 1.0], atol=1e-2)
    np.testing.assert_allclose(proj_u[1], [0.0, 0.0], atol=1e-2)


def test_projection_invalid_input_types():
    """Verify input validation for projection methods."""
    space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(2, 2), coefficients_shape=(3, 3))
    ctrl = np.zeros((3, 3, 3))

    with pytest.raises(TypeError, match="must be a numpy array or a CSDL variable"):
        space._project("not_an_array", ctrl)


def test_low_level_projection_functions():
    """Verify low-level projection functions (residuals, Jacobians, and nearest neighbor search)."""
    import jax.numpy as jnp
    from lsdo_function_spaces.core.spaces.non_cython_bsplines.b_spline_patch_projection import (
        _nearest_idx_for_point,
        brute_force_nn,
        compute_projection_residual,
        compute_projection_jacobian,
        compute_point_to_bspline_projection,
        res_fun_single,
        newton_solve_single,
    )

    degrees = (1, 1)
    knots = (
        (0.0, 0.0, 1.0, 1.0),
        (0.0, 0.0, 1.0, 1.0),
    )
    # Unit square flat patch at z = 0
    coeffs = np.array([
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
    ])

    # 1. Nearest neighbor functions
    bsp_pts = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    q_pt = jnp.array([0.9, 0.9, 0.0])
    idx = _nearest_idx_for_point(bsp_pts, q_pt)
    assert int(idx) == 1

    q_batch = jnp.array([[0.1, 0.1, 0.0], [0.9, 0.9, 0.0]])
    batch_indices = brute_force_nn(bsp_pts, q_batch)
    np.testing.assert_array_equal(batch_indices, [0, 1])

    # 2. Residual and Jacobian computation
    pt = jnp.array([0.5, 0.5, 1.0])
    para_u = jnp.array([[0.5, 0.5]])
    res, bsp_eval, surface_jac, _ = compute_projection_residual(
        point_in_space=pt,
        para_coords=para_u,
        degrees=degrees,
        coefficients=jnp.array(coeffs),
        knot_vectors=knots,
    )
    assert res.shape == (2,)

    jac = compute_projection_jacobian(
        point_in_space=pt,
        para_coords=para_u,
        degrees=degrees,
        coefficients=jnp.array(coeffs),
        knot_vectors=knots,
        bsp_eval=bsp_eval,
        surface_jacobian=surface_jac,
    )
    assert jac.shape == (2, 2)

    # 3. compute_point_to_bspline_projection
    final_coords, final_res, converged, final_i, J, mask, _ = compute_point_to_bspline_projection(
        point=pt,
        degrees=degrees,
        coefficients=jnp.array(coeffs),
        para_coords=jnp.array([0.4, 0.4]),
        knots=knots,
    )
    assert bool(converged)
    np.testing.assert_allclose(final_coords, [0.5, 0.5], atol=1e-4)

    # 4. res_fun_single and newton_solve_single
    single_res = res_fun_single(jnp.array([0.5, 0.5]), pt, jnp.array(coeffs), degrees=degrees, knots=knots)
    assert single_res.shape == (2,)

    solved = newton_solve_single(pt, jnp.array(coeffs), jnp.array([0.4, 0.4]), degrees=degrees, knots=knots)
    np.testing.assert_allclose(solved, [0.5, 0.5], atol=1e-4)


def test_function_set_project():
    """Verify projection of points onto a FunctionSet."""
    s1 = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))
    ctrl1 = np.array([
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
    ])
    f1 = lfs.Function(space=s1, coefficients=ctrl1, name="patch_0")
    fset = lfs.FunctionSet([f1])

    query_pts = np.array([[0.3, 0.6, 2.0]])
    res = fset.project(query_pts, num_workers=1, do_pickles=False)
    assert len(res) == 1
    patch_idx, uv = res[0]
    assert patch_idx == 0
    np.testing.assert_allclose(uv, [0.3, 0.6], atol=1e-3)


def test_projection_operation_csdl():
    """Verify ProjectionOperation CSDL custom operation and its VJP."""
    from lsdo_function_spaces.core.spaces.non_cython_bsplines.b_spline_patch_projection import ProjectionOperation

    degrees = (1, 1)
    knots = (
        (0.0, 0.0, 1.0, 1.0),
        (0.0, 0.0, 1.0, 1.0),
    )
    coeffs = np.array([
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
    ])
    query_pts = np.array([[0.3, 0.7, 2.0]])

    pts_var = csdl.Variable(value=query_pts)
    coeffs_var = csdl.Variable(value=coeffs)

    proj_op = ProjectionOperation(
        num_parametric_dimensions=2,
        degrees=degrees,
        knots=knots,
        num_grid_search_points=5,
    )

    uvs = proj_op.evaluate(points=pts_var, control_points=coeffs_var)
    assert isinstance(uvs, csdl.Variable)
    np.testing.assert_allclose(uvs.value[0], [0.3, 0.7], atol=1e-3)

    # Test derivative through ProjectionOperationVJP
    d_uv = csdl.derivative(csdl.sum(uvs), coeffs_var)
    assert d_uv is not None


