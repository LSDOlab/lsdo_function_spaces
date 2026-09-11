"""Unit tests for Function class evaluation, copy, refit, projection, plotting, and operations."""

import pytest
import numpy as np
import csdl_alpha as csdl
import lsdo_function_spaces as lfs
from lsdo_function_spaces import operations as ops


def test_function_initialization_and_copy():
    """Test Function creation with numpy and CSDL variables, and verify copy."""
    space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(2, 2), coefficients_shape=(4, 4))
    ctrl = np.ones((4, 4, 3))
    fn = lfs.Function(space=space, coefficients=ctrl, name="test_fn")

    assert fn.name == "test_fn"
    assert fn.num_physical_dimensions == 3
    assert isinstance(fn.coefficients, csdl.Variable)

    fn_copy = fn.copy()
    assert fn_copy.name == fn.name
    np.testing.assert_allclose(fn_copy.coefficients.value, fn.coefficients.value)


def test_function_evaluation_non_csdl():
    """Test Function evaluation in non-CSDL mode."""
    space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(2,))
    # Line from (0, 0) to (1, 10)
    ctrl = np.array([[0.0, 0.0], [1.0, 10.0]])
    fn = lfs.Function(space=space, coefficients=ctrl)

    u = np.array([[0.0], [0.5], [1.0]])
    res = fn.evaluate(u, non_csdl=True)
    expected = np.array([[0.0, 0.0], [0.5, 5.0], [1.0, 10.0]])
    np.testing.assert_allclose(res, expected, atol=1e-12)


def test_function_get_matrix_vector():
    """Test Function get_matrix_vector method."""
    space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(2,))
    ctrl = np.array([[0.0, 0.0], [1.0, 10.0]])
    fn = lfs.Function(space=space, coefficients=ctrl)

    u = np.array([[0.0], [0.5], [1.0]])
    mv = fn.get_matrix_vector(u)
    assert mv is not None


def test_function_refit():
    """Test refitting a function into a new function space."""
    space1 = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(2, 2), coefficients_shape=(4, 4))
    ctrl1 = np.random.uniform(0.0, 5.0, (4, 4, 3))
    fn1 = lfs.Function(space=space1, coefficients=ctrl1)

    space2 = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(3, 3), coefficients_shape=(6, 6))
    fn2 = fn1.refit(new_function_space=space2)

    # Both should evaluate similarly across test points
    test_u = np.array([[0.25, 0.25], [0.5, 0.5], [0.75, 0.75]])
    v1 = fn1.evaluate(test_u).value
    v2 = fn2.evaluate(test_u).value
    np.testing.assert_allclose(v1, v2, atol=0.2)


def test_function_operations():
    """Test arithmetic operations on functions (add, sub, mult, div, pow, neg)."""
    space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(3,))
    c1 = np.array([[1.0], [2.0], [3.0]])
    c2 = np.array([[4.0], [5.0], [6.0]])

    f1 = lfs.Function(space=space, coefficients=c1)
    f2 = lfs.Function(space=space, coefficients=c2)

    u = np.array([[0.5]])

    # Addition
    f_sum = f1 + f2
    np.testing.assert_allclose(f_sum.evaluate(u).value, f1.evaluate(u).value + f2.evaluate(u).value)

    # Subtraction
    f_sub = f1 - f2
    np.testing.assert_allclose(f_sub.evaluate(u).value, f1.evaluate(u).value - f2.evaluate(u).value)

    # Multiplication
    f_mul = f1 * f2
    np.testing.assert_allclose(f_mul.evaluate(u).value, f1.evaluate(u).value * f2.evaluate(u).value)

    # Division
    f_div = f1 / f2
    np.testing.assert_allclose(f_div.evaluate(u).value, f1.evaluate(u).value / f2.evaluate(u).value)

    # Negation
    f_neg = -f1
    np.testing.assert_allclose(f_neg.evaluate(u).value, -f1.evaluate(u).value)


def test_function_project_and_refine():
    """Test Function.project() and refine_projection() methods."""
    # Create flat surface z = 0 on [0, 1] x [0, 1]
    space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))
    ctrl = np.zeros((2, 2, 3))
    ctrl[0, 0, :2] = [0.0, 0.0]
    ctrl[0, 1, :2] = [0.0, 1.0]
    ctrl[1, 0, :2] = [1.0, 0.0]
    ctrl[1, 1, :2] = [1.0, 1.0]
    surf = lfs.Function(space=space, coefficients=ctrl, name="flat_surface")

    # Project point above the surface with direction [0, 0, -1]
    query_pts = np.array([[0.3, 0.7, 5.0]])
    dir_vec = np.array([0.0, 0.0, -1.0])

    uv = surf.project(query_pts, direction=dir_vec, do_pickles=False)
    np.testing.assert_allclose(uv[0], [0.3, 0.7], atol=1e-4)

    # Test refine_projection
    refined_uv = surf.refine_projection(
        points=query_pts,
        parametric_coordinates=uv,
        direction=dir_vec,
        do_pickles=False,
    )
    np.testing.assert_allclose(refined_uv[0], [0.3, 0.7], atol=1e-4)

    # Trigger refinement loop
    refined_uv2 = surf.refine_projection(
        points=query_pts,
        parametric_coordinates=np.array([[0.1, 0.1]]),
        direction=dir_vec,
        projection_tolerance=1e-3,
        do_pickles=False,
    )
    np.testing.assert_allclose(refined_uv2[0], [0.3, 0.7], atol=1e-3)


def test_function_integrate():
    """Test Function.integrate() over a 2D domain."""
    space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))
    # Area mapping: unit square
    area_ctrl = np.zeros((2, 2, 3))
    area_ctrl[0, 0, :2] = [0.0, 0.0]
    area_ctrl[0, 1, :2] = [0.0, 1.0]
    area_ctrl[1, 0, :2] = [1.0, 0.0]
    area_ctrl[1, 1, :2] = [1.0, 1.0]
    area_fn = lfs.Function(space=space, coefficients=area_ctrl)

    # Scalar function to integrate: f(u,v) = 2.0
    val_ctrl = np.ones((2, 2, 1)) * 2.0
    val_fn = lfs.Function(space=space, coefficients=val_ctrl)

    integral, centers = val_fn.integrate(area_fn, grid_n=4, quadrature_order=2)
    assert integral is not None
    # Area = 1, function = 2, total integral approx 2
    total = np.sum(integral.value)
    np.testing.assert_allclose(total, 2.0, rtol=0.1)


def test_function_plotting_variations():
    """Test Function plot methods across 1D curve, 2D surface, and 3D volume."""
    # 1D Curve plotting
    s1 = lfs.BSplineSpace(num_parametric_dimensions=1, degree=2, coefficients_shape=(4,))
    c1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0]])
    f1 = lfs.Function(space=s1, coefficients=c1)
    elems1 = f1.plot(show=False)
    assert len(elems1) > 0
    elems1_coeff = f1.plot_curve(point_type='coefficients', show=False)
    assert len(elems1_coeff) > 0

    # 2D Surface plotting with wireframe and point_cloud
    s2 = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(2, 2), coefficients_shape=(3, 3))
    c2 = np.zeros((3, 3, 3))
    f2 = lfs.Function(space=s2, coefficients=c2, name="surface_test")
    elems2_func = f2.plot(plot_types=['function'], show=False)
    elems2_wire = f2.plot(plot_types=['wireframe'], show=False)
    elems2_pts = f2.plot(plot_types=['point_cloud'], show=False)
    assert len(elems2_func) > 0
    assert len(elems2_wire) > 0
    assert len(elems2_pts) > 0
    elems2_coeff = f2.plot_surface(point_type='coefficients', show=False)
    assert len(elems2_coeff) > 0

    # 3D Volume plotting
    s3 = lfs.BSplineSpace(num_parametric_dimensions=3, degree=1, coefficients_shape=(2, 2, 2))
    c3 = np.zeros((2, 2, 2, 3))
    f3 = lfs.Function(space=s3, coefficients=c3, name="volume_test")
    elems3_eval = f3.plot(show=False)
    assert len(elems3_eval) > 0
    elems3_coeff = f3.plot_volume(point_type='coefficients', show=False)
    assert len(elems3_coeff) > 0


def test_function_project_caching_and_sections():
    """Verify Function.project caching mechanism and sectioned grid search."""
    space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))
    ctrl = np.zeros((2, 2, 3))
    ctrl[0, 0, :2] = [0.0, 0.0]
    ctrl[0, 1, :2] = [0.0, 1.0]
    ctrl[1, 0, :2] = [1.0, 0.0]
    ctrl[1, 1, :2] = [1.0, 1.0]
    fn = lfs.Function(space=space, coefficients=ctrl, name="cached_surf")

    pts = np.array([[0.25, 0.75, 1.0]])
    # 1. Project with do_pickles=True and grid_search_evaluation_cutoff=5 (to trigger sectioned grid search)
    coords1 = fn.project(pts, do_pickles=True, grid_search_evaluation_cutoff=5, force_reproject=True)
    np.testing.assert_allclose(coords1[0], [0.25, 0.75], atol=1e-3)

    # 1b. Directional projection with small subtraction cutoff
    coords1_dir = fn.project(
        pts,
        direction=np.array([0.0, 0.0, -1.0]),
        grid_search_subtraction_cutoff=1,
        force_reproject=True,
        do_pickles=False,
    )
    np.testing.assert_allclose(coords1_dir[0], [0.25, 0.75], atol=1e-3)

    # 2. Project again without force_reproject to hit the pickle-loading cache branch
    coords2 = fn.project(pts, do_pickles=True, force_reproject=False)
    np.testing.assert_allclose(coords2[0], [0.25, 0.75], atol=1e-3)

    # 3. Distance bounds
    d = fn._compute_distance_bounds(np.array([2.0, 2.0, 0.0]))
    assert d >= 0.0

