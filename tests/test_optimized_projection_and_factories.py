"""Tests for optimized projection algorithms and factory-based B-spline evaluators."""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from lsdo_function_spaces.core.spaces.non_cython_bsplines.compute_basis_matrix_numpy_factory import (
    make_bspline_evaluator_numpy,
    make_bspline_space_cache,
    evaluate_b_spline_numpy,
)
from lsdo_function_spaces.core.spaces.non_cython_bsplines.compute_basis_matrix_jax_factory import (
    make_bspline_evaluator_jax,
    compute_basis_matrix_jax,
    evaluate_b_spline_jax as eval_jax_baseline,
)
from lsdo_function_spaces.core.spaces.non_cython_bsplines.b_spline_patch_projection_optimized import (
    project_points_gauss_newton_numpy,
    project_points_lm_numpy,
    make_projector_lm_numpy,
    make_projector_lm_jax,
    LMParams,
)
from lsdo_function_spaces.core.spaces.non_cython_bsplines.b_spline_patch_projection_non_differentiable import (
    compute_point_to_bspline_projection,
)


@pytest.fixture
def surface_setup():
    degrees = (2, 2)
    n_ctrl = (4, 4)
    # Open uniform knot vectors on [0, 1]
    kv_u = np.array([0., 0., 0., 0.5, 1., 1., 1.])
    kv_v = np.array([0., 0., 0., 0.5, 1., 1., 1.])
    knots = (kv_u, kv_v)

    # Control points on a 4x4 grid creating a smooth dome surface
    u_grid, v_grid = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 4), indexing="ij")
    z = -(u_grid - 0.5)**2 - (v_grid - 0.5)**2 + 1.0
    coeffs = np.stack([u_grid, v_grid, z], axis=-1)
    return degrees, knots, coeffs


def test_numpy_factory_evaluator(surface_setup):
    degrees, knots, coeffs = surface_setup
    eval_fn = make_bspline_evaluator_numpy(degrees, knots)
    
    us = np.array([
        [0.2, 0.3],
        [0.5, 0.5],
        [0.8, 0.6],
    ])
    pts = eval_fn(us, coeffs)
    assert pts.shape == (3, 3)

    # Direct evaluation function
    pts_direct = evaluate_b_spline_numpy(us, degrees, knots, coeffs)
    np.testing.assert_allclose(pts, pts_direct, atol=1e-12)

    # With derivatives
    eval_du = make_bspline_evaluator_numpy(degrees, knots, der_orders=(1, 0))
    pts_du = eval_du(us, coeffs)
    assert pts_du.shape == (3, 3)


def test_jax_factory_evaluator(surface_setup):
    degrees, knots, coeffs = surface_setup
    eval_fn = make_bspline_evaluator_jax(degrees, knots, jit=False)
    
    us = jnp.array([
        [0.2, 0.3],
        [0.5, 0.5],
    ])
    coeffs_jnp = jnp.array(coeffs)
    pts = eval_fn(us, coeffs_jnp)
    assert pts.shape == (2, 3)

    # Test baseline matrix
    B = compute_basis_matrix_jax(us, degrees, knots)
    pts_baseline = eval_jax_baseline(us, degrees, knots, coeffs_jnp)
    np.testing.assert_allclose(np.array(pts), np.array(pts_baseline), atol=1e-5)


def test_optimized_gauss_newton_and_lm_numpy(surface_setup):
    degrees, knots, coeffs = surface_setup
    eval_fn = make_bspline_evaluator_numpy(degrees, knots)

    # Known target parametric points
    true_u = np.array([
        [0.3, 0.4],
        [0.7, 0.2],
    ])
    target_pts = eval_fn(true_u, coeffs)

    # Initial guess offset from true
    u0s = np.array([
        [0.25, 0.35],
        [0.65, 0.25],
    ])

    # Gauss-Newton numpy
    u_gn, conv_gn = project_points_gauss_newton_numpy(
        points=target_pts,
        u0s=u0s,
        coeffs=coeffs,
        degrees=degrees,
        knot_vectors=knots,
        max_iter=30,
        tol_grad=1e-8,
    )
    assert np.all(conv_gn)
    np.testing.assert_allclose(u_gn, true_u, atol=1e-3)

    # Levenberg-Marquardt numpy
    u_lm, conv_lm, lam, res = project_points_lm_numpy(
        points=target_pts,
        u0s=u0s,
        coeffs=coeffs,
        degrees=degrees,
        knot_vectors=knots,
        params=LMParams(max_iter=30, tol_grad=1e-8),
    )
    assert np.all(conv_lm)
    np.testing.assert_allclose(u_lm, true_u, atol=1e-3)


def test_optimized_lm_jax(surface_setup):
    degrees, knots, coeffs = surface_setup
    eval_fn = make_bspline_evaluator_numpy(degrees, knots)

    true_u = np.array([[0.4, 0.6]])
    target_pts = eval_fn(true_u, coeffs)
    u0s = np.array([[0.5, 0.5]])

    proj_jax = make_projector_lm_jax(
        degrees=degrees,
        knots=knots,
        params=LMParams(max_iter=25, tol_grad=1e-6),
        jit=False,
    )
    u_star, conv = proj_jax(
        jnp.array(target_pts),
        jnp.array(u0s),
        jnp.array(coeffs),
    )
    assert bool(conv[0])
    np.testing.assert_allclose(np.array(u_star), true_u, atol=1e-3)


def test_non_differentiable_projection(surface_setup):
    degrees, knots, coeffs = surface_setup
    eval_fn = make_bspline_evaluator_numpy(degrees, knots)

    true_u = np.array([[0.35, 0.45]])
    target_pt = eval_fn(true_u, coeffs)[0]
    u0 = np.array([0.4, 0.5])

    final_coords, final_res, converged, iters, J, mask, surf_jac = compute_point_to_bspline_projection(
        point=jnp.array(target_pt),
        degrees=degrees,
        coefficients=jnp.array(coeffs),
        para_coords=jnp.array(u0),
        knots=tuple(jnp.array(k) for k in knots),
        max_iter=50,
        tol=1e-8,
    )
    assert bool(converged)
    np.testing.assert_allclose(np.array(final_coords), true_u[0], atol=1e-3)

