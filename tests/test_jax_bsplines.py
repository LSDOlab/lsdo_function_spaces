"""Unit tests for JAX-accelerated B-spline evaluation and stencil operations."""

import pytest
import numpy as np
import jax.numpy as jnp
from lsdo_function_spaces.core.spaces.non_cython_bsplines.compute_basis_matrix_jax_stencil import (
    compute_basis_matrix_jax,
    compute_basis_stencil_jax,
    apply_basis_stencil_jax,
    evaluate_b_spline_jax,
    evaluate_b_spline_jax_fast,
    stencil_to_bcoo,
)


def test_jax_basis_matrix_and_stencil_evaluation():
    """Test JAX sparse basis matrix and stencil evaluation."""
    degrees = (2, 2)
    knots = (
        (0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0),
        (0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0),
    )
    # 4 x 4 control points
    num_cp_x = len(knots[0]) - degrees[0] - 1
    num_cp_y = len(knots[1]) - degrees[1] - 1
    n_ctrl = num_cp_x * num_cp_y

    coeffs = np.ones((num_cp_x, num_cp_y, 3))

    us = jnp.array([
        [0.25, 0.25],
        [0.5, 0.5],
        [0.75, 0.75],
    ])

    # 1. BCOO matrix
    bcoo_mat = compute_basis_matrix_jax(us, degrees, knots)
    dense_mat = bcoo_mat.todense()
    # Partition of unity: rows sum to 1
    np.testing.assert_allclose(np.sum(dense_mat, axis=1), 1.0, atol=1e-10)

    # 2. Stencil evaluation
    cols, w, total_n = compute_basis_stencil_jax(us, degrees, knots)
    assert total_n == n_ctrl

    vals = apply_basis_stencil_jax(cols, w, jnp.array(coeffs))
    np.testing.assert_allclose(vals, np.ones((3, 3)), atol=1e-10)

    # 3. Fast evaluate
    fast_vals = evaluate_b_spline_jax_fast(us, degrees, knots, jnp.array(coeffs))
    np.testing.assert_allclose(fast_vals, np.ones((3, 3)), atol=1e-10)

    # 4. Standard evaluate
    std_vals = evaluate_b_spline_jax(us, degrees, knots, jnp.array(coeffs))
    np.testing.assert_allclose(std_vals, np.ones((3, 3)), atol=1e-10)

    # 5. stencil_to_bcoo
    bcoo_from_stencil = stencil_to_bcoo(cols, w, total_n)
    np.testing.assert_allclose(bcoo_from_stencil.todense(), dense_mat, atol=1e-10)


def test_jax_stencil_with_derivatives():
    """Test JAX stencil evaluation with first derivatives."""
    degrees = (2, 2)
    knots = (
        (0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        (0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
    )
    # Linear function in x: coeffs_x = [0, 0.5, 1.0] for all y
    coeffs = np.zeros((3, 3, 1))
    coeffs[:, :, 0] = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])

    us = jnp.array([[0.4, 0.6]])
    # Derivative w.r.t first parametric coordinate u
    der_val = evaluate_b_spline_jax(us, degrees, knots, jnp.array(coeffs), der_orders=(1, 0))
    # df/du of linear ramp across [0, 1] is 1.0
    np.testing.assert_allclose(der_val, [[1.0]], atol=1e-5)

