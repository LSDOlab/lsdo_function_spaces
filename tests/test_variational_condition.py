"""Tests for Gauss-Lobatto quadrature and VariationalInequalityCondition in lsdo_function_spaces."""

import numpy as np
import scipy.sparse as sps
import csdl_alpha as csdl
import pytest

import lsdo_function_spaces as lfs
from lsdo_function_spaces.core.interface import (
    _gauss_lobatto_1d,
    Discretization,
    InterfaceDomain,
    Interface,
    generate_element_quadrature,
    VariationalInequalityCondition,
)
from lsdo_function_spaces.core.spaces.b_spline_space import BSplineSpace


def test_gauss_lobatto_1d():
    """Verify Gauss-Lobatto 1D roots, endpoints, and weight sums on [-1, 1]."""
    for N in [2, 3, 4, 5]:
        x, w = _gauss_lobatto_1d(N)
        assert len(x) == N
        assert len(w) == N
        # Endpoints must be -1 and 1
        assert np.isclose(x[0], -1.0)
        assert np.isclose(x[-1], 1.0)
        # Weights must be positive and sum to 2.0 (length of interval)
        assert np.all(w > 0)
        assert np.isclose(np.sum(w), 2.0)

    # Exactness test: N-point Gauss-Lobatto integrates polynomials up to degree 2N-3 exactly
    # For N=4, degree up to 5: integral of x^4 on [-1, 1] is 2/5 = 0.4
    x4, w4 = _gauss_lobatto_1d(4)
    integral_x4 = np.sum(w4 * (x4**4))
    assert np.isclose(integral_x4, 0.4, atol=1e-14)


def test_discretization_gauss_lobatto():
    """Verify 2D surface discretization with Gauss-Lobatto includes corners."""
    disc = Discretization(num_points=3, scheme="gauss_lobatto")
    domain = InterfaceDomain.surface(u_bounds=(0.0, 1.0), v_bounds=(0.0, 1.0))
    nodes, weights = disc.generate_samples(domain)

    assert nodes.shape == (9, 2)
    assert weights.shape == (9,)
    assert np.isclose(np.sum(weights), 1.0)

    # Check that all 4 corners [0,0], [0,1], [1,0], [1,1] are present
    corners = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    for c in corners:
        dists = np.linalg.norm(nodes - c, axis=1)
        assert np.min(dists) < 1e-14


def test_generate_element_quadrature():
    """Verify composite element quadrature across unique knot spans with deduplication."""
    # 2D B-spline with 2 knot spans in u and 2 in v
    knots_u = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
    knots_v = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
    space = BSplineSpace(
        num_parametric_dimensions=2,
        degree=(1, 1),
        coefficients_shape=(3, 3),
        knots=(knots_u, knots_v),
    )

    nodes, weights = generate_element_quadrature(
        space=space,
        scheme="gauss_lobatto",
        degree_elevation=1,
    )

    # Weights must sum to total parametric volume (1.0)
    assert np.isclose(np.sum(weights), 1.0, atol=1e-12)
    assert np.all(weights > 0)
    # Check bounds
    assert np.all(nodes >= -1e-14)
    assert np.all(nodes <= 1.0 + 1e-14)


def test_constant_preservation():
    """
    Mathematical proof verification:
    If constraint violation is uniform c_g ≡ C across the domain,
    basis_weighted_ks MUST reproduce R_m ≡ C to machine precision.
    """
    space = BSplineSpace(
        num_parametric_dimensions=2,
        degree=(1, 1),
        coefficients_shape=(2, 2),
    )

    condition = VariationalInequalityCondition(
        space=space,
        method="basis_weighted_ks",
        rho=50.0,
        discretization_scheme="gauss_lobatto",
    )

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    # Test positive, negative, and zero constant values
    for C in [-2.5, 0.0, 1.75, 10.0]:
        c_const = csdl.Variable(value=np.full((condition.num_quad_points,), C))
        R = condition.evaluate_projection(c_const)
        res_val = R.value
        assert np.allclose(res_val, C, atol=1e-12), f"Failed for C={C}: max diff = {np.max(np.abs(res_val - C))}"


def test_dimension_decoupling():
    """
    Verify that increasing quadrature resolution does NOT increase the number of constraints.
    Constraint count N_psi must strictly equal the number of test basis functions.
    """
    space = BSplineSpace(
        num_parametric_dimensions=2,
        degree=(1, 1),
        coefficients_shape=(2, 2),  # 4 basis functions
    )

    cond_coarse = VariationalInequalityCondition(
        space=space,
        degree_elevation=0,
    )
    cond_fine = VariationalInequalityCondition(
        space=space,
        degree_elevation=3,
    )

    # Coarse vs Fine quadrature point counts differ
    assert cond_fine.num_quad_points > cond_coarse.num_quad_points

    # But constraint counts are IDENTICAL and equal to 4
    assert cond_coarse.num_basis_functions == 4
    assert cond_fine.num_basis_functions == 4

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    c_coarse = csdl.Variable(value=np.zeros(cond_coarse.num_quad_points))
    c_fine = csdl.Variable(value=np.zeros(cond_fine.num_quad_points))

    R_coarse = cond_coarse.evaluate_projection(c_coarse)
    R_fine = cond_fine.evaluate_projection(c_fine)

    assert R_coarse.shape == (4,)
    assert R_fine.shape == (4,)


def test_rectified_galerkin():
    """Verify rectified_galerkin weak penalty formulation."""
    space = BSplineSpace(
        num_parametric_dimensions=2,
        degree=(1, 1),
        coefficients_shape=(2, 2),
    )
    cond = VariationalInequalityCondition(
        space=space,
        method="rectified_galerkin",
    )

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    # Satisfied condition (all non-positive) -> zero residual
    c_neg = csdl.Variable(value=np.full((cond.num_quad_points,), -0.5))
    R_neg = cond.evaluate_projection(c_neg)
    assert np.allclose(R_neg.value, 0.0)

    # Violating condition (positive) -> positive residual
    c_pos = csdl.Variable(value=np.full((cond.num_quad_points,), 2.0))
    R_pos = cond.evaluate_projection(c_pos)
    assert np.all(R_pos.value > 0.0)
