"""Unit tests for Optimization problem setup and NewtonOptimizer."""

import pytest
import numpy as np
import csdl_alpha as csdl
import lsdo_function_spaces as lfs
from lsdo_function_spaces.core.optimization import Optimization, NewtonOptimizer


def test_optimization_basic_setup():
    """Test creating an Optimization object and adding objective and design variables."""
    opt = Optimization()
    assert opt.design_variables == []
    assert opt.constraints == []

    x = csdl.Variable(value=np.array([2.0, 3.0]), name="x")
    opt.add_design_variable(x, initial_value=np.array([1.0, 1.0]))
    assert len(opt.design_variables) == 1
    np.testing.assert_allclose(x.value, [1.0, 1.0])

    # Objective: f(x) = x[0]^2 + x[1]^2
    obj = csdl.vdot(x, x)
    opt.add_objective(obj)
    assert opt.objective is obj

    grad = opt.compute_objective_gradient()
    assert grad is not None


def test_optimization_with_penalized_constraints():
    """Test optimization problem with penalized constraints."""
    opt = Optimization()
    x = csdl.Variable(value=np.array([1.0, 2.0]), name="x")
    opt.add_design_variable(x)

    obj = csdl.vdot(x, x)
    opt.add_objective(obj)

    # Constraint: x[0] + x[1] - 1 = 0 with penalty
    c = csdl.sum(x) - 1.0
    c.add_name("linear_constraint")
    opt.add_constraint(c, penalty=10.0)

    lagrangian = opt.compute_lagrangian()
    assert lagrangian is not None

    grad_lag = opt.compute_lagrangian_gradient()
    assert grad_lag is not None

    jac = opt.compute_constraint_jacobian()
    assert jac is not None

    opt.setup()
    assert len(opt.state_residual_pairs) >= 1


def test_optimization_with_lagrange_multipliers():
    """Test optimization problem setup with unpenalized constraints (Lagrange multipliers)."""
    opt = Optimization()
    x = csdl.Variable(value=np.array([1.0, 2.0]), name="x")
    opt.add_design_variable(x, initial_value=np.array([0.5, 0.5]))

    obj = csdl.vdot(x, x)
    opt.add_objective(obj)

    c = csdl.sum(x) - 1.0
    c.add_name("multiplier_constraint")
    opt.add_constraint(c, penalty=None)

    opt.setup()
    assert len(opt.state_residual_pairs) == 2  # 1 constraint multiplier + 1 design variable


def test_newton_optimizer_init_and_add():
    """Test NewtonOptimizer initialization and adding optimization problem."""
    optimizer = NewtonOptimizer()
    assert not optimizer.has_been_setup

    opt = Optimization()
    x = csdl.Variable(value=np.array([1.0, 2.0]), name="x")
    opt.add_design_variable(x)
    obj = csdl.vdot(x, x)
    opt.add_objective(obj)

    optimizer.add_optimization(opt)
    assert optimizer.optimization is opt
    optimizer.setup()
    assert optimizer.has_been_setup

