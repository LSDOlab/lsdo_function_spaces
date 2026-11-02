import numpy as np
import pytest
import scipy.sparse as sps
import csdl_alpha as csdl
import lsdo_function_spaces as lfs


def test_interface_domain():
    d_pt = lfs.InterfaceDomain.point(num_points=1)
    assert d_pt.dimension == 0
    assert d_pt.is_finite
    assert d_pt.num_copies == 1

    d_multi = lfs.InterfaceDomain.point(num_points=3)
    assert d_multi.dimension == 0
    assert d_multi.is_finite
    assert d_multi.num_copies == 3

    d_curve = lfs.InterfaceDomain.interval(0.0, 1.0)
    assert d_curve.dimension == 1
    assert not d_curve.is_finite
    assert d_curve.num_copies == 1


def test_marker_point_and_variable():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))
    c1 = np.array([[[0., 0., 0.], [0., 1., 0.]], [[1., 0., 0.], [1., 1., 0.]]])
    f = lfs.Function(space=space, coefficients=c1, name="test_f")

    # Constant point marker
    m_const = lfs.Marker(entity=f, parametric_coordinates=np.array([0.5, 0.5]), name="m_const")
    assert m_const.is_constant
    assert m_const.domain.dimension == 0
    assert m_const.domain.num_copies == 1
    assert m_const.patches == {(f, None)}

    val_const = m_const.evaluate()
    assert np.allclose(val_const.value, [0.5, 0.5, 0.0])

    # Variable point marker (csdl.Variable)
    var_coord = csdl.Variable(value=np.array([[0.25, 0.75]]))
    m_var = lfs.Marker(entity=f, parametric_coordinates=var_coord, name="m_var")
    assert not m_var.is_constant

    val_var = m_var.evaluate()
    assert np.allclose(val_var.value.flatten(), [0.25, 0.75, 0.0])


def test_marker_multipoint_projection():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))
    c1 = np.array([[[0., 0., 0.], [0., 1., 0.]], [[1., 0., 0.], [1., 1., 0.]]])
    f = lfs.Function(space=space, coefficients=c1)

    pts = np.array([[0.2, 0.3, 0.0], [0.7, 0.8, 0.0]])
    m_multi = lfs.Marker.from_points(entity=f, points=pts, name="m_multi")
    assert m_multi.domain.num_copies == 2
    assert m_multi.domain.dimension == 0

    eval_pts = m_multi.evaluate()
    assert np.allclose(eval_pts.value, pts)


def test_curve_marker_with_function():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    # Surface patch [0, 1]^2 -> R^3
    space_surf = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))
    c_surf = np.array([[[0., 0., 0.], [0., 10., 0.]], [[10., 0., 0.], [10., 10., 0.]]])
    f_surf = lfs.Function(space=space_surf, coefficients=c_surf, name="surf")

    # Curve alpha: [0, 1] -> [0, 1]^2 (a 1D B-spline with 2D codomain)
    space_curve = lfs.BSplineSpace(num_parametric_dimensions=1, degree=(1,), coefficients_shape=(2,))
    # Diagonal across the surface parameter space from (0, 0) to (1, 1)
    c_curve = np.array([[0.0, 0.0], [1.0, 1.0]])
    alpha_fn = lfs.Function(space=space_curve, coefficients=c_curve, name="diagonal_curve")

    m_curve = lfs.Marker.from_curve(entity=f_surf, curve_map=alpha_fn, name="diagonal_marker")
    assert m_curve.domain.dimension == 1
    assert m_curve.domain.num_copies == 1
    assert not m_curve.domain.is_finite

    # Evaluate at curve midpoint s = 0.5 -> parametric (0.5, 0.5) -> physical (5.0, 5.0, 0.0)
    eval_mid = m_curve.evaluate(sample_parameters=np.array([[0.5]]))
    assert np.allclose(eval_mid.value.flatten(), [5.0, 5.0, 0.0])


def test_interface_trace_operator_function_and_function_set():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    # Function 1
    s1 = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))
    c1 = np.array([[[0., 0., 0.], [0., 1., 0.]], [[1., 0., 0.], [1., 1., 0.]]])
    f1 = lfs.Function(space=s1, coefficients=c1, name="f1")

    # FunctionSet 2 (two patches)
    s2 = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))
    c2_0 = np.array([[[0., 0., 0.], [0., 1., 0.]], [[1., 0., 0.], [1., 1., 0.]]])
    c2_1 = np.array([[[1., 0., 0.], [1., 1., 0.]], [[2., 0., 0.], [2., 1., 0.]]])
    f2_0 = lfs.Function(space=s2, coefficients=c2_0, name="p0")
    f2_1 = lfs.Function(space=s2, coefficients=c2_1, name="p1")
    fset2 = lfs.FunctionSet(functions={0: f2_0, 1: f2_1}, name="fset2")

    # Markers at 2 discrete points
    m1 = lfs.Marker(entity=f1, parametric_coordinates=np.array([[0.2, 0.3], [0.8, 0.9]]), name="m1")
    m2 = lfs.Marker(
        entity=fset2,
        parametric_coordinates=[(0, np.array([0.2, 0.3])), (1, np.array([0.8, 0.9]))],
        name="m2",
    )

    inter = lfs.Interface(markers=[m1, m2], name="test_interface")
    assert inter.domain.num_copies == 2
    assert inter.patches == {(f1, None), (fset2, 0), (fset2, 1)}

    # Trace operator for f1
    T1 = inter.get_trace_operator(0)
    assert sps.issparse(T1)
    assert T1.shape == (2, 4)  # 2 points x 4 control points

    # Trace operator for fset2
    T2 = inter.get_trace_operator(1)
    assert sps.issparse(T2)
    assert T2.shape == (2, 8)  # 2 points x 8 total control points across 2 patches

    # Verify T @ a matches direct evaluation
    val1_direct = m1.evaluate().value
    a1_flat = c1.reshape((4, 3))
    val1_trace = T1.toarray() @ a1_flat
    assert np.allclose(val1_direct, val1_trace)

    val2_direct = m2.evaluate().value
    a2_flat = np.vstack([c2_0.reshape((4, 3)), c2_1.reshape((4, 3))])
    val2_trace = T2.toarray() @ a2_flat
    assert np.allclose(val2_direct, val2_trace)


def test_collocation_condition_and_selectors():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))
    c1 = np.array([[[0., 0., 0.], [0., 1., 0.]], [[1., 0., 0.], [1., 1., 0.]]])
    c2 = np.array([[[0., 0.5, 0.], [0., 1.5, 0.]], [[1., 0.5, 0.], [1., 1.5, 0.]]])

    f1 = lfs.Function(space=space, coefficients=c1)
    f2 = lfs.Function(space=space, coefficients=c2)

    # Hip joint: x and z match, y is offset/free
    m1 = lfs.Marker(entity=f1, parametric_coordinates=np.array([0.5, 0.5]), name="m1")
    m2 = lfs.Marker(entity=f2, parametric_coordinates=np.array([0.5, 0.5]), name="m2")

    inter = lfs.Interface(markers=[m1, m2], name="hip_interface")

    # Constrain x (idx 0) and z (idx 2) only
    cond_xz = lfs.CollocationCondition(
        interface=inter,
        selectors={0: [0, 2], 1: [0, 2]},
        name="hip_xz",
    )
    assert cond_xz.is_linear

    res = cond_xz.evaluate_residual()
    # f1 evaluates to [0.5, 0.5, 0.0]; f2 evaluates to [0.5, 1.0, 0.0]
    # x and z match (diff is 0.0); y differs by 0.5 but is unselected
    assert np.allclose(res.value, [0.0, 0.0])


def test_function_network_patch_level_graph():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))
    c = np.zeros((2, 2, 3))

    # Multi-patch wing: Patch 0 (inboard), Patch 1 (outboard)
    p0 = lfs.Function(space=space, coefficients=c.copy(), name="p0")
    p1 = lfs.Function(space=space, coefficients=c.copy(), name="p1")
    wing = lfs.FunctionSet(functions={0: p0, 1: p1}, name="wing")

    # Fuselage: Patch 0
    fuse_p0 = lfs.Function(space=space, coefficients=c.copy(), name="fuse_p0")
    fuselage = lfs.FunctionSet(functions={0: fuse_p0}, name="fuselage")

    # Seam between wing patch 0 and wing patch 1
    m_w0 = lfs.Marker(entity=wing, parametric_coordinates=(0, np.array([1.0, 0.5])), name="w0_seam")
    m_w1 = lfs.Marker(entity=wing, parametric_coordinates=(1, np.array([0.0, 0.5])), name="w1_seam")
    seam_inter = lfs.Interface(markers=[m_w0, m_w1], name="wing_seam")
    seam_cond = lfs.CollocationCondition(interface=seam_inter, name="seam_c0")

    # Joint between wing patch 0 and fuselage patch 0
    m_w_root = lfs.Marker(entity=wing, parametric_coordinates=(0, np.array([0.0, 0.5])), name="w_root")
    m_fuse = lfs.Marker(entity=fuselage, parametric_coordinates=(0, np.array([0.5, 0.5])), name="fuse_mount")
    root_inter = lfs.Interface(markers=[m_w_root, m_fuse], name="wing_fuselage_joint")
    root_cond = lfs.CollocationCondition(interface=root_inter, name="root_attach")

    net = lfs.FunctionNetwork(name="aircraft_net")
    net.add_condition(seam_cond)
    net.add_condition(root_cond)

    # Component groupings
    net.component_groupings = {
        "wing": [(wing, 0), (wing, 1)],
        "fuselage": [(fuselage, 0)],
    }

    # Connected components
    comps = net.compute_connected_components()
    assert len(comps) == 1  # All 3 patches connected into one structure
    assert len(comps[0]) == 3

    # Classification
    internal, between = net.classify_interfaces()
    assert len(internal) == 1
    assert internal[0] is seam_inter  # Wing seam is internal
    assert len(between) == 1
    assert between[0] is root_inter   # Wing-fuselage attachment is between components

    # Cycle rank
    b1_multi, b1_macro = net.compute_cycle_rank()
    # Tree topology: 3 patch nodes, 2 interfaces -> 0 cycles
    assert b1_multi == 0
    assert b1_macro == 0


def test_freedom_analysis_rigid_vs_deformable():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))
    c1 = np.zeros((2, 2, 3))
    c2 = np.zeros((2, 2, 3))

    f1 = lfs.Function(space=space, coefficients=c1, name="f1")
    f2 = lfs.Function(space=space, coefficients=c2, name="f2")

    # Two connection points
    m1 = lfs.Marker(entity=f1, parametric_coordinates=np.array([[0.2, 0.2], [0.8, 0.8]]), name="m1")
    m2 = lfs.Marker(entity=f2, parametric_coordinates=np.array([[0.2, 0.2], [0.8, 0.8]]), name="m2")

    inter = lfs.Interface(markers=[m1, m2], name="two_point_inter")
    cond = lfs.CollocationCondition(interface=inter, name="two_pt_colloc")

    net = lfs.FunctionNetwork()
    net.add_condition(cond)

    # Total variables: 2 functions x 4 control points x 3 dims = 24 variables
    # Total constraints: 2 points x 3 dims = 6 constraints
    report = net.analyze_freedom()
    assert report.num_variables == 24
    assert report.num_constraints == 6
    # For deformable B-splines, both points displace independently -> rank = 6, redundancy = 0, LICQ holds
    assert report.rank == 6
    assert report.redundancy == 0
    assert report.is_licq_satisfied



def test_marker_from_point_and_points_on_function_set():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))
    # Patch 0: [0, 1] x [0, 1] in XY plane
    c0 = np.array([[[0., 0., 0.], [0., 1., 0.]], [[1., 0., 0.], [1., 1., 0.]]])
    # Patch 1: [1, 2] x [0, 1] in XY plane
    c1 = np.array([[[1., 0., 0.], [1., 1., 0.]], [[2., 0., 0.], [2., 1., 0.]]])

    f0 = lfs.Function(space=space, coefficients=c0, name="p0")
    f1 = lfs.Function(space=space, coefficients=c1, name="p1")
    fset = lfs.FunctionSet(functions={0: f0, 1: f1}, name="two_patch_fset")

    # Physical point on Patch 0
    p_on_0 = np.array([0.5, 0.5, 0.0])
    m_p0 = lfs.Marker.from_point(entity=fset, point=p_on_0, name="marker_on_patch_0")
    assert m_p0.domain.num_copies == 1
    # Check that it resolved to patch 0
    patch_idx, uv = m_p0.parametric_coordinates
    assert patch_idx == 0
    assert np.allclose(uv, [0.5, 0.5])
    assert np.allclose(m_p0.evaluate().value, p_on_0)

    # Physical point on Patch 1
    p_on_1 = np.array([1.5, 0.5, 0.0])
    m_p1 = lfs.Marker.from_point(entity=fset, point=p_on_1, name="marker_on_patch_1")
    assert m_p1.domain.num_copies == 1
    patch_idx1, uv1 = m_p1.parametric_coordinates
    assert patch_idx1 == 1
    assert np.allclose(uv1, [0.5, 0.5])
    assert np.allclose(m_p1.evaluate().value, p_on_1)

    # Multi-point projection across both patches
    pts_both = np.array([[0.5, 0.5, 0.0], [1.5, 0.5, 0.0]])
    m_multi = lfs.Marker.from_points(entity=fset, points=pts_both, name="multi_patch_marker")
    assert m_multi.domain.num_copies == 2
    assert m_multi.patches == {(fset, 0), (fset, 1)}
    eval_both = m_multi.evaluate().value
    assert np.allclose(eval_both, pts_both)


def test_sliding_state_type_markers_and_boundary_penalties():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))
    c1 = np.array([[[0., 0., 0.], [0., 1., 0.]], [[1., 0., 0.], [1., 1., 0.]]])
    c2 = np.array([[[0.5, 0., 0.], [0.5, 1., 0.]], [[1.5, 0., 0.], [1.5, 1., 0.]]])

    f1 = lfs.Function(space=space, coefficients=c1, name="wing")
    f2 = lfs.FunctionSet(functions={0: lfs.Function(space=space, coefficients=c2)}, space=space, name="fuselage")

    # Check parametric bounds
    assert f1.parametric_bounds == ((0.0, 1.0), (0.0, 1.0))
    assert f2.parametric_bounds == {0: ((0.0, 1.0), (0.0, 1.0))}

    # Fixed marker on f1
    m1 = lfs.Marker(entity=f1, parametric_coordinates=np.array([1.0, 0.5]), name="m_fixed")
    assert m1.is_constant is True
    assert m1.compute_boundary_violation() is None

    # Sliding marker on f2 (FunctionSet)
    u2_var = csdl.Variable(value=np.array([[0.2, 0.3]]))
    m2 = lfs.Marker(entity=f2, parametric_coordinates=[(0, u2_var)], name="m_sliding")
    assert m2.is_constant is False
    assert m2.parametric_bounds == {0: ((0.0, 1.0), (0.0, 1.0))}

    # Interface and condition
    interface = lfs.Interface(markers=[m1, m2], name="sliding_joint")
    cond = lfs.CollocationCondition(interface=interface)
    assert cond.is_linear is False  # Because m2 has variable coordinates

    res = cond.evaluate_residual()
    # At u2=(0.2, 0.3), f2 pos is [0.7, 0.3, 0.0]
    # f1 pos is [1.0, 0.5, 0.0]
    # Residual = f2 - f1 = [-0.3, -0.2, 0.0]
    np.testing.assert_allclose(res.value, [[-0.3, -0.2, 0.0]], atol=1e-6)

    # Gradient of squared residual wrt sliding coordinate u2_var
    loss = csdl.sum(res**2)
    grad = csdl.derivative(loss, u2_var)
    # d/du2 loss = 2*(-0.3)*(1.0) = -0.6
    # d/dv2 loss = 2*(-0.2)*(1.0) = -0.4
    np.testing.assert_allclose(grad.value, [[-0.6, -0.4]], atol=1e-5)

    # Boundary violation testing
    # Inside bounds: penalty should be ~0
    v_in = m2.compute_boundary_violation()
    assert v_in.value < 1e-6

    # Outside bounds: penalty and non-zero gradient
    u_out_var = csdl.Variable(value=np.array([[-0.1, 1.2]]))
    m_out = lfs.Marker(entity=f1, parametric_coordinates=u_out_var)
    v_out = m_out.compute_boundary_violation()
    assert v_out.value > 0.01
    grad_v = csdl.derivative(v_out, u_out_var)
    # Should push u up (negative grad wrt negative violation) and v down (positive grad)
    assert grad_v.value[0, 0] < 0.0
    assert grad_v.value[0, 1] > 0.0


def test_continuous_interface_and_discretization():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    # 1. Domain and Discretization tests
    surf_domain = lfs.InterfaceDomain.surface(u_bounds=(0.0, 2.0), v_bounds=(0.0, 3.0))
    assert surf_domain.dimension == 2
    assert surf_domain.is_finite is False

    discr_gl = lfs.Discretization(num_points=(4, 4), scheme="gauss_legendre")
    nodes_gl, weights_gl = discr_gl.generate_samples(surf_domain)
    assert nodes_gl.shape == (16, 2)
    assert np.isclose(np.sum(weights_gl), 6.0)

    # Exact integration of f(u, v) = u^2 * v on [0, 2] x [0, 3] -> (8/3) * (9/2) = 12.0
    exact_integral = 12.0
    numerical_integral = float(np.sum(weights_gl * (nodes_gl[:, 0]**2 * nodes_gl[:, 1])))
    assert np.isclose(numerical_integral, exact_integral)

    # Chebyshev and uniform sampling
    discr_cheb = lfs.Discretization(num_points=5, scheme="chebyshev")
    nodes_ch, _ = discr_cheb.generate_samples(lfs.InterfaceDomain.interval())
    assert nodes_ch.shape == (5, 1)

    discr_unif = lfs.Discretization(num_points=5, scheme="uniform")
    nodes_un, _ = discr_unif.generate_samples(lfs.InterfaceDomain.interval())
    np.testing.assert_allclose(nodes_un.flatten(), [0.0, 0.25, 0.5, 0.75, 1.0])

    # 2. Continuous 1D Seam between two surface patches
    space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))
    # Patch 1: [0, 1] x [0, 1]
    c1 = np.array([[[0., 0., 0.], [0., 1., 0.]], [[1., 0., 0.], [1., 1., 0.]]])
    # Patch 2: [1, 2] x [0, 1]
    c2 = np.array([[[1., 0., 0.], [1., 1., 0.]], [[2., 0., 0.], [2., 1., 0.]]])

    p1 = lfs.Function(space=space, coefficients=c1, name="patch1")
    p2 = lfs.Function(space=space, coefficients=c2, name="patch2")

    # Marker 1 on right edge of patch 1: u=1.0, v=s
    m1 = lfs.Marker(
        entity=p1,
        parametric_coordinates=lambda s: np.array([1.0, s[0]]),
        domain=lfs.InterfaceDomain.interval(0.0, 1.0),
        name="p1_right_edge",
    )
    # Marker 2 on left edge of patch 2: u=0.0, v=s
    m2 = lfs.Marker(
        entity=p2,
        parametric_coordinates=lambda s: np.array([0.0, s[0]]),
        domain=lfs.InterfaceDomain.interval(0.0, 1.0),
        name="p2_left_edge",
    )

    discr_seam = lfs.Discretization(num_points=6, scheme="gauss_legendre")
    seam_inter = lfs.Interface(markers=[m1, m2], discretization=discr_seam, name="seam_interface")
    assert seam_inter.domain.dimension == 1

    # Collocation condition with quadrature weights
    seam_cond = lfs.CollocationCondition(interface=seam_inter, weighted=True, name="seam_l2_condition")
    res = seam_cond.evaluate_residual()
    assert res.shape == (6, 3)
    # Since patches meet perfectly along x=1.0, residual must be exactly 0
    np.testing.assert_allclose(res.value, np.zeros((6, 3)), atol=1e-8)

    # Trace operator verification on continuous interface
    T1 = seam_inter.get_trace_operator(0)
    T2 = seam_inter.get_trace_operator(1)
    assert T1.shape == (6, 4)
    assert T2.shape == (6, 4)

    # Freedom analysis on continuous seam
    net = lfs.FunctionNetwork(name="continuous_seam_network")
    net.add_condition(seam_cond)
    report = net.analyze_freedom()
    # 2 patches * 4 control points * 3 dims = 24 variables
    assert report.num_variables == 24
    # 6 collocation points * 3 dims = 18 constraints
    assert report.num_constraints == 18
    # Rank along degree-1 linear edge is 2 control points * 3 dims = 6
    assert report.rank == 6
    assert report.redundancy == 12
    assert report.nullity == 18


def test_c1_and_g1_continuity_conditions():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))
    # Planar flat patch 1: [0, 1] x [0, 1]
    c1 = np.array([[[0., 0., 0.], [0., 1., 0.]], [[1., 0., 0.], [1., 1., 0.]]])
    # Planar flat patch 2: [1, 2] x [0, 1]
    c2 = np.array([[[1., 0., 0.], [1., 1., 0.]], [[2., 0., 0.], [2., 1., 0.]]])

    p1 = lfs.Function(space=space, coefficients=c1, name="patch1")
    p2 = lfs.Function(space=space, coefficients=c2, name="patch2")

    m1 = lfs.Marker(entity=p1, parametric_coordinates=lambda s: np.array([1.0, s[0]]), domain=lfs.InterfaceDomain.interval(), name="m1")
    m2 = lfs.Marker(entity=p2, parametric_coordinates=lambda s: np.array([0.0, s[0]]), domain=lfs.InterfaceDomain.interval(), name="m2")

    discr = lfs.Discretization(num_points=4, scheme="gauss_legendre")
    inter = lfs.Interface(markers=[m1, m2], discretization=discr, name="seam")

    # 1. Transversal derivative matching (d/du across the seam)
    c1_cond = lfs.TransversalDerivativeCondition(interface=inter, derivative_orders=(1, 0), weighted=True, name="c1_cond")
    res_c1 = c1_cond.evaluate_residual()
    assert res_c1.shape == (4, 3)
    # Patches have identical dx/du = 1.0, dy/du = 0, dz/du = 0
    np.testing.assert_allclose(res_c1.value, np.zeros((4, 3)), atol=1e-8)
    assert c1_cond.is_linear is True

    # 2. Surface normal alignment (G1 continuity)
    g1_cond = lfs.NormalAlignmentCondition(interface=inter, weighted=True, name="g1_cond")
    res_g1 = g1_cond.evaluate_residual()
    assert res_g1.shape == (4, 3)
    # Both patches have unit surface normal [0, 0, 1] everywhere
    np.testing.assert_allclose(res_g1.value, np.zeros((4, 3)), atol=1e-8)
    assert g1_cond.is_linear is False

    # 3. Sensitivity verification under patch perturbation
    c2_tilted = np.array([[[1., 0., 0.], [1., 1., 0.]], [[2., 0., 0.4], [2., 1., 0.4]]])
    p2_tilted = lfs.Function(space=space, coefficients=c2_tilted, name="p2_tilted")
    m2_tilted = lfs.Marker(entity=p2_tilted, parametric_coordinates=lambda s: np.array([0.0, s[0]]), domain=lfs.InterfaceDomain.interval())
    inter_tilted = lfs.Interface(markers=[m1, m2_tilted], discretization=discr)

    c1_tilted_cond = lfs.TransversalDerivativeCondition(interface=inter_tilted, derivative_orders=(1, 0))
    loss_c1 = csdl.sum(c1_tilted_cond.evaluate_residual()**2)
    assert loss_c1.value > 0.0
    grad_c1 = csdl.derivative(loss_c1, p2_tilted.coefficients)
    assert grad_c1 is not None

    g1_tilted_cond = lfs.NormalAlignmentCondition(interface=inter_tilted)
    loss_g1 = csdl.sum(g1_tilted_cond.evaluate_residual()**2)
    assert loss_g1.value > 0.0
    grad_g1 = csdl.derivative(loss_g1, p2_tilted.coefficients)
    assert grad_g1 is not None


def test_clearance_condition_and_product_domains():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    # 1. Product domain
    d1 = lfs.InterfaceDomain.interval(0.0, 1.0)
    d2 = lfs.InterfaceDomain.interval(2.0, 5.0)
    d_prod = lfs.InterfaceDomain.product(d1, d2)
    assert d_prod.dimension == 2
    np.testing.assert_allclose(d_prod.bounds, [[0.0, 1.0], [2.0, 5.0]])

    # 2. Clearance between two planar functions separated by dz = 0.5
    space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))
    c1 = np.array([[[0., 0., 0.], [0., 1., 0.]], [[1., 0., 0.], [1., 1., 0.]]])
    c2 = np.array([[[0., 0., 0.5], [0., 1., 0.5]], [[1., 0., 0.5], [1., 1., 0.5]]])

    f1 = lfs.Function(space=space, coefficients=c1, name="floor")
    f2 = lfs.Function(space=space, coefficients=c2, name="ceiling")

    m1 = lfs.Marker(entity=f1, parametric_coordinates=np.array([0.5, 0.5]), name="m_floor")
    m2 = lfs.Marker(entity=f2, parametric_coordinates=np.array([0.5, 0.5]), name="m_ceiling")
    inter = lfs.Interface(markers=[m1, m2], name="clearance_interface")

    # Min clearance d_min = 0.2 < 0.5 -> Satisfied
    cond_sat = lfs.ClearanceCondition(interface=inter, min_clearance=0.2, mode="violation", name="clearance_sat")
    viol_sat = cond_sat.evaluate_residual()
    assert np.allclose(viol_sat.value, 0.0, atol=1e-3)
    assert cond_sat.is_linear is False

    # Margin mode for satisfied clearance: dist^2 - d_min^2 = 0.25 - 0.04 = 0.21 > 0
    cond_margin = lfs.ClearanceCondition(interface=inter, min_clearance=0.2, mode="margin", name="margin_cond")
    margin_val = cond_margin.evaluate_residual()
    assert np.isclose(margin_val.value.flatten()[0], 0.21, atol=1e-5)

    # Min clearance d_min = 0.8 > 0.5 -> Violated by 0.8^2 - 0.5^2 = 0.64 - 0.25 = 0.39
    cond_viol = lfs.ClearanceCondition(interface=inter, min_clearance=0.8, mode="violation", name="clearance_viol")
    viol_val = cond_viol.evaluate_residual()
    assert np.isclose(viol_val.value.flatten()[0], 0.39, atol=1e-3)

    # Gradient flow wrt f2 coefficients
    loss = csdl.sum(viol_val**2)
    grad = csdl.derivative(loss, f2.coefficients)
    assert grad is not None
    assert np.max(np.abs(grad.value)) > 0.0


def test_automated_adjacency_discovery():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))
    # Two patches touching at x=1.0: patch 0 right edge (u=1) and patch 1 left edge (u=0)
    c0 = np.array([[[0., 0., 0.], [0., 1., 0.]], [[1., 0., 0.], [1., 1., 0.]]])
    c1 = np.array([[[1., 0., 0.], [1., 1., 0.]], [[2., 0., 0.], [2., 1., 0.]]])

    wing = lfs.FunctionSet(
        functions={0: lfs.Function(space=space, coefficients=c0), 1: lfs.Function(space=space, coefficients=c1)},
        space=space,
        name="wing",
    )

    # 1. Discover interfaces on FunctionSet
    discovered = wing.discover_interfaces(tolerance=1e-5, search_n=6)
    assert len(discovered) == 1
    inter = discovered[0]
    assert inter.domain.dimension == 1
    assert inter.provenance.source_type == "detected"
    assert np.isclose(inter.provenance.measured_gap, 0.0)
    assert inter.provenance.relative_orientation == 1
    assert len(inter.markers) == 2

    # 2. Automated registration and condition generation in FunctionNetwork
    net = lfs.FunctionNetwork(name="wing_network")
    seams = net.discover_internal_seams(wing, tolerance=1e-5, search_n=6, create_conditions=True)
    assert len(seams) == 1
    assert len(net.interfaces) == 1
    assert len(net.conditions) == 1

    # Network topology: both patches now in 1 connected component
    comps = net.compute_connected_components()
    assert len(comps) == 1
    assert len(comps[0]) == 2

    # Residuals along the discovered continuous seam
    res = net.evaluate_residuals()
    assert res.shape == (6, 3)
    np.testing.assert_allclose(res.value, np.zeros((6, 3)), atol=1e-8)


def test_network_rebind_and_copy():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))
    c1 = np.array([[[0., 0., 0.], [0., 1., 0.]], [[1., 0., 0.], [1., 1., 0.]]])
    c2 = np.array([[[1., 0., 0.], [1., 1., 0.]], [[2., 0., 0.], [2., 1., 0.]]])

    p1 = lfs.Function(space=space, coefficients=c1, name="p1")
    p2 = lfs.Function(space=space, coefficients=c2, name="p2")

    m1 = lfs.Marker.from_parametric(p1, np.array([1.0, 0.5]), name="m1")
    m2 = lfs.Marker.from_parametric(p2, np.array([0.0, 0.5]), name="m2")

    iface = lfs.Interface(markers=[m1, m2], name="test_iface")
    cond = lfs.CollocationCondition(interface=iface, name="test_cond")

    net = lfs.FunctionNetwork(name="orig_net")
    net.add_function(p1)
    net.add_function(p2)
    net.add_condition(cond)
    net.component_groupings = {
        "group1": [(p1, None)],
        "group2": [(p2, None)],
    }

    # 1. Test deep copy
    net_copy = net.copy(name="copied_net")
    assert net_copy.name == "copied_net"
    assert len(net_copy.functions) == 2
    assert net_copy.functions[0] is not p1
    assert net_copy.functions[1] is not p2
    assert len(net_copy.conditions) == 1
    assert net_copy.conditions[0] is not cond
    assert net_copy.conditions[0].interface.markers[0].entity is net_copy.functions[0]
    assert net_copy.conditions[0].interface.markers[1].entity is net_copy.functions[1]

    # Verify grouping rebind in copy
    assert net_copy.component_groupings["group1"][0][0] is net_copy.functions[0]
    assert net_copy.component_groupings["group2"][0][0] is net_copy.functions[1]

    # 2. Test rebind with explicit target map
    p1_deformed = lfs.Function(space=space, coefficients=c1 + 1.0, name="p1_deformed")
    p2_deformed = lfs.Function(space=space, coefficients=c2 + 1.0, name="p2_deformed")
    target_map = {p1: p1_deformed, p2: p2_deformed}

    rebound_net = net.rebind(target_map, name="rebound_net")
    assert rebound_net.functions[0] is p1_deformed
    assert rebound_net.functions[1] is p2_deformed
    assert rebound_net.conditions[0].interface.markers[0].entity is p1_deformed
    assert rebound_net.conditions[0].interface.markers[1].entity is p2_deformed

    # Residual on rebound network
    res = rebound_net.evaluate_residuals()
    np.testing.assert_allclose(res.value.flatten(), [0.0, 0.0, 0.0])


def test_higher_dimensional_marker_from_function():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    # Two 2D surface patches in 3D: [0, 1]^2 -> R^3
    space_target = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))
    c1 = np.array([[[0., 0., 0.], [0., 1., 0.]], [[1., 0., 0.], [1., 1., 0.]]])
    c2 = np.array([[[0., 0., 0.], [0., 1., 0.]], [[1., 0., 0.], [1., 1., 0.]]])

    s1 = lfs.Function(space=space_target, coefficients=c1, name="surface_1")
    s2 = lfs.Function(space=space_target, coefficients=c2, name="surface_2")

    # Define a 2D surface marker map alpha: [0, 1]^2 -> [0, 1]^2
    map_space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))
    c_map = np.array([
        [[0.2, 0.3], [0.2, 0.7]],
        [[0.8, 0.3], [0.8, 0.7]],
    ])
    alpha_fn = lfs.Function(space=map_space, coefficients=c_map, name="subsurface_map")

    # 1. Create 2D Marker using from_function
    m1 = lfs.Marker.from_function(entity=s1, function_map=alpha_fn, name="subsurface_m1")
    m2 = lfs.Marker.from_function(entity=s2, function_map=alpha_fn, name="subsurface_m2")

    assert m1.domain.dimension == 2
    assert m1.domain.is_finite is False
    assert m2.domain.dimension == 2
    np.testing.assert_allclose(m1.domain.bounds, [[0.0, 1.0], [0.0, 1.0]])

    # 2. Evaluate marker with default Discretization sampling (10 x 10 = 100 points)
    eval_m1 = m1.evaluate()
    assert eval_m1.shape == (100, 3)

    # 3. Create 2D surface Interface with explicit tensor-product quadrature (4 x 5 = 20 nodes)
    discr_2d = lfs.Discretization(num_points=(4, 5), scheme="gauss_legendre")
    iface_2d = lfs.Interface(markers=[m1, m2], discretization=discr_2d, name="2d_contact_interface")
    assert iface_2d.domain.dimension == 2

    nodes, weights = iface_2d.get_samples()
    assert nodes.shape == (20, 2)
    assert weights.shape == (20,)
    assert np.isclose(np.sum(weights), 1.0)

    # 4. Enforce 2D CollocationCondition across the entire sub-surface area
    cond_2d = lfs.CollocationCondition(interface=iface_2d, weighted=True, name="2d_surface_bonding")
    res_2d = cond_2d.evaluate_residual()
    assert res_2d.shape == (20, 3)
    np.testing.assert_allclose(res_2d.value, np.zeros((20, 3)), atol=1e-8)

    # 5. Check trace operator caching and freedom analysis on 2D interface
    T1 = iface_2d.get_trace_operator(0)
    T2 = iface_2d.get_trace_operator(1)
    assert T1.shape == (20, 4)
    assert T2.shape == (20, 4)

    net_2d = lfs.FunctionNetwork(name="2d_interface_network")
    net_2d.add_condition(cond_2d)
    report_2d = net_2d.analyze_freedom()
    # 2 surfaces * 4 control points * 3 dims = 24 vars
    assert report_2d.num_variables == 24
    # 20 points * 3 dims = 60 constraints
    assert report_2d.num_constraints == 60
    # Over 2D surface with degree (1, 1), 4 control points * 3 dims = 12 independent DOFs
    assert report_2d.rank == 12
    assert report_2d.redundancy == 48
    assert report_2d.nullity == 12


def test_ex_connectivity_network_pipeline():
    """Verifies that the standalone example script ex_connectivity_network.py executes cleanly."""
    from examples.basic_examples.ex_connectivity_network import (
        run_synthetic_connectivity_showcase,
        run_cad_seam_discovery_showcase,
    )
    import os

    net_synthetic = run_synthetic_connectivity_showcase(plot=False)
    assert len(net_synthetic.functions) == 4
    assert len(net_synthetic.interfaces) == 3
    assert len(net_synthetic.conditions) == 4

    stp_path = os.path.join(os.path.dirname(__file__), "..", "examples", "import_files_for_examples", "rectangular_wing.stp")
    net_cad = run_cad_seam_discovery_showcase(stp_path=stp_path)
    assert len(net_cad.conditions) == 18








