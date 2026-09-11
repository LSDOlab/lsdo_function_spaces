"""Unit tests for FunctionSet and FunctionSetSpace multi-surface representations."""

import pytest
import numpy as np
import csdl_alpha as csdl
import lsdo_function_spaces as lfs


def test_function_set_initialization():
    """Test initializing a FunctionSet from a list and dict of Functions."""
    s1 = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(2, 2), coefficients_shape=(3, 3))
    s2 = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(2, 2), coefficients_shape=(3, 3))

    f1 = lfs.Function(space=s1, coefficients=np.zeros((3, 3, 3)), name="surf_1")
    f2 = lfs.Function(space=s2, coefficients=np.ones((3, 3, 3)), name="surf_2")

    # From list
    fset_list = lfs.FunctionSet([f1, f2], name="multi_surf")
    assert len(fset_list.functions) == 2
    assert fset_list.name == "multi_surf"
    assert isinstance(fset_list.space, lfs.FunctionSetSpace)

    # From dict
    fset_dict = lfs.FunctionSet({0: f1, 1: f2})
    assert len(fset_dict.functions) == 2


def test_function_set_evaluate():
    """Test evaluation of a FunctionSet with paired (patch_index, coordinate) tuples."""
    s1 = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(2,))
    s2 = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(2,))

    f1 = lfs.Function(space=s1, coefficients=np.array([[0.0], [2.0]]))
    f2 = lfs.Function(space=s2, coefficients=np.array([[10.0], [20.0]]))

    fset = lfs.FunctionSet([f1, f2])
    coords = [(0, np.array([0.5])), (1, np.array([0.5]))]

    res = fset.evaluate(coords)
    assert isinstance(res, csdl.Variable)
    np.testing.assert_allclose(res.value, [1.0, 15.0])


def test_function_set_copy():
    """Test copying a FunctionSet."""
    s1 = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(2, 2), coefficients_shape=(3, 3))
    f1 = lfs.Function(space=s1, coefficients=np.ones((3, 3, 3)), name="orig")
    fset = lfs.FunctionSet([f1], name="set_orig")

    fset_copy = fset.copy()
    assert fset_copy.name == "set_orig"
    assert len(fset_copy.functions) == 1


def test_function_set_stack_unstack():
    """Test stack_coefficients and unstack_coefficients."""
    s1 = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(2, 2), coefficients_shape=(3, 3))
    s2 = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(2, 2), coefficients_shape=(4, 4))

    f1 = lfs.Function(space=s1, coefficients=np.ones((3, 3, 3)) * 2.0)
    f2 = lfs.Function(space=s2, coefficients=np.ones((4, 4, 3)) * 3.0)
    fset = lfs.FunctionSet([f1, f2])

    stacked = fset.stack_coefficients()
    assert isinstance(stacked, csdl.Variable)
    assert stacked.shape == (25, 3)

    # Unstack with new values
    new_vals = csdl.Variable(value=np.zeros(stacked.shape))
    fset.unstack_coefficients(new_vals)
    np.testing.assert_allclose(fset.functions[0].coefficients.value, 0.0)
    np.testing.assert_allclose(fset.functions[1].coefficients.value, 0.0)


def test_function_set_subset_and_search():
    """Test get_function_indices, search_for_function_indices, and create_subset."""
    s = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))
    f1 = lfs.Function(space=s, coefficients=np.zeros((2, 2, 3)), name="wing_upper")
    f2 = lfs.Function(space=s, coefficients=np.zeros((2, 2, 3)), name="wing_lower")
    f3 = lfs.Function(space=s, coefficients=np.zeros((2, 2, 3)), name="tail_fin")
    fset = lfs.FunctionSet([f1, f2, f3])

    # get_function_indices
    indices = fset.get_function_indices(["wing_upper", "tail_fin"])
    assert indices == [0, 2]

    # search_for_function_indices
    wing_indices = fset.search_for_function_indices(["wing"])
    assert wing_indices == [0, 1]

    # create_subset
    subset = fset.create_subset(function_indices=[0, 1], name="wing_only")
    assert len(subset.functions) == 2
    assert subset.name == "wing_only"


def test_function_set_generate_grid_and_normals():
    """Test generate_parametric_grid and evaluate_normals on FunctionSet."""
    s = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))
    ctrl = np.zeros((2, 2, 3))
    ctrl[0, 0, :2] = [0.0, 0.0]
    ctrl[0, 1, :2] = [0.0, 1.0]
    ctrl[1, 0, :2] = [1.0, 0.0]
    ctrl[1, 1, :2] = [1.0, 1.0]
    # Flat horizontal surface, normal should point along +z or -z
    f = lfs.Function(space=s, coefficients=ctrl)
    fset = lfs.FunctionSet([f])

    grid = fset.generate_parametric_grid(grid_resolution=(3, 3))
    assert len(grid) == 9

    normals = fset.evaluate_normals([(0, np.array([0.5, 0.5]))])
    assert isinstance(normals, csdl.Variable)
    # Unit normal along z axis
    norm_val = np.abs(normals.value[0])
    np.testing.assert_allclose(norm_val, [0.0, 0.0, 1.0], atol=1e-6)


def test_function_set_integrate():
    """Test FunctionSet integrate."""
    s = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))
    ctrl = np.zeros((2, 2, 3))
    ctrl[0, 0, :2] = [0.0, 0.0]
    ctrl[0, 1, :2] = [0.0, 1.0]
    ctrl[1, 0, :2] = [1.0, 0.0]
    ctrl[1, 1, :2] = [1.0, 1.0]
    area_f = lfs.Function(space=s, coefficients=ctrl)
    area_set = lfs.FunctionSet([area_f])

    val_f = lfs.Function(space=s, coefficients=np.ones((2, 2, 1)) * 3.0)
    val_set = lfs.FunctionSet([val_f])

    integral, centers = val_set.integrate(area_set, grid_n=4)
    assert integral is not None
    total = np.sum(integral.value)
    np.testing.assert_allclose(total, 3.0, rtol=0.1)


def test_function_set_operations():
    """Test arithmetic operations between FunctionSets."""
    s = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(2,))
    f1 = lfs.Function(space=s, coefficients=np.array([[1.0], [2.0]]))
    f2 = lfs.Function(space=s, coefficients=np.array([[3.0], [4.0]]))

    set1 = lfs.FunctionSet([f1])
    set2 = lfs.FunctionSet([f2])

    sum_set = set1 + set2
    diff_set = set1 - set2
    prod_set = set1 * set2
    div_set = set1 / set2

    u = np.array([[0.5]])
    np.testing.assert_allclose(sum_set.functions[0].evaluate(u).value, [5.0])
    np.testing.assert_allclose(diff_set.functions[0].evaluate(u).value, [-2.0])
    np.testing.assert_allclose(prod_set.functions[0].evaluate(u).value, [1.5 * 3.5])
    np.testing.assert_allclose(div_set.functions[0].evaluate(u).value, [1.5 / 3.5])


def test_function_set_plotting():
    """Test FunctionSet plot and plot_but_good methods off-screen."""
    s = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))
    ctrl = np.zeros((2, 2, 3))
    ctrl[1, 1] = [1.0, 1.0, 0.5]
    f = lfs.Function(space=s, coefficients=ctrl, name="test_patch")
    fset = lfs.FunctionSet([f])

    elems = fset.plot(interactive=False)
    assert len(elems) > 0

    # plot_but_good
    fset.plot_but_good(show=False, grid_n=5)


def test_find_best_surface_chunked():
    """Test find_best_surface_chunked with various options (extrema, direction, priority)."""
    from lsdo_function_spaces.core.function_set import find_best_surface_chunked

    s1 = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))
    ctrl1 = np.zeros((2, 2, 3))
    ctrl1[0, 0, :2] = [0.0, 0.0]
    ctrl1[0, 1, :2] = [0.0, 1.0]
    ctrl1[1, 0, :2] = [1.0, 0.0]
    ctrl1[1, 1, :2] = [1.0, 1.0]
    f1 = lfs.Function(space=s1, coefficients=ctrl1)

    s2 = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))
    ctrl2 = np.zeros((2, 2, 3))
    ctrl2[0, 0, :2] = [10.0, 10.0]
    ctrl2[0, 1, :2] = [10.0, 11.0]
    ctrl2[1, 0, :2] = [11.0, 10.0]
    ctrl2[1, 1, :2] = [11.0, 11.0]
    f2 = lfs.Function(space=s2, coefficients=ctrl2)

    funcs = {0: f1, 1: f2}
    pts = np.array([[0.5, 0.5, 1.0]])

    # 1. Standard projection
    res1 = find_best_surface_chunked(pts, functions=funcs, options={})
    assert len(res1) == 1
    assert res1[0][0] == 0  # Closest to patch 0

    # 2. Extrema projection
    res2 = find_best_surface_chunked(pts, functions=funcs, options={'extrema': True})
    assert len(res2) == 1
    assert res2[0][0] == 0

    # 3. With direction and priority_inds
    opts = {
        'direction': np.array([0.0, 0.0, 1.0]),
        'priority_inds': [0],
        'priority_eps': 0.1,
    }
    res3 = find_best_surface_chunked(pts, functions=funcs, options=opts)
    assert len(res3) == 1
    assert res3[0][0] == 0


def test_function_set_find_surface_connections():
    """Test finding topology connections between adjacent B-spline patches."""
    s1 = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))
    c1 = np.zeros((2, 2, 3))
    c1[0, 0, :2] = [0.0, 0.0]
    c1[0, 1, :2] = [0.0, 1.0]
    c1[1, 0, :2] = [1.0, 0.0]
    c1[1, 1, :2] = [1.0, 1.0]
    f1 = lfs.Function(space=s1, coefficients=c1, name="patch_left")

    s2 = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))
    c2 = np.zeros((2, 2, 3))
    # Sharing edge at x = 1
    c2[0, 0, :2] = [1.0, 0.0]
    c2[0, 1, :2] = [1.0, 1.0]
    c2[1, 0, :2] = [2.0, 0.0]
    c2[1, 1, :2] = [2.0, 1.0]
    f2 = lfs.Function(space=s2, coefficients=c2, name="patch_right")

    fset = lfs.FunctionSet([f1, f2])
    conns = fset.find_surface_connections()
    assert len(conns) >= 1


def test_function_set_refit():
    """Test refitting a FunctionSet into a new function space."""
    s = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))
    c = np.zeros((2, 2, 3))
    f = lfs.Function(space=s, coefficients=c)
    fset = lfs.FunctionSet([f])

    new_s = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(2, 2), coefficients_shape=(3, 3))
    fset_refit = fset.refit(new_s, grid_resolution=(4, 4))
    assert isinstance(fset_refit, lfs.FunctionSet)
    assert len(fset_refit.functions) == 1

