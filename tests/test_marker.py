import numpy as np
import pytest
import csdl_alpha as csdl
import lsdo_function_spaces as lfs


def test_marker_on_function():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))
    c1 = np.array([[[0., 0., 0.], [0., 1., 0.]], [[1., 0., 0.], [1., 1., 0.]]])
    f = lfs.Function(space=space, coefficients=c1, name='test_func')

    # Add marker with direct parametric coordinates
    m1 = f.add_marker(np.array([0.5, 0.5]), name='center')
    assert m1.name == 'center'
    assert m1.name in f.markers
    assert f.markers['center'] is m1

    # Evaluate physical value
    pos = m1.evaluate()
    assert np.allclose(pos.value, [0.5, 0.5, 0.0])

    # Evaluate derivatives df/du and df/dv via evaluate_derivatives and evaluate(parametric_derivative_orders=...)
    df_du = m1.evaluate_derivatives((1, 0))
    df_dv = m1.evaluate_derivatives((0, 1))
    assert np.allclose(df_du.value, [1.0, 0.0, 0.0])
    assert np.allclose(df_dv.value, [0.0, 1.0, 0.0])

    df_du_eval = m1.evaluate(parametric_derivative_orders=(1, 0))
    df_dv_eval = m1.evaluate(parametric_derivative_orders=(0, 1))
    assert np.allclose(df_du_eval.value, [1.0, 0.0, 0.0])
    assert np.allclose(df_dv_eval.value, [0.0, 1.0, 0.0])

    # Marker with optional name
    m2 = f.add_marker(np.array([0.25, 0.75]))
    assert m2.name is not None
    assert m2.name in f.markers


def test_marker_on_function_set():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))
    c1 = np.array([[[0., 0., 0.], [0., 1., 0.]], [[1., 0., 0.], [1., 1., 0.]]])
    f = lfs.Function(space=space, coefficients=c1)
    fs = lfs.FunctionSet(functions={0: f}, name='test_set')

    # Add marker with direct parametric coordinates tuple (function_index, uv)
    m1 = fs.add_marker((0, np.array([0.5, 0.5])), name='mid')
    assert m1.name == 'mid'
    assert m1.name in fs.markers
    assert fs.markers['mid'] is m1

    pos = m1.evaluate()
    assert np.allclose(pos.value, [0.5, 0.5, 0.0])

    # Derivatives on function set via evaluate_derivatives and evaluate(parametric_derivative_orders=...)
    df_du = m1.evaluate_derivatives((1, 0))
    df_dv = m1.evaluate_derivatives((0, 1))
    assert np.allclose(df_du.value, [1.0, 0.0, 0.0])
    assert np.allclose(df_dv.value, [0.0, 1.0, 0.0])

    df_du_eval = m1.evaluate(parametric_derivative_orders=(1, 0))
    df_dv_eval = m1.evaluate(parametric_derivative_orders=(0, 1))
    assert np.allclose(df_du_eval.value, [1.0, 0.0, 0.0])
    assert np.allclose(df_dv_eval.value, [0.0, 1.0, 0.0])

    # Marker with list of tuples format and optional name
    m2 = fs.add_marker([(0, np.array([0.25, 0.75]))])
    assert m2.name is not None
    assert m2.name in fs.markers
    pos2 = m2.evaluate()
    assert np.allclose(pos2.value, [0.25, 0.75, 0.0])

    # Marker with single tuple format
    m3 = fs.add_marker((0, np.array([0.1, 0.2])), name='tuple_marker')
    assert m3.name == 'tuple_marker'
    assert m3.name in fs.markers
    pos3 = m3.evaluate()
    assert np.allclose(pos3.value, [0.1, 0.2, 0.0])
