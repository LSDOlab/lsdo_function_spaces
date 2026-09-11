"""Unit tests for PyVista plotting utilities in utils/plotting_functions.py."""

import pytest
import numpy as np
import pyvista as pv
import lsdo_function_spaces as lfs
from lsdo_function_spaces.utils.plotting_functions import (
    _normalize_points,
    _normalize_grid,
    _extract_scalars,
    _make_plot_element,
    _flatten_plotting_elements,
    plot_points,
    plot_curve,
    plot_surface,
    get_surface_mesh,
    make_scalar_bar_element,
    show_plot,
)


def test_normalize_grid():
    """Test 2D grid normalization to 3D grid."""
    grid_2d = np.ones((5, 5, 2))
    norm_grid = _normalize_grid(grid_2d)
    assert norm_grid.shape == (5, 5, 3)
    np.testing.assert_allclose(norm_grid[:, :, 2], 0.0)

    # 3D grid unchanged
    grid_3d = np.ones((4, 4, 3))
    assert _normalize_grid(grid_3d).shape == (4, 4, 3)

    # Invalid dimension raises
    with pytest.raises(ValueError, match="must have 3 or fewer physical dimensions"):
        _normalize_grid(np.ones((4, 4, 4)))


def test_flatten_and_make_element():
    """Test _flatten_plotting_elements and _make_plot_element."""
    mesh = pv.PolyData()
    elem1 = _make_plot_element(mesh, color="red")
    elem2 = _make_plot_element(mesh, color="blue")

    nested = [elem1, [elem2]]
    flat = _flatten_plotting_elements(nested)
    assert len(flat) == 2
    assert flat[0]["kwargs"]["color"] == "red"
    assert flat[1]["kwargs"]["color"] == "blue"


def test_plot_points_and_curve():
    """Test plot_points and plot_curve off-screen."""
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 0.0, 1.0]])
    scalars = np.array([0.1, 0.5, 0.9])

    # Points with scalar coloring
    elems_pts = plot_points(pts, color=scalars, color_map="viridis", show=False)
    assert len(elems_pts) == 1

    # Curve with string color
    elems_crv = plot_curve(pts, color="#FF0000", line_width=2.0, show=False)
    assert len(elems_crv) == 1


def test_plot_surface_and_mesh():
    """Test plot_surface and get_surface_mesh off-screen."""
    u = np.linspace(0, 1, 6)
    v = np.linspace(0, 1, 6)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    zz = uu * vv
    grid = np.stack([uu, vv, zz], axis=-1)

    # Plot surface as function and wireframe
    elems_surf = plot_surface(grid, plot_types=["function", "wireframe"], show=False)
    assert len(elems_surf) == 2

    # get_surface_mesh
    s = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))
    ctrl = np.zeros((2, 2, 3))
    surf_fn = lfs.Function(space=s, coefficients=ctrl)
    vertices, faces = get_surface_mesh(surf_fn, grid_n=5)
    assert len(vertices) == 25
    assert len(faces) == 16


def test_make_scalar_bar_element_and_show():
    """Test make_scalar_bar_element and show_plot off-screen."""
    bar = make_scalar_bar_element(color_min=0.0, color_max=10.0, color_map="plasma")
    assert bar["kwargs"]["show_scalar_bar"] is True
    assert bar["kwargs"]["cmap"] == "plasma"

    # show_plot with elements (runs headless)
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    elems = plot_points(pts, show=False)
    # Add scalar bar
    elems.append(bar)
    # Should execute without error
    show_plot(elems, title="Test Plot", interactive=False)
