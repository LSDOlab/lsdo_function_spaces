"""Unit tests for utility functions, geometry construction helpers, and CAD import."""

import os
import pytest
import numpy as np
import csdl_alpha as csdl
import lsdo_function_spaces as lfs
from lsdo_function_spaces.utils.plotting_functions import (
    _normalize_points,
    _normalize_grid,
    _extract_scalars,
)

SAMPLE_STP = os.path.join(
    os.path.dirname(__file__), "..", "examples", "import_files_for_examples", "rectangular_wing.stp"
)


def test_create_b_spline_from_corners():
    """Test constructing a 3D B-spline volume from bounding corners."""
    # 2x2x2 corner bounding box
    corners = np.zeros((2, 2, 2, 3))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                corners[i, j, k, :] = [float(i), float(j), float(k)]

    bspline = lfs.create_b_spline_from_corners(
        corners,
        degree=(2, 2, 2),
        num_coefficients=(4, 4, 4),
        name="test_box",
    )
    assert bspline.name == "test_box"
    assert bspline.space.num_parametric_dimensions == 3
    assert bspline.coefficients.shape == (4, 4, 4, 3)


def test_create_enclosure_block():
    """Test constructing an enclosure volume around point cloud."""
    pts = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 5.0, 2.0],
        [5.0, 2.5, 1.0],
    ])
    block = lfs.create_enclosure_block(
        points=pts,
        num_coefficients=(4, 4, 4),
        degree=(2, 2, 2),
        name="enclosure",
    )
    assert block.name == "enclosure"
    assert block.coefficients.shape == (4, 4, 4, 3)


def test_import_file_step():
    """Test parsing an OpenVSP STEP CAD file into a FunctionSet."""
    if not os.path.exists(SAMPLE_STP):
        pytest.skip("Sample STEP file not found.")

    fset = lfs.import_file(SAMPLE_STP, parallelize=False, name="imported_wing")
    assert fset.name == "imported_wing"
    assert len(fset.functions) > 0

    # Ensure all surfaces are pure-Python BSplineSpace
    for fn in fset.functions.values():
        assert isinstance(fn.space, lfs.BSplineSpace)


def test_plotting_utilities_normalization():
    """Test point normalization and scalar extraction utilities."""
    # 2D points -> padded to 3D with zeros
    pts2d = np.array([[1.0, 2.0], [3.0, 4.0]])
    norm_pts = _normalize_points(pts2d)
    assert norm_pts.shape == (2, 3)
    np.testing.assert_allclose(norm_pts[:, 2], 0.0)

    # 1D scalars
    color_1d = np.array([0.5, 0.8])
    scalars, is_rgb = _extract_scalars(color_1d, 2)
    assert not is_rgb
    np.testing.assert_allclose(scalars, [0.5, 0.8])

    # Validation errors on invalid shapes
    with pytest.raises(ValueError, match="must have 3 or fewer physical dimensions"):
        _normalize_points(np.zeros((5, 4)))


def test_create_b_spline_from_corners_scalar_args_and_multi_section():
    """Test scalar args, explicit knot vectors, and multi-section grid interpolation."""
    # 3x2 corners (3 control points along axis 0 tests intermediate section linspace)
    corners = np.zeros((3, 2, 2))
    corners[0, 0] = [0.0, 0.0]
    corners[0, 1] = [0.0, 1.0]
    corners[1, 0] = [0.5, 0.0]
    corners[1, 1] = [0.5, 1.0]
    corners[2, 0] = [1.0, 0.0]
    corners[2, 1] = [1.0, 1.0]

    kv0 = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    bspline = lfs.create_b_spline_from_corners(
        corners,
        degree=2,
        num_coefficients=3,
        knot_vectors=(kv0, kv0),
        name="test_corners_scalar",
    )
    assert bspline.coefficients.shape == (5, 3, 2)


def test_create_enclosure_block_scalar_args():
    """Test scalar degree and num_coefficients for enclosure blocks."""
    pts = np.array([[0.0, 0.0], [1.0, 1.0]])
    block = lfs.create_enclosure_block(
        points=pts,
        num_parametric_dimensions=2,
        num_coefficients=4,
        degree=2,
        name="enclosure_2d",
    )
    assert block.coefficients.shape == (4, 4, 2)

