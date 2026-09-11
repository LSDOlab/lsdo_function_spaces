"""Unit tests for STEP / CAD file I/O and parsing utilities in utils/file_io.py."""

import os
import tempfile
import pytest
import numpy as np
import lsdo_function_spaces as lfs
from lsdo_function_spaces.utils.file_io import (
    _to_float,
    _split_top_level_commas,
    _strip_outer_parens,
    _parse_int_list,
    _parse_float_list,
    _parse_cartesian_point,
    _parse_bspline_surface_with_knots,
    _expand_knots,
    _normalize_knots_affine,
    _space_key,
    _read_step_entities,
    import_file,
    import_file_patched,
)


def test_parsing_primitive_helpers():
    """Test string parsing and token extraction helpers."""
    assert _to_float("1.23D+02") == 123.0
    assert _to_float("-4.56d-01") == -0.456
    assert _strip_outer_parens("(hello, world)") == "hello, world"
    assert _strip_outer_parens("no_parens") == "no_parens"

    # _split_top_level_commas
    res = _split_top_level_commas("a, (b, c), 'd, e', f")
    assert res == ["a", "(b, c)", "'d, e'", "f"]

    # _parse_int_list
    assert _parse_int_list("(1, 2, 3, 4)") == [1, 2, 3, 4]

    # _parse_float_list
    assert _parse_float_list("(1.0, 2.5, -3.14)") == [1.0, 2.5, -3.14]


def test_parse_cartesian_point():
    """Test parsing CARTESIAN_POINT entities."""
    rhs = "CARTESIAN_POINT('origin', (0.0, 1.5, -2.0))"
    pt = _parse_cartesian_point(rhs)
    assert pt is not None
    np.testing.assert_allclose(pt, [0.0, 1.5, -2.0])

    assert _parse_cartesian_point("VERTEX_POINT(...)") is None


def test_parse_bspline_surface_with_knots():
    """Test parsing B_SPLINE_SURFACE_WITH_KNOTS entity string."""
    rhs = (
        "B_SPLINE_SURFACE_WITH_KNOTS('test_surf', 1, 1, "
        "((#10, #11), (#12, #13)), "
        ".UNSPECIFIED., .F., .F., .F., "
        "(2, 2), (2, 2), (0.0, 1.0), (0.0, 1.0), .UNSPECIFIED.)"
    )
    parsed = _parse_bspline_surface_with_knots(rhs)
    assert parsed is not None
    name, deg_u, deg_v, cp_ids, u_mults, v_mults, u_knots, v_knots = parsed
    assert name == "test_surf"
    assert deg_u == 1
    assert deg_v == 1
    assert cp_ids.shape == (2, 2)
    assert u_mults == [2, 2]
    assert v_mults == [2, 2]
    assert u_knots == [0.0, 1.0]
    assert v_knots == [0.0, 1.0]


def test_knot_expansion_and_normalization():
    """Test expanding knot multiplicities and affine normalizing."""
    base_knots = [0.0, 0.5, 1.0]
    mults = [3, 1, 3]
    expanded = _expand_knots(base_knots, mults)
    expected = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(expanded, expected)

    # Affine normalization from [10, 20] to [0, 1]
    unnormalized = np.array([10.0, 10.0, 15.0, 20.0, 20.0])
    normed = _normalize_knots_affine(unnormalized)
    np.testing.assert_allclose(normed, [0.0, 0.0, 0.5, 1.0, 1.0])

    # Constant knot vector edge case (degenerate: returns copy)
    const_knots = np.array([5.0, 5.0])
    np.testing.assert_allclose(_normalize_knots_affine(const_knots), [5.0, 5.0])


def test_space_key():
    """Test generation of cache keys for B-spline spaces."""
    key = _space_key(2, 3, [3, 3], [4, 4], [0.0, 1.0], [0.0, 1.0])
    assert isinstance(key, str)
    assert len(key) == 32


def test_read_step_entities_multiline():
    """Test reading multiline STEP entities from a temporary file."""
    content = (
        "ISO-10303-21;\n"
        "HEADER;\n"
        "ENDSEC;\n"
        "DATA;\n"
        "#1 = CARTESIAN_POINT('pt1', (1.0,\n"
        "  2.0,\n"
        "  3.0));\n"
        "#2 = CARTESIAN_POINT('pt2', (4.0, 5.0, 6.0));\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n"
    )
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".stp", delete=False) as f:
        f.write(content)
        temp_path = f.name

    try:
        entities = _read_step_entities(temp_path)
        assert 1 in entities
        assert 2 in entities
        pt1 = _parse_cartesian_point(entities[1])
        np.testing.assert_allclose(pt1, [1.0, 2.0, 3.0])
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_import_file_invalid():
    """Test error handling when importing file without B-splines."""
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".stp", delete=False) as f:
        f.write("ISO-10303-21; DATA; #1=CARTESIAN_POINT('',(0,0,0)); ENDSEC;")
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="No B_SPLINE_SURFACE_WITH_KNOTS"):
            import_file(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_import_file_synthetic_complete():
    """Test end-to-end import of a synthetic STEP file containing a B-spline surface."""
    step_content = """ISO-10303-21;
HEADER;
ENDSEC;
DATA;
#10 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#11 = CARTESIAN_POINT('', (0.0, 1.0, 0.0));
#12 = CARTESIAN_POINT('', (1.0, 0.0, 0.0));
#13 = CARTESIAN_POINT('', (1.0, 1.0, 0.0));
#20 = B_SPLINE_SURFACE_WITH_KNOTS('synthetic_patch', 1, 1,
  ((#10, #11), (#12, #13)),
  .UNSPECIFIED., .F., .F., .F.,
  (2, 2), (2, 2), (0.0, 1.0), (0.0, 1.0), .UNSPECIFIED.);
ENDSEC;
END-ISO-10303-21;
"""
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".stp", delete=False) as f:
        f.write(step_content)
        temp_path = f.name

    try:
        fset = import_file(temp_path, name="synth_set")
        assert fset.name == "synth_set"
        assert len(fset.functions) == 1
        fn = list(fset.functions.values())[0]
        assert fn.name == "synthetic_patch"
        assert fn.space.degree == (1, 1)

        # Test import_file_patched deprecation warning
        with pytest.deprecated_call():
            fset_deprecated = import_file_patched(temp_path, name="synth_dep")
            assert fset_deprecated is not None
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
