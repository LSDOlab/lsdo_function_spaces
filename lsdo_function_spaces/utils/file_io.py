"""STEP file importer (patched).

Correctness fixes:
- Robust entity parsing across line wraps (entities end with ';')
- Correct float parsing for knots/control points (supports E/e and D/d exponents)
- Correct knot expansion with multiplicities
- Order-preserving control point lookup via id->xyz dictionary

Performance improvements:
- Single pass read: builds entity dict and point dict once
- Avoids pandas and global regex scans
- Caches BSplineSpace objects by a stable hash key

Intended primarily for OpenVSP-exported STEP files containing B_SPLINE_SURFACE_WITH_KNOTS.

Public API:
- import_file(...)
- _check_if_load_stored_import(...)
"""

from __future__ import annotations

import os
import re
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import lsdo_function_spaces as lfs
import csdl_alpha as csdl


# ---------------------------- Numeric parsing -----------------------------

_FLOAT_RE = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][+-]?\d+)?")
_INT_RE = re.compile(r"[+-]?\d+")


def _to_float(tok: str) -> float:
    return float(tok.replace('D', 'E').replace('d', 'e'))


# ---------------------------- STEP parsing -------------------------------

def _read_step_entities(file_name: str) -> Dict[int, str]:
    """Return a dict {id: rhs_text} where rhs_text excludes the trailing ';'.

    Handles entities that span multiple lines by accumulating until ';'.
    """
    entities: Dict[int, str] = {}
    cur_id: Optional[int] = None
    buf: List[str] = []

    with open(file_name, 'r') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if cur_id is None:
                if not line.startswith('#'):
                    continue
                eq = line.find('=')
                if eq < 0:
                    continue
                try:
                    cur_id = int(line[1:eq].strip())
                except ValueError:
                    cur_id = None
                    continue
                rest = line[eq+1:].strip()
                buf = [rest]
            else:
                buf.append(line)

            if buf and buf[-1].endswith(';'):
                joined = ' '.join(buf).strip()
                joined = joined[:-1].strip()  # drop trailing ';'
                entities[cur_id] = joined
                cur_id = None
                buf = []

    return entities


def _split_top_level_commas(s: str) -> List[str]:
    """Split by commas not inside parentheses or quoted strings."""
    out: List[str] = []
    depth = 0
    start = 0
    in_quote = False
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "'":
            if in_quote:
                if i + 1 < len(s) and s[i + 1] == "'":
                    i += 2
                    continue
                in_quote = False
            else:
                in_quote = True
        elif not in_quote:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            elif ch == ',' and depth == 0:
                out.append(s[start:i].strip())
                start = i + 1
        i += 1
    out.append(s[start:].strip())
    return out


def _strip_outer_parens(s: str) -> str:
    s = s.strip()
    if s.startswith('(') and s.endswith(')'):
        return s[1:-1].strip()
    return s


def _parse_int_list(step_list: str) -> List[int]:
    body = _strip_outer_parens(step_list)
    toks = _INT_RE.findall(body)
    return [int(t) for t in toks]


def _parse_float_list(step_list: str) -> List[float]:
    body = _strip_outer_parens(step_list)
    toks = _FLOAT_RE.findall(body)
    return [_to_float(t) for t in toks]


def _parse_cartesian_point(rhs: str) -> Optional[np.ndarray]:
    """Parse CARTESIAN_POINT('',(x,y,z)) -> np.array([x,y,z])"""
    if not rhs.startswith('CARTESIAN_POINT'):
        return None
    # grab all floats; take the last 3
    vals = [_to_float(t) for t in _FLOAT_RE.findall(rhs)]
    if len(vals) < 3:
        return None
    return np.array(vals[-3:], dtype=float)


def _parse_control_point_grid(field: str) -> np.ndarray:
    """Parse control point grid '((#1,#2),(#3,#4))' -> int array shape (nu,nv)."""
    body = _strip_outer_parens(field)
    rows = _split_top_level_commas(body)
    grid: List[List[int]] = []
    for r in rows:
        r_body = _strip_outer_parens(r)
        ids = [int(x[1:]) for x in re.findall(r"#\d+", r_body)]
        if ids:
            grid.append(ids)
    if not grid:
        raise ValueError('Failed to parse control point grid.')
    # Ensure rectangular
    n0 = len(grid[0])
    for rr in grid:
        if len(rr) != n0:
            raise ValueError('Control point grid is not rectangular.')
    return np.array(grid, dtype=int)


def _detect_knot_fields(fields: List[str]) -> Tuple[List[int], List[int], List[float], List[float]]:
    """Identify (u_mults, v_mults, u_knots, v_knots) from argument fields.

    OpenVSP typically uses indices 8..11, but exporters vary. We look for
    two int-lists followed by two float-lists among the parenthesized fields.
    """
    # First try the OpenVSP expected slots
    try:
        u_mults = _parse_int_list(fields[8])
        v_mults = _parse_int_list(fields[9])
        u_knots = _parse_float_list(fields[10])
        v_knots = _parse_float_list(fields[11])
        if len(u_mults) == len(u_knots) and len(v_mults) == len(v_knots):
            return u_mults, v_mults, u_knots, v_knots
    except Exception:
        pass

    # Otherwise scan for pattern: int list, int list, float list, float list
    parenth_idxs = [i for i, f in enumerate(fields) if f.strip().startswith('(')]
    parsed_int: Dict[int, List[int]] = {}
    parsed_float: Dict[int, List[float]] = {}

    for i in parenth_idxs:
        txt = fields[i]
        # Heuristic: if contains '.' or 'E'/'D' treat as float list
        if any(c in txt for c in ['.', 'E', 'e', 'D', 'd']):
            try:
                parsed_float[i] = _parse_float_list(txt)
            except Exception:
                continue
        else:
            try:
                parsed_int[i] = _parse_int_list(txt)
            except Exception:
                continue

    # Find int,int,float,float sequence
    idxs = sorted(parenth_idxs)
    for a in idxs:
        if a not in parsed_int:
            continue
        for b in idxs:
            if b <= a or b not in parsed_int:
                continue
            for c in idxs:
                if c <= b or c not in parsed_float:
                    continue
                for d in idxs:
                    if d <= c or d not in parsed_float:
                        continue
                    u_mults = parsed_int[a]
                    v_mults = parsed_int[b]
                    u_knots = parsed_float[c]
                    v_knots = parsed_float[d]
                    if len(u_mults) == len(u_knots) and len(v_mults) == len(v_knots):
                        return u_mults, v_mults, u_knots, v_knots

    raise ValueError('Could not locate knot and multiplicity fields in B_SPLINE_SURFACE_WITH_KNOTS.')


def _parse_bspline_surface_with_knots(rhs: str) -> Optional[Tuple[str, int, int, np.ndarray, List[int], List[int], List[float], List[float]]]:
    """Parse B_SPLINE_SURFACE_WITH_KNOTS entity RHS.

    Returns:
        (name, deg_u, deg_v, cp_id_grid, u_mults, v_mults, u_knots, v_knots)
    """
    if not rhs.startswith('B_SPLINE_SURFACE_WITH_KNOTS'):
        return None

    args = rhs[len('B_SPLINE_SURFACE_WITH_KNOTS'):].strip()
    args = _strip_outer_parens(args)
    fields = _split_top_level_commas(args)
    if len(fields) < 8:
        return None

    name = fields[0].strip().strip("'")
    deg_u = int(fields[1].strip())
    deg_v = int(fields[2].strip())

    cp_grid = _parse_control_point_grid(fields[3])

    u_mults, v_mults, u_knots, v_knots = _detect_knot_fields(fields)

    return name, deg_u, deg_v, cp_grid, u_mults, v_mults, u_knots, v_knots


# ---------------------------- Knot helpers -------------------------------

def _expand_knots(base_knots: List[float], mults: List[int]) -> np.ndarray:
    base = np.asarray(base_knots, dtype=float)
    m = np.asarray(mults, dtype=int)
    if base.shape[0] != m.shape[0]:
        raise ValueError('Knot values and multiplicities must have same length.')
    return np.repeat(base, m)


def _normalize_knots_affine(k: np.ndarray, eps: float = 1e-14) -> np.ndarray:
    """Affine-map knots to [0,1] using endpoints. If degenerate, returns copy."""
    k = np.asarray(k, dtype=float)
    denom = k[-1] - k[0]
    if abs(denom) < eps:
        return k.copy()
    return (k - k[0]) / denom


def _space_key(deg_u: int, deg_v: int, u_mults: List[int], v_mults: List[int], u_knots: List[float], v_knots: List[float]) -> str:
    """Create a stable key for caching BSplineSpace objects."""
    payload = (
        str(deg_u) + '|' + str(deg_v) + '|'
        + ','.join(map(str, u_mults)) + '|' + ','.join(map(str, v_mults)) + '|'
        + ','.join(f'{x:.17g}' for x in u_knots) + '|' + ','.join(f'{x:.17g}' for x in v_knots)
    ).encode('utf-8')
    return hashlib.md5(payload).hexdigest()


# -----------------------------------------------------------------------------
# Stored import helper
# -----------------------------------------------------------------------------

def _check_if_load_stored_import(
    file_name: str,
    name: str = 'geometry',
    parallelize: bool = True,
) -> Optional[lfs.FunctionSet]:
    """Load a previously stored import if it exists.

    Note: parallelize is retained for backward-compatibility with the old API.
    """
    fn = os.path.basename(file_name)
    fn_wo_ext = fn[:fn.rindex('.')]
    file_path = f"stored_files/imports/{fn_wo_ext}_stored_import.pickle"
    path = Path(file_path)

    if not path.is_file():
        return None

    with open(file_path, 'rb') as handle:
        function_set = pickle.load(handle)

    # Re-wrap coefficients as csdl Variables (pickle stores numpy arrays)
    for function in function_set.functions.values():
        function.coefficients = csdl.Variable(value=np.asarray(function.coefficients))

    # Invalidate cache if it was built with an incompatible class.
    for function in function_set.functions.values():
        if not isinstance(function.space, lfs.BSplineSpace):
            return None

    # Optionally rename set
    if hasattr(function_set, 'name'):
        function_set.name = name

    return function_set


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def import_file(
    file_name: str,
    parallelize: bool = True,
    normalize_knots: bool = True,
    name: str = 'imported_geometry',
) -> lfs.FunctionSet:
    """Import OpenVSP STEP file containing B_SPLINE_SURFACE_WITH_KNOTS.

    Parameters
    ----------
    file_name : str
        STEP file path.
    parallelize : bool
        Retained for compatibility; parsing is typically faster single-threaded.
    normalize_knots : bool
        If True, affine-normalize each knot vector to [0,1].
    name : str
        Name for the returned FunctionSet.

    Returns
    -------
    lfs.FunctionSet
        A set of lfs.Function objects, one per B-spline surface.
    """

    # Quick existence check
    with open(file_name, 'r') as f:
        content = f.read(200000)  # partial read for quick check
        if 'B_SPLINE_SURFACE_WITH_KNOTS' not in content:
            raise ValueError('No B_SPLINE_SURFACE_WITH_KNOTS found in file (or file not compatible).')

    # Stored import
    loaded = _check_if_load_stored_import(file_name, name=name, parallelize=parallelize)
    if loaded is not None:
        return loaded

    print('Importing OpenVSP file:', file_name)

    entities = _read_step_entities(file_name)

    # Build point map: STEP id -> xyz
    point_map: Dict[int, np.ndarray] = {}
    for eid, rhs in entities.items():
        if rhs.startswith('CARTESIAN_POINT'):
            xyz = _parse_cartesian_point(rhs)
            if xyz is not None:
                point_map[eid] = xyz

    # Parse surfaces
    surfaces: List[Tuple[str, int, int, np.ndarray, List[int], List[int], List[float], List[float]]] = []
    for eid, rhs in entities.items():
        if rhs.startswith('B_SPLINE_SURFACE_WITH_KNOTS'):
            parsed = _parse_bspline_surface_with_knots(rhs)
            if parsed is not None:
                surfaces.append(parsed)

    if not surfaces:
        raise ValueError('No parsable B_SPLINE_SURFACE_WITH_KNOTS entities found.')

    # Cache spaces
    space_cache: Dict[str, lfs.BSplineSpace] = {}
    functions: List[lfs.Function] = []

    for (surf_name, deg_u, deg_v, cp_ids, u_mults, v_mults, u_knots_base, v_knots_base) in surfaces:
        # Expand and optionally normalize knots
        ku = _expand_knots(u_knots_base, u_mults)
        kv = _expand_knots(v_knots_base, v_mults)
        if normalize_knots:
            ku = _normalize_knots_affine(ku)
            kv = _normalize_knots_affine(kv)

        order_u = deg_u + 1
        order_v = deg_v + 1
        # Control point grid gives coefficient shape directly (nu, nv)
        coeff_shape = (cp_ids.shape[0], cp_ids.shape[1])

        # Basic consistency check: for clamped open knot vectors,
        # len(knots) = n + p + 1, where p=degree, n=#ctrlpts-1
        # Here: n_ctrl = coeff_shape[0] etc.
        expected_ku = coeff_shape[0] + deg_u + 1
        expected_kv = coeff_shape[1] + deg_v + 1
        if ku.size != expected_ku or kv.size != expected_kv:
            # Don't hard-fail: some exporters may include different specs.
            # But warn to surface possible mismatches.
            print(
                f"[WARN] Knot length mismatch for '{surf_name}': "
                f"len(ku)={ku.size} expected={expected_ku}; "
                f"len(kv)={kv.size} expected={expected_kv}"
            )

        # Create/get space
        skey = _space_key(deg_u, deg_v, u_mults, v_mults, u_knots_base, v_knots_base)
        if skey in space_cache:
            space = space_cache[skey]
        else:
            space = lfs.BSplineSpace(
                num_parametric_dimensions=2,
                degree=(deg_u, deg_v),
                coefficients_shape=coeff_shape,
                knots=tuple([ku, kv]),
            )
            space_cache[skey] = space

        # Assemble control points in exact order
        nu, nv = coeff_shape
        ctrl = np.zeros((nu, nv, 3), dtype=float)
        missing = 0
        for i in range(nu):
            for j in range(nv):
                pid = int(cp_ids[i, j])
                p = point_map.get(pid)
                if p is None:
                    missing += 1
                    continue
                ctrl[i, j, :] = p
        if missing:
            raise ValueError(f"Missing {missing} CARTESIAN_POINT references while building '{surf_name}'.")

        coeffs = csdl.Variable(value=ctrl)
        fn = lfs.Function(space=space, coefficients=coeffs, name=f"{surf_name}")
        functions.append(fn)

    fset = lfs.FunctionSet(functions, name=name)

    # Store import to disk (convert csdl vars to numpy arrays)
    fn_base = os.path.basename(file_name)
    fn_wo_ext = fn_base[:fn_base.rindex('.')]
    store_path = f"stored_files/imports/{fn_wo_ext}_stored_import.pickle"
    Path('stored_files/imports').mkdir(parents=True, exist_ok=True)

    with open(store_path, 'wb+') as handle:
        fset_copy = fset.copy()
        for key, function in fset.functions.items():
            function_copy = function.copy()
            # csdl.Variable -> ndarray
            val = function.coefficients.value if hasattr(function.coefficients, 'value') else function.coefficients
            function_copy.coefficients = np.asarray(val).copy()
            fset_copy.functions[key] = function_copy
        pickle.dump(fset_copy, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Complete import')
    return fset


def import_file_patched(*args, **kwargs):
    """Deprecated alias for import_file.
    
    .. deprecated:: 1.0.0
        Use :func:`import_file` instead.
    """
    import warnings
    warnings.warn(
        "import_file_patched is deprecated; use import_file instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return import_file(*args, **kwargs)

