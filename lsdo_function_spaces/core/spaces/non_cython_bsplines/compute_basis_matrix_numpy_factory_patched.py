
import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple


@dataclass(frozen=True)
class BSplineSpaceCache:
    """Cached, spline-space-dependent data for fast repeated evaluation.

    Notes
    -----
    This cache assumes:
      - degrees and knot_vectors are fixed
      - control-net grid shape (n_ctrls per dim) is fixed by knot_vectors and degrees
      - flattening is C-order with strides computed accordingly
    """
    degrees: Tuple[int, ...]
    knot_vectors: Tuple[np.ndarray, ...]
    n_ctrls: Tuple[int, ...]
    dim: int
    offs: np.ndarray          # (L, dim) local offset combinations
    strides: np.ndarray       # (dim,) C-order strides
    L: int                    # number of nonzeros per row (prod(p_i+1))
    n_total: int              # total control points (prod(n_ctrls))


def _as_tuple_int(x: Sequence[int]) -> Tuple[int, ...]:
    return tuple(int(v) for v in x)


def _as_tuple_arrays(knot_vectors: Sequence[np.ndarray]) -> Tuple[np.ndarray, ...]:
    return tuple(np.asarray(kv) for kv in knot_vectors)


def _normalize_us_numpy(us: np.ndarray, expected_dim: Optional[int] = None) -> np.ndarray:
    """Normalize parametric coordinates to shape (M, dim).

    Accepts either a single point of shape (dim,) or a batch of shape (M, dim).
    If expected_dim is provided, the last axis must match it.
    """
    us = np.asarray(us, dtype=float)
    if us.ndim == 1:
        if expected_dim is None:
            raise ValueError("For 1D input, expected_dim must be provided to disambiguate shape.")
        if us.shape[0] != expected_dim:
            raise ValueError(f"Single parametric point has dim={us.shape[0]}, expected {expected_dim}")
        us = us.reshape(1, expected_dim)
    elif us.ndim == 2:
        if expected_dim is not None and us.shape[1] != expected_dim:
            raise ValueError(f"us has dim={us.shape[1]}, expected {expected_dim}")
    else:
        raise ValueError(f"us must have shape (dim,) or (M, dim); got shape {us.shape}")
    return us


def make_bspline_space_cache(
    degrees: Sequence[int],
    knot_vectors: Sequence[np.ndarray],
) -> BSplineSpaceCache:
    """Precompute offsets + strides for a fixed tensor-product B-spline space."""
    degrees = _as_tuple_int(degrees)
    knot_vectors = _as_tuple_arrays(knot_vectors)
    dim = len(degrees)
    if len(knot_vectors) != dim:
        raise ValueError(f"degrees has dim={dim} but knot_vectors has len={len(knot_vectors)}")

    n_ctrls = []
    for p, U in zip(degrees, knot_vectors):
        num_cps = len(U) - p - 1
        if num_cps <= 0:
            raise ValueError("Invalid knot vector / degree combination: num control points <= 0")
        n_ctrls.append(int(num_cps))
    n_ctrls = tuple(n_ctrls)

    # Local offset combinations offs of shape (L, dim), L = prod_i (p_i+1)
    grids = [np.arange(p + 1, dtype=int) for p in degrees]
    mesh = np.meshgrid(*grids, indexing="ij")
    offs = np.stack([g.ravel() for g in mesh], axis=-1)  # (L, dim)
    L = int(offs.shape[0])

    # C-order strides for flattening a grid of shape n_ctrls
    strides = np.empty(dim, dtype=int)
    acc = 1
    for i in range(dim - 1, -1, -1):
        strides[i] = acc
        acc *= n_ctrls[i]

    n_total = int(np.prod(np.array(n_ctrls, dtype=int)))

    return BSplineSpaceCache(
        degrees=degrees,
        knot_vectors=knot_vectors,
        n_ctrls=n_ctrls,
        dim=dim,
        offs=offs,
        strides=strides,
        L=L,
        n_total=n_total,
    )


def compute_basis_stencil_numpy(
    us: np.ndarray,
    degrees: Sequence[int],
    knot_vectors: Sequence[np.ndarray],
    der_orders: Optional[Sequence[int]] = None,
    cache: Optional[BSplineSpaceCache] = None,
) -> Tuple[np.ndarray, np.ndarray, BSplineSpaceCache]:
    """Compute the per-point sparse stencil (cols, weights) without building a sparse matrix.

    Returns
    -------
    cols : ndarray, shape (M, L)
        Flattened control-point indices for each query point.
    w : ndarray, shape (M, L)
        Corresponding tensor-product basis weights (or requested derivative weights).
    cache : BSplineSpaceCache
        The cache used/created (useful to reuse across calls).
    """
    expected_dim = len(degrees) if cache is None else cache.dim
    us = _normalize_us_numpy(us, expected_dim=expected_dim)
    M, dim = us.shape

    if cache is None:
        cache = make_bspline_space_cache(degrees, knot_vectors)
    else:
        # Basic sanity check
        if dim != cache.dim:
            raise ValueError(f"us has dim={dim} but cache.dim={cache.dim}")

    degrees_t = cache.degrees
    knot_vectors_t = cache.knot_vectors

    if der_orders is None:
        der_orders = (0,) * dim
    der_orders = _as_tuple_int(der_orders)
    if len(der_orders) != dim:
        raise ValueError(f"der_orders must have length {dim}, got {len(der_orders)}")
    if any(n < 0 for n in der_orders):
        raise ValueError(f"der_orders must be nonnegative, got {der_orders}")

    spans = []
    Ns = []
    n_ctrls = cache.n_ctrls

    # Per-dimension basis (or derivative) values at each point
    for i in range(dim):
        p = degrees_t[i]
        U = knot_vectors_t[i]
        n = der_orders[i]
        num_cps = n_ctrls[i]
        zero_derivative = n > p

        # span: U[span] <= u < U[span+1]
        span = np.searchsorted(U, us[:, i], side="right") - 1
        span = np.clip(span, p, len(U) - p - 2)  # valid: [p, num_cps-1]
        spans.append(span)

        if zero_derivative:
            Ns.append(np.zeros((M, p + 1), dtype=float))
            continue

        # Build NDU table for all M points: (M, p+1, p+1)
        ndu = np.zeros((M, p + 1, p + 1), dtype=float)
        left = np.zeros((M, p + 1), dtype=float)
        right = np.zeros((M, p + 1), dtype=float)

        ndu[:, 0, 0] = 1.0
        for j in range(1, p + 1):
            left[:, j] = us[:, i] - U[span + 1 - j]
            right[:, j] = U[span + j] - us[:, i]
            saved = np.zeros(M, dtype=float)

            for r in range(j):
                ndu[:, j, r] = right[:, r + 1] + left[:, j - r]
                temp = ndu[:, r, j - 1] / ndu[:, j, r]
                ndu[:, r, j] = saved + right[:, r + 1] * temp
                saved = left[:, j - r] * temp

            ndu[:, j, j] = saved

        # Zero-th derivative basis: last column of NDU
        if n == 0:
            Ns.append(ndu[:, :, p])  # (M, p+1)
            continue

        # Full derivative table ders: (M, n+1, p+1)
        ders = np.zeros((M, n + 1, p + 1), dtype=float)
        ders[:, 0, :] = ndu[:, :, p]

        # Alg A2.3 (Piegl & Tiller The NURBS book) for derivatives
        a = np.zeros((2, M, p + 1), dtype=float)
        for r in range(p + 1):
            a[0, :, 0] = 1.0
            for k in range(1, n + 1):
                d = np.zeros(M, dtype=float)
                rk = r - k
                pk = p - k

                if rk >= 0:
                    a[1, :, 0] = a[0, :, 0] / ndu[:, pk + 1, rk]
                    d += a[1, :, 0] * ndu[:, rk, pk]
                else:
                    a[1, :, 0] = 0.0

                j1 = 1 if rk >= -1 else -rk
                j2 = k - 1 if (r - 1) <= pk else p - r
                for j in range(j1, j2 + 1):
                    a[1, :, j] = (a[0, :, j] - a[0, :, j - 1]) / ndu[:, pk + 1, rk + j]
                    d += a[1, :, j] * ndu[:, rk + j, pk]

                if r <= pk:
                    a[1, :, k] = -a[0, :, k - 1] / ndu[:, pk + 1, r]
                    d += a[1, :, k] * ndu[:, r, pk]
                else:
                    a[1, :, k] = 0.0

                ders[:, k, r] = d

                # swap rows
                a[0, :, :], a[1, :, :] = a[1, :, :], a[0, :, :]

        # Multiply by factorial factors: p*(p-1)*...*(p-k+1)
        fact = 1.0
        for k in range(1, n + 1):
            fact *= (p - (k - 1))
            ders[:, k, :] *= fact

        # Use the requested derivative order in this dim: ders[:, n, :]
        Ns.append(ders[:, n, :])  # (M, p+1)

    offs = cache.offs  # (L, dim)
    L = cache.L

    # cols: (M, L)
    cols = np.zeros((M, L), dtype=int)
    for i in range(dim):
        base = (spans[i] - degrees_t[i])[:, None] + offs[None, :, i]
        cols += base * cache.strides[i]

    # weights: (M, L) = product_i Ns_i[:, offs[:,i]]
    w = np.ones((M, L), dtype=float)
    for i in range(dim):
        w *= Ns[i][:, offs[:, i]]

    return cols, w, cache


def apply_basis_stencil_numpy(
    cols: np.ndarray,
    w: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    """Apply a basis stencil (cols, w) to coefficients via gather + contraction.

    Parameters
    ----------
    cols : (M, L) int
    w : (M, L) float
    coefficients : (..., dim_out)
        Control point coefficients on a tensor grid.

    Returns
    -------
    values : (M, dim_out)
    """
    coeffs = np.asarray(coefficients)
    dim_out = coeffs.shape[-1]
    coeffs_flat = coeffs.reshape(-1, dim_out)  # C-order flatten
    gathered = coeffs_flat[cols]               # (M, L, dim_out)
    # Weighted sum over L
    return np.einsum("ml,mld->md", w, gathered)


def compute_basis_matrix_numpy(
    us: np.ndarray,
    degrees: Sequence[int],
    knot_vectors: Sequence[np.ndarray],
    der_orders: Optional[Sequence[int]] = None,
    cache: Optional[BSplineSpaceCache] = None,
) -> sp.coo_matrix:
    """Compatibility API: build a SciPy COO basis matrix.

    This is slower than the stencil apply path because it constructs a sparse matrix object.
    Prefer `compute_basis_stencil_numpy` + `apply_basis_stencil_numpy` for performance.
    """
    cols, w, cache = compute_basis_stencil_numpy(us, degrees, knot_vectors, der_orders, cache=cache)
    expected_dim = len(degrees) if cache is None else cache.dim
    us = _normalize_us_numpy(us, expected_dim=expected_dim)
    M = us.shape[0]
    L = cache.L

    rows = np.repeat(np.arange(M, dtype=int), L)
    cols_flat = cols.reshape(-1)
    data = w.reshape(-1)
    return sp.coo_matrix((data, (rows, cols_flat)), shape=(M, cache.n_total))


def evaluate_b_spline_numpy(
    us: np.ndarray,
    degrees: Sequence[int],
    knot_vectors: Sequence[np.ndarray],
    coefficients: np.ndarray,
    der_orders: Optional[Sequence[int]] = None,
    cache: Optional[BSplineSpaceCache] = None,
) -> np.ndarray:
    """Fast evaluation using stencil gather+einsum (same output as sparse-matmul path)."""
    cols, w, cache = compute_basis_stencil_numpy(us, degrees, knot_vectors, der_orders, cache=cache)
    return apply_basis_stencil_numpy(cols, w, coefficients)


def make_bspline_evaluator_numpy(
    degrees: Sequence[int],
    knot_vectors: Sequence[np.ndarray],
    der_orders: Optional[Sequence[int]] = None,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Factory that caches spline-space data and returns a fast evaluator.

    The returned function has signature:
        eval_fn(us, coefficients) -> values

    It reuses the cached offsets/strides/control-net sizing across calls.
    """
    cache = make_bspline_space_cache(degrees, knot_vectors)
    degrees_t = cache.degrees
    knot_vectors_t = cache.knot_vectors
    if der_orders is None:
        der_orders_t = (0,) * cache.dim
    else:
        der_orders_t = _as_tuple_int(der_orders)

    def eval_fn(us: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
        cols, w, _ = compute_basis_stencil_numpy(
            us,
            degrees_t,
            knot_vectors_t,
            der_orders_t,
            cache=cache,
        )
        return apply_basis_stencil_numpy(cols, w, coefficients)

    return eval_fn


if __name__ == "__main__":
    # Quick sanity + speed demo (mirrors your original main)
    np.random.seed(42)
    import time

    num_cp_x = 10
    num_cp_y = 8
    px = 3
    py = 2

    knots_x = np.concatenate([np.zeros(px), np.linspace(0, 1, num_cp_x - px + 1), np.ones(px)])
    knots_y = np.concatenate([np.zeros(py), np.linspace(0, 1, num_cp_y - py + 1), np.ones(py)])
    knots = (knots_x, knots_y)
    degrees = (px, py)

    coeffs_x, coeffs_y = np.meshgrid(
        np.linspace(0, 5, num_cp_x),
        np.linspace(0, 2, num_cp_y),
        indexing="ij",
    )
    coeffs = np.stack((coeffs_x, coeffs_y, 0.2 * np.random.rand(num_cp_x, num_cp_y)), axis=-1)

    num_para_coords = 500
    u1, v1 = np.meshgrid(np.linspace(0, 1, num_para_coords), np.linspace(0, 1, num_para_coords), indexing="ij")
    us1 = np.stack((u1.ravel(), v1.ravel()), axis=-1)

    der_orders = (1, 1)

    # Old-style (sparse matrix) via compatibility API
    t1 = time.perf_counter()
    B = compute_basis_matrix_numpy(us1, degrees, knots, der_orders)
    y_sparse = (B @ coeffs.reshape(-1, coeffs.shape[-1])).reshape(-1, 3)
    t2 = time.perf_counter()

    # New fast path via factory
    eval_fn = make_bspline_evaluator_numpy(degrees, knots, der_orders)
    t3 = time.perf_counter()
    y_fast = eval_fn(us1, coeffs).reshape(-1, 3)
    t4 = time.perf_counter()

    max_err = np.max(np.abs(y_sparse - y_fast))
    print(f"sparse time: {t2 - t1:.6f}s")
    print(f"fast time:   {t4 - t3:.6f}s")
    print(f"% speedup:   {(t2 - t1) / (t4 - t3):.2f}x")
    print(f"max abs err: {max_err:.3e}")
