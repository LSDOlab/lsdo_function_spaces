import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO


def compute_basis_matrix_jax(us, degrees, knot_vectors, der_orders=None):
    """
    JAX-jit-compiled sparse basis-matrix builder with parametric derivatives.

    us            : jnp.ndarray, shape (M, d)
    degrees       : tuple of length d
    knot_vectors  : tuple of length d, each a jnp.ndarray
    der_orders    : tuple of length d, derivative order in each dim
    """
    us = jnp.atleast_2d(us)
    M, dim = us.shape

    # Ensure knot_vectors are JAX arrays (knot_vectors may be passed as
    # Python tuples of tuples to be hashable for jax.jit static args).
    knot_vectors = tuple(jnp.array(U) for U in knot_vectors)

    if der_orders is None:
        der_orders = tuple([0] * dim)

    # 1) per-dim spans, ndu & ders up to der_orders[i]
    n_ctrls = []
    spans   = []
    Ns      = []   # will hold ders[:, n_i, :] per dimension

    for i in range(dim):
        p = degrees[i]
        U = knot_vectors[i]
        n = der_orders[i]

        # number of control points in this dim
        num_cp = len(U) - p - 1
        n_ctrls.append(num_cp)

        # find spans: shape (M,)
        span = jnp.searchsorted(U, us[:, i], side="right") - 1
        span = jnp.clip(span, p, num_cp - 1)
        spans.append(span)

        # build ndu table: shape (M, p+1, p+1)
        ndu = jnp.zeros((M, p+1, p+1))
        ndu = ndu.at[:, 0, 0].set(1.0)
        left  = jnp.zeros((M, p+1))
        right = jnp.zeros((M, p+1))

        for j in range(1, p+1):
            # vectorized distances
            left  = left.at[:, j].set(us[:, i] - U[span + 1 - j])
            right = right.at[:, j].set(U[span + j] - us[:, i])
            saved = jnp.zeros((M,))

            for r in range(j):
                # lower triangle
                ndu = ndu.at[:, j, r].set(right[:, r+1] + left[:, j-r])
                temp = ndu[:, r, j-1] / ndu[:, j, r]
                # upper triangle
                ndu = ndu.at[:, r, j].set(saved + right[:, r+1] * temp)
                saved = left[:, j-r] * temp

            ndu = ndu.at[:, j, j].set(saved)

        # now DersBasisFuns up to order n (Algorithm A2.3)
        # ders shape (M, n+1, p+1)
        ders = jnp.zeros((M, n+1, p+1))
        ders = ders.at[:, 0, :].set(ndu[:, :, p])

        # a buffer for alternating
        a = jnp.zeros((2, M, p+1))
        for r in range(p+1):
            # initialize a row
            a = a.at[0, :, 0].set(1.0)

            for k in range(1, n+1):
                d   = jnp.zeros((M,))
                rk  = r - k
                pk  = p - k

                # first term
                a = a.at[1, :, 0].set(
                    jnp.where(rk >= 0, a[0, :, 0] / ndu[:, pk+1, rk], 0.0)
                )
                d = d + jnp.where(rk >= 0,
                    a[1, :, 0] * ndu[:, rk, pk], 0.0)

                # inner terms
                j1 = 1   if rk >= -1 else -rk
                j2 = k-1 if (r-1) <= pk else p - r
                for j in range(j1, j2+1):
                    val = (a[0, :, j] - a[0, :, j-1]) / ndu[:, pk+1, rk+j]
                    a = a.at[1, :, j].set(val)
                    d = d + val * ndu[:, rk+j, pk]

                # last term
                a = a.at[1, :, k].set(
                    jnp.where(r <= pk, -a[0, :, k-1] / ndu[:, pk+1, r], 0.0)
                )
                d = d + jnp.where(r <= pk,
                    a[1, :, k] * ndu[:, r, pk], 0.0)

                ders = ders.at[:, k, r].set(d)
                # swap rows in a
                a = a.at[0].set(a[1])
                a = a.at[1].set(0.0)

        # scale derivatives by factorial factors
        for k in range(1, n+1):
            factor = jnp.prod(jnp.arange(p, p-k, -1))
            ders = ders.at[:, k, :].multiply(factor)

        # pick off the highest derivative requested
        Ns.append(ders[:, n, :])

    # 2) build all local-offset combinations (static)
    grids = [jnp.arange(p+1) for p in degrees]
    mesh  = jnp.meshgrid(*grids, indexing="ij")
    offs  = jnp.stack([g.ravel() for g in mesh], axis=-1)  # (L, d)
    L     = offs.shape[0]

    # 3) compute C-order strides (python ints)
    strides = np.empty(dim, int)
    acc = 1
    for i in range(dim-1, -1, -1):
        strides[i] = acc
        acc *= n_ctrls[i]

    # 4) assemble sparse COO entries
    rows = jnp.repeat(jnp.arange(M), L)

    # columns
    cols = 0
    for i in range(dim):
        base = (spans[i] - degrees[i])[:, None] + offs[None, :, i]
        cols = cols + base * strides[i]
    cols = cols.ravel()

    # data = product over dims of Ns[i][:, offs[:,i]]
    data = jnp.ones((M, L))
    for i in range(dim):
        data = data * Ns[i][:, offs[:, i]]
    data = data.ravel()

    # shape as python ints
    shape = (M, int(np.prod(n_ctrls)))
    Bcoo = BCOO((data, jnp.stack((rows, cols), axis=-1)), shape=shape)
    return Bcoo


def compute_basis_stencil_jax(us, degrees, knot_vectors, der_orders=None):
    """
    Build a *stencil* representation of the B-spline basis operator without
    constructing a BCOO sparse matrix.

    Returns
    -------
    cols : jnp.ndarray, shape (M, L), int32
        Column indices into the flattened control-point array.
    w    : jnp.ndarray, shape (M, L), float
        Basis weights (or derivative weights).
    n_ctrl : int
        Total number of control points N (so coeffs_flat has shape (N, ...)).

    Notes
    -----
    L = prod_i (degrees[i] + 1) nonzeros per row.
    To get values, use `apply_basis_stencil_jax(cols, w, coeffs)`.
    """
    us = jnp.atleast_2d(us)
    M, dim = us.shape

    # Convert static, hashable knot_vectors (tuples of floats) to JAX arrays
    knot_vectors = tuple(jnp.array(U) for U in knot_vectors)

    if der_orders is None:
        der_orders = tuple([0] * dim)

    # 1) per-dim spans and derivative basis values
    n_ctrls = []
    spans   = []
    Ns      = []

    for i in range(dim):
        p = degrees[i]
        U = knot_vectors[i]
        n = der_orders[i]

        num_cp = len(U) - p - 1
        n_ctrls.append(num_cp)

        span = jnp.searchsorted(U, us[:, i], side="right") - 1
        span = jnp.clip(span, p, num_cp - 1)
        spans.append(span)

        # ndu table (Algorithm A2.2, Piegl & Tiller)
        ndu = jnp.zeros((M, p+1, p+1))
        ndu = ndu.at[:, 0, 0].set(1.0)
        left  = jnp.zeros((M, p+1))
        right = jnp.zeros((M, p+1))

        for j in range(1, p+1):
            left  = left.at[:, j].set(us[:, i] - U[span + 1 - j])
            right = right.at[:, j].set(U[span + j] - us[:, i])
            saved = jnp.zeros((M,))

            for r in range(j):
                ndu = ndu.at[:, j, r].set(right[:, r+1] + left[:, j-r])
                temp = ndu[:, r, j-1] / ndu[:, j, r]
                ndu = ndu.at[:, r, j].set(saved + right[:, r+1] * temp)
                saved = left[:, j-r] * temp

            ndu = ndu.at[:, j, j].set(saved)

        # DersBasisFuns (Algorithm A2.3)
        ders = jnp.zeros((M, n+1, p+1))
        ders = ders.at[:, 0, :].set(ndu[:, :, p])

        a = jnp.zeros((2, M, p+1))
        for r in range(p+1):
            a = a.at[0, :, 0].set(1.0)

            for k in range(1, n+1):
                d   = jnp.zeros((M,))
                rk  = r - k
                pk  = p - k

                a = a.at[1, :, 0].set(jnp.where(rk >= 0, a[0, :, 0] / ndu[:, pk+1, rk], 0.0))
                d = d + jnp.where(rk >= 0, a[1, :, 0] * ndu[:, rk, pk], 0.0)

                j1 = 1   if rk >= -1 else -rk
                j2 = k-1 if (r-1) <= pk else p - r
                for j in range(j1, j2+1):
                    val = (a[0, :, j] - a[0, :, j-1]) / ndu[:, pk+1, rk+j]
                    a = a.at[1, :, j].set(val)
                    d = d + val * ndu[:, rk+j, pk]

                a = a.at[1, :, k].set(jnp.where(r <= pk, -a[0, :, k-1] / ndu[:, pk+1, r], 0.0))
                d = d + jnp.where(r <= pk, a[1, :, k] * ndu[:, r, pk], 0.0)

                ders = ders.at[:, k, r].set(d)
                a = a.at[0].set(a[1])
                a = a.at[1].set(0.0)

        for k in range(1, n+1):
            factor = jnp.prod(jnp.arange(p, p-k, -1))
            ders = ders.at[:, k, :].multiply(factor)

        Ns.append(ders[:, n, :])  # (M, p+1)

    # 2) local offset combinations (L, dim)
    grids = [jnp.arange(p+1) for p in degrees]
    mesh  = jnp.meshgrid(*grids, indexing="ij")
    offs  = jnp.stack([g.ravel() for g in mesh], axis=-1)  # (L, dim)
    L     = offs.shape[0]

    # 3) C-order strides (python ints; safe if degrees/knot_vectors are static)
    strides = np.empty(dim, int)
    acc = 1
    for i in range(dim-1, -1, -1):
        strides[i] = acc
        acc *= n_ctrls[i]

    # 4) cols: (M, L)
    cols = 0
    for i in range(dim):
        base = (spans[i] - degrees[i])[:, None] + offs[None, :, i]
        cols = cols + base * strides[i]
    cols = cols.astype(jnp.int32)

    # 5) weights: (M, L)
    w = jnp.ones((M, L))
    for i in range(dim):
        w = w * Ns[i][:, offs[:, i]]

    n_ctrl = int(np.prod(n_ctrls))
    return cols, w, n_ctrl


def apply_basis_stencil_jax(cols, w, coeffs):
    """
    Apply a stencil basis operator (cols, w) to coefficients.

    Parameters
    ----------
    cols : (M, L) int32
    w    : (M, L) float
    coeffs : (..., P) control points; will be flattened over control-point axes.

    Returns
    -------
    out : (M, P)
    """
    coeffs2 = coeffs.reshape((-1, coeffs.shape[-1]))  # (N, P)
    gathered = coeffs2[cols]                          # (M, L, P)
    return jnp.einsum("ml,mlp->mp", w, gathered)


def evaluate_b_spline_jax_fast(us, degrees, knot_vectors, coeffs, der_orders=None):
    """
    Fast B-spline evaluation: build (cols, w) stencil and apply via gather+einsum.
    This avoids constructing a BCOO sparse matrix each call.
    """
    # If knot_vectors were passed as static Python tuples (to be hashable),
    # convert them to JAX arrays here. compute_basis_stencil_jax also does
    # this conversion, but doing it here keeps explicit intent.
    knot_vectors = tuple(jnp.array(U) for U in knot_vectors)
    cols, w, _ = compute_basis_stencil_jax(us, degrees, knot_vectors, der_orders)
    return apply_basis_stencil_jax(cols, w, coeffs)


# If you want the old "matrix" object for debugging or compatibility:
def stencil_to_bcoo(cols, w, n_ctrl):
    """
    Convert stencil representation to a BCOO sparse matrix.

    cols : (M, L)
    w    : (M, L)
    n_ctrl : int
    """
    M, L = cols.shape
    rows = jnp.repeat(jnp.arange(M, dtype=jnp.int32), L)
    data = w.reshape(-1)
    colr = cols.reshape(-1)
    idx  = jnp.stack((rows, colr), axis=-1)
    return BCOO((data, idx), shape=(M, n_ctrl))


def evaluate_b_spline_jax(us, degrees, knot_vectors, coeffs, der_orders=None):
    """
    Evaluate B-spline basis functions at parameter u with derivatives.

    Parameters:
    -----------
    us : jnp.ndarray, shape (M, d)
        Parameter values where the B-spline basis functions are evaluated.
    degrees : tuple of int
        Degrees of the B-spline in each dimension.
    knot_vectors : tuple of jnp.ndarray
        Knot vectors for each dimension.
    coeffs : jnp.ndarray, shape (N, num_phys_dims)
        Coefficients of the B-spline basis functions.
    der_orders : tuple of int, optional
        Derivative orders for each dimension. If None, defaults to (0,) * d.
    """
    num_phys_dims = coeffs.shape[-1]
    ndim = len(degrees)
    if der_orders is None:
        der_orders = (0,) * ndim

    # Ensure knot_vectors are JAX arrays when passed as static Python tuples
    knot_vectors = tuple(jnp.array(U) for U in knot_vectors)
    # Compute the basis matrix using JAX
    cols, w, _ = compute_basis_stencil_jax(us, degrees, knot_vectors, der_orders)

    # Apply without constructing a sparse matrix
    return apply_basis_stencil_jax(cols, w, coeffs)


if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    import lsdo_function_spaces as lfs
    import time

    jax.config.update("jax_enable_x64", True)  # Use 64-bit precision for JAX

    # Define the B-spline space parameters
    num_cp_x = 10
    num_cp_y = 8
    nx = num_cp_x - 1  # Number of control points - 1
    ny = num_cp_y - 1  # Number of control points - 1
    px = 3  # Degree of the B-spline
    py = 2  # Degree of the B-spline
    p = (px, py)
    coefficients_shape = (num_cp_x, num_cp_y)
    derivative_orders = (1, 1)  # derivative orders for the evaluation

    # Build knot vectors as Python tuples of floats so they are hashable
    # and can be used as static arguments to jax.jit.
    knots = tuple(
        tuple(np.concatenate([
            np.zeros(p[i]),
            np.linspace(0, 1, coefficients_shape[i] - p[i] + 1),
            np.ones(p[i])
        ]).tolist())
        for i in range(len(p))
    )

    # Define the coefficients
    coeffs_x, coeffs_y = np.meshgrid(np.linspace(0, 5, num_cp_x), np.linspace(0, 2, num_cp_y), indexing='ij')
    coeffs = np.array(np.stack((coeffs_x, coeffs_y, 0.2 * np.random.rand(num_cp_x, num_cp_y)), axis=-1))
    coeffs_jnp = jnp.array(coeffs.reshape(-1, 3))

    eval_jit_old = jax.jit(
        evaluate_b_spline_jax,
        static_argnames=('degrees', 'knot_vectors', 'der_orders')
    )

    eval_jit_new = jax.jit(
        evaluate_b_spline_jax_fast,
        static_argnames=('degrees', 'knot_vectors', 'der_orders')
    )

    # Create parameter coordinates for evaluation
    num_para_coords = 500  # NOTE: the actual number is squared
    u1, v1 = np.meshgrid(np.linspace(0, 1, num_para_coords), np.linspace(0, 1, num_para_coords), indexing='ij')
    us1 = np.array(np.stack((u1.flatten(), v1.flatten()), axis=-1))
    us1_jnp = jnp.array(us1)

    b_spline_eval_old = eval_jit_old(
        us1_jnp,
        p,
        knots,
        coeffs_jnp,
        derivative_orders
    ).block_until_ready()

    t1 = time.perf_counter()
    b_spline_eval_old = eval_jit_old(
        us1_jnp,
        p,
        knots,
        coeffs_jnp,
        derivative_orders
    )
    t2 = time.perf_counter()
    print(f"Time to evaluate old JAX B-spline: {t2 - t1:.6f} seconds")

    b_spline_eval_fast = eval_jit_new(
        us1_jnp,
        p,
        knots,
        coeffs_jnp,
        derivative_orders
    ).block_until_ready()
    t3 = time.perf_counter()
    b_spline_eval_fast = eval_jit_new(
        us1_jnp,
        p,
        knots,
        coeffs_jnp,
        derivative_orders
    ).block_until_ready()
    t4 = time.perf_counter()
    print(f"Time to evaluate fast JAX B-spline: {t4 - t3:.6f} seconds")

    # print("Old eval:", b_spline_eval_old)
    # print("Fast eval:", b_spline_eval_fast)
    # print("Difference:", jnp.max(jnp.abs(b_spline_eval_old - b_spline_eval_fast)))
