import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO


def compute_basis_matrix_jax(us, degrees, knot_vectors, der_orders=None):
    """Original matrix-builder API (returns a BCOO sparse matrix).

    This is kept for backwards compatibility with the baseline file you shared.

    Parameters
    ----------
    us : jnp.ndarray, shape (M, d)
    degrees : tuple[int]
    knot_vectors : tuple[jnp.ndarray]
    der_orders : tuple[int] | None
    """
    knot_vectors = tuple(jnp.array(U) for U in knot_vectors)
    us = jnp.atleast_2d(us)
    M, dim = us.shape

    if der_orders is None:
        der_orders = tuple([0] * dim)

    n_ctrls = []
    spans = []
    Ns = []  # ders[:, n_i, :] per dimension

    for i in range(dim):
        p = degrees[i]
        U = knot_vectors[i]
        n = der_orders[i]

        num_cp = len(U) - p - 1
        n_ctrls.append(num_cp)

        span = jnp.searchsorted(U, us[:, i], side="right") - 1
        span = jnp.clip(span, p, num_cp - 1)
        spans.append(span)

        ndu = jnp.zeros((M, p + 1, p + 1))
        ndu = ndu.at[:, 0, 0].set(1.0)
        left = jnp.zeros((M, p + 1))
        right = jnp.zeros((M, p + 1))

        for j in range(1, p + 1):
            left = left.at[:, j].set(us[:, i] - U[span + 1 - j])
            right = right.at[:, j].set(U[span + j] - us[:, i])
            saved = jnp.zeros((M,))

            for r in range(j):
                ndu = ndu.at[:, j, r].set(right[:, r + 1] + left[:, j - r])
                temp = ndu[:, r, j - 1] / ndu[:, j, r]
                ndu = ndu.at[:, r, j].set(saved + right[:, r + 1] * temp)
                saved = left[:, j - r] * temp

            ndu = ndu.at[:, j, j].set(saved)

        # DersBasisFuns up to order n (Algorithm A2.3)
        ders = jnp.zeros((M, n + 1, p + 1))
        ders = ders.at[:, 0, :].set(ndu[:, :, p])

        a = jnp.zeros((2, M, p + 1))
        for r in range(p + 1):
            a = a.at[0, :, 0].set(1.0)

            for k in range(1, n + 1):
                d = jnp.zeros((M,))
                rk = r - k
                pk = p - k

                a = a.at[1, :, 0].set(
                    jnp.where(rk >= 0, a[0, :, 0] / ndu[:, pk + 1, rk], 0.0)
                )
                d = d + jnp.where(rk >= 0, a[1, :, 0] * ndu[:, rk, pk], 0.0)

                j1 = 1 if rk >= -1 else -rk
                j2 = k - 1 if (r - 1) <= pk else p - r
                for j in range(j1, j2 + 1):
                    val = (a[0, :, j] - a[0, :, j - 1]) / ndu[:, pk + 1, rk + j]
                    a = a.at[1, :, j].set(val)
                    d = d + val * ndu[:, rk + j, pk]

                a = a.at[1, :, k].set(
                    jnp.where(r <= pk, -a[0, :, k - 1] / ndu[:, pk + 1, r], 0.0)
                )
                d = d + jnp.where(r <= pk, a[1, :, k] * ndu[:, r, pk], 0.0)

                ders = ders.at[:, k, r].set(d)
                a = a.at[0].set(a[1])
                a = a.at[1].set(0.0)

        for k in range(1, n + 1):
            factor = jnp.prod(jnp.arange(p, p - k, -1))
            ders = ders.at[:, k, :].multiply(factor)

        Ns.append(ders[:, n, :])

    # local-offset combinations
    grids = [jnp.arange(p + 1) for p in degrees]
    mesh = jnp.meshgrid(*grids, indexing="ij")
    offs = jnp.stack([g.ravel() for g in mesh], axis=-1)  # (L, d)
    L = offs.shape[0]

    # C-order strides (python ints)
    strides = np.empty(dim, int)
    acc = 1
    for i in range(dim - 1, -1, -1):
        strides[i] = acc
        acc *= n_ctrls[i]

    rows = jnp.repeat(jnp.arange(M), L)

    cols = 0
    for i in range(dim):
        base = (spans[i] - degrees[i])[:, None] + offs[None, :, i]
        cols = cols + base * strides[i]
    cols = cols.ravel()

    data = jnp.ones((M, L))
    for i in range(dim):
        data = data * Ns[i][:, offs[:, i]]
    data = data.ravel()

    shape = (M, int(np.prod(n_ctrls)))
    return BCOO((data, jnp.stack((rows, cols), axis=-1)), shape=shape)


def evaluate_b_spline_jax(us, degrees, knot_vectors, coeffs, der_orders=None):
    """Baseline evaluation API (matrix-vector): y = B(us) @ coeffs_flat."""
    num_phys_dims = coeffs.shape[-1]
    ndim = len(degrees)
    if der_orders is None:
        der_orders = (0,) * ndim

    B = compute_basis_matrix_jax(us, degrees, knot_vectors, der_orders)
    return B @ coeffs.reshape(-1, num_phys_dims)


def make_bspline_evaluator(degrees, knot_vectors, der_orders=None, *, jit=True):
    """Create a fast B-spline evaluator compiled once per spline space.

    This returns a callable with signature:

        eval_fn(us, coeffs) -> values

    where:
      - us: (M, d) parametric coordinates
      - coeffs: (..., num_phys_dims) control points / coefficients (any shape),
                flattened internally in C-order, exactly like the baseline.

    The returned function computes the same result as `evaluate_b_spline_jax(...)`
    from the original file (same basis math and coefficient flattening), but uses
    a stencil-style gather+einsum instead of constructing a sparse BCOO matrix.

    Notes
    -----
    * `degrees`, `knot_vectors`, `der_orders` are closed over, so JAX compiles once
      per unique spline space. Only `us` and `coeffs` are dynamic arguments.
    * Output matches the baseline up to floating-point roundoff.
    """
    degrees = tuple(int(p) for p in degrees)
    dim = len(degrees)

    knot_vectors = tuple(jnp.asarray(U) for U in knot_vectors)
    if der_orders is None:
        der_orders = (0,) * dim
    der_orders = tuple(int(n) for n in der_orders)

    # control-point grid sizes per dim
    n_ctrls = tuple(int(len(knot_vectors[i]) - degrees[i] - 1) for i in range(dim))

    # Precompute local offsets (L, dim) and per-dim offset column vectors
    grids = [np.arange(p + 1, dtype=np.int32) for p in degrees]
    mesh = np.meshgrid(*grids, indexing="ij")
    offs_np = np.stack([g.ravel() for g in mesh], axis=-1).astype(np.int32)  # (L, dim)
    offs = jnp.asarray(offs_np, dtype=jnp.int32)
    L = int(offs_np.shape[0])
    offs_cols = [offs[:, i] for i in range(dim)]  # each (L,)

    # C-order strides (python ints -> embedded constant array)
    strides_np = np.empty(dim, dtype=np.int32)
    acc = 1
    for i in range(dim - 1, -1, -1):
        strides_np[i] = acc
        acc *= n_ctrls[i]
    strides = jnp.asarray(strides_np, dtype=jnp.int32)

    def _basis_1d(u, p, U, n):
        """Compute span and the n-th derivative basis vector (size p+1) for M points."""
        u = jnp.atleast_1d(u)
        M = u.shape[0]
        num_cp = int(len(U) - p - 1)

        span = jnp.searchsorted(U, u, side="right") - 1
        span = jnp.clip(span, p, num_cp - 1)

        ndu = jnp.zeros((M, p + 1, p + 1))
        ndu = ndu.at[:, 0, 0].set(1.0)
        left = jnp.zeros((M, p + 1))
        right = jnp.zeros((M, p + 1))

        for j in range(1, p + 1):
            left = left.at[:, j].set(u - U[span + 1 - j])
            right = right.at[:, j].set(U[span + j] - u)
            saved = jnp.zeros((M,))

            for r in range(j):
                ndu = ndu.at[:, j, r].set(right[:, r + 1] + left[:, j - r])
                temp = ndu[:, r, j - 1] / ndu[:, j, r]
                ndu = ndu.at[:, r, j].set(saved + right[:, r + 1] * temp)
                saved = left[:, j - r] * temp

            ndu = ndu.at[:, j, j].set(saved)

        ders = jnp.zeros((M, n + 1, p + 1))
        ders = ders.at[:, 0, :].set(ndu[:, :, p])

        a = jnp.zeros((2, M, p + 1))
        for r in range(p + 1):
            a = a.at[0, :, 0].set(1.0)

            for k in range(1, n + 1):
                d = jnp.zeros((M,))
                rk = r - k
                pk = p - k

                a = a.at[1, :, 0].set(
                    jnp.where(rk >= 0, a[0, :, 0] / ndu[:, pk + 1, rk], 0.0)
                )
                d = d + jnp.where(rk >= 0, a[1, :, 0] * ndu[:, rk, pk], 0.0)

                j1 = 1 if rk >= -1 else -rk
                j2 = k - 1 if (r - 1) <= pk else p - r
                for j in range(j1, j2 + 1):
                    val = (a[0, :, j] - a[0, :, j - 1]) / ndu[:, pk + 1, rk + j]
                    a = a.at[1, :, j].set(val)
                    d = d + val * ndu[:, rk + j, pk]

                a = a.at[1, :, k].set(
                    jnp.where(r <= pk, -a[0, :, k - 1] / ndu[:, pk + 1, r], 0.0)
                )
                d = d + jnp.where(r <= pk, a[1, :, k] * ndu[:, r, pk], 0.0)

                ders = ders.at[:, k, r].set(d)
                a = a.at[0].set(a[1])
                a = a.at[1].set(0.0)

        for k in range(1, n + 1):
            factor = jnp.prod(jnp.arange(p, p - k, -1))
            ders = ders.at[:, k, :].multiply(factor)

        return span, ders[:, n, :]

    def _eval(us, coeffs):
        us = jnp.atleast_2d(us)
        M = us.shape[0]

        num_phys_dims = int(coeffs.shape[-1])
        coeffs_flat = coeffs.reshape(-1, num_phys_dims)

        spans = []
        Ns = []
        for i in range(dim):
            span_i, N_i = _basis_1d(us[:, i], degrees[i], knot_vectors[i], der_orders[i])
            spans.append(span_i)
            Ns.append(N_i)  # (M, p_i+1)

        cols = jnp.zeros((M, L), dtype=jnp.int32)
        for i in range(dim):
            base = (spans[i] - degrees[i])[:, None] + offs_cols[i][None, :]
            cols = cols + base.astype(jnp.int32) * strides[i]

        w = jnp.ones((M, L), dtype=coeffs_flat.dtype)
        for i in range(dim):
            w = w * Ns[i][:, offs_cols[i]]

        gathered = coeffs_flat[cols]  # (M, L, P)
        return jnp.einsum("ml,mlp->mp", w, gathered)

    return jax.jit(_eval) if jit else _eval


make_bspline_evaluator_jax = make_bspline_evaluator



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
    derivative_orders = (2, 1)  # derivative orders for the evaluation
    knots = tuple(
        tuple(
            np.concatenate([
                np.zeros(p[i]),
                np.linspace(0, 1, coefficients_shape[i] - p[i] + 1),
                np.ones(p[i])
            ]).tolist()
        )
        for i in range(len(p))
    )

    # Define the coefficients
    coeffs_x, coeffs_y = np.meshgrid(np.linspace(0, 5, num_cp_x), np.linspace(0, 2, num_cp_y), indexing='ij')
    coeffs = np.array(np.stack((coeffs_x, coeffs_y, 0.2 * np.random.rand(num_cp_x, num_cp_y)), axis=-1))
    coeffs_jnp = jnp.array(coeffs.reshape(-1, 3))

    # Create parameter coordinates for evaluation
    num_para_coords = 500  # NOTE: the actual number is squared
    u1, v1 = np.meshgrid(np.linspace(0, 1, num_para_coords), np.linspace(0, 1, num_para_coords), indexing='ij')
    us1 = np.array(np.stack((u1.flatten(), v1.flatten()), axis=-1))
    us1_jnp = jnp.array(us1)


    eval_fn = make_bspline_evaluator(
        degrees=p,
        knot_vectors=knots,
        der_orders=derivative_orders,   # or None
        jit=True,
    )

    t1 = time.perf_counter()
    b_spline_eval = eval_fn(
        us1_jnp,
        coeffs_jnp,
    ).block_until_ready()
    t2 = time.perf_counter()
    print(f"Time to evaluate JAX B-spline (stencil): {t2 - t1:.6f} seconds")

    t3 = time.perf_counter()
    b_spline_eval = eval_fn(
        us1_jnp,
        coeffs_jnp,
    ).block_until_ready()
    t4 = time.perf_counter()
    print(f"Time to evaluate JAX B-spline (stencil, 2nd call): {t4 - t3:.6f} seconds")

    eval_jit_old = jax.jit(
        evaluate_b_spline_jax,
        static_argnames=('degrees', 'knot_vectors', 'der_orders')
    )

    t5 = time.perf_counter()
    b_spline_eval_old = eval_jit_old(
        us1_jnp,
        p,
        knots,
        coeffs_jnp,
        derivative_orders
    ).block_until_ready()
    t6 = time.perf_counter()
    print(f"Time to evaluate 'old' JAX B-spline (baseline): {t6 - t5:.6f} seconds")

    t1 = time.perf_counter()
    b_spline_eval_old = eval_jit_old(
        us1_jnp,
        p,
        knots,
        coeffs_jnp,
        derivative_orders
    )
    t2 = time.perf_counter()
    print(f"Time to evaluate 'old' JAX B-spline: {t2 - t1:.6f} seconds")

    # Verify that both methods give the same result
    assert jnp.allclose(b_spline_eval, b_spline_eval_old, atol=1e-10), "Results from both methods do not match!"
    print("Results from both methods match!")
    