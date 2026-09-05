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

    # Compute the basis matrix using JAX
    basis_matrix = compute_basis_matrix_jax(us, degrees, knot_vectors, der_orders)

    # Evaluate the B-spline basis functions
    return basis_matrix @ coeffs.reshape(-1, num_phys_dims)