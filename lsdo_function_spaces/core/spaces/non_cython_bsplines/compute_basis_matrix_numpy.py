import numpy as np
import scipy.sparse as sp

def compute_basis_matrix_numpy(us, degrees, knot_vectors, der_orders=None):
    """
    Vectorized computation of nonzero B-spline basis funcs and their derivatives
    for M parameter values, up to derivative order n (n ≤ p).

    Parameters
    ----------
    us : ndarray, shape (M,)
        Parameter values.
    degrees : tuple(int)
        Degree of the spline.
    knot_vectors : tuple(ndarray)
        Knot vectors.
    der_orders : tuple(int), 
        Maximum derivative order (n ≤ p).

    Returns
    -------
    basis_mat : sparse.coo_matrix
        Sparse basis matrix 
    """
    us = us.reshape(us.shape[0], -1)  # Ensure us is 2D
    M, dim = us.shape

    if der_orders is None:
        der_orders = [0] * dim
    elif len(der_orders) != dim:
        if len(der_orders) == 1:
            der_orders = der_orders * dim
        else:
            raise ValueError("der_orders must be either a single int or a tuple of ints with length equal to the number of dimensions.")

    # 1) find spans for all us
    #    span m satisfies U[i] ≤ u_m < U[i+1]
    n_ctrls = []
    spans   = []
    Ns      = []

    for i in range(dim):
        p = degrees[i]
        U = knot_vectors[i]
        n = der_orders[i]

        num_cps = len(U) - p - 1
        n_ctrls.append(num_cps)

        span = np.searchsorted(U, us[:, i], side="right") - 1
        span = np.clip(span, p, len(U)-p-2)  # clamp to valid [p, n_ctrl-1]
        spans.append(span)

        # 2) build the ndu table for all M points at once: shape (M, p+1, p+1)
        ndu   = np.zeros((M, p+1, p+1))
        left  = np.zeros((M, p+1))
        right = np.zeros((M, p+1))

        ndu[:,0,0] = 1.0
        for j in range(1, p+1):
            # left and right distances
            left[:, j]  = us[:, i] - U[span + 1 - j]
            right[:, j] = U[span + j] - us[:, i]
            saved = np.zeros(M)

            for r in range(j):
                ndu[:, j, r] = right[:, r+1] + left[:, j-r]
                temp = ndu[:, r, j-1] / ndu[:, j, r]
                ndu[:, r, j] = saved + right[:, r+1] * temp
                saved = left[:, j-r] * temp

            ndu[:, j, j] = saved

        # 3) allocate ders array and load zero-th derivatives
        ders = np.zeros((M, n+1, p+1))
        ders[:, 0, :] = ndu[:, :, p]

        if n == 0:
            # If no derivatives requested, just return the zero-th order
            Ns.append(ders[:, 0, :])
            continue

        # 4) compute derivatives via Alg A2.3
        a = np.zeros((2, M, p+1))
        for r in range(p+1):
            a[0, :, 0] = 1.0
            for k in range(1, n+1):
                if k > p:
                    break  # No derivatives beyond degree
                d   = np.zeros(M)
                rk  = r - k
                pk  = p - k

                # first term
                if rk >= 0:
                    a[1, :, 0] = a[0, :, 0] / ndu[:, pk+1, rk]
                    d += a[1, :, 0] * ndu[:, rk, pk]
                else:
                    a[1, :, 0] = 0.0

                # inner terms
                j1 = 1     if rk >= -1 else -rk
                j2 = k-1   if (r-1) <= pk else p - r
                for j in range(j1, j2+1):
                    a[1, :, j] = (a[0, :, j] - a[0, :, j-1]) / ndu[:, pk+1, rk+j]
                    d += a[1, :, j] * ndu[:, rk+j, pk]

                # last term
                if r <= pk:
                    a[1, :, k] = -a[0, :, k-1] / ndu[:, pk+1, r]
                    d += a[1, :, k] * ndu[:, r, pk]
                else:
                    a[1, :, k] = 0.0

                ders[:, k, r] = d

                # swap rows for next k
                a[0, :, :], a[1, :, :] = a[1, :, :], a[0, :, :]

        # 5) multiply through by factorial factors
        #    ders[k, :, :] *= p*(p-1)*...*(p-k+1)
        for k in range(1, n+1):
            ders[:, k, :] *= np.prod(np.arange(p, p-k, -1))

        Ns.append(ders[:, -1, :])

    # 2) build all local-offset combinations offs of shape (L, d),
    #    where L = prod_i (p_i+1)
    grids = [np.arange(p+1) for p in degrees]
    mesh = np.meshgrid(*grids, indexing="ij")
    offs = np.stack([g.ravel() for g in mesh], axis=-1)  # shape (L, d)
    L   = offs.shape[0]

    # 3) compute C-order strides for flattening a grid of shape n_ctrls
    #    stride[i] = prod(n_ctrls[i+1:])
    strides = np.empty(dim, int)
    acc = 1
    for i in range(dim-1, -1, -1):
        strides[i] = acc
        acc *= n_ctrls[i]

    # 4) assemble sparse entries
    # rows: 0..M-1 repeated each L times
    rows = np.repeat(np.arange(M), L)

    # cols: sum_i [ (spans[i] - p_i + offs[:,i]) * strides[i] ], broadcasted over M×L
    cols = np.zeros((M, L), int)
    for i in range(dim):
        base = (spans[i] - degrees[i])[:,None] + offs[None,:,i]
        cols += base * strides[i]
    cols = cols.ravel()

    # data: product over i of Ns[i][k, offs[:,i]]
    data = np.ones((M, L))
    for i in range(dim):
        data *= Ns[i][:, offs[:,i]]
    data = data.ravel()

    # 5) build the COO
    shape = (M, np.prod(n_ctrls))
    B_coo = sp.coo_matrix((data, (rows, cols)), shape=shape)
    return B_coo


def evaluate_b_spline_numpy(
    us,
    degrees,
    knot_vectors,
    coefficients,
    der_orders=None,
):
    """
    Evaluate B-spline at given parametric coordinates using numpy.

    Parameters
    ----------
    us : ndarray, shape (M, d)
        Parametric coordinates.
    degrees : tuple(int)
        Degrees of the B-spline in each dimension.
    knot_vectors : tuple(ndarray)
        Knot vectors for each dimension.
    coefficients : ndarray, shape (n_ctrls_1, n_ctrls_2, ..., dim_out)
        Control point coefficients.
    der_orders : tuple(int), optional
        Derivative orders for each dimension. If None, defaults to (0,) * d.    
    Returns
    -------
    values : ndarray, shape (M, dim_out)
        Evaluated B-spline values at the given parametric coordinates.
    """
    # if der_orders is None:
    #     der_orders = (0,) * len(degrees)

    B = compute_basis_matrix_numpy(us, degrees, knot_vectors, der_orders)
    return B @ coefficients.reshape(-1, coefficients.shape[-1])


if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    import time

    num_cp_x = 10
    num_cp_y = 8
    nx = num_cp_x - 1  # Number of control points - 1
    ny = num_cp_y - 1  # Number of control points - 1
    px = 3  # Degree of the B-spline
    py = 2  # Degree of the B-spline
    
    knots_x = np.concatenate(
        [np.zeros(px), 
         np.linspace(0, 1, num_cp_x - px + 1), 
         np.ones(px)]
    )
    # print("knots_x", knots_x)
    knots_y = np.concatenate(
        [np.zeros(py), 
         np.linspace(0, 1, num_cp_y - py + 1), 
         np.ones(py)]
    )

    knots = (knots_x, knots_y)  # Knot vectors for each dimension
    knots_jnp = (np.array(knots_x), np.array(knots_y))  # JAX-compatible knot vectors

     # Degrees of the B-spline in each dimension
    p = (px, py) 

    # create control points
    coeffs_x, coeffs_y = np.meshgrid(np.linspace(0, 5, num_cp_x), np.linspace(0, 2, num_cp_y), indexing='ij')
    coeffs = np.array(np.stack((coeffs_x, coeffs_y, 0.2 * np.random.rand(num_cp_x, num_cp_y)), axis=-1))

    num_para_coords = 500 # NOTE: the actual number is squared
    u1, v1 = np.meshgrid(np.linspace(0, 1, num_para_coords), np.linspace(0, 1, num_para_coords), indexing='ij')
    us1 = np.array(np.stack((u1.flatten(), v1.flatten()), axis=-1))

    der_orders = (0, 0)  # Derivative orders for each dimension

    t1 = time.perf_counter()
    b_spline_eval = evaluate_b_spline_numpy(
        us1, 
        p, 
        knots, 
        coeffs, der_orders
    ).reshape(-1, 3)
    t2 = time.perf_counter()
    print(f"Time to evaluate numpy B-spline: {t2 - t1:.6f} seconds")
    print("b_spline_eval", b_spline_eval.shape)
   