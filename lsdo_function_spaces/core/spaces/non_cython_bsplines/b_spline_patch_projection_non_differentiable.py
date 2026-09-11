import jax 
import jax.numpy as jnp
from lsdo_function_spaces.core.spaces.non_cython_bsplines.compute_basis_matrix_jax import evaluate_b_spline_jax

def compute_projection_residual(
    point_in_space,
    para_coords,
    degrees,
    coefficients,
    knot_vectors,
):
    """
    Compute the residual of the projection of a point in space onto a B-spline surface.

    Parameters:
    -----------
    point_in_space : jnp.ndarray, shape (d,)
        The point in space to project.
    para_coords : jnp.ndarray, shape (M, d)
        Parameter coordinates for the B-spline surface.
    degrees : tuple of int
        Degrees of the B-spline in each dimension.
    coefficients : jnp.ndarray, shape (N, num_phys_dims)
        Coefficients of the B-spline basis functions.
    knot_vectors : tuple of jnp.ndarray
        Knot vectors for each dimension.

    Returns:
    --------
    residual : jnp.ndarray, shape (M,)
        The residuals of the projection.
    """
    n_dims = len(degrees)
    n_phys_dims = coefficients.shape[-1]
    point_in_space = point_in_space.reshape(-1, n_phys_dims)

    bsp_eval = evaluate_b_spline_jax(
        us=para_coords, 
        degrees=degrees,
        knot_vectors=knot_vectors,
        coeffs=coefficients,
    )

    surface_jacobian = [
        evaluate_b_spline_jax(
            us=para_coords,
            degrees=degrees,
            knot_vectors=knot_vectors,
            coeffs=coefficients,
            der_orders=tuple(
                    int(i == j) for j in range(n_dims)
                )
            ) for i in range(n_dims)
    ]
    surface_jacobian_array = jnp.array(surface_jacobian).squeeze().T

    diff = bsp_eval - point_in_space

    residuals = [jnp.sum(diff * d, axis=1) for d in surface_jacobian]  # shape: (num_pts,) for each
    res = jnp.array(residuals).reshape(n_dims, )

    return res, bsp_eval, surface_jacobian, surface_jacobian_array

def compute_projection_jacobian(
    point_in_space,
    para_coords,
    degrees,
    coefficients,
    knot_vectors,
    bsp_eval,
    surface_jacobian,
):
    """
    Compute the Jacobian of the projection of a point in space onto a B-spline surface.

    Parameters:
    -----------
    point_in_space : jnp.ndarray, shape (d,)
        The point in space to project.
    para_coords : jnp.ndarray, shape (M, d)
        Parameter coordinates for the B-spline surface.
    degrees : tuple of int
        Degrees of the B-spline in each dimension.
    coefficients : jnp.ndarray, shape (N, num_phys_dims)
        Coefficients of the B-spline basis functions.
    knot_vectors : tuple of jnp.ndarray
        Knot vectors for each dimension.
    bsp_eval : jnp.ndarray, shape (M, num_phys_dims)
        B-spline evaluation at the parameter coordinates.
    surface_jacobian : list of jnp.ndarray, shape (M, num_phys_dims)
        Jacobian of the B-spline surface at the parameter coordinates.

    Returns:
    --------
    jacobian : jnp.ndarray, shape (M, d)
        The Jacobian of the projection.
    """
    n_dims = len(degrees)

    diff = (bsp_eval - point_in_space).flatten()

    surface_hessian_tensor = [
        [evaluate_b_spline_jax(
                us=para_coords,
                degrees=degrees,
                knot_vectors=knot_vectors,
                coeffs=coefficients,
                der_orders=tuple((i == k) + (j == k) for k in range(n_dims))
            ) for j in range(n_dims)
        ] for i in range(n_dims)
    ]

    projection_jacobian = jnp.zeros((n_dims, n_dims), dtype=bsp_eval.dtype)

    for i in range(n_dims):
        for j in range(n_dims):
            inner = jnp.sum(surface_jacobian[i] * surface_jacobian[j], axis=1) + jnp.sum(diff * surface_hessian_tensor[i][j], axis=1)
            projection_jacobian = projection_jacobian.at[i, j].set(inner[0])
 
    return projection_jacobian

def compute_point_to_bspline_projection(
    point,
    degrees,
    coefficients,
    para_coords,
    knots,
    max_iter=100,
    tol=1e-12,
):
    """Project a point onto an n-dimensional B-spline using Newton iteration."""
    n_dims = len(degrees) # parametric dimensions
 
    para_coords = para_coords.reshape(n_dims, )

    # knots = tuple([knots[i] for i in range(n_dims)])
 
    def body(state):
        i, para_coords, _, _, _, _, _ = state
 
        res, bsp_eval, surface_jacobian, surface_jacobian_array = compute_projection_residual(
            point, para_coords, degrees, coefficients, knots
        )
        J = compute_projection_jacobian(
            point, para_coords, degrees, coefficients, knots, bsp_eval, surface_jacobian, 
        )
 
        grad = res
        # Applying active-set like approach
        # Inactive directions are those where the gradient is zero or the parameter is at the boundary
        inactive_mask = jnp.logical_or(
            jnp.logical_and(para_coords <= 0.0, grad > 0.0),
            jnp.logical_and(para_coords >= 1.0, grad < 0.0),
        )
        inactive_mask = jnp.logical_or(inactive_mask, grad == 0.0)
        active_mask = ~inactive_mask
 
        # Mask Jacobian and residual using active_mask
        J_masked = J * (active_mask[:, None] & active_mask[None, :])
        res_masked = res * active_mask
 
        res_norm = jnp.linalg.norm(res_masked)
        
        # NOTE: we are solving an augmented system (always 2x2)
        # Ideally we would remove the inactive directions from the system
        # However, for JAX, the dimensions must be fixed for jit compilation
        step = -jnp.linalg.solve(J_masked + jnp.eye(n_dims) * (~active_mask), res_masked)
        step = step * active_mask  # zero out inactive directions
 
        para_coords_new = jnp.clip(para_coords + step, 0.0, 1.0)
        converged_new = res_norm < tol
 
        return (i + 1, para_coords_new, res_masked, converged_new, J, active_mask, surface_jacobian_array)
 
    def cond(state):
        i, _, _, converged, _, _, _ = state
        return (i < max_iter) & (~converged)
    
    dim = len(degrees)
    bool_init = jnp.zeros_like(para_coords, dtype=jnp.bool_)
    dS_dxi_init = jnp.zeros((3, dim))
    init_state = (0, para_coords, jnp.zeros_like(para_coords), False, jnp.zeros((dim, dim)), bool_init, dS_dxi_init)
    final_i, final_coords, final_res, final_converged, J, mask, surface_jacobian = jax.lax.while_loop(cond, body, init_state)
 
    return final_coords, final_res, final_converged, final_i, J, mask, surface_jacobian

