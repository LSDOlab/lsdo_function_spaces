import jax 
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax import custom_jvp, custom_vjp
from lsdo_function_spaces.core.spaces.non_cython_bsplines.compute_basis_matrix_jax_stencil import evaluate_b_spline_jax
from lsdo_function_spaces.core.spaces.non_cython_bsplines.compute_basis_matrix_numpy import compute_basis_matrix_numpy
import csdl_alpha as csdl
import time 


jax.config.update("jax_enable_x64", True)


@partial(jax.jit, static_argnames=("degrees",  "knot_vectors"))
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

    # Convert static, hashable knot_vectors (tuples of floats) to JAX arrays
    knot_vectors = tuple(jnp.array(U) for U in knot_vectors)

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

@partial(jax.jit, static_argnames=("degrees",  "knot_vectors"))
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

    # Convert static, hashable knot_vectors (tuples of floats) to JAX arrays
    knot_vectors = tuple(jnp.array(U) for U in knot_vectors)

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

@partial(jax.jit, static_argnames=("degrees", "knots"))
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

    # Convert static, hashable knots (tuples of floats) to JAX arrays
    knots = tuple(jnp.array(k) for k in knots)

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

@partial(jax.jit, static_argnames=("degrees", "knots"))
def res_fun_single(para_coords, point, coefficients, degrees=None, knots=None):
    """Compute the residual for a single point."""
    return jax.jit(compute_projection_residual, static_argnums=(2, ))(
        point_in_space=point,
        para_coords=para_coords,
        degrees=degrees,
        coefficients=coefficients,
    knot_vectors=knots,
    )[0]

@partial(jax.jit, static_argnames=("degrees", "knots"))
def newton_solve_single(point, coefficients, para_coords_0, degrees=None, knots=None):
    """Newton's method to solve the projection for a single point."""
    return compute_point_to_bspline_projection(
        point=point,
        degrees=degrees,
        coefficients=coefficients,
        para_coords=para_coords_0,
    knots=knots,
    )[0]

@jax.custom_jvp
@jax.custom_vjp
@partial(jax.jit, static_argnames=("degrees", "knots"))
def implicit_solve_single(point, coefficients, para_coords_0, degrees=None, knots=None):
    return newton_solve_single(
        point=point,
        coefficients=coefficients,
        para_coords_0=para_coords_0,
        degrees=degrees,
        knots=knots,
    )
    
@partial(jax.jit, static_argnames=("degrees", "knots"))
def fwd(point, coefficients, para_coords_0, degrees=None, knots=None):
    para_coord_star = implicit_solve_single(
        point=point,
        coefficients=coefficients,
        para_coords_0=para_coords_0,
        degrees=degrees,
        knots=knots,
    )
    return para_coord_star, (para_coord_star, point, coefficients)

@partial(jax.jit, static_argnames=("degrees", "knots"))
def bwd(primals, x_bar, degrees=None, knots=None):
    n_dims = len(degrees)
    
    para_coord_star, point, coefficients = primals
    # Compute the Jacobian of the projection
    res, bsp_eval, surface_jacobian, _ = compute_projection_residual(
        point_in_space=point,
        para_coords=para_coord_star,
        degrees=degrees,
        coefficients=coefficients,
    knot_vectors=knots,
    )

    grad = res
    # Applying active-set like approach
    # Inactive directions are those where the gradient is zero or the parameter is at the boundary
    inactive_mask = jnp.logical_or(
        jnp.logical_and(para_coord_star.flatten() <= 0.0, grad > 0.0),
        jnp.logical_and(para_coord_star.flatten() >= 1.0, grad < 0.0),
    )
    # inactive_mask = jnp.logical_or(inactive_mask, grad == 0.0)
    active_mask = ~inactive_mask
    jax.lax.stop_gradient(active_mask)  # ensure mask is not differentiated

    J = compute_projection_jacobian(
        point_in_space=point,
        para_coords=para_coord_star,
        degrees=degrees,
        coefficients=coefficients,
        knot_vectors=knots,
        bsp_eval=bsp_eval,
        surface_jacobian=surface_jacobian,
    )

    # Mask Jacobian and residual using active_mask
    J_masked = J * (active_mask[:, None] & active_mask[None, :])
    x_bar_masked = x_bar * active_mask
    J_masked = J_masked + jnp.eye(n_dims) * (~active_mask)  # add identity to avoid singularity

    # lambda_ = jnp.linalg.solve(J.T, x_bar)
    lambda_ = jnp.linalg.solve(J_masked.T, x_bar_masked)

    def res_fun_wrapped(point, coefficients):
        res = res_fun_single(
            para_coords=para_coord_star,
            point=point,
            coefficients=coefficients,
            degrees=degrees,
            knots=knots,
        )
        return res.squeeze()
    
    _, vjp_res = jax.vjp(res_fun_wrapped, point, coefficients)
    dpoint, dcoeffs = vjp_res(-lambda_)

    return (dpoint, dcoeffs, None)
implicit_solve_single.defvjp(fwd, bwd)


@implicit_solve_single.defjvp
def iv_jvp(primals, tangents):
    """
    primals: tuple of positional arguments passed to implicit_solve_single
    tangents: tuple of corresponding tangents (same length as primals)
    """
    # --- Defensive unpacking of primals (support 3..5 entries) ---
    if len(primals) < 3:
        raise ValueError("iv_jvp expected at least (point, coefficients, para_coords_0) primals")

    point = primals[0]
    coefficients = primals[1]
    para_coords_0 = primals[2]

    degrees = None
    knots = None
    if len(primals) >= 4:
        degrees = primals[3]
    if len(primals) >= 5:
        knots = primals[4]

    # --- Defensive unpacking of tangents (fill missing with zeros) ---
    # tangents may be shorter if some primals are static/non-differentiable
    def safe_get(tup, idx, fill):
        try:
            return tup[idx]
        except Exception:
            return fill

    point_dot = safe_get(tangents, 0, jnp.zeros_like(point))
    coeffs_dot = safe_get(tangents, 1, jnp.zeros_like(coefficients))

    # --- Compute the primal solution (call with the same kwargs your forward used) ---
    # Use keyword args to be robust to ordering; implicit_solve_single must accept them.
    if (degrees is None) and (knots is None):
        para_coord_star = implicit_solve_single(point, coefficients, para_coords_0)
    else:
        para_coord_star = implicit_solve_single(
            point=point,
            coefficients=coefficients,
            para_coords_0=para_coords_0,
            degrees=degrees,
            knots=knots,
        )

    # --- Build residual wrapper evaluated at the converged parametric location ---
    # res_fun_single(para_coords, point, coefficients, degrees, knots) -> residual (n_dim,)
    def res_wrapped(p_in, coeffs_in):
        # call the same residual function you use elsewhere; make sure shapes consistent
        return res_fun_single(
            para_coords=para_coord_star,
            point=p_in,
            coefficients=coeffs_in,
            degrees=degrees,
            knots=knots,
        ).squeeze()

    # Directional derivative of residual w.r.t. (point, coefficients) in directions (point_dot, coeffs_dot)
    _, res_dir = jax.jvp(res_wrapped, (point, coefficients), (point_dot, coeffs_dot))

    # Evaluate Jacobian J = dr/dpara (size = n_param_dims x n_param_dims)
    res_val, bsp_eval, surface_jacobian, _ = compute_projection_residual(
        point_in_space=point,
        para_coords=para_coord_star,
        degrees=degrees,
        coefficients=coefficients,
        knot_vectors=knots if (knots is not None) else None,
    )

    # Build active mask as in forward
    grad = res_val
    inactive_mask = jnp.logical_or(
        jnp.logical_and(para_coord_star.flatten() <= 0.0, grad > 0.0),
        jnp.logical_and(para_coord_star.flatten() >= 1.0, grad < 0.0),
    )
    # inactive_mask = jnp.logical_or(inactive_mask, grad == 0.0)
    active_mask = ~inactive_mask
    jax.lax.stop_gradient(active_mask)

    J = compute_projection_jacobian(
        point_in_space=point,
        para_coords=para_coord_star,
        degrees=degrees,
        coefficients=coefficients,
        knot_vectors=knots if (knots is not None) else None,
        bsp_eval=bsp_eval,
        surface_jacobian=surface_jacobian,
    )

    n_dims = J.shape[0]
    J_masked = J * (active_mask[:, None] & active_mask[None, :])
    J_aug = J_masked + jnp.eye(n_dims, dtype=J.dtype) * (~active_mask)
    res_dir_masked = res_dir * active_mask

    # Solve for parametric tangent: J * para_dot = - res_dir
    # para_dot = - jnp.linalg.solve(J, res_dir)
    para_dot = - jnp.linalg.solve(J_aug, res_dir_masked)

    # return primal and tangent (exact pair required by defjvp)
    return para_coord_star, para_dot


def _nearest_idx_for_point(bsp_pts: jnp.ndarray, query_pt: jnp.ndarray) -> jnp.ndarray:
    """
    Find the index of the nearest neighbor in bsp_pts to a single query_pt,
    using a fori_loop to keep memory footprint O(1) per point.
    """
    def body(i, carry):
        best_i, best_d2 = carry
        # compute squared distance to bsp_pts[i]
        d2 = jnp.sum((bsp_pts[i] - query_pt) ** 2)
        # update if smaller
        better = d2 < best_d2
        new_best_i = jax.lax.select(better, i, best_i)
        new_best_d2 = jax.lax.select(better, d2, best_d2)
        return (new_best_i, new_best_d2)

    # initialize with idx=0, distance=∞
    init = (jnp.array(0, dtype=jnp.int64), jnp.array(jnp.inf, dtype=bsp_pts.dtype))
    best_idx, _ = jax.lax.fori_loop(0, bsp_pts.shape[0], body, init)
    return best_idx

def brute_force_nn(bsp_pts: jnp.ndarray, 
                   queries: jnp.ndarray) -> jnp.ndarray:
    """
    For each row in queries (shape [Q, D]), returns the index of the closest
    point in bsp_pts (shape [P, D]). Result is an array of shape [Q].
    """
    # vmapped over the first axis of queries
    return jax.vmap(lambda q: _nearest_idx_for_point(bsp_pts, q))(queries)

class ProjectionOperationVJPVJP(csdl.experimental.CustomExplicitOperationBeta):
    def __init__(
            self,
            num_parametric_dimensions,
            degrees,
            knots,
            num_grid_search_points=100,
        ):
        super().__init__()
        self.num_parametric_dimensions = num_parametric_dimensions
        self.degrees = degrees
        self.knots = knots
        self.num_grid_search_points = num_grid_search_points

        def sensitivity_analysis_single(
            point,
            point_dot,
            coefficients,
            coefficients_dot,
            x_bar,
            para_coords_closest,
            degrees,
            knots,
        ):
            # Build vjp-wrapper for a single point. It returns (dpoint, dcoeffs) for the current x_bar.
            def vjp_wrapper(point_in, coeffs_in, xbar_in):
                # 1) compute primal para_coord_star for this point (use the same initial guess logic as the forward op)
                #    We reuse para_coords_closest[i] as the initial guess for Newton.
                para0 = jnp.array(para_coords_closest)
                para_star = implicit_solve_single(
                    point=point_in,
                    coefficients=coeffs_in,
                    para_coords_0=para0,
                    degrees=degrees,
                    knots=knots,
                )

                # 2) build primals tuple in the same shape ordering used by bwd:
                primals = (para_star, point_in, coeffs_in)

                # 3) call bwd to get the first-order VJP outputs (dpoint, dcoeffs)
                #    bwd returns (dpoint, dcoeffs, None)
                dpt, dcoeffs, _ = bwd(primals=primals, x_bar=xbar_in, degrees=degrees, knots=knots)
                # Return both outputs so jvp can linearize them both
                return dpt, dcoeffs

            # Now linearize vjp_wrapper via jax.jvp.
            # primals for vjp_wrapper: (point, coefficients, x_bar)
            primals_vw = (point, coefficients, x_bar)
            # tangents: how the inputs change — we use d_d_points[i] and d_d_control_points; assume x_bar perturbation = 0
            tangents_vw = (point_dot, coefficients_dot, jnp.zeros_like(x_bar))

            # perform jvp; it returns (primal_outs, tangent_outs)
            (primal_outs), (tangent_outs) = jax.jvp(vjp_wrapper, primals_vw, tangents_vw)

            # tangent_outs is a tuple (dpoint_dot, dcoeffs_dot)
            dpoint_dot, dcoeffs_dot = tangent_outs

            para0 = jnp.array(para_coords_closest)
            # primals and tangents for the implicit solve
            primals_solve = (point, coefficients, para0, degrees, knots)                # match ordering your defjvp expects
            tangents_solve = (point_dot, coefficients_dot, jnp.zeros_like(para0))

            # call iv_jvp directly: it returns (primal_para_star, para_dot)
            para_star_primal, para_dot = iv_jvp(primals_solve, tangents_solve)

            return dpoint_dot, dcoeffs_dot, para_dot

        self.sensitivity_analysis_single = sensitivity_analysis_single  

        self._cache = {
            "points" : None,
            "control_points" : None,
            "para_coords" : None,
        }

        # Perform grid search
        self.n = len(self.degrees)

        def generate_parametric_grid(N):
            samples_1d = []
            for U in self.knots:
                # take only the unique knots in sorted order
                knots = np.unique(U)
                # for each interval [knots[j], knots[j+1]), sample N points
                pts = []
                for j in range(len(knots) - 1):
                    a, b = knots[j], knots[j+1]
                    pts.append(np.linspace(a, b, N, endpoint=False))
                # finally include the very last knot
                pts.append(np.array([knots[-1]]))
                samples_1d.append(np.concatenate(pts))

            # build the d‐dimensional tensor grid
            mesh = np.meshgrid(*samples_1d, indexing="ij")
            # flatten each coordinate array and stack into shape (M, d)
            coord_arrays = [m.flatten() for m in mesh]
            grid = np.stack(coord_arrays, axis=-1)
            return grid
    
        self.para_grid = generate_parametric_grid(num_grid_search_points)

    def evaluate(self, inputs, d_outputs):
        points = inputs["points"].reshape(-1, 3)
        control_points = inputs["control_points"]
        d_para_coords = inputs["d_para_coords"]

        d_points = d_outputs["d_points"]
        d_control_points = d_outputs["d_control_points"]

        # Inputs from 1st VJP
        self.declare_input("points", points)
        self.declare_input("control_points", control_points)
        self.declare_input("d_para_coords", d_para_coords)

        # 2nd order "V"s
        self.declare_input("d_d_points", d_points)
        self.declare_input("d_d_control_points", d_control_points)

        d_d_points = self.create_output("d_vjp_points", points.shape)
        d_d_control_points = self.create_output("d_vjp_control_points", control_points.shape)
        d_d_para_cords = self.create_output("d_d_para_cords", d_para_coords.shape)

        d_inputs = {
            "points": d_d_points,
            "control_points": d_d_control_points,
            "d_para_coords": d_d_para_cords,
        }

        return d_inputs
    
    def compute(self, inputs, outputs):
        points = jnp.array(inputs["points"]).reshape((-1, 3))
        coefficients = jnp.array(inputs["control_points"])
        d_para_coords = jnp.array(inputs["d_para_coords"])

        d_d_points = jnp.array(inputs["d_d_points"])
        d_d_control_points = jnp.array(inputs["d_d_control_points"])

        save_name = f"points_shape_{points.shape}_coefficients_shape_{coefficients.shape}"
        para_coords = self.para_grid

        if f"{save_name}_basis_mat" in self._cache:
            basis_mat = self._cache[f"{save_name}_basis_mat"]
            # print("Using cached basis matrix.")
        else:
            # convert stored knots (hashable tuples) back to numpy arrays for
            # the NumPy implementation
            knot_vectors_np = tuple(np.array(U) for U in self.knots)
            basis_mat = compute_basis_matrix_numpy(
                us=para_coords,
                degrees=self.degrees,
                knot_vectors=knot_vectors_np,
            )
            self._cache[f"{save_name}_basis_mat"] = basis_mat
        bsp_eval = basis_mat @ coefficients.reshape(-1, coefficients.shape[-1])


        if f"{save_name}_nearest_indices" in self._cache:
            nearest_indices_fun = self._cache[f"{save_name}_nearest_indices"]
            # print("Using cached nearest indices REVERSE (2nd order VJP).")
        else:
            nearest_indices_fun = jax.jit(brute_force_nn)
            self._cache[f"{save_name}_nearest_indices"] = nearest_indices_fun

        nearest_indices = nearest_indices_fun(jnp.array(bsp_eval), points)
        para_coords_closest = para_coords[nearest_indices]


        if f"{save_name}_sensitivity_analysis" in self._cache:
            sensitivity_analysis_jitted = self._cache[f"{save_name}_sensitivity_analysis"]
            # print("Using cached sensitivity analysis single.")
        else:
            sensitivity_analysis_single = self.sensitivity_analysis_single
            sensitivity_analysis_batched = jax.vmap(
                sensitivity_analysis_single,
                in_axes=(0, 0, None, None, 0, 0, None, None),
            )
            # sensitivity_analysis_jitted = jax.jit(sensitivity_analysis_batched, static_argnums=(6, ))
            sensitivity_analysis_jitted = sensitivity_analysis_batched

            self._cache[f"{save_name}_sensitivity_analysis"] = sensitivity_analysis_jitted

        t1 = time.time()
        dpoint_dot, dcoeffs_dot, para_dot = sensitivity_analysis_jitted(
            points,
            d_d_points,
            coefficients,
            d_d_control_points,
            d_para_coords,
            para_coords_closest,
            self.degrees,
            self.knots,
        )
        t2 = time.time()
        print(f"Second-order sensitivity analysis took {t2 - t1:.4f} seconds.")

        outputs["d_vjp_points"] = jax.device_get(dpoint_dot)
        outputs["d_vjp_control_points"] = jax.device_get(jnp.sum(dcoeffs_dot, axis=0))  # accumulate control points
        outputs["d_d_para_cords"] = jax.device_get(para_dot)

class ProjectionOperationVJP(csdl.experimental.CustomExplicitOperationBeta):
    def __init__(
            self,
            num_parametric_dimensions,
            degrees,
            knots,
            num_grid_search_points=100,
        ):
        super().__init__()
        self.num_parametric_dimensions = num_parametric_dimensions
        self.degrees = degrees
        self.knots = knots
        self.num_grid_search_points = num_grid_search_points

        def sensitivity_analysis_single(
            point,
            coefficients,
            para_coords_closest,
            d_para_coords,
        ):
            para_coords_star = implicit_solve_single(
                point, coefficients, jnp.array(para_coords_closest),
                degrees=self.degrees, knots=self.knots
            )

            para_coords_star = para_coords_star.reshape(-1, len(self.degrees))

            primals = (para_coords_star, point, coefficients)

            dpoint, dcoeffs, _ = bwd(
                primals=primals,
                x_bar=d_para_coords,
                degrees=self.degrees,
                knots=self.knots,
            )

            return dpoint, dcoeffs
        self.sensitivity_analysis_single = sensitivity_analysis_single

        self._cache = {
            "points" : None,
            "control_points" : None,
            "para_coords" : None,
        }

        # Perform grid search
        self.n = len(self.degrees)

        def generate_parametric_grid(N):
            samples_1d = []
            for U in self.knots:
                # take only the unique knots in sorted order
                knots = np.unique(U)
                # for each interval [knots[j], knots[j+1]), sample N points
                pts = []
                for j in range(len(knots) - 1):
                    a, b = knots[j], knots[j+1]
                    pts.append(np.linspace(a, b, N, endpoint=False))
                # finally include the very last knot
                pts.append(np.array([knots[-1]]))
                samples_1d.append(np.concatenate(pts))

            # build the d‐dimensional tensor grid
            mesh = np.meshgrid(*samples_1d, indexing="ij")
            # flatten each coordinate array and stack into shape (M, d)
            coord_arrays = [m.flatten() for m in mesh]
            grid = np.stack(coord_arrays, axis=-1)
            return grid
    
        self.para_grid = generate_parametric_grid(num_grid_search_points)

    def evaluate(self, inputs, d_outputs):
        # print("evaluate first")
        points = inputs["points"].reshape(-1, 3)
        control_points = inputs["control_points"]
        d_para_coords = d_outputs["para_coords"]

        self.declare_input("points", points)
        self.declare_input("control_points", control_points)
        self.declare_input("d_para_coords", d_para_coords)

        d_points = self.create_output("d_points", points.shape)
        d_control_points = self.create_output("d_control_points", control_points.shape)

        self.declare_vjp_function(
            ProjectionOperationVJPVJP,
            num_parametric_dimensions=self.num_parametric_dimensions,
            knots=self.knots,
            degrees=self.degrees,
            num_grid_search_points=self.num_grid_search_points,
        )

        d_inputs = {
            "points": d_points,
            "control_points": d_control_points,
        }

        return d_inputs
    
    def compute(self, inputs, outputs):
        points = jnp.array(inputs["points"]).reshape((-1, 3))
        coefficients = jnp.array(inputs["control_points"])
        d_para_coords = jnp.array(inputs["d_para_coords"])

        save_name = f"points_shape_{points.shape}_coefficients_shape_{coefficients.shape}"
        para_coords = self.para_grid

        if f"{save_name}_basis_mat" in self._cache:
            basis_mat = self._cache[f"{save_name}_basis_mat"]
            # print("Using cached basis matrix.")
        else:
            knot_vectors_np = tuple(np.array(U) for U in self.knots)
            basis_mat = compute_basis_matrix_numpy(
                us=para_coords,
                degrees=self.degrees,
                knot_vectors=knot_vectors_np,
            )
            self._cache[f"{save_name}_basis_mat"] = basis_mat
        bsp_eval = basis_mat @ coefficients.reshape(-1, coefficients.shape[-1])

        if f"{save_name}_nearest_indices" in self._cache:
            nearest_indices_fun = self._cache[f"{save_name}_nearest_indices"]
            # print("Using cached nearest indices REVERSE.")
        else:
            nearest_indices_fun = jax.jit(brute_force_nn)
            self._cache[f"{save_name}_nearest_indices"] = nearest_indices_fun

        nearest_indices = nearest_indices_fun(jnp.array(bsp_eval), points)
        para_coords_closest = para_coords[nearest_indices]

        if f"{save_name}_sensitivity_analysis" in self._cache:
            sensitivity_analysis_jitted = self._cache[f"{save_name}_sensitivity_analysis"]
            # print("Using cached sensitivity analysis single.")
        else:
            sensitivity_analysis_single = self.sensitivity_analysis_single
            sensitivity_analysis_batched = jax.vmap(
                sensitivity_analysis_single,
                in_axes=(0, None, 0, 0),
            )
            # sensitivity_analysis_jitted = jax.jit(sensitivity_analysis_batched, static_argnums=(2, ))
            sensitivity_analysis_jitted = sensitivity_analysis_batched

            self._cache[f"{save_name}_sensitivity_analysis"] = sensitivity_analysis_jitted

        dpoint, dcoeffs = sensitivity_analysis_jitted(
            points,
            coefficients,
            para_coords_closest,
            d_para_coords,
        )
        

        outputs["d_points"] = jax.device_get(dpoint)
        outputs["d_control_points"] = jax.device_get(jnp.sum(dcoeffs, axis=0)) # accumulate control points

class ProjectionOperation(csdl.experimental.CustomExplicitOperationBeta):
    def __init__(
            self,
            num_parametric_dimensions,
            degrees,
            knots,
            num_grid_search_points=100,
        ):
        super().__init__()
        self.num_parametric_dimensions = num_parametric_dimensions
        self.degrees = degrees
        self.knots = knots
        self.num_grid_search_points = num_grid_search_points
        self.res_jac = None
        self.mask = None
        self.final_coords = None

        def point_to_b_spline_projection_single(point, para_coords, coefficients):
            return compute_point_to_bspline_projection(
                point=point,
                degrees=degrees,
                coefficients=coefficients,
                para_coords=para_coords,
                knots=knots,
            )
        self.point_to_b_spline_projection_batched = jax.vmap(point_to_b_spline_projection_single, in_axes=(0, 0, None), out_axes=0)

        self._cache = {
            "points" : None,
            "control_points" : None,
            "para_coords" : None,
        }

        # Perform grid search
        self.n = len(self.degrees)

        def generate_parametric_grid(N):
            samples_1d = []
            for U in self.knots:
                # take only the unique knots in sorted order
                knots = np.unique(U)
                # for each interval [knots[j], knots[j+1]), sample N points
                pts = []
                for j in range(len(knots) - 1):
                    a, b = knots[j], knots[j+1]
                    pts.append(np.linspace(a, b, N, endpoint=False))
                # finally include the very last knot
                pts.append(np.array([knots[-1]]))
                samples_1d.append(np.concatenate(pts))

            # build the d‐dimensional tensor grid
            mesh = np.meshgrid(*samples_1d, indexing="ij")
            # flatten each coordinate array and stack into shape (M, d)
            coord_arrays = [m.flatten() for m in mesh]
            grid = np.stack(coord_arrays, axis=-1)
            return grid
    
        self.para_grid = generate_parametric_grid(num_grid_search_points)

    def evaluate(self, points, control_points):
        points = points.reshape(-1, 3)
        num_points = points.shape[0]
        self.declare_input("points", points)
        self.declare_input("control_points", control_points)
        
        para_coords = self.create_output("para_coords", (num_points, self.num_parametric_dimensions))

        self.declare_vjp_function(
            ProjectionOperationVJP,
            num_parametric_dimensions=self.num_parametric_dimensions,
            knots=self.knots,
            degrees=self.degrees,
            num_grid_search_points=self.num_grid_search_points,
        )

        return para_coords

    def compute(self, inputs, outputs):
        points = jnp.array(inputs["points"])
        control_points = jnp.array(inputs["control_points"])

        save_name = f"points_shape_{points.shape}_control_points_shape_{control_points.shape}"
        para_coords = self.para_grid

        if f"{save_name}_basis_mat" in self._cache:
            basis_mat = self._cache[f"{save_name}_basis_mat"]
            # print("Using cached basis matrix.")
        else:
            knot_vectors_np = tuple(np.array(U) for U in self.knots)
            basis_mat = compute_basis_matrix_numpy(
                us=para_coords,
                degrees=self.degrees,
                knot_vectors=knot_vectors_np,
            )
            self._cache[f"{save_name}_basis_mat"] = basis_mat
        bsp_eval = basis_mat @ control_points.reshape(-1, control_points.shape[-1])

        if f"{save_name}_nearest_indices" in self._cache:
            nearest_indices_fun = self._cache[f"{save_name}_nearest_indices"]
            print("Using cached nearest indices FORWARD.")
        else:
            nearest_indices_fun = jax.jit(brute_force_nn)
            self._cache[f"{save_name}_nearest_indices"] = nearest_indices_fun

        nearest_indices = nearest_indices_fun(jnp.array(bsp_eval), points)
        para_coords_closest = para_coords[nearest_indices]


        if f"{save_name}_projection_fun" in self._cache:
            projection_fun = self._cache[f"{save_name}_projection_fun"]
            # print("Using cached projection function.")
        else:
            projection_fun = jax.jit(
                self.point_to_b_spline_projection_batched,
            )
            self._cache[f"{save_name}_projection_fun"] = projection_fun

        final_coords, final_res, converged, final_i, J, mask, _ = projection_fun(
            jnp.array(points),
            jnp.array(para_coords_closest),
            jnp.array(control_points),
        )

        self.mask = mask

        if not converged.all():
            print(f"Warning: {np.sum(~converged)} out of {len(converged)} projection points did not converge.")
            print("Initial guess for these points was:", para_coords_closest[~converged])
            print("Final parameter coordinates for these points were:", final_coords[~converged])
            print("Final residuals for these points were:", final_res[~converged])
            print("Final iteration counts for these points were:", final_i[~converged])
            print("Jacobian for these points was:", J[~converged])

        outputs["para_coords"] = jax.device_get(final_coords)

if __name__ == "__main__":
    import numpy as np
    np.random.seed(42)  # For reproducibility


    rec = csdl.Recorder(inline=True)
    rec.start()

    # Define the B-spline space parameters
    num_cp_x = 10
    num_cp_y = 8
    nx = num_cp_x - 1  # Number of control points - 1
    ny = num_cp_y - 1  # Number of control points - 1
    px = 3  # Degree of the B-spline
    py = 2  # Degree of the B-spline
    p = (px, py)
    coefficients_shape = (num_cp_x, num_cp_y)
    derivative_orders = (1, 0)  # # derivatives for the evaluation

    knots_x = np.concatenate(
        [np.zeros(px), 
         np.linspace(0, 1, num_cp_x - px + 1), 
         np.ones(px)]
    )
    knots_y = np.concatenate(
        [np.zeros(py), 
         np.linspace(0, 1, num_cp_y - py + 1), 
         np.ones(py)]
    )

    # Make knot vectors hashable Python tuples so they can be used as static
    # arguments to jax.jit. Functions in this module convert them back to
    # JAX arrays when needed.
    knots = (
        tuple(knots_x.tolist()),
        tuple(knots_y.tolist()),
    )
    knots_jnp = (jnp.array(knots_x), jnp.array(knots_y))  # JAX-compatible knot vectors


    test_dv = csdl.Variable(
        name="test_design_variable",
        value=np.array([0.25, 0., 0.5]),
    )
    # test_dv.set_as_design_variable()

    coeffs_x, coeffs_y = np.meshgrid(np.linspace(0, 5, num_cp_x), np.linspace(0, 2, num_cp_y), indexing='ij')
    coeffs = np.array(np.stack((coeffs_x, coeffs_y, 0.2 * np.random.rand(num_cp_x, num_cp_y)), axis=-1))
    coeffs_jnp = jnp.array(coeffs)
    coeffs_csdl = csdl.Variable(name="coefficients", value=coeffs)
    coeffs_csdl = coeffs_csdl
    # test_dv = csdl.expand(test_dv, out_shape=coeffs_csdl.shape, action="k->ijk")
    # coeffs_csdl = coeffs_csdl + test_dv
    coeffs_csdl.set_as_design_variable()


    random_points_in_space =  np.random.rand(100, 3) # [1, :].reshape(-1, 3)  # Random points in space
    random_points_in_space[:, 0] *= 5
    random_points_in_space[:, 1] *= 2
    random_points_in_space[:, 2] = 5

    random_points_in_space_csdl = csdl.Variable(name="points_to_project", value=random_points_in_space)
    random_points_in_space_csdl.set_as_design_variable()

    projection_op = ProjectionOperation(
        num_parametric_dimensions=2,
        degrees=p,
        knots=knots,
        num_grid_search_points=100,
    )

    uvs = projection_op.evaluate(
        points=random_points_in_space_csdl,
        control_points=coeffs_csdl,
    )
    uvs_sum = csdl.sum(uvs)
    d_uvs_sum_d_coeffs = csdl.derivative(uvs_sum, coeffs_csdl)
    d_uvs_sum_d_coeffs_sum = csdl.sum(d_uvs_sum_d_coeffs)
    d_uvs_sum_d_coeffs_sum.name = "projection_derivative (objective)"
    d_uvs_sum_d_coeffs_sum.set_as_objective()

    sim = csdl.experimental.JaxSimulator(recorder=rec, gpu=False)
    # sim = csdl.experimental.PySimulator(recorder=rec)
    sim.check_optimization_derivatives(step_size=1e-8, raise_on_error=False)

    print("Parametric coordinates:", uvs.value)

