
"""
Optimized B-spline surface projection (point -> parametric coords) for:
  1) High-efficiency Gauss-Newton / Levenberg–Marquardt (LM) distance minimization
  2) JAX implementation (jit + vmap friendly)
  3) NumPy vectorized implementation

Key idea (efficiency):
- Avoid second derivatives (surface Hessians) entirely.
- Minimize squared distance:  f(ξ) = 1/2 || S(ξ) - p ||^2
  with residual r = S(ξ) - p  (phys-dim, e.g. 3)
  Jacobian J = dS/dξ          (phys-dim x n_param, e.g. 3x2)
- Gauss-Newton step: (J^T J) δ = - J^T r
- LM step: (J^T J + λ I) δ = - J^T r

This is usually faster and more robust than Newton on the orthogonality conditions,
because it only requires first derivatives and yields symmetric SPD 2x2 systems.

Dependencies:
- JAX path uses an evaluator callable for S and partials:
    evaluate_b_spline_jax(us, degrees, knot_vectors, coeffs, der_orders=None)
  If you have the factory version, you can substitute it for extra performance.

- NumPy path expects an evaluator for S and partials; you can plug in your
  compute_basis_matrix_numpy_factory stencil-based evaluator.

Both implementations implement:
- box constraints ξ ∈ [0,1]^n via clipping
- simple active-set-like masking near bounds to keep steps feasible
- early stopping (optional) using a convergence mask
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as _np

# ----------------------------
# JAX implementation
# ----------------------------
try:
    import jax
    import jax.numpy as jnp
except Exception:  # pragma: no cover
    jax = None
    jnp = None

# Import your JAX evaluator if available.
try:
    from lsdo_function_spaces.core.spaces.non_cython_bsplines.compute_basis_matrix_jax_stencil import (
        evaluate_b_spline_jax,
    )
except Exception:  # pragma: no cover
    evaluate_b_spline_jax = None


@dataclass(frozen=True)
class LMParams:
    max_iter: int = 50
    tol_grad: float = 1e-12          # ||g|| threshold
    tol_step: float = 1e-12          # ||δ|| threshold
    lambda0: float = 1e-3            # initial damping
    lambda_min: float = 1e-12
    lambda_max: float = 1e12
    lambda_up: float = 10.0          # multiply λ when step not accepted
    lambda_down: float = 0.1         # multiply λ when step accepted
    accept_ratio: float = 1e-4       # predicted vs actual improvement threshold
    # bounds handling:
    use_active_set: bool = True
    bound_eps: float = 0.0           # treat u<=eps as lower-active, u>=1-eps as upper-active


def _ensure_knot_arrays_jax(knots):
    # Allow knots as tuple-of-tuples floats; convert once.
    return tuple(jnp.array(k) for k in knots)


def _eval_surface_and_jac_jax(
    u: "jnp.ndarray",
    degrees: Tuple[int, ...],
    coeffs: "jnp.ndarray",
    knots: Tuple[Sequence[float], ...],
) -> Tuple["jnp.ndarray", "jnp.ndarray"]:
    """
    Evaluate S(u) and J(u) where J is (phys_dim, n_param).
    u: (n_param,)
    returns:
      S: (phys_dim,)
      J: (phys_dim, n_param)
    """
    n_param = len(degrees)
    uu = u.reshape(1, n_param)

    # knots are expected to already be JAX arrays when called from the factory;
    # if you pass Python tuples-of-floats directly, they will be converted once in the factory.
    S = evaluate_b_spline_jax(us=uu, degrees=degrees, knot_vectors=knots, coeffs=coeffs)[0]  # (phys,)
    # partials
    dS_list = []
    for i in range(n_param):
        der = tuple(1 if k == i else 0 for k in range(n_param))
        dS_i = evaluate_b_spline_jax(us=uu, degrees=degrees, knot_vectors=knots, coeffs=coeffs, der_orders=der)[0]
        dS_list.append(dS_i)
    J = jnp.stack(dS_list, axis=1)  # (phys, n_param)
    return S, J


def _solve_2x2_spd_jax(A: "jnp.ndarray", b: "jnp.ndarray") -> "jnp.ndarray":
    """
    Solve A x = b for A (2,2) SPD, vector b (2,).
    Uses explicit formula for speed and stability in small systems.
    """
    a00, a01 = A[0, 0], A[0, 1]
    a10, a11 = A[1, 0], A[1, 1]
    det = a00 * a11 - a01 * a10
    # If det is tiny, fallback to linalg.solve (rare if λ>0)
    def _fallback(_):
        return jnp.linalg.solve(A, b)
    def _explicit(_):
        inv = jnp.array([[ a11, -a01],
                         [-a10,  a00]]) / det
        return inv @ b
    return jax.lax.cond(jnp.abs(det) < 1e-30, _fallback, _explicit, operand=0)


def _active_mask_jax(u: "jnp.ndarray", g: "jnp.ndarray", eps: float):
    # active where free to move
    lower_active = u <= eps
    upper_active = u >= (1.0 - eps)
    # if at lower bound and gradient wants to decrease u (g < 0? depends on sign)
    # We solve δ from A δ = -g, so desired move is roughly -g.
    # If u at lower bound and -g would be negative -> g positive blocks.
    block_lower = lower_active & (g > 0.0)
    block_upper = upper_active & (g < 0.0)
    inactive = block_lower | block_upper | (g == 0.0)
    return ~inactive


def project_point_gauss_newton_jax(
    point: "jnp.ndarray",
    u0: "jnp.ndarray",
    degrees: Tuple[int, ...],
    coeffs: "jnp.ndarray",
    knots: Tuple[Sequence[float], ...],
    *,
    max_iter: int = 50,
    tol_grad: float = 1e-12,
    tol_step: float = 1e-12,
    use_active_set: bool = True,
    bound_eps: float = 0.0,
):
    """
    Fast Gauss-Newton projection for one point.
    Returns u*, converged, n_iter
    """
    knots = _ensure_knot_arrays_jax(knots)
    if evaluate_b_spline_jax is None:
        raise ImportError("evaluate_b_spline_jax not found; check your imports.")

    n_param = len(degrees)
    u0 = u0.reshape(n_param,)

    def body(i, state):
        u, converged = state
        S, J = _eval_surface_and_jac_jax(u, degrees, coeffs, knots)
        r = S - point
        g = J.T @ r  # (n_param,)

        # active-set masking (keeps a fixed-size solve)
        if use_active_set:
            active = _active_mask_jax(u, g, bound_eps)
        else:
            active = jnp.ones((n_param,), dtype=bool)

        # Build normal equations A = J^T J
        A = J.T @ J  # (n_param,n_param)
        # Mask inactive directions by adding identity on inactive coords.
        I_inactive = jnp.diag((~active).astype(A.dtype))
        A_mask = A * (active[:, None] & active[None, :]) + I_inactive
        g_mask = g * active

        # Solve A δ = -g
        rhs = -g_mask
        if n_param == 2:
            delta = _solve_2x2_spd_jax(A_mask, rhs)
        else:
            delta = jnp.linalg.solve(A_mask, rhs)

        delta = delta * active
        u_new = jnp.clip(u + delta, 0.0, 1.0)

        g_norm = jnp.linalg.norm(g_mask)
        d_norm = jnp.linalg.norm(delta)

        converged_new = (g_norm < tol_grad) | (d_norm < tol_step)
        return (u_new, converged_new)

    # Use fori_loop with an early-stop emulation: keep updating but freeze when converged
    def scan_body(carry, i):
        u, converged = carry
        u_new, conv_new = body(i, (u, converged))
        u_out = jnp.where(converged, u, u_new)
        conv_out = converged | conv_new
        return (u_out, conv_out), None

    (u_star, conv), _ = jax.lax.scan(scan_body, (u0, False), jnp.arange(max_iter))
    n_iter = max_iter  # scan doesn't easily return first-stop without extra work
    return u_star, conv, n_iter


def project_point_lm_jax(
    point: "jnp.ndarray",
    u0: "jnp.ndarray",
    degrees: Tuple[int, ...],
    coeffs: "jnp.ndarray",
    knots: Tuple[Sequence[float], ...],
    params: LMParams = LMParams(),
):
    """
    Levenberg–Marquardt projection for one point (distance minimization).

    Uses a simple acceptance test based on actual decrease in 0.5||r||^2
    and predicted decrease from quadratic model.

    Returns:
      u*, converged, n_iter, final_lambda
    """
    knots = _ensure_knot_arrays_jax(knots)
    if evaluate_b_spline_jax is None:
        raise ImportError("evaluate_b_spline_jax not found; check your imports.")

    n_param = len(degrees)
    u0 = u0.reshape(n_param,)

    def one_iter(carry, _):
        u, lam, converged = carry

        S, J = _eval_surface_and_jac_jax(u, degrees, coeffs, knots)
        r = S - point
        f = 0.5 * (r @ r)

        g = J.T @ r  # (n_param,)
        A = J.T @ J  # (n_param,n_param)
        A_lm = A + lam * jnp.eye(n_param, dtype=A.dtype)

        # active-set
        if params.use_active_set:
            active = _active_mask_jax(u, g, params.bound_eps)
        else:
            active = jnp.ones((n_param,), dtype=bool)
        I_inactive = jnp.diag((~active).astype(A.dtype))
        A_mask = A_lm * (active[:, None] & active[None, :]) + I_inactive
        g_mask = g * active

        rhs = -g_mask
        if n_param == 2:
            delta = _solve_2x2_spd_jax(A_mask, rhs)
        else:
            delta = jnp.linalg.solve(A_mask, rhs)
        delta = delta * active

        u_trial = jnp.clip(u + delta, 0.0, 1.0)

        # Evaluate trial objective
        S_t, _ = _eval_surface_and_jac_jax(u_trial, degrees, coeffs, knots)
        r_t = S_t - point
        f_t = 0.5 * (r_t @ r_t)

        # Predicted reduction (quadratic model): m(0)-m(δ) ≈ -g^T δ - 0.5 δ^T A δ
        # Use undamped A (GN Hessian approx) for prediction.
        pred = -(g_mask @ delta) - 0.5 * (delta @ (A @ delta))
        act = f - f_t

        # Accept if actual improvement positive and ratio decent
        ratio = jnp.where(pred > 0, act / pred, 0.0)
        accept = (act > 0) & (ratio > params.accept_ratio)

        u_new = jnp.where(accept, u_trial, u)
        lam_new = jnp.where(accept, lam * params.lambda_down, lam * params.lambda_up)
        lam_new = jnp.clip(lam_new, params.lambda_min, params.lambda_max)

        g_norm = jnp.linalg.norm(g_mask)
        d_norm = jnp.linalg.norm(delta)
        converged_new = (g_norm < params.tol_grad) | (d_norm < params.tol_step)

        # Freeze if converged already
        u_out = jnp.where(converged, u, u_new)
        lam_out = jnp.where(converged, lam, lam_new)
        conv_out = converged | converged_new
        return (u_out, lam_out, conv_out), None

    (u_star, lam_star, conv), _ = jax.lax.scan(
        one_iter, (u0, params.lambda0, False), xs=None, length=params.max_iter
    )
    return u_star, conv, params.max_iter, lam_star


def make_projector_lm_jax(
    degrees: Tuple[int, ...],
    knots: Tuple[Sequence[float], ...],
    *,
    params: LMParams = LMParams(),
    jit: bool = True,
):
    """
    Factory that returns a batched projector:
      proj(points, u0s, coeffs) -> (u*, converged)

    points: (M, phys_dim)
    u0s:    (M, n_param)
    coeffs: (n_ctrl, phys_dim) or (..control net.., phys_dim) flattened inside evaluate_b_spline_jax
    """
    knots = _ensure_knot_arrays_jax(knots)
    if jax is None:
        raise ImportError("JAX is not available.")
    if evaluate_b_spline_jax is None:
        raise ImportError("evaluate_b_spline_jax not found; check your imports.")

    def _single(point, u0, coeffs):
        return project_point_lm_jax(point, u0, degrees, coeffs, knots, params)

    vmapped = jax.vmap(_single, in_axes=(0, 0, None))

    def proj(points, u0s, coeffs):
        u_star, conv, _, _ = vmapped(points, u0s, coeffs)
        return u_star, conv

    if jit:
        # degrees/knots captured in closure => static
        proj = jax.jit(proj)
    return proj


# ----------------------------
# NumPy implementation
# ----------------------------
try:
    # If you have your optimized stencil evaluator factory, you can swap it in.
    from lsdo_function_spaces.core.spaces.non_cython_bsplines.compute_basis_matrix_numpy_factory_patched import make_bspline_evaluator_numpy
except Exception:  # pragma: no cover
    make_bspline_evaluator_numpy = None


def _solve_2x2_spd_numpy(A: _np.ndarray, b: _np.ndarray) -> _np.ndarray:
    """
    Vectorized solve for batches of 2x2 systems:
      A: (M,2,2) SPD-ish
      b: (M,2)
    returns x: (M,2)
    """
    a00 = A[:, 0, 0]
    a01 = A[:, 0, 1]
    a10 = A[:, 1, 0]
    a11 = A[:, 1, 1]
    det = a00 * a11 - a01 * a10
    # safe inverse
    inv00 = a11 / det
    inv01 = -a01 / det
    inv10 = -a10 / det
    inv11 = a00 / det
    x0 = inv00 * b[:, 0] + inv01 * b[:, 1]
    x1 = inv10 * b[:, 0] + inv11 * b[:, 1]
    return _np.stack([x0, x1], axis=1)


def _active_mask_numpy(u: _np.ndarray, g: _np.ndarray, eps: float) -> _np.ndarray:
    lower_active = u <= eps
    upper_active = u >= (1.0 - eps)
    block_lower = lower_active & (g > 0.0)
    block_upper = upper_active & (g < 0.0)
    inactive = block_lower | block_upper | (g == 0.0)
    return ~inactive


def _build_masked_lm_system_numpy(
    A: _np.ndarray,
    g: _np.ndarray,
    active: _np.ndarray,
    lam: _np.ndarray,
) -> tuple[_np.ndarray, _np.ndarray]:
    """Build the masked LM linear systems.

    Active coordinates use the damped GN system. Inactive coordinates are
    frozen with an identity diagonal and zero RHS so the system remains SPD.
    """
    M, n_param, _ = A.shape
    A_lm = A.copy()
    idx = _np.arange(n_param)
    A_lm[:, idx, idx] += lam[:, None]

    mask2 = active[:, :, None] & active[:, None, :]
    A_mask = A_lm * mask2.astype(A.dtype)
    for k in range(n_param):
        A_mask[:, k, k] += (~active[:, k]).astype(A.dtype)

    rhs = -(g * active)
    return A_mask, rhs


def make_projector_lm_numpy(
    degrees: Tuple[int, ...],
    knot_vectors: Tuple[_np.ndarray, ...],
    der_orders: Optional[Tuple[int, ...]] = None,
):
    """
    Factory returning evaluators for S(u) and partials using your stencil-based NumPy evaluator.

    Returns:
      eval_S(us, coeffs)      -> (M, phys_dim)
      eval_partials(us, coeffs) -> (M, phys_dim, n_param)
    """
    if make_bspline_evaluator_numpy is None:
        raise ImportError("make_bspline_evaluator_numpy not found; ensure compute_basis_matrix_numpy_factory.py is on path.")

    eval_S = make_bspline_evaluator_numpy(degrees=degrees, knot_vectors=knot_vectors, der_orders=None)

    n_param = len(degrees)
    eval_d = []
    for i in range(n_param):
        der = tuple(1 if k == i else 0 for k in range(n_param))
        eval_d.append(make_bspline_evaluator_numpy(degrees=degrees, knot_vectors=knot_vectors, der_orders=der))

    def eval_partials(us, coeffs):
        # stack along last axis -> (M, phys_dim, n_param)
        parts = [fn(us, coeffs) for fn in eval_d]  # list of (M, phys)
        return _np.stack(parts, axis=2)

    return eval_S, eval_partials


def project_points_gauss_newton_numpy(
    points: _np.ndarray,          # (M, phys_dim)
    u0s: _np.ndarray,             # (M, n_param)
    coeffs: _np.ndarray,
    degrees: Tuple[int, ...],
    knot_vectors: Tuple[_np.ndarray, ...],
    *,
    max_iter: int = 50,
    tol_grad: float = 1e-12,
    tol_step: float = 1e-12,
    use_active_set: bool = True,
    bound_eps: float = 0.0,
):
    """
    Vectorized Gauss-Newton projection for many points (NumPy).
    Returns:
      u: (M, n_param)
      converged: (M,)
    """
    M = points.shape[0]
    n_param = len(degrees)

    eval_S, eval_partials = make_projector_lm_numpy(degrees, knot_vectors)

    u = _np.clip(u0s.copy(), 0.0, 1.0)
    converged = _np.zeros((M,), dtype=bool)

    for _ in range(max_iter):
        S = eval_S(u, coeffs)                         # (M, phys)
        J = eval_partials(u, coeffs)                  # (M, phys, n_param)
        r = S - points                                # (M, phys)

        # g = J^T r -> (M, n_param)
        g = _np.einsum("mpk,mp->mk", J, r)

        if use_active_set:
            active = _active_mask_numpy(u, g, bound_eps)  # (M, n_param)
        else:
            active = _np.ones_like(u, dtype=bool)

        # A = J^T J -> (M, n_param, n_param)
        A = _np.einsum("mpk, mpl -> mkl", J, J)

        # Mask inactive dirs by adding identity on inactive coords
        # For n_param==2, do it explicitly.
        if n_param != 2:
            # generic: add diag(~active)
            for k in range(n_param):
                A[:, k, k] += (~active[:, k]).astype(A.dtype)
            # also zero off-diags for inactive pairs
            mask2 = active[:, :, None] & active[:, None, :]
            A = A * mask2
            rhs = -(g * active)
            # solve batch using np.linalg.solve (works for small n_param)
            delta = _np.linalg.solve(A, rhs[..., None]).squeeze(-1)
        else:
            mask2 = active[:, :, None] & active[:, None, :]
            A_mask = A * mask2
            A_mask[:, 0, 0] += (~active[:, 0]).astype(A.dtype)
            A_mask[:, 1, 1] += (~active[:, 1]).astype(A.dtype)

            rhs = -(g * active)
            delta = _solve_2x2_spd_numpy(A_mask, rhs)
            delta *= active

        u_new = _np.clip(u + delta, 0.0, 1.0)

        g_norm = _np.linalg.norm(g * active, axis=1)
        d_norm = _np.linalg.norm(delta, axis=1)
        conv_new = (g_norm < tol_grad) | (d_norm < tol_step)

        # freeze those already converged
        u = _np.where(converged[:, None], u, u_new)
        converged = converged | conv_new

        if converged.all():
            break

    return u, converged


def project_points_lm_numpy(
    points: _np.ndarray,
    u0s: _np.ndarray,
    coeffs: _np.ndarray,
    degrees: Tuple[int, ...],
    knot_vectors: Tuple[_np.ndarray, ...],
    *,
    params: LMParams = LMParams(),
):
    """
    Vectorized Levenberg–Marquardt projection for many points (NumPy).
    Uses per-point damping λ (vector of length M).

    Returns:
      u: (M, n_param)
      converged: (M,)
      lam: (M,)
      residual: (M, phys_dim) with residual = S(u) - points at the returned iterate
    """
    M = points.shape[0]
    n_param = len(degrees)

    eval_S, eval_partials = make_projector_lm_numpy(degrees, knot_vectors)

    u = _np.clip(u0s.copy(), 0.0, 1.0)
    lam = _np.full((M,), params.lambda0, dtype=float)
    converged = _np.zeros((M,), dtype=bool)

    for _ in range(params.max_iter):
        S = eval_S(u, coeffs)                         # (M, phys)
        J = eval_partials(u, coeffs)                  # (M, phys, n_param)
        r = S - points                                # (M, phys)
        f = 0.5 * _np.einsum("mp,mp->m", r, r)       # (M,)

        g = _np.einsum("mpk,mp->mk", J, r)           # (M, n_param)
        A = _np.einsum("mpk,mpl->mkl", J, J)         # (M, n_param, n_param)

        if params.use_active_set:
            active = _active_mask_numpy(u, g, params.bound_eps)
        else:
            active = _np.ones_like(u, dtype=bool)

        A_mask, rhs = _build_masked_lm_system_numpy(A, g, active, lam)

        if n_param == 2:
            delta = _solve_2x2_spd_numpy(A_mask, rhs)
        else:
            delta = _np.linalg.solve(A_mask, rhs[..., None]).squeeze(-1)
        delta *= active

        u_trial = _np.clip(u + delta, 0.0, 1.0)
        delta_eff = u_trial - u

        S_t = eval_S(u_trial, coeffs)
        r_t = S_t - points
        f_t = 0.5 * _np.einsum("mp,mp->m", r_t, r_t)

        # Predicted reduction from the *damped* local LM model evaluated at the
        # effective (possibly clipped) step.
        g_mask = g * active
        A_active = A * (active[:, :, None] & active[:, None, :]).astype(A.dtype)
        g_dot_d = _np.einsum("mk,mk->m", g_mask, delta_eff)
        Ad = _np.einsum("mkl,ml->mk", A_active, delta_eff)
        dAd = _np.einsum("mk,mk->m", delta_eff, Ad)
        d2 = _np.einsum("mk,mk->m", delta_eff, delta_eff)
        pred = -(g_dot_d + 0.5 * dAd + 0.5 * lam * d2)

        act = f - f_t
        # ratio = _np.where(pred > 0.0, act / pred, -_np.inf)
        pred_tol = 1e-14
        pred_safe = _np.where(_np.isfinite(pred) & (pred > pred_tol), pred, _np.nan)
        ratio = act / pred_safe
        ratio = _np.where(_np.isfinite(ratio), ratio, _np.inf)

        accept = (act > 0.0) & (ratio > params.accept_ratio)

        u_new = _np.where(accept[:, None], u_trial, u)
        lam_new = _np.where(accept, lam * params.lambda_down, lam * params.lambda_up)
        lam_new = _np.clip(lam_new, params.lambda_min, params.lambda_max)

        accepted_step_norm = _np.linalg.norm(_np.where(accept[:, None], delta_eff, 0.0), axis=1)
        g_norm = _np.linalg.norm(g_mask, axis=1)
        conv_new = (g_norm < params.tol_grad) | (accept & (accepted_step_norm < params.tol_step))

        u = _np.where(converged[:, None], u, u_new)
        lam = _np.where(converged, lam, lam_new)
        converged = converged | conv_new

        if converged.all():
            break

    residual = g_norm # eval_S(u, coeffs) - points
    return u, converged, lam, residual


if __name__ == "__main__":
    import numpy as np
    np.random.seed(42)  # For reproducibility

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

    coeffs_x, coeffs_y = np.meshgrid(np.linspace(0, 5, num_cp_x), np.linspace(0, 2, num_cp_y), indexing='ij')
    coeffs = np.array(np.stack((coeffs_x, coeffs_y, 0.2 * np.random.rand(num_cp_x, num_cp_y)), axis=-1))
    coeffs_jnp = jnp.array(coeffs)

    random_points_in_space =  np.random.rand(100, 3) # [1, :].reshape(-1, 3)  # Random points in space
    random_points_in_space[:, 0] *= 5
    random_points_in_space[:, 1] *= 2
    random_points_in_space[:, 2] = 5
