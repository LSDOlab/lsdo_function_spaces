# Inverse Point & Surface Projection

A critical operation in geometric modeling and multidisciplinary mesh mapping is **point projection**: finding the parametric coordinates $\mathbf{u}^* = (u^*, v^*)$ on a spline surface $\mathbf{S}(u, v)$ closest to an arbitrary query point $\mathbf{P} \in \mathbb{R}^3$, or projected along a specified direction $\mathbf{d}$ {cite:p}`fletcher2026implicit`.

---

## Closest-Point Orthogonal Projection

The unconstrained orthogonal projection problem seeks the parameter values minimizing Euclidean distance:

$$\min_{\mathbf{u}} f(\mathbf{u}) = \frac{1}{2} \|\mathbf{S}(\mathbf{u}) - \mathbf{P}\|^2$$

subject to the box bounds:

$$u_{\min} \le u \le u_{\max}, \quad v_{\min} \le v \le v_{\max}$$

### First-Order Optimality Conditions

At an interior local minimum, the gradient $\nabla f(\mathbf{u}) = \mathbf{0}$, yielding the orthogonality conditions:

$$\mathbf{r}(\mathbf{u}) = \begin{bmatrix} (\mathbf{S}(\mathbf{u}) - \mathbf{P}) \cdot \mathbf{S}_u(\mathbf{u}) \\ (\mathbf{S}(\mathbf{u}) - \mathbf{P}) \cdot \mathbf{S}_v(\mathbf{u}) \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

where $\mathbf{S}_u = \frac{\partial \mathbf{S}}{\partial u}$ and $\mathbf{S}_v = \frac{\partial \mathbf{S}}{\partial v}$ are the surface tangent vectors.

---

## Newton-Raphson Iterative Solution

The nonlinear system $\mathbf{r}(\mathbf{u}) = \mathbf{0}$ is solved iteratively using Newton-Raphson iterations:

$$\mathbf{u}^{(k+1)} = \mathbf{u}^{(k)} - \mathbf{J}(\mathbf{u}^{(k)})^{-1} \mathbf{r}(\mathbf{u}^{(k)})$$

where the Jacobian $\mathbf{J} \in \mathbb{R}^{2 \times 2}$ is given by:

$$\mathbf{J} = \begin{bmatrix} \|\mathbf{S}_u\|^2 + (\mathbf{S} - \mathbf{P}) \cdot \mathbf{S}_{uu} & \mathbf{S}_u \cdot \mathbf{S}_v + (\mathbf{S} - \mathbf{P}) \cdot \mathbf{S}_{uv} \\ \mathbf{S}_u \cdot \mathbf{S}_v + (\mathbf{S} - \mathbf{P}) \cdot \mathbf{S}_{uv} & \|\mathbf{S}_v\|^2 + (\mathbf{S} - \mathbf{P}) \cdot \mathbf{S}_{vv} \end{bmatrix}$$

---

## Two-Stage Robust Algorithm

To ensure global convergence and avoid local minima or boundary divergence, `lsdo_function_spaces` implements a robust two-stage algorithm:

1. **Sectioned Grid Search Initial Guess**:
   The surface is evaluated over a discretely sampled grid. For each query point $\mathbf{P}_i$, the closest grid evaluation index is selected as the initial seed $\mathbf{u}^{(0)}$.
2. **Damped Newton Iteration with Boundary Projection**:
   Newton steps are applied with backtracking line search. If an update step steps outside the parametric domain $[0, 1]^2$, the coordinate is projected back to the boundary.
3. **Adaptive Refinement**:
   Any points exceeding the specified projection tolerance are automatically refined using higher grid density seeds until convergence.
