# B-Spline Mathematical Formulations

B-splines provide a powerful, numerically stable basis for geometric modeling and functional approximations due to their compact support, partition of unity, and convex hull properties {cite:p}`piegl1997nurbs`.

---

## Univariate B-Spline Basis Functions

Given a non-decreasing knot vector $U = \{u_0, u_1, \dots, u_{m}\}$, the $i$-th B-spline basis function of degree $p$ (order $k = p + 1$) is defined recursively by the Cox-de Boor recursion formula:

### Degree 0 ($p = 0$)

$$N_{i,0}(u) = \begin{cases} 1 & \text{if } u_i \le u < u_{i+1} \\ 0 & \text{otherwise} \end{cases}$$

### Degree $p \ge 1$

$$N_{i,p}(u) = \frac{u - u_i}{u_{i+p} - u_i} N_{i, p-1}(u) + \frac{u_{i+p+1} - u}{u_{i+p+1} - u_{i+1}} N_{i+1, p-1}(u)$$

where quotients with zero denominators are defined as zero.

### Essential Properties

1. **Partition of Unity**: For all $u \in [u_p, u_{m-p}]$, $\sum_{i=0}^{n} N_{i,p}(u) = 1$.
2. **Compact Local Support**: $N_{i,p}(u) = 0$ if $u \notin [u_i, u_{i+p+1}]$.
3. **Continuity**: At a knot of multiplicity $k$, the basis function has $C^{p-k}$ continuity across the knot.

---

## Multivariate Tensor-Product B-Splines

For multidimensional parametric spaces $\mathbf{u} = (u_1, u_2, \dots, u_d)$, the multivariate basis is formed via tensor products:

$$N_{\mathbf{i}, \mathbf{p}}(\mathbf{u}) = \prod_{j=1}^d N_{i_j, p_j}(u_j)$$

Given control points $\mathbf{P}_{\mathbf{i}} \in \mathbb{R}^m$, a multidimensional curve, surface, or volume is evaluated as:

$$\mathbf{x}(\mathbf{u}) = \sum_{i_1=0}^{n_1} \dots \sum_{i_d=0}^{n_d} N_{\mathbf{i}, \mathbf{p}}(\mathbf{u}) \mathbf{P}_{\mathbf{i}}$$

In matrix notation, evaluating $M$ query points simultaneously yields a sparse linear mapping:

$$\mathbf{X} = \mathbf{B}(\mathbf{u}) \mathbf{P}$$

where $\mathbf{B}(\mathbf{u}) \in \mathbb{R}^{M \times N_{\text{total}}}$ is the sparse B-spline basis matrix.

---

## Derivatives of B-Splines

The derivative of a degree-$p$ B-spline basis function with respect to the parametric coordinate $u$ is given by:

$$\frac{d}{du} N_{i,p}(u) = \frac{p}{u_{i+p} - u_i} N_{i, p-1}(u) - \frac{p}{u_{i+p+1} - u_{i+1}} N_{i+1, p-1}(u)$$

In `lsdo_function_spaces`, arbitrary derivative orders are evaluated analytically using vector-optimized derivative matrices, avoiding finite-difference approximations.
