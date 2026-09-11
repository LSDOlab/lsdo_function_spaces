# Differentiable Computational Graph Integration

In gradient-based Multidisciplinary Design Optimization (MDO), all geometric modeling operations must provide analytic sensitivities (derivatives) with respect to design variables {cite:p}`fletcher2026implicit`.

---

## Symbolic Operations & Vector-Jacobian Products (VJP)

`lsdo_function_spaces` bridges Python numerical algorithms with the **CSDL Alpha** computational graph via custom differentiable operations:

1. `BSplineEvalCustomOp`: Evaluates a B-spline surface or volume when both control point coefficients $\mathbf{P}$ and query parametric coordinates $\mathbf{u}$ are CSDL variables.
2. `BasisMatrixCustomOp`: Differentiably builds the sparse basis matrix $\mathbf{B}(\mathbf{u})$ as a function of parametric locations.
3. `ProjectionOperation`: Differentiably maps physical points onto spline surfaces, computing exact adjoint pullbacks across the implicit projection manifold.

---

## Reverse-Mode Automatic Differentiation (Adjoint Method)

When evaluating a scalar objective $\mathcal{J}$ downstream of a function evaluation:

$$\mathcal{J} = g(\mathbf{x}(\mathbf{u}, \mathbf{P}))$$

the reverse-mode automatic differentiation computes the sensitivity of $\mathcal{J}$ with respect to the inputs via the Vector-Jacobian Product (VJP):

$$\bar{\mathbf{P}} = \left(\frac{\partial \mathbf{x}}{\partial \mathbf{P}}\right)^T \bar{\mathbf{x}} = \mathbf{B}(\mathbf{u})^T \bar{\mathbf{x}}$$

$$\bar{\mathbf{u}} = \left(\frac{\partial \mathbf{x}}{\partial \mathbf{u}}\right)^T \bar{\mathbf{x}} = \left(\sum_{i} \mathbf{P}_i \frac{\partial N_i(\mathbf{u})}{\partial \mathbf{u}}\right)^T \bar{\mathbf{x}}$$

where $\bar{\mathbf{x}} = \frac{\partial \mathcal{J}}{\partial \mathbf{x}}$ is the incoming adjoint vector (cotangent).

---

## JAX JIT-Accelerated Kernels

Underneath CSDL execution, `lsdo_function_spaces` employs pure-Python JAX primitives:
- Pre-allocated stencil lookup algorithms.
- Batched COO sparse matrix generation (`jax.experimental.sparse.BCOO`).
- Just-In-Time (`jax.jit`) compilation for zero-overhead GPU and multi-core CPU execution.
