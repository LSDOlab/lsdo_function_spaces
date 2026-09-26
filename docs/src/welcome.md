# Welcome to lsdo_function_spaces (v1.0.0)

![LSDO Lab](/src/images/lsdolab.png "LSDO Lab")

**lsdo_function_spaces** is a high-performance, pure-Python library for continuous function representations, multivariate B-splines, scattered data interpolation, and CAD geometry processing tailored for Multidisciplinary Design Optimization (MDO).

It integrates directly with [CSDL](https://github.com/LSDOlab/csdl) and CSDL Alpha, providing analytic derivative-compatible function spaces, JAX-accelerated evaluations, and inverse projection solvers for gradient-based design.

---

## Key Capabilities

- **Pure-Python Multivariate B-Splines**: Arbitrary dimension ($d \ge 1$), arbitrary degree tensor-product B-splines with 100% pure Python and JAX backends (zero C/Cython compiler dependencies).
- **Inverse Point & Surface Projection**: Robust two-stage Newton-Raphson nonlinear solvers with adaptive refinement and exact adjoint sensitivity computation.
- **Multi-Patch Geometries & CAD Processing**: `FunctionSet` and `FunctionSetSpace` for parsing and manipulating complex multi-surface assemblies from OpenVSP and CAD formats (STEP, IGES).
- **Generalized Function Spaces**: Inverse Distance Weighting (IDW), Radial Basis Functions (RBF), polynomial spaces, and linear triangulations.
- **Analytic Sensitivities & CSDL Alpha**: First-class computational graph integration supporting forward and reverse Vector-Jacobian Products (VJPs) for seamless gradient propagation.

---

## Cite us
```none
@article{lsdo_function_spaces,
    author  = {Andrew Fletcher},
    title   = {lsdo_function_spaces: Continuous Function Spaces and Differentiable B-Spline Representations for MDO},
    year    = {2024}
}
```

```{toctree}
:maxdepth: 1
:hidden:

src/getting_started
src/background
src/tutorials
src/examples
src/api
```
