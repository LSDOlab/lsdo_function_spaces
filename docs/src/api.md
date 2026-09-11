# API Reference

This section provides the complete reference for the `lsdo_function_spaces` application programming interface (API), automatically generated from source code docstrings and type annotations.

---

## Package Architecture Overview

`lsdo_function_spaces` is structured into core function representations, space definitions, and CAD/geometric utilities:

| Submodule / Class | Description |
|:---|:---|
| **[`BSplineSpace`](autoapi/lsdo_function_spaces/core/spaces/b_spline_space/index)** | Multidimensional tensor-product B-spline space with pure-Python and JAX evaluation backends. |
| **[`Function`](autoapi/lsdo_function_spaces/core/function/index)** | Function instance pairing a function space with coefficient arrays; handles evaluation, projection, and arithmetic. |
| **[`FunctionSet`](autoapi/lsdo_function_spaces/core/function_set/index)** | Multi-patch collection representing assemblies of connected surfaces or volumes (e.g., CAD models). |
| **[`FunctionSetSpace`](autoapi/lsdo_function_spaces/core/function_set_space/index)** | Space definition encompassing sets of heterogeneous or multi-patch function spaces. |
| **[`IDWSpace`](autoapi/lsdo_function_spaces/core/spaces/idw_space/index)** | Inverse Distance Weighting interpolation for scattered data with optional k-NN sparsity. |
| **[`RBFSpace`](autoapi/lsdo_function_spaces/core/spaces/rbf_space/index)** | Radial Basis Function interpolation with Gaussian, multiquadric, and inverse multiquadric kernels. |
| **[`PolynomialSpace`](autoapi/lsdo_function_spaces/core/spaces/polynomial_space/index)** | Multivariate polynomial function space. |
| **[`LinearTriangulationSpace`](autoapi/lsdo_function_spaces/core/spaces/tri_space/index)** | Piecewise linear finite-element shape functions on unstructured surface triangulations. |
| **[`ConstantSpace`](autoapi/lsdo_function_spaces/core/spaces/constant_space/index)** | Constant field representation over a given domain. |
| **[`ConditionalSpace`](autoapi/lsdo_function_spaces/core/spaces/conditional_space/index)** | Piecewise conditional function spaces switched based on parameter thresholds. |
| **[`file_io`](autoapi/lsdo_function_spaces/utils/file_io/index)** | CAD parsers importing STEP/IGES geometries into `FunctionSet` representations. |
| **[`utility_functions`](autoapi/lsdo_function_spaces/utils/utility_functions/index)** | Geometry construction helpers (`create_b_spline_from_corners`, `create_enclosure_block`). |

---

## Full Auto-Generated API

Browse the complete hierarchy of modules, classes, and methods below:

```{toctree}
:maxdepth: 2
:titlesonly:

autoapi/lsdo_function_spaces/index
```
