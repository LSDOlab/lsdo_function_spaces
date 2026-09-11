# lsdo_function_spaces

[![Documentation Status](https://readthedocs.org/projects/lsdo-function-spaces/badge/?version=latest)](https://lsdo-function-spaces.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/LSDOlab/lsdo_function_spaces/actions/workflows/actions.yml/badge.svg)](https://github.com/LSDOlab/lsdo_function_spaces/actions)
![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL_v3-blue.svg)](LICENSE.txt)

**lsdo_function_spaces** is a pure-Python library for constructing continuous, high-dimensional, differentiable function representations (B-splines, tensor-product splines, scattered data spaces, and multivariate polynomials) tailored for gradient-based Multidisciplinary Design Optimization (MDO) and scientific computing.

Developed by the [Large-Scale Design Optimization (LSDO) Lab](https://lsdo.eng.ucsd.edu/) at the University of California, San Diego.

---

## Key Features

* **Differentiable Function Spaces**: Unifies Cox-de Boor B-spline curves, surfaces, and volumes, Shepard inverse distance weighting (IDW), radial basis functions (RBF), and multivariate polynomials under an extensible functional abstraction.
* **100% Pure Python & JAX Accelerated**: Zero Cython or C compiler dependencies. High-performance vectorized evaluation and projection algorithms implemented cleanly in NumPy with optional JAX JIT acceleration.
* **Vectorized Inverse Point Projection**: Robust, vectorized Gauss-Newton and Levenberg-Marquardt algorithms with active-set bounds handling for projecting point clouds onto parametric spline curves, surfaces, and volumes.
* **End-to-End CSDL Graph Integration**: Custom Vector-Jacobian Products (VJPs) and automatic differentiation through `csdl_alpha`, enabling exact analytic adjoint sensitivities across complex computational graphs.
* **Interactive & Headless 3D Visualization**: Native PyVista support for visualizing spline control meshes, evaluated surface geometries, and discrete point clouds.

---

## Architecture Overview

| Subpackage | Key Classes / Functions | Description |
|:---|:---|:---|
| **`lsdo_function_spaces.core.spaces`** | `BSplineSpace`, `FunctionSpace`, `ShepardSpace`, `PolynomialSpace`, `RBFSpace` | Core function space representations, basis function evaluations, and tensor-product constructions. |
| **`lsdo_function_spaces.core.function`** | `Function`, `FunctionSet` | Concrete functional instances binding coefficient vectors to function spaces, supporting evaluation and composition. |
| **`lsdo_function_spaces.core.spaces.non_cython_bsplines`** | `project_points_gauss_newton_numpy`, `project_points_lm_numpy`, `LMParams` | Vectorized inverse point projection engines utilizing active-set Levenberg-Marquardt and Gauss-Newton solvers. |
| **`lsdo_function_spaces.core.b_spline_csdl_custom_ops`** | Custom CSDL evaluation operations | Differentiable computational graph operations providing exact forward and reverse-mode derivative evaluations. |

---

## Quickstart

### Creating and Evaluating a B-spline Surface

```python
import numpy as np
import csdl_alpha as csdl
from lsdo_function_spaces import BSplineSpace, Function

# 1. Define a 2D B-spline function space (degrees 3x3, 6x6 control points)
space = BSplineSpace(
    num_dimensions=2,
    order=(4, 4),
    num_coefficients=(6, 6),
)

# 2. Assign control point coefficients (e.g. parabolic saddle surface)
u = np.linspace(0, 1, 6)
v = np.linspace(0, 1, 6)
U, V = np.meshgrid(u, v, indexing="ij")
coefficients = np.stack([U, V, U**2 - V**2], axis=-1).reshape(-1, 3)

func = Function(space=space, coefficients=coefficients)

# 3. Evaluate surface at query parametric coordinates
eval_pts = np.array([
    [0.2, 0.3],
    [0.5, 0.5],
    [0.8, 0.9],
])
physical_coords = func.evaluate(eval_pts)
print("Evaluated physical coordinates:\n", physical_coords)
```

### Inverse Parametric Point Projection

```python
import numpy as np
from lsdo_function_spaces.core.spaces.non_cython_bsplines.b_spline_patch_projection_optimized import (
    project_points_lm_numpy,
    LMParams,
)

# Project arbitrary 3D points onto a B-spline surface patch
points = np.array([[0.25, 0.35, 0.1]])
u0s = np.array([[0.5, 0.5]])  # Initial parametric guess

u_opt, converged, lam, res = project_points_lm_numpy(
    points=points,
    u0s=u0s,
    coeffs=coefficients,
    degrees=(3, 3),
    knot_vectors=space.knot_vectors,
    params=LMParams(max_iter=50, tol_grad=1e-8),
)
print("Converged parametric coordinate:", u_opt)
```

---

## Installation

### Prerequisites & Installation
`lsdo_function_spaces` requires Python $\ge 3.9$ and standard scientific Python packages:

```sh
# 1. Install prerequisites
pip install numpy scipy jax networkx
pip install git+https://github.com/LSDOlab/CSDL_alpha.git@dev_andrew

# 2. Install lsdo_function_spaces (User)
pip install lsdo_function_spaces
# Or install development version directly from GitHub:
pip install git+https://github.com/LSDOlab/lsdo_function_spaces.git
```

### For Developers
Clone the repository and install in editable mode with testing and documentation dependencies:
```sh
git clone https://github.com/LSDOlab/lsdo_function_spaces.git
cd lsdo_function_spaces
pip install -e ".[test,docs]"
```

---

## Testing

Run the full test suite with coverage:
```sh
pytest -v tests/ --cov=lsdo_function_spaces --cov-report=term-missing
```

For environments without a physical display (e.g. CI runners), run with `xvfb`:
```sh
xvfb-run --auto-servernum pytest -v tests/
```

---

## Documentation

Build the HTML documentation locally:
```sh
sphinx-build -b html docs docs/_build/html -q -W
```
View the generated documentation by opening `docs/_build/html/index.html` in any web browser.

---

## License

This project is licensed under the terms of the **GNU Lesser General Public License v3.0** ([LGPL-3.0](LICENSE.txt)).
