# Release Notes

## lsdo_function_spaces 1.0.0 (September 2026)

`lsdo_function_spaces` 1.0.0 is the first major stable release of the LSDO lab's high-dimensional differentiable function representations library for gradient-based Multidisciplinary Design Optimization (MDO). This release completes a major repository modernization: transitioning to 100% pure Python (eliminating Cython and compiled C dependencies), providing full Python 3.9–3.14 and NumPy 2.x support, consolidating the B-spline space API, expanding unit test coverage to 77% across 79 tests, adding comprehensive mathematical documentation, and establishing automated CI/CD and PyPI release pipelines.

---

### Highlights

* **100% Pure Python & Universal Wheel**: Completely eliminated compiled Cython and C extensions (`lsdo_b_splines_cython`) in favor of vectorized NumPy routines and JAX JIT acceleration, enabling a universal `py3-none-any` wheel installable on any platform without a C compiler.
* **Extended Python & NumPy Support**: Full compatibility across Python 3.9, 3.10, 3.11, 3.12, 3.13, and 3.14, and full compatibility with NumPy 2.x.
* **Consolidated B-Spline API**: Unified B-spline space implementations into canonical `BSplineSpace`, deprecating legacy `BSplineSpaceNew` and `import_file_patched` while retaining backward-compatible aliases with `DeprecationWarning`.
* **Standard-Compliant Packaging**: Packaged via PEP 517/518/621 `pyproject.toml` with GNU LGPL-3.0 licensing, modern `.gitattributes`, and strict setuptools package discovery restricted to `lsdo_function_spaces*`.
* **Comprehensive Test Suite**: 79 automated unit tests achieving 77% test coverage across all function spaces, point projection algorithms, and evaluation graphs.
* **Automated CI/CD & Publishing**: GitHub Actions multi-version test matrix (Python 3.9–3.14) with headless display (`xvfb`), documentation build verification (`-q -W`), and PyPI Trusted Publishing (OIDC).

---

### Core Architecture & Features

* **Unified Function Spaces (`BSplineSpace`, `ShepardSpace`, `PolynomialSpace`, `RBFSpace`)**:
  * Tensor-product B-spline curves, surfaces, and volume spaces with Cox-de Boor basis recursion and non-uniform knot vectors.
  * Shepard inverse distance weighting (IDW) spaces for multidimensional interpolation.
  * Multivariate polynomial function spaces with customizable term degrees.
  * Radial basis function (RBF) spaces supporting multiquadric, thin-plate spline, and Gaussian kernels.
* **Functional Abstractions (`Function`, `FunctionSet`)**:
  * Bind coefficient vectors to function spaces to evaluate functions and derivatives at arbitrary parametric coordinates.
  * Seamless composition of multiple function instances across shared or distinct spaces.
* **Vectorized Inverse Point Projection (`project_points_gauss_newton_numpy`, `project_points_lm_numpy`)**:
  * Fast vectorized Gauss-Newton and Levenberg-Marquardt algorithms for projecting arbitrary spatial point clouds onto parametric spline patches.
  * Active-set bounding ensures projected coordinates remain strictly within valid parametric intervals $[0, 1]^d$.
* **Differentiable Graph Integration (`CSDL_alpha`)**:
  * Custom CSDL operations providing exact analytic Vector-Jacobian Products (VJPs) for backpropagation.
  * Efficient forward evaluation and reverse-mode derivative propagation through the computational graph.

---

### API Consolidations & Deprecations

* **`BSplineSpaceNew`**: Deprecated alias of `BSplineSpace`. Emits `DeprecationWarning` upon instantiation and passes all arguments through to `BSplineSpace`.
* **`import_file_patched`**: Deprecated utility alias delegating to `import_file`, emitting `DeprecationWarning`.
* **Clean Visualization Backend**: Replaced legacy `vedo` dependencies with `pyvista`, enabling both interactive window rendering and headless off-screen rendering for automated environments.

---

### Documentation & Infrastructure

* **Mathematical Background Articles**:
  * `1_b_spline_theory.md`: In-depth Cox-de Boor recursion, knot vectors, and tensor-product geometry.
  * `2_surface_projection.md`: Inverse point projection formulations, active-set Newton-Raphson, and Levenberg-Marquardt damping.
  * `3_scattered_data_spaces.md`: Mathematical formulations of Shepard IDW and RBF kernel spaces.
  * `4_differentiable_graph_integration.md`: Vector-Jacobian products, reverse-mode automatic differentiation, and JAX compilation.
* **Sphinx & AutoAPI**: Automated API documentation generated directly from docstrings with 0 warnings (`sphinx -b html docs docs/_build/html -q -W`).
* **PyPI Release Pipeline**: Tag-driven deployment workflow (`.github/workflows/publish.yml`) configured with Trusted Publishing.
