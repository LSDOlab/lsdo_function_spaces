# Release Notes

## lsdo_function_spaces 1.1.0 (October 2026)

`lsdo_function_spaces` 1.1.0 introduces major performance and architectural enhancements to the LSDO lab's function representation and function space framework for Multidisciplinary Design Optimization (MDO). This release features a matrix-free B-spline evaluation engine with exact analytic reverse-mode sensitivities for dynamic parametric coordinates, accelerated and dimension-agnostic surface projections with convex-hull bounding box pruning, automatic memory safety for large-scale queries, and modernized tag-driven CI testing.

---

### Highlights

* **Matrix-Free B-Spline Custom Operations**: Replaces sparse COO/BCOO basis matrix assembly with matrix-free local tensor contraction, reducing JAX compilation overhead by **4.6x** and accelerating steady-state VJPs by up to **73x**.
* **Parametric Coordinates as Design Variables**: Full support for parametric coordinates ($u$) as first-class differentiable CSDL variables with exact analytic reverse-mode VJPs, enabling simultaneous optimization of both surface control points and parametric locations.
* **Accelerated Multi-Patch Projections**: Prunes 60%–90% of redundant patch evaluations using provable Euclidean distance lower bounds derived from the B-spline convex hull property with zero false negatives.
* **$5.5\times$ Memory Reduction in Grid Search**: Replaced 3D broadcasted array allocations with 2D BLAS-3 GEMM matrix multiplications, speeding up grid searches by 4x and reducing peak memory by 82%.
* **Dimension-Agnostic Orthogonal Projection**: Replaced 3D-specific cross-products with general orthogonal vector decomposition $\|r_{\perp}\|^2 = \|r\|_2^2 - \langle r, \hat{d} \rangle^2$, supporting arbitrary dimensions $\mathbb{R}^{d_{\text{phys}}}$ ($d_{\text{phys}} \ge 2$).
* **Automatic ~8 GB Memory Safety**: Added automatic batch chunking capped at $10^9$ `float64` elements (~8 GB RAM) so massive queries run reliably without out-of-memory errors or manual cutoff tuning.

---

### Core Architecture & Features

* **Matrix-Free Tensor Evaluator (`b_spline_csdl_custom_ops_v2.py`)**:
  * Evaluates multi-dimensional tensor-product B-spline curves, surfaces, and volumes via localized span contraction (`evaluate_b_spline_matrix_free`).
  * Computes pointwise reverse-mode VJPs ($\bar{u}_m = \bar{y}_m \cdot J_m$) and scatters $\bar{C}$ locally into active control point spans, eliminating $O(M^2)$ computational graph coupling.
  * Preserves the legacy `b_spline_csdl_custom_ops.py` module for backward compatibility while wiring `b_spline_space.py` to use v2.
* **Bounding Box (AABB) Distance Pruning (`FunctionSet.project`)**:
  * Precomputes physical bounding boxes $\text{AABB}(C) = [\min C, \max C]$ per patch.
  * Evaluates the $O(1)$ distance lower bound $d(x, \text{AABB}_k) \le \min_u \|S_k(u) - x\|_2$.
  * Safely skips candidate patches where $d(x, \text{AABB}_k) \ge d_{\text{best}}(x)$, dramatically speeding up multi-patch geometries like wings and fuselages.
* **Active-Set Newton Stabilization**:
  * Two-stage stabilized polish with eigenvalue clamping and Levenberg-Marquardt damping prevents singular matrix updates near boundaries or on degenerate patches.

---

### API Changes & Enhancements

* **Streamlined `project` Parameter Order**:
  * Reordered arguments across `Function.project`, `FunctionSet.project`, and `Function.refine_projection` to place `use_line_search` immediately after `direction`, followed by `projection_tolerance`:
    ```python
    def project(self, points, direction=None, use_line_search=False, projection_tolerance=None, ...)
    ```
* **Memory Management Cutoffs**:
  * Added `MAX_GRID_SEARCH_MEMORY_ELEMENTS = 1_000_000_000` (~8 GB RAM).
  * `grid_search_evaluation_cutoff` and `grid_search_subtraction_cutoff` default to `None` and automatically apply the 8 GB limit, while manual cutoffs are retained for backward compatibility.
* **Dimension-Agnostic Refinement**:
  * Updated `Function.refine_projection` to dynamically handle arbitrary physical dimensions (`points.shape[-1]`) rather than assuming 3D.

---

### Bug Fixes & Stability

* **Direction Normalization**: Ensured unit-norm normalization of `direction` at entry of `Function.project`, preventing Newton iteration drift during directional ray projections.
* **STEP CAD Entity Parser Fallback**: Added robust heuristic scanning in `_detect_knot_fields` when knot multiplicity and knot lists appear with integer formatting or non-standard entity slots.
* **Rational Surface Detection**: Added explicit `NotImplementedError` when rational NURBS (`RATIONAL_B_SPLINE_SURFACE`) entities are encountered.
* **CSDL Recorder Initialization**: Added `_ensure_active_recorder()` when loading pickled geometries to prevent unpickling errors when a CSDL recorder is inactive.

---

### Documentation & Infrastructure

* **GitHub Actions Tag-Driven CI**: Updated `.github/workflows/actions.yml` to automatically trigger test matrix runs across Python 3.9, 3.10, 3.11, 3.12, 3.13, and 3.14 upon pushing version tags (`v*`).
* **Sphinx Documentation Clean Build**: Verified HTML documentation generation with zero warnings or errors (`sphinx-build -q -W`).

---

## lsdo_function_spaces 1.0.0 (September 2024)

`lsdo_function_spaces` 1.0.0 was the first major stable release of the LSDO lab's functional representation framework. This release eliminated Cython build requirements in favor of a 100% pure Python and JAX architecture, integrated with CSDL Alpha for automatic differentiation, expanded unit test coverage, and established modern packaging and documentation.

---

### Highlights

* **Pure Python Architecture**: Removed all Cython build and compilation dependencies, enabling seamless pip installation across all platforms.
* **JAX-Accelerated B-Splines**: Introduced vectorized B-spline basis evaluation and forward sensitivity calculation in JAX.
* **CSDL Alpha Integration**: Native support for CSDL Alpha graph variables, vector operations, and automatic differentiation.
* **Sphinx AutoAPI Documentation**: Complete online documentation hosted on ReadTheDocs with API auto-generation.
