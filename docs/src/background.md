# Mathematical & Theoretical Background

`lsdo_function_spaces` provides high-performance, continuously differentiable mathematical function representations for engineering analysis and Multidisciplinary Design Optimization (MDO).

This section details the mathematical theory, basis formulations, numerical algorithms, and automatic differentiation mechanics underpinning the package.

---

## Overview of Supported Function Spaces

In engineering design optimization, physical geometries, aerodynamic fields, and structural properties must often be represented as smooth, continuous functions of spatial or parametric coordinates:

$$\mathbf{y} = \mathbf{f}(\mathbf{u}; \boldsymbol{\alpha})$$

where $\mathbf{u} \in \mathbb{R}^d$ is a parametric or spatial coordinate vector, $\boldsymbol{\alpha} \in \mathbb{R}^N$ represents trainable or optimizable coefficients (e.g. control points), and $\mathbf{y} \in \mathbb{R}^m$ is the physical output (e.g., Cartesian coordinates, pressure, or thickness).

```
                      ┌─────────────────────────────────────────┐
                      │        lsdo_function_spaces             │
                      └────────────────────┬────────────────────┘
                                           │
         ┌─────────────────────────────────┼─────────────────────────────────┐
         │                                 │                                 │
         ▼                                 ▼                                 ▼
┌──────────────────┐             ┌──────────────────┐             ┌──────────────────┐
│  B-Spline Space  │             │  Scattered Data  │             │ Analytic Spaces  │
│  - Tensor product│             │  - IDW           │             │  - Polynomial    │
│  - Multi-patch   │             │  - RBF (Gaussian,│             │  - Triangulation │
│  - CAD import    │             │    multiquadric) │             │  - Constant      │
└────────┬─────────┘             └─────────┬────────┘             └─────────┬────────┘
         │                                 │                                │
         └─────────────────────────────────┼────────────────────────────────┘
                                           │
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │    Automatic Differentiation & CSDL     │
                      │  - Forward & Reverse VJP operators      │
                      │  - Fast JAX JIT-compiled evaluation     │
                      └─────────────────────────────────────────┘
```

---

## Background Articles

```{toctree}
:maxdepth: 2
:numbered: 1

background/1_b_spline_theory
background/2_surface_projection
background/3_scattered_data_spaces
background/4_differentiable_graph_integration
```

---

## Bibliography

```{bibliography} references.bib
:style: unsrt
```
