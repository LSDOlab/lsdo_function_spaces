# Getting Started

This guide provides instructions for installing `lsdo_function_spaces` and demonstrates basic usage examples.

---

## Installation

### Installation via pip

Install the package directly using `pip`:

```sh
pip install lsdo_function_spaces
```

Or install the latest development version directly from GitHub:

```sh
pip install git+https://github.com/LSDOlab/lsdo_function_spaces.git
```

### Developer Installation

To install for development with testing and documentation dependencies:

```sh
git clone https://github.com/LSDOlab/lsdo_function_spaces.git
cd lsdo_function_spaces
pip install -e ".[test,docs]"
```

Verify the installation by running the test suite:

```sh
pytest tests/
```

---

## Quickstart Examples

### 1. Creating and Evaluating a B-Spline Surface

```python
import numpy as np
import lsdo_function_spaces as lfs

# 1. Define a 2D B-spline space of degree (2, 2) with 4x4 control points
space = lfs.BSplineSpace(
    num_parametric_dimensions=2,
    degree=(2, 2),
    coefficients_shape=(4, 4),
)

# 2. Assign control points defining a curved surface in 3D
ctrl = np.zeros((4, 4, 3))
x = np.linspace(0.0, 1.0, 4)
y = np.linspace(0.0, 1.0, 4)
X, Y = np.meshgrid(x, y, indexing='ij')
ctrl[:, :, 0] = X
ctrl[:, :, 1] = Y
ctrl[:, :, 2] = np.sin(np.pi * X) * np.cos(np.pi * Y)

surf = lfs.Function(space=space, coefficients=ctrl, name="curved_patch")

# 3. Evaluate at query parametric coordinates (u, v)
query_uv = np.array([
    [0.2, 0.3],
    [0.5, 0.5],
    [0.8, 0.9],
])
eval_pts = surf.evaluate(parametric_coordinates=query_uv, non_csdl=True)
print("Evaluated points shape:", eval_pts.shape)
```

### 2. Point Projection onto Spline Surfaces

```python
# Project physical 3D points back onto the spline surface to find (u, v)
target_pts = np.array([
    [0.5, 0.5, 0.9],
    [0.2, 0.3, 0.5],
])

uv_projected = surf.project(points=target_pts, do_pickles=False)
print("Projected parametric coordinates:\n", uv_projected)
```

### 3. Importing STEP CAD Files

```python
# Import an OpenVSP STEP CAD file as a multi-patch FunctionSet
fset = lfs.import_file("examples/import_files_for_examples/rectangular_wing.stp")

print(f"Imported {len(fset.functions)} surface patches:")
for name, patch in fset.functions.items():
    print(f"  Patch '{name}': degree={patch.space.degree}, shape={patch.coefficients.shape}")
```
