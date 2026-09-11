# Scattered Data Function Spaces

In addition to structured tensor-product B-splines, `lsdo_function_spaces` provides functional representations for unstructured, scattered datasets.

---

## Inverse Distance Weighting (IDW)

Inverse Distance Weighting {cite:p}`shepard1968two` interpolates a scalar or vector quantity from a set of known points $\mathbf{x}_i \in \mathbb{R}^d$ with values $f_i$ to an arbitrary query coordinate $\mathbf{x}$:

$$f(\mathbf{x}) = \sum_{i=1}^N w_i(\mathbf{x}) f_i$$

where the normalized weights $w_i(\mathbf{x})$ are defined by:

$$w_i(\mathbf{x}) = \frac{\frac{1}{(d(\mathbf{x}, \mathbf{x}_i) + \epsilon)^p}}{\sum_{j=1}^N \frac{1}{(d(\mathbf{x}, \mathbf{x}_j) + \epsilon)^p}}$$

Here:
- $d(\mathbf{x}, \mathbf{x}_i) = \|\mathbf{x} - \mathbf{x}_i\|_2$ is the Euclidean distance,
- $p \ge 1$ is the distance weighting exponent (commonly $p = 2$),
- $\epsilon > 0$ is a small smoothing regularization parameter preventing numerical singularities when $\mathbf{x} \to \mathbf{x}_i$.

### Localized & Sparse IDW

For large point sets ($N > 10^4$), evaluating dense all-to-all distances becomes computationally prohibitive ($\mathcal{O}(N \cdot M)$). `IDWSpace` supports local $k$-nearest-neighbor (k-NN) queries and compact radius-bounded support, yielding sparse evaluation matrices:

$$\mathbf{f}_{\text{eval}} = \mathbf{W}_{\text{sparse}} \mathbf{f}_{\text{source}}$$

---

## Radial Basis Functions (RBF)

Radial Basis Function interpolation models an unknown field as a linear combination of radially symmetric kernel functions centered at data nodes $\mathbf{x}_i$:

$$f(\mathbf{x}) = \sum_{i=1}^N c_i \, \phi(\|\mathbf{x} - \mathbf{x}_i\|_2)$$

### Kernel Types

| Kernel Name | Mathematical Form $\phi(r)$ | Properties |
|:---|:---|:---|
| **Gaussian** | $\exp\left(-(\epsilon r)^2\right)$ | Infinitely smooth ($C^\infty$), local decay |
| **Multiquadric** | $\sqrt{1 + (\epsilon r)^2}$ | Conditionally positive definite |
| **Inverse Multiquadric** | $\frac{1}{\sqrt{1 + (\epsilon r)^2}}$ | Strictly positive definite |
| **Linear** | $r$ | Polyharmonic |
| **Thin-Plate Spline** | $r^2 \ln(r)$ | Minimizes bending energy in 2D |

Fitting the coefficients $\mathbf{c} = [c_1, \dots, c_N]^T$ requires solving the linear system:

$$\boldsymbol{\Phi} \mathbf{c} = \mathbf{y}$$

where $\Phi_{ij} = \phi(\|\mathbf{x}_i - \mathbf{x}_j\|)$. Once fitted, evaluation at new points is evaluated as a simple matrix-vector product.
