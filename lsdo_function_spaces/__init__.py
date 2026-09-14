"""lsdo_function_spaces: Functional representations and function spaces for MDO."""

import multiprocessing

__version__ = "1.0.0"
num_workers = multiprocessing.cpu_count()

# NumPy 2.x compatibility patch for CSDL_alpha SetVarIndex
try:
    import numpy as _np
    from csdl_alpha.src.operations.set_get.setindex import SetVarIndex as _SetVarIndex

    def _safe_compute_inline(self, x, y, *slice_args):
        x_updated = x.copy()
        if getattr(y, "size", None) == 1:
            x_updated[self.slice.evaluate(*slice_args)] = y.item() if isinstance(y, _np.ndarray) else y
        else:
            x_updated[self.slice.evaluate(*slice_args)] = y
        return x_updated

    _SetVarIndex.compute_inline = _safe_compute_inline
except Exception:
    pass

# Core representations
from .core.function import Function
from .core.function_set import FunctionSet
from .core.function_space import FunctionSpace, LinearFunctionSpace
from .core.function_set_space import FunctionSetSpace

# Function spaces
from .core.spaces.b_spline_space import BSplineSpace, BSplineSpaceNew
from .core.spaces.polynomial_space import PolynomialSpace
from .core.spaces.conditional_space import ConditionalSpace
from .core.spaces.idw_space import IDWFunctionSpace
from .core.spaces.constant_space import ConstantSpace
from .core.spaces.rbf_space import RBFFunctionSpace
from .core.spaces.tri_space import LinearTriangulationSpace

# Utilities
from .utils.plotting_functions import (
    plot_points,
    plot_curve,
    plot_surface,
    show_plot,
)
from .utils.file_io import import_file, import_file_patched
from .utils.utility_functions import (
    create_b_spline_from_corners,
    create_enclosure_block,
)

# Operations
from .core.operations import operations

__all__ = [
    "__version__",
    "num_workers",
    "Function",
    "FunctionSet",
    "FunctionSpace",
    "LinearFunctionSpace",
    "FunctionSetSpace",
    "BSplineSpace",
    "BSplineSpaceNew",
    "PolynomialSpace",
    "ConditionalSpace",
    "ConstantSpace",
    "IDWFunctionSpace",
    "RBFFunctionSpace",
    "LinearTriangulationSpace",
    "plot_points",
    "plot_curve",
    "plot_surface",
    "show_plot",
    "import_file",
    "import_file_patched",
    "create_b_spline_from_corners",
    "create_enclosure_block",
    "operations",
]
