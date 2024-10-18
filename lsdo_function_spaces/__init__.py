__version__ = '0.1.4'
import multiprocessing
num_workers = multiprocessing.cpu_count()

# core stuff
from .core.function import Function
from .core.function_set import FunctionSet
from .core.function_space import FunctionSpace, LinearFunctionSpace

# spaces
from .core.function_set_space import FunctionSetSpace
from .core.spaces.b_spline_space import BSplineSpace
from .core.spaces.polynomial_space import PolynomialSpace
from .core.spaces.conditional_space import ConditionalSpace
from .core.spaces.idw_space import IDWFunctionSpace
from .core.spaces.constant_space import ConstantSpace

# utilities
from .utils.plotting_functions import plot_points, plot_curve, plot_surface, show_plot
from .utils.file_io import import_file
from .utils.utility_functions import create_b_spline_from_corners, create_enclosure_block
