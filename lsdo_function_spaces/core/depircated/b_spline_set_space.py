import numpy as np
import scipy.sparse as sps
import lsdo_function_spaces as lfs

from dataclasses import dataclass

from lsdo_b_splines_cython.cython.basis_matrix_curve_py import get_basis_curve_matrix
from lsdo_b_splines_cython.cython.basis_matrix_surface_py import get_basis_surface_matrix
from lsdo_b_splines_cython.cython.basis_matrix_volume_py import get_basis_volume_matrix
from lsdo_b_splines_cython.cython.get_open_uniform_py import get_open_uniform

@dataclass
class BSplineSetSpace(lfs.FunctionSpace):
    '''
    Class for representing the space of BSplineFunctions of a particular degree.

    Attributes
    ----------
    num_parametric_dimensions : dict[int, int]
        The number of parametric dimensions/variables for each B-spline that is a part of this B-spline set.
    b_spline_spaces : list[lfs.BSplineSpace]
        The function spaces that make up this B-spline set.
    index_to_space : dict[int, int]
        A mapping from the index of the B-spline to the index of the B-spline space in the b_spline_spaces list.
    name_to_index : dict[str, int]
        A mapping from the name of the B-spline to the index of the B-spline in the b_spline_spaces list.
    coefficients_shape : tuple
        The shape/structure that the coefficients are arranged in. Since this is a set, this should be (num_coefficients,num_physical_dimensions).
    knots : np.ndarray = None -- shape=(num_knots,)
        The knot vector for the B-spline. If None, open uniform knot vectors will be generated.
    knot_indices : list[np.ndarray] = None -- shape of list=(num_parametric_dimensions,), shape of inner np.ndarray=(num_knots_in_that_dimension,)
        The indices of the knots for each parametric dimension. If None, the indices will be generated from the knot vector.
    connections : list[int, list[int]] = None
        The connections between the B-splines in the set. If None, the B-splines are assumed to be independent.

    Methods
    -------
    compute_basis_matrix(parametric_coordinates: np.ndarray, parametric_derivative_orders: np.ndarray = None) -> sps.csc_matrix:
        Computes the basis matrix for the given parametric coordinates and derivative orders.
    '''
    num_parametric_dimensions : dict[int, int]
    spaces : list[lfs.BSplineSpace]
    index_to_space : dict[int, int]
    name_to_index : dict[str, int]
    # knots : np.ndarray = None
    # knot_indices : dict[int, list[np.ndarray]] = None  # outer list is for parametric dimensions, inner list is for knot indices
    connections : list[int, list[int]] = None

    def __post_init__(self):
        # if self.knots is None:
        #     num_knots = 0
        #     self.knots = []
        #     self.knot_indices = {}

        #     for index in self.index_to_space:
        #         space = self.spaces[index]
        #         self.knot_indices[index] = []
        #         for i in range(space.num_parametric_dimensions):
        #             dimension_num_knots = len(space.knot_indices[i])
        #             self.knot_indices[index].append(np.arange(num_knots, num_knots + dimension_num_knots))
        #             self.knots.append(space.knots[i])
        #             num_knots += dimension_num_knots

        self.index_to_coefficient_indices = {}
        num_coefficients = 0
        for index in self.index_to_space:
            space = self.spaces[self.index_to_space[index]]
            sub_function_num_coefficients = np.prod(space.coefficients_shape[:-1])
            self.index_to_coefficient_indices[index] = list(np.arange(num_coefficients, num_coefficients + sub_function_num_coefficients))
            num_coefficients += sub_function_num_coefficients
        self.coefficients_shape = (num_coefficients, self.spaces[0].coefficients_shape[-1])

    def compute_basis_matrix(self, parametric_coordinates: list[tuple[int, np.ndarray]], parametric_derivative_orders: np.ndarray = None,
                                   expansion_factor:int=None) -> sps.csc_matrix:
        '''
        Evaluates the basis functions of the B-spline at the given parametric coordinates and assembles it into a sparse matrix.

        Parameters
        ----------
        parametric_coordinates : list[tuple[int, np.ndarray]] -- list of tuples of the form (index, parametric_coordinates) for each point to evaluate.
            The parametric coordinates at which to evaluate the basis functions.
            list indices correspond to points, tuple str is the B-spline name, and tuple np.ndarray is the parametric coordinates for that point.
        parametric_derivative_orders : np.ndarray = None -- shape=(num_points, num_parametric_dimensions,)
            The derivative orders for each parametric dimension.
        expansion_factor : int = None
            The number of times to repeat the basis functions in the basis matrix. This is useful if coefficients are flattened and 
            operations are restricted to matrix-vector products. If used, the expansion factor is usually the number of physical dimensions.
            If None, the basis matrix will not be expanded.

        Returns
        -------
        basis_matrix : sps.csc_matrix
            The basis matrix evaluated at the given parametric coordinates (Evaluation of the appropriate basis functions and assembled into a matrix)
        '''

        if expansion_factor is None:
            expansion_factor = 1

        basis_matrix_rows = []
        for i, parametric_coordinate in enumerate(parametric_coordinates):
            index, parametric_coordinate = parametric_coordinate
            space = self.spaces[self.index_to_space[index]]
            basis_matrix = space.compute_basis_matrix(parametric_coordinates=parametric_coordinate,
                                                      parametric_derivative_orders=parametric_derivative_orders[i],
                                                      expansion_factor=expansion_factor)
            basis_matrix_rows.append(basis_matrix)

        basis_matrix = sps.vstack(basis_matrix_rows, format='csc')
        return basis_matrix



if __name__ == "__main__":
    import csdl_alpha as csdl
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    num_coefficients1 = 10
    num_coefficients2 = 5
    degree1 = 4
    degree2 = 3
    
    space_of_cubic_b_spline_surfaces_with_10_cp = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(degree1,degree1),
                                                              coefficients_shape=(num_coefficients1,num_coefficients1, 3))
    space_of_quadratic_b_spline_surfaces_with_5_cp = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(degree2,degree2),
                                                              coefficients_shape=(num_coefficients2,num_coefficients2, 3))
    b_spline_spaces = [space_of_cubic_b_spline_surfaces_with_10_cp, space_of_quadratic_b_spline_surfaces_with_5_cp]
    index_to_space = {0:0, 1:1}
    name_to_index = {'space_of_cubic_b_spline_surfaces_with_10_cp':0, 'quadratic_b_spline_surfaces_5_cp':1}
    num_parametric_dimensions = {0:2, 1:2}

    b_spline_set_space = BSplineSetSpace(num_parametric_dimensions=num_parametric_dimensions, spaces=b_spline_spaces,
                                            index_to_space=index_to_space, name_to_index=name_to_index)

    coefficients_line = np.linspace(0., 1., num_coefficients1)
    coefficients_y, coefficients_x = np.meshgrid(coefficients_line,coefficients_line)
    coefficients1 = np.stack((coefficients_x, coefficients_y, 0.1*np.random.rand(num_coefficients1,num_coefficients1)), axis=-1)

    coefficients_line = np.linspace(0., 1., num_coefficients2)
    coefficients_y, coefficients_x = np.meshgrid(coefficients_line,coefficients_line)
    coefficients_y += 1.5
    coefficients2 = np.stack((coefficients_x, coefficients_y, 0.1*np.random.rand(num_coefficients2,num_coefficients2)), axis=-1)

    coefficients = np.vstack((coefficients1.reshape((-1,3)), coefficients2.reshape((-1,3))))
    coefficients = csdl.Variable(value=coefficients)

    my_b_spline_surface_set = lfs.Function(space=b_spline_set_space, coefficients=coefficients)
    my_b_spline_surface_set.plot()