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
    num_parametric_dimensions : int
        The number of parametric dimensions/variables of a function from this function space.
    degree : tuple
        The degree of the B-spline in each parametric dimension.
    coefficients_shape : tuple
        The shape/structure that the coefficients are arranged in. For a surface (num_parametric_dimensions=2), the shape would be:
          (nu,nv,num_physical_dimensions)
    knots : np.ndarray = None -- shape=(num_knots,)
        The knot vector for the B-spline. If None, an open uniform knot vector will be generated.
    knot_indices : list[np.ndarray] = None -- shape of list=(num_parametric_dimensions,), shape of inner np.ndarray=(num_knots_in_that_dimension,)
        The indices of the knots for each parametric dimension. If None, the indices will be generated from the knot vector.

    Methods
    -------
    compute_basis_matrix(parametric_coordinates: np.ndarray, parametric_derivative_orders: np.ndarray = None) -> sps.csc_matrix:
        Computes the basis matrix for the given parametric coordinates and derivative orders.
    '''
    b_spline_spaces : list[lfs.BSplineSpace]
    degree : tuple
    coefficients_shape : tuple
    knots : np.ndarray = None
    knot_indices : list[np.ndarray] = None  # outer list is for parametric dimensions, inner list is for knot indices

    def compute_basis_matrix(self, parametric_coordinates: np.ndarray, parametric_derivative_orders: np.ndarray = None,
                                   expansion_factor:int=None) -> sps.csc_matrix:
        '''
        Evaluates the basis functions of the B-spline at the given parametric coordinates and assembles it into a sparse matrix.

        Parameters
        ----------
        parametric_coordinates : np.ndarray -- shape=(num_points, num_parametric_dimensions)
            The parametric coordinates at which to evaluate the basis functions.
        parametric_derivative_orders : np.ndarray = None -- shape=(num_points, num_parametric_dimensions,)
            The derivative orders for each parametric dimension.
        expansion_factor : int = None
            The number of times to repeat the basis functions in the basis matrix. This is useful if coefficients are flattened and 
            operations are restricted to matrix-vector products. If used, the expansion factor is usually the number of physical dimensions.
            If None, the basis matrix will not be expanded.
        '''
        if len(parametric_coordinates.shape) == 1:
            parametric_coordinates = parametric_coordinates.reshape((1, -1))

        if expansion_factor is None:
            expansion_factor = 1

        # if len(parametric_coordinates.shape) == 1:
        #     num_points = parametric_coordinates.shape[0]
        #     num_parametric_dimensions = 1
        # else:
        num_points = np.prod(parametric_coordinates.shape[:-1])
        num_parametric_dimensions = parametric_coordinates.shape[-1]

        if parametric_derivative_orders is None:
            parametric_derivative_orders = (0,)*num_parametric_dimensions
        if isinstance(parametric_derivative_orders, int):
            parametric_derivative_orders = (parametric_derivative_orders,)*num_parametric_dimensions
        elif len(parametric_derivative_orders) == 1 and num_parametric_dimensions != 1:
            parametric_derivative_orders = parametric_derivative_orders*num_parametric_dimensions

        order_multiplied = 1
        for i in range(len(self.degree)):
            order_multiplied *= (self.degree[i] + 1)

        data = np.zeros(num_points * order_multiplied) 
        row_indices = np.zeros(len(data), np.int32)
        col_indices = np.zeros(len(data), np.int32)

        num_coefficient_elements = np.prod(self.coefficients_shape[:-1])

        if num_parametric_dimensions == 1:
            u_vec = parametric_coordinates[:,0].copy()
            order_u = self.degree[0] + 1
            if self.knots is None:
                knots_u = np.zeros(self.coefficients_shape[0]+order_u)
                get_open_uniform(order_u, self.coefficients_shape[0], knots_u)
            elif len(self.knots.shape) == 1:
                knots_u = self.knots[:self.coefficients_shape[0]+order_u]

            get_basis_curve_matrix(order_u, self.coefficients_shape[0], parametric_derivative_orders[0], u_vec, knots_u,
                len(u_vec), data, row_indices, col_indices)
        elif num_parametric_dimensions == 2:
            u_vec = parametric_coordinates[:,0].copy()
            v_vec = parametric_coordinates[:,1].copy()
            order_u = self.degree[0] + 1
            order_v = self.degree[1] + 1
            if self.knots is None:
                knots_u = np.zeros(self.coefficients_shape[0]+order_u)
                get_open_uniform(order_u, self.coefficients_shape[0], knots_u)
                knots_v = np.zeros(self.coefficients_shape[1]+order_v)
                get_open_uniform(order_v, self.coefficients_shape[1], knots_v)
            elif len(self.knots.shape) == 1:
                knots_u = self.knots[:self.coefficients_shape[0]+order_u]
                knots_v = self.knots[self.coefficients_shape[0]+order_u:]

            get_basis_surface_matrix(order_u, self.coefficients_shape[0], parametric_derivative_orders[0], u_vec, knots_u, 
                order_v, self.coefficients_shape[1], parametric_derivative_orders[1], v_vec, knots_v, 
                len(u_vec), data, row_indices, col_indices)
        elif num_parametric_dimensions == 3:
            u_vec = parametric_coordinates[:,0].copy()
            v_vec = parametric_coordinates[:,1].copy()
            w_vec = parametric_coordinates[:,2].copy()
            order_u = self.degree[0] + 1
            order_v = self.degree[1] + 1
            order_w = self.degree[2] + 1
            if self.knots is None:
                knots_u = np.zeros(self.coefficients_shape[0]+order_u)
                get_open_uniform(order_u, self.coefficients_shape[0], knots_u)
                knots_v = np.zeros(self.coefficients_shape[1]+order_v)
                get_open_uniform(order_v, self.coefficients_shape[1], knots_v)
                knots_w = np.zeros(self.coefficients_shape[1]+order_w)
                get_open_uniform(order_w, self.coefficients_shape[1], knots_w)
            elif len(self.knots.shape) == 1:
                knots_u = self.knots[:self.coefficients_shape[0]+order_u]
                knots_v = self.knots[self.coefficients_shape[0]+order_u : 
                                self.coefficients_shape[0]+order_u + self.coefficients_shape[1]+order_v]
                knots_w = self.knots[self.coefficients_shape[0]+order_u + self.coefficients_shape[1]+order_v:]

            get_basis_volume_matrix(order_u, self.coefficients_shape[0], parametric_derivative_orders[0], u_vec, knots_u,
                order_v, self.coefficients_shape[1], parametric_derivative_orders[1], v_vec, knots_v, 
                order_w, self.coefficients_shape[2], parametric_derivative_orders[2], w_vec, knots_w, 
                len(u_vec), data, row_indices, col_indices)
            
        basis_matrix = sps.csc_matrix((data, (row_indices, col_indices)), shape=(len(u_vec), num_coefficient_elements))

        if expansion_factor > 1:
            expanded_basis = sps.lil_matrix((len(u_vec)*expansion_factor, num_coefficient_elements*expansion_factor))
            for i in range(expansion_factor):
                input_indices = np.arange(i, num_coefficient_elements*expansion_factor, expansion_factor)
                output_indices = np.arange(i, len(u_vec)*expansion_factor, expansion_factor)
                expanded_basis[np.ix_(output_indices, input_indices)] = basis_matrix
            return expanded_basis.tocsc()
        else:
            return basis_matrix



if __name__ == "__main__":
    import csdl_alpha as csdl
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    num_coefficients = 5
    space_of_linear_25_cp_b_spline_surfaces = BSplineSpace(num_parametric_dimensions=2, degree=(2,2), 
                                                           coefficients_shape=(num_coefficients,num_coefficients,3))
    
    import lsdo_function_spaces as lfs
    coefficients_line = np.linspace(0., 1., num_coefficients)
    coefficients_y, coefficients_x = np.meshgrid(coefficients_line,coefficients_line)
    coefficients = np.stack((coefficients_x, coefficients_y, 0.1*np.random.rand(num_coefficients,num_coefficients)), axis=-1)
    coefficients = csdl.Variable(value=coefficients.reshape((num_coefficients,num_coefficients,3)))
    b_spline = lfs.Function(space=space_of_linear_25_cp_b_spline_surfaces, coefficients=coefficients)

    # b_spline.plot()

    parametric_coordinates = np.array([
        [0., 0.],
        [0., 1.],
        [1., 0.],
        [1., 1.],
        [0.5, 0.5],
        [0.25, 0.75]
    ])


    print('points: ', b_spline.evaluate(parametric_coordinates=parametric_coordinates, parametric_derivative_order=(0,0)).value)
    print('derivative wrt u:', b_spline.evaluate(parametric_coordinates=parametric_coordinates, parametric_derivative_order=(1,0)).value)
    print('second derivative wrt u: ', b_spline.evaluate(parametric_coordinates=parametric_coordinates, parametric_derivative_order=(2,0)).value)

    projecting_points_z = np.zeros((6,))
    projecting_points = np.stack((parametric_coordinates[:,0], parametric_coordinates[:,1], projecting_points_z), axis=-1)

    import time
    t1 = time.time()
    for i in range(100):
        projected_points_parametric = b_spline.project(points=projecting_points, plot=True, grid_search_density_parameter=1)
    t2 = time.time()
    print('average time: ', (t2-t1)/100)
    projected_points = b_spline.evaluate(parametric_coordinates=projected_points_parametric).value

    import vedo
    b_spline_plot = b_spline.plot(show=False, opacity=0.8)
    projected_points_plot = vedo.Points(projected_points, r=10, c='g')
    projecting_points_plot = vedo.Points(projecting_points, r=10, c='r')
    vedo.show(b_spline_plot, projected_points_plot, projecting_points_plot, axes=1, viewup='z')


    # num_fitting_points = 25
    # u_vec = np.einsum('i,j->ij', np.linspace(0., 1., num_fitting_points), np.ones(num_fitting_points)).flatten().reshape((-1,1))
    # v_vec = np.einsum('i,j->ij', np.ones(num_fitting_points), np.linspace(0., 1., num_fitting_points)).flatten().reshape((-1,1))
    # parametric_coordinates = np.hstack((u_vec, v_vec))

    # grid_points = b_spline.evaluate(parametric_coordinates=parametric_coordinates, parametric_derivative_order=(0,0), plot=True
    #                                 ).value.reshape((num_fitting_points,num_fitting_points,3))

    # from lsdo_geo.splines.b_splines.b_spline_functions import fit_b_spline

    # new_b_spline = fit_b_spline(fitting_points=grid_points, parametric_coordinates=parametric_coordinates, num_coefficients=(15,),
    #                             order=(5,), regularization_parameter=1.e-3)
    # new_b_spline.plot()