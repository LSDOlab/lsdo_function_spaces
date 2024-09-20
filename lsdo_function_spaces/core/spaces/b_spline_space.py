import numpy as np
import scipy.sparse as sps
from lsdo_function_spaces import FunctionSpace, Function
import csdl_alpha as csdl

from dataclasses import dataclass

from lsdo_b_splines_cython.cython.basis_matrix_curve_py import get_basis_curve_matrix
from lsdo_b_splines_cython.cython.basis_matrix_surface_py import get_basis_surface_matrix
from lsdo_b_splines_cython.cython.basis_matrix_volume_py import get_basis_volume_matrix
from lsdo_b_splines_cython.cython.get_open_uniform_py import get_open_uniform

class BSplineSpace(FunctionSpace):
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
          (nu,nv)
    knots : np.ndarray = None -- shape=(num_knots,)
        The knot vector for the B-spline. If None, an open uniform knot vector will be generated.
    knot_indices : list[np.ndarray] = None -- shape of list=(num_parametric_dimensions,), shape of inner np.ndarray=(num_knots_in_that_dimension,)
        The indices of the knots for each parametric dimension. If None, the indices will be generated from the knot vector.

    Methods
    -------
    compute_basis_matrix(parametric_coordinates: np.ndarray, parametric_derivative_orders: np.ndarray = None) -> sps.csc_matrix:
        Computes the basis matrix for the given parametric coordinates and derivative orders.
    '''
    def __init__(self, num_parametric_dimensions:int, degree:tuple, coefficients_shape:tuple, knots:np.ndarray=None, knot_indices:list[np.ndarray]=None):
        # TODO: replace num_parametric_dimensions with len(coefficients_shape)
        self.degree = degree
        self.knots = knots
        self.knot_indices = knot_indices
        super().__init__(num_parametric_dimensions, coefficients_shape)

    # def __post_init__(self):
        # super().__post_init__()
        if isinstance(self.degree, int):
            self.degree = (self.degree,)*self.num_parametric_dimensions

        for i in range(self.num_parametric_dimensions):
            if self.degree[i] < 0:
                raise ValueError(f'Degree in axis {i} must be non-negative.')
            if self.degree[i] > self.coefficients_shape[i]:
                raise ValueError(f'Degree in axis {i} must be less than the number of coefficients in each dimension.')

        if self.knots is None:
            # If knots are None, generate open uniform knot vectors
            self.knots = np.array([])
            num_knots = 0
            self.knot_indices = []
            for i in range(self.num_parametric_dimensions):
                dimension_num_knots = self.coefficients_shape[i] + self.degree[i] + 1
                num_knots += dimension_num_knots

                knots_i = np.zeros((dimension_num_knots,))
                get_open_uniform(order=self.degree[i]+1, num_coefficients=self.coefficients_shape[i], knot_vector=knots_i)
                self.knot_indices.append(np.arange(len(self.knots), len(self.knots) + dimension_num_knots))
                # self.knots.append(knots_i)
                self.knots = np.hstack((self.knots, knots_i))
        elif self.knot_indices is None:
            self.knot_indices = []
            knot_index = 0
            for i in range(self.num_parametric_dimensions):
                num_knots_i = self.coefficients_shape[i] + self.degree[i] + 1
                self.knot_indices.append(np.arange(knot_index, knot_index + num_knots_i))
                knot_index += num_knots_i

    def stitch(self, self_face, self_coeffs, other, other_face, other_coeffs):
        """
        Stitch two IDW function spaces together.

        Parameters
        ----------
        self_face : int
            The face of the current function space.
        other : IDWFunctionSpace
            The other function space to stitch.
        other_face : int
            The face of the other function space.

        Returns
        -------
        IDWFunctionSpace
            The stitched function space.

        """
        # TODO: triple/quad intersections don't work with this - eg, corners

        ind_array = np.arange(np.prod(self.coefficients_shape)).reshape(self.coefficients_shape)

        if len(self_coeffs.shape) > 2:
            self_coeffs = self_coeffs.reshape((-1, self.num_physical_dimensions))
        if len(other_coeffs.shape) > 2:
            other_coeffs = other_coeffs.reshape((-1, other.num_physical_dimensions))

        if self_face == 1:
            self_inds = ind_array[:,0]
        elif self_face == 2:
            self_inds = ind_array[-1,:]
        elif self_face == 3:
            self_inds = ind_array[:,-1]
        elif self_face == 4:
            self_inds = ind_array[0,:]
        self_inds = [int(ind) for ind in self_inds]

        if other_face == 1:
            other_inds = ind_array[:,0]
        elif other_face == 2:
            other_inds = ind_array[-1,:]
        elif other_face == 3:
            other_inds = ind_array[:,-1]
        elif other_face == 4:
            other_inds = ind_array[0,:]
        other_inds = [int(ind) for ind in other_inds]

        
        for i, j in csdl.frange(vals=(self_inds, other_inds)):
            self_face_coeffs = self_coeffs[i]
            other_face_coeffs = other_coeffs[j]
            average_coeffs = (self_face_coeffs + other_face_coeffs)/2
            self_coeffs = self_coeffs.set(csdl.slice[i], average_coeffs)
            other_coeffs = other_coeffs.set(csdl.slice[j], average_coeffs)
        
        return self_coeffs, other_coeffs



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
            parametric_coordinates = parametric_coordinates.reshape((-1, self.num_parametric_dimensions))
        elif len(parametric_coordinates.shape) > 2:
            parametric_coordinates = parametric_coordinates.reshape((-1, self.num_parametric_dimensions))


        if expansion_factor is None:
            expansion_factor = 1

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

        num_coefficient_elements = np.prod(self.coefficients_shape)

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
                knots_u = self.knots[:self.coefficients_shape[0]+order_u]   # This should probably use knot_indices
                knots_v = self.knots[self.coefficients_shape[0]+order_u:]   # This should probably use knot_indices

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
                knots_u = self.knots[:self.coefficients_shape[0]+order_u]   # This should probably use knot_indices
                knots_v = self.knots[self.coefficients_shape[0]+order_u : 
                                self.coefficients_shape[0]+order_u + self.coefficients_shape[1]+order_v]    # This should probably use knot_indices
                knots_w = self.knots[self.coefficients_shape[0]+order_u + self.coefficients_shape[1]+order_v:]  # This should probably use knot_indices

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
        

    def _compute_distance_bounds(self, point:np.ndarray, function:Function, direction=None) -> float:
        '''
        Computes the distance bounds for the given point.
        '''
        if not hasattr(function, 'bounding_box'):
            coefficients = function.coefficients.value.reshape((-1, function.num_physical_dimensions))
            function.bounding_box = np.zeros((2, coefficients.shape[-1]))
            if self.num_parametric_dimensions == 1:
                function.bounding_box[0, 0] = np.min(coefficients)
                function.bounding_box[1, 0] = np.max(coefficients)
            else:
                function.bounding_box[0, :] = np.min(coefficients, axis=0)
                function.bounding_box[1, :] = np.max(coefficients, axis=0)

        if direction is None:
            neg = function.bounding_box[0] - point
            pos = point - function.bounding_box[1]
            distance_vector = np.maximum(np.maximum(neg, pos), 0)
            return np.linalg.norm(distance_vector)
        else:
            closest_point = np.zeros((len(point),))
            for i in range(len(point)):
                if point[i] < function.bounding_box[0, i]:
                    closest_point[i] = function.bounding_box[0, i]
                elif point[i] > function.bounding_box[1, i]:
                    closest_point[i] = function.bounding_box[1, i]
                else:
                    closest_point[i] = point[i]
            t = np.dot(direction, (closest_point - point)) / np.dot(direction, direction)
            closest_point_on_line = point + t * direction
            return np.linalg.norm(closest_point_on_line - closest_point)

        
    def _generate_projection_grid_search_resolution(self, grid_search_density_parameter=1):
        '''
        Generates the resolution of the grid search for projection.
        '''
        grid_search_resolution = []
        for dimension_length in self.coefficients_shape:
            grid_search_resolution.append(int(dimension_length*grid_search_density_parameter))
        return tuple(grid_search_resolution)


def test_single_surface():
    import csdl_alpha as csdl
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    num_coefficients = 5
    space_of_linear_25_cp_b_spline_surfaces = BSplineSpace(num_parametric_dimensions=2, degree=(2,2), 
                                                           coefficients_shape=(num_coefficients,num_coefficients))
    
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


    print('points: ', b_spline.evaluate(parametric_coordinates=parametric_coordinates, parametric_derivative_orders=(0,0)).value)
    print('derivative wrt u:', b_spline.evaluate(parametric_coordinates=parametric_coordinates, parametric_derivative_orders=(1,0)).value)
    print('second derivative wrt u: ', b_spline.evaluate(parametric_coordinates=parametric_coordinates, parametric_derivative_orders=(2,0)).value)

    # projecting_points_z = np.zeros((6,))
    # projecting_points = np.stack((parametric_coordinates[:,0], parametric_coordinates[:,1], projecting_points_z), axis=-1)

    num_points = 50
    x_coordinates = np.random.rand(num_points)
    y_coordinates = np.random.rand(num_points)
    z_coordinates = np.zeros((num_points,))
    projecting_points = np.stack((x_coordinates, y_coordinates, z_coordinates), axis=-1)

    import time
    num_trials = 1
    t1 = time.time()
    for i in range(num_trials):
        projected_points_parametric = b_spline.project(points=projecting_points, plot=False, grid_search_density_parameter=1)
    t2 = time.time()
    print('average time: ', (t2-t1)/num_trials)
    projected_points = b_spline.evaluate(parametric_coordinates=projected_points_parametric, plot=False).value

    import vedo
    # b_spline_plot = b_spline.plot(show=False, opacity=0.8)
    # projected_points_plot = vedo.Points(projected_points, r=10, c='g')
    # projecting_points_plot = vedo.Points(projecting_points, r=10, c='r')
    # vedo.show(b_spline_plot, projected_points_plot, projecting_points_plot, axes=1, viewup='z')

    new_b_spline_space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1,1), coefficients_shape=(4,4))
    new_b_spline = b_spline.refit(new_function_space=new_b_spline_space)
    # new_b_spline.plot()
    projected_points_parametric = new_b_spline.project(points=projecting_points, plot=False, grid_search_density_parameter=1)
    projected_points = new_b_spline.evaluate(parametric_coordinates=projected_points_parametric).value
    # new_b_spline_plot = new_b_spline.plot(show=False, opacity=0.8)
    # projected_points_plot = vedo.Points(projected_points, r=10, c='g')
    # projecting_points_plot = vedo.Points(projecting_points, r=10, c='r')
    # plotter = vedo.Plotter()
    # plotter.show(new_b_spline_plot, projected_points_plot, projecting_points_plot, axes=1, viewup='z')

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

def test_multiple_surfaces():
    import lsdo_function_spaces as lfs
    import csdl_alpha as csdl
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    num_coefficients1 = 10
    num_coefficients2 = 5
    degree1 = 4
    degree2 = 3
    
    # Create functions that make up set
    space_of_cubic_b_spline_surfaces_with_10_cp = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(degree1,degree1),
                                                              coefficients_shape=(num_coefficients1,num_coefficients1))
    space_of_quadratic_b_spline_surfaces_with_5_cp = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(degree2,degree2),
                                                              coefficients_shape=(num_coefficients2,num_coefficients2))

    coefficients_line = np.linspace(0., 1., num_coefficients1)
    coefficients_y, coefficients_x = np.meshgrid(coefficients_line,coefficients_line)
    coefficients1 = np.stack((coefficients_x, coefficients_y, 0.1*np.random.rand(num_coefficients1,num_coefficients1)), axis=-1)
    coefficients1 = coefficients1.reshape((num_coefficients1,num_coefficients1,3))

    b_spline1 = lfs.Function(space=space_of_cubic_b_spline_surfaces_with_10_cp, coefficients=coefficients1, name='b_spline1')

    coefficients_line = np.linspace(0., 1., num_coefficients2)
    coefficients_y, coefficients_x = np.meshgrid(coefficients_line,coefficients_line)
    coefficients_y += 1.5
    coefficients2 = np.stack((coefficients_x, coefficients_y, 0.1*np.random.rand(num_coefficients2,num_coefficients2)), axis=-1)
    coefficients2 = coefficients2.reshape((num_coefficients2,num_coefficients2,3))

    b_spline2 = lfs.Function(space=space_of_quadratic_b_spline_surfaces_with_5_cp, coefficients=coefficients2, name='b_spline2')

    # Make function set and plot
    my_b_spline_surface_set = lfs.FunctionSet(functions=[b_spline1, b_spline2], function_names=['b_spline1', 'b_spline2'])


    num_points = 10000
    x_coordinates = np.random.rand(num_points)
    y_coordinates = np.random.rand(num_points)
    z_coordinates = np.zeros((num_points,))
    projecting_points_1 = np.stack((x_coordinates, y_coordinates, z_coordinates), axis=-1)

    projecting_points_2 = np.stack((x_coordinates, y_coordinates+1.5, z_coordinates), axis=-1)

    projecting_points = np.vstack((projecting_points_1, projecting_points_2))

    import time
    import vedo
    num_trials = 1
    t1 = time.time()
    for i in range(num_trials):
        projected_points_parametric = my_b_spline_surface_set.project(points=projecting_points, plot=False, grid_search_density_parameter=1)
    t2 = time.time()
    print('average time: ', (t2-t1)/num_trials)
    projected_points = my_b_spline_surface_set.evaluate(parametric_coordinates=projected_points_parametric, plot=False).value
    # new_b_spline_plot = my_b_spline_surface_set.plot(show=False, opacity=0.8)
    # projected_points_plot = vedo.Points(projected_points, r=10, c='g')
    # projecting_points_plot = vedo.Points(projecting_points, r=10, c='r')
    # plotter = vedo.Plotter()
    # plotter.show(new_b_spline_plot, projected_points_plot, projecting_points_plot, axes=1, viewup='z')


if __name__ == '__main__':
    test_single_surface()
    test_multiple_surfaces()