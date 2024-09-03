from __future__ import annotations

from dataclasses import dataclass
import csdl_alpha as csdl
import numpy as np
import scipy.sparse as sps

import pickle
from pathlib import Path
import string
import random

# from lsdo_function_spaces.core.function_space import FunctionSpace
import lsdo_function_spaces as lfs



@dataclass
class Function:
    '''
    Function class. This class is used to represent a function in a given function space. The function space is used to evaluate the function at
    given coordinates, refit the function, and project points onto the function.

    Attributes
    ----------
    space : lfs.FunctionSpace
        The function space in which the function resides.
    coefficients : csdl.Variable -- shape=coefficients_shape
        The coefficients of the function.
    name : str = None
        If applicable, the name of the function.
    '''
    space: lfs.FunctionSpace
    coefficients: csdl.Variable
    name : str = None

    def __post_init__(self):
        if not isinstance(self.coefficients, csdl.Variable):
            self.coefficients = csdl.Variable(value=self.coefficients)

        if len(self.coefficients.shape) == 1:
            self.num_physical_dimensions = 1
        else:
            self.num_physical_dimensions = self.coefficients.shape[-1]


    def _compute_distance_bounds(self, point, direction=None):
        return self.space._compute_distance_bounds(point, self, direction=direction)

    def copy(self) -> lfs.Function:
        '''
        Returns a copy of the function.
        '''
        return lfs.Function(space=self.space, coefficients=self.coefficients, name=self.name)

    def get_matrix_vector(self, parametric_coordinates:np.ndarray, parametric_derivative_orders:list[tuple]=None, coefficients:csdl.Variable=None,
                 non_csdl:bool=False):
        '''
        Gets the basis matrix and coefficients whose product is the function evaluated at the given coordinates.

        Parameters
        ----------
        parametric_coordinates : np.ndarray -- shape=(num_points, num_parametric_dimensions)
            The coordinates at which to evaluate the function.
        parametric_derivative_order : tuple = None -- shape=(num_points,num_parametric_dimensions)
            The order of the parametric derivatives to evaluate.
        coefficients : csdl.Variable = None -- shape=coefficients_shape
            The coefficients of the function.
        plot : bool = False
            Whether or not to plot the function with the points from the result of the evaluation.
        non_csdl : bool = False
            If true, will run numpy computations instead of csdl computations, and return a numpy array.

        Returns
        -------
        basis_matrix : np.ndarray | sps.csr_matrix
            The basis matrix evaluated at the given coordinates.
        coefficients : csdl.Variable
            The coefficients of the function.
        '''
        if coefficients is None:
            coefficients = self.coefficients

        if non_csdl and isinstance(coefficients, csdl.Variable):
            coefficients = coefficients.value

        basis_matrix = self.space.compute_basis_matrix(parametric_coordinates, parametric_derivative_orders)
        if coefficients.shape != (basis_matrix.shape[1], self.num_physical_dimensions):
            coefficients = coefficients.reshape((basis_matrix.shape[1], self.num_physical_dimensions))

        return basis_matrix, coefficients

    def evaluate(self, parametric_coordinates:np.ndarray, parametric_derivative_orders:list[tuple]=None, coefficients:csdl.Variable=None,
                 plot:bool=False, non_csdl:bool=False) -> csdl.Variable:
        '''
        Evaluates the function.

        Parameters
        ----------
        parametric_coordinates : np.ndarray -- shape=(num_points, num_parametric_dimensions)
            The coordinates at which to evaluate the function.
        parametric_derivative_order : tuple = None -- shape=(num_points,num_parametric_dimensions)
            The order of the parametric derivatives to evaluate.
        coefficients : csdl.Variable = None -- shape=coefficients_shape
            The coefficients of the function.
        plot : bool = False
            Whether or not to plot the function with the points from the result of the evaluation.
        non_csdl : bool = False
            If true, will run numpy computations instead of csdl computations, and return a numpy array.

        Returns
        -------
        function_values : csdl.Variable
            The function evaluated at the given coordinates.
        '''
        if coefficients is None:
            coefficients = self.coefficients

        if non_csdl and isinstance(coefficients, csdl.Variable):
            coefficients = coefficients.value

        basis_matrix = self.space.compute_basis_matrix(parametric_coordinates, parametric_derivative_orders)
        # # values = basis_matrix @ coefficients
        if isinstance(coefficients, csdl.Variable) and sps.issparse(basis_matrix):
            if coefficients.shape != (basis_matrix.shape[1], coefficients.size//basis_matrix.shape[1]):
                coefficients = coefficients.reshape((basis_matrix.shape[1], coefficients.size//basis_matrix.shape[1]))
            # NOTE: TEMPORARY IMPLEMENTATION SINCE CSDL ONLY SUPPORTS SPARSE MATVECS AND NOT MATMATS
            values = csdl.Variable(value=np.zeros((basis_matrix.shape[0], coefficients.shape[1])))
            for i in csdl.frange(coefficients.shape[1]):
                coefficients_column = coefficients[:,i].reshape((coefficients.shape[0],1))
                values = values.set(csdl.slice[:,i], csdl.sparse.matvec(basis_matrix, coefficients_column).reshape((basis_matrix.shape[0],)))
        else:
            values = basis_matrix @ coefficients.reshape((basis_matrix.shape[1], self.num_physical_dimensions))

        if len(parametric_coordinates.shape) == 1:
            pass    # Come back to this case

        if values.shape != parametric_coordinates.shape[:-1] + (self.num_physical_dimensions,):
            values = values.reshape(parametric_coordinates.shape[:-1] + (self.num_physical_dimensions,))

        if values.shape[0] == 1:
            values = values[0]  # Get rid of the extra dimension if only one point is evaluated
        if values.shape[-1] == 1 and len(values.shape) > 1:
            values = values.reshape(values.shape[:-1])   # Get rid of the extra dimension if only one physical dimension is evaluated
        elif values.shape[-1] == 1:
            values = values[0]
        
            # values = csdl.sparse.matvec or matmat(basis_matrix, coefficients_reshaped)
        # elif isinstance(coefficients, csdl.Variable):
        #     values = csdl.matvec(basis_matrix, coefficients)
        # else:
        #     values = basis_matrix.dot(coefficients.reshape((basis_matrix.shape[1], -1)))

        # values = basis_matrix @ coefficients.reshape((basis_matrix.shape[1], -1))

        if plot:
            # Plot the function
            plotting_elements = self.plot(opacity=0.8, show=False)
            # Plot the evaluated points
            if non_csdl:
                vals = values
            else:
                vals = values.value
            lfs.plot_points(vals, color='#C69214', size=10, additional_plotting_elements=plotting_elements)

        return values
    
    # def integrate(self, area, grid_n=10):
    #     # Generate parametric grid
    #     # parametric_grid = self.space.generate_parametric_grid(grid_n)
    #     parametric_grid = np.zeros((grid_n, grid_n, 2))
    #     for i in range(grid_n):
    #         for j in range(grid_n):
    #             parametric_grid[i,j] = np.array([i/(grid_n-1), j/(grid_n-1)])

    #     # print('parametric grid shape', parametric_grid.shape)
    #     # parametric_grid = parametric_grid.reshape(grid_n, grid_n, -1)        

    #     # Get the parametric coordinates of the grid center points
    #     grid_centers = np.zeros((grid_n-1, grid_n-1, self.space.num_parametric_dimensions))
    #     for i in range(grid_n-1):
    #         for j in range(grid_n-1):
    #             grid_centers[i,j] = (parametric_grid[i+1, j] + parametric_grid[i, j] + parametric_grid[i, j+1] + parametric_grid[i+1, j+1])/4
    #     # Evaluate grid of points
    #     grid_values = area.evaluate(parametric_coordinates=parametric_grid.reshape(-1,2)).reshape((grid_n, grid_n, -1))
    #     grid_center_values = self.evaluate(parametric_coordinates=grid_centers.reshape(-1,2)).reshape((grid_n-1, grid_n-1, self.num_physical_dimensions))

    #     values = csdl.Variable(value=np.zeros((grid_n-1, grid_n-1)))
    #     for i in csdl.frange(grid_n-1):
    #         for j in csdl.frange(grid_n-1):
    #             # Compute the area of the quadrilateral
    #             area_1 = csdl.norm(csdl.cross(grid_values[i+1,j]-grid_values[i,j], grid_values[i,j+1]-grid_values[i,j]))/2
    #             area_2 = csdl.norm(csdl.cross(grid_values[i,j+1]-grid_values[i+1,j+1], grid_values[i+1,j]-grid_values[i+1,j+1]))/2
    #             area = area_1 + area_2

    #             values = values.set(csdl.slice[i,j], grid_center_values[i,j]*area)

    #     return values.reshape((-1, self.num_physical_dimensions)), grid_centers.reshape(-1, self.space.num_parametric_dimensions)
    
    def integrate(self, area, grid_n=10, quadrature_order=2):
        """
        Integrate the function over the area (2D). Uses gaussian quadrature for the integration.
        """

        # Generate parametric grid
        parametric_grid = np.zeros((grid_n, grid_n, 2))
        for i in range(grid_n):
            for j in range(grid_n):
                parametric_grid[i,j] = np.array([i/(grid_n-1), j/(grid_n-1)])

        # get quadrature points and weights
        quadrature_points, quadrature_weights = np.polynomial.legendre.leggauss(quadrature_order)
        quadrature_points = (quadrature_points + 1)/2
        quadrature_weights = quadrature_weights/2
        quadrature_coords = np.zeros((quadrature_order**2, 2))
        quadrature_coord_weights = np.zeros((quadrature_order**2,))
        for i in range(quadrature_order):
            for j in range(quadrature_order):
                quadrature_coords[i*quadrature_order+j] = np.array([quadrature_points[i], quadrature_points[j]])
                quadrature_coord_weights[i*quadrature_order+j] = quadrature_weights[i]*quadrature_weights[j]
        quadrature_coord_weights = csdl.Variable(value=quadrature_coord_weights)

        # get the parametric coordinates of the quadrature points
        quadrature_parametric_coords = np.zeros((grid_n-1, grid_n-1, quadrature_order**2, 2))
        for i in range(grid_n-1):
            for j in range(grid_n-1):
                for k in range(quadrature_order**2):
                    quadrature_parametric_coords[i,j,k] = parametric_grid[i,j] + quadrature_coords[k]/(grid_n-1)

        # evaluate the function at the quadrature points
        quadrature_values = self.evaluate(parametric_coordinates=quadrature_parametric_coords.reshape(-1,2)).reshape((grid_n-1, grid_n-1, quadrature_order**2, self.num_physical_dimensions))

        # compute the integral
        values = csdl.Variable(value=np.zeros((grid_n-1, grid_n-1, self.num_physical_dimensions)))
        for i in csdl.frange(grid_n-1):
            for j in csdl.frange(grid_n-1):
                for k in csdl.frange(quadrature_order**2):
                    values = values.set(csdl.slice[i,j], values[i,j] + quadrature_values[i,j,k]*quadrature_coord_weights[k])

        # compute areas of the quadrilaterals
        grid_values = area.evaluate(parametric_coordinates=parametric_grid.reshape(-1,2)).reshape((grid_n, grid_n, -1))
        output = csdl.Variable(value=np.zeros((grid_n-1, grid_n-1)))
        for i in csdl.frange(grid_n-1):
            for j in csdl.frange(grid_n-1):
                area_1 = csdl.norm(csdl.cross(grid_values[i+1,j]-grid_values[i,j], grid_values[i,j+1]-grid_values[i,j]) + 1e-8)/2
                area_2 = csdl.norm(csdl.cross(grid_values[i,j+1]-grid_values[i+1,j+1], grid_values[i+1,j]-grid_values[i+1,j+1]) + 1e-8)/2
                output = output.set(csdl.slice[i,j], (area_1+area_2)*values[i,j])

        # Get the parametric coordinates of the grid center points
        grid_centers = np.zeros((grid_n-1, grid_n-1, self.space.num_parametric_dimensions))
        for i in range(grid_n-1):
            for j in range(grid_n-1):
                grid_centers[i,j] = (parametric_grid[i+1, j] + parametric_grid[i, j] + parametric_grid[i, j+1] + parametric_grid[i+1, j+1])/4

        return output.reshape((-1, self.num_physical_dimensions)), grid_centers.reshape(-1, self.space.num_parametric_dimensions) 



    def refit(self, new_function_space:lfs.FunctionSpace, grid_resolution:tuple=None, 
              parametric_coordinates:np.ndarray=None, parametric_derivative_orders:np.ndarray=None,
              regularization_parameter:float=None) -> Function:
        '''
        Optimally refits the function. Either a grid resolution or parametric coordinates must be provided. 
        If both are provided, the parametric coordinates will be used. If derivatives are used, the parametric derivative orders must be provided.

        NOTE: this method will not overwrite the coefficients or function space in this object. 
        It will return a new function object with the refitted coefficients.

        Parameters
        ----------
        new_function_space : FunctionSpace
            The new function space that the function will be picked from.
        grid_resolution : tuple = None -- shape=(num_parametric_dimensions,)
            The resolution of the grid to refit the function.
        parametric_coordinates : np.ndarray = None -- shape=(num_points, num_parametric_dimensions)
            The coordinates at which to refit the function.
        parametric_derivative_orders : np.ndarray = None -- shape=(num_points, num_parametric_dimensions)
            The orders of the parametric derivatives to refit.

        Returns
        -------
        Function
            The refitted function with the new function space and new coefficients.
        '''

        if parametric_coordinates is None and grid_resolution is None:
            # raise ValueError("Either grid resolution or parametric coordinates must be provided.")
            grid_resolution = (100,)*self.space.num_parametric_dimensions
        if parametric_coordinates is not None and grid_resolution is not None:
            print("Warning: Both grid resolution and parametric coordinates were provided. Using parametric coordinates.")
            # raise Warning("Both grid resolution and parametric coordinates were provided. Using parametric coordinates.")
        
        if parametric_coordinates is None:
            # if grid_resolution is not None: # Don't need this line because we already error checked at the beginning.
            mesh_grid_input = []
            for dimension_index in range(self.space.num_parametric_dimensions):
                mesh_grid_input.append(np.linspace(0., 1., grid_resolution[dimension_index]))

            parametric_coordinates_tuple = np.meshgrid(*mesh_grid_input, indexing='ij')
            for dimensions_index in range(self.space.num_parametric_dimensions):
                parametric_coordinates_tuple[dimensions_index] = parametric_coordinates_tuple[dimensions_index].reshape((-1,1))

            parametric_coordinates = np.hstack(parametric_coordinates_tuple)

        # JUST CALL self.evaluate()!
        # basis_matrix = self.space.compute_basis_matrix(parametric_coordinates, parametric_derivative_orders)
        # coefficients_reshaped = self.coefficients.reshape((self.coefficients.size//self.num_physical_dimensions,  self.num_physical_dimensions))
        # fitting_values = csdl.Variable(value=np.zeros((parametric_coordinates.shape[0], self.num_physical_dimensions)))
        # for i in range(self.num_physical_dimensions):
        #     fitting_values = fitting_values.set(csdl.slice[:,i], csdl.sparse.matvec(basis_matrix, 
        #                                                             coefficients_reshaped[:,i].reshape((coefficients_reshaped.shape[0],1))).flatten())
        # # fitting_values = basis_matrix.dot(self.coefficients.value.reshape((-1,self.num_physical_dimensions)))
        fitting_values = self.evaluate(parametric_coordinates, parametric_derivative_orders=parametric_derivative_orders)
        
        coefficients = new_function_space.fit(
            values=fitting_values,
            parametric_coordinates=parametric_coordinates,
            parametric_derivative_orders=parametric_derivative_orders,
            regularization_parameter=regularization_parameter)
        
        new_function = Function(space=new_function_space, coefficients=coefficients)
        return new_function


    def project(self, points:np.ndarray, direction:np.ndarray=None, grid_search_density_parameter:int=1, 
                max_newton_iterations:int=100, newton_tolerance:float=1e-6, plot:bool=False, force_reproject:bool=False, do_pickles=True) -> csdl.Variable:
        '''
        Projects a set of points onto the function. The points to project must be provided. If a direction is provided, the projection will find
        the points on the function that are closest to the axis defined by the direction. If no direction is provided, the projection will find the
        points on the function that are closest to the points to project. The grid search density parameter controls the density of the grid search
        used to find the initial guess for the Newton iterations. The max newton iterations and newton tolerance control the convergence of the
        Newton iterations. If plot is True, a plot of the projection will be displayed.

        NOTE: Distance is measured by the 2-norm.

        Parameters
        ----------
        points : np.ndarray -- shape=(num_points, num_phyiscal_dimensions)
            The points to project onto the function.
        direction : np.ndarray = None -- shape=(num_parametric_dimensions,)
            The direction of the projection.
        grid_search_density_parameter : int = 1
            The density of the grid search used to find the initial guess for the Newton iterations.
        max_newton_iterations : int = 100
            The maximum number of Newton iterations.
        newton_tolerance : float = 1e-6
            The tolerance for the Newton iterations.
        plot : bool = False
            Whether or not to plot the projection.
        '''
        if isinstance(points, csdl.Variable):
            points = points.value

        if do_pickles:
            output = self._check_whether_to_load_projection(points, direction, 
                                                            grid_search_density_parameter, 
                                                            max_newton_iterations, 
                                                            newton_tolerance,
                                                            force_reproject)
            if isinstance(output, np.ndarray):
                parametric_coordinates = output
                if plot:
                    projection_results = self.evaluate(parametric_coordinates).value
                    plotting_elements = []
                    plotting_elements.append(lfs.plot_points(points, color='#00629B', size=10, show=False))
                    plotting_elements.append(lfs.plot_points(projection_results, color='#C69214', size=10, show=False))
                    self.plot(opacity=0.8, additional_plotting_elements=plotting_elements, show=True)
                return parametric_coordinates
            else:
                name_space_dict, long_name_space = output

        num_physical_dimensions = points.shape[-1]

        points = points.reshape((-1, num_physical_dimensions))

        # grid_search_resolution = 10*grid_search_density_parameter//self.space.num_parametric_dimensions + 1
        if not hasattr(self, '_grid_searches'):
            self._grid_searches = {}
        if grid_search_density_parameter not in self._grid_searches:
            grid_search_resolution = self.space._generate_projection_grid_search_resolution(grid_search_density_parameter)
            if grid_search_resolution is None:
                grid_search_resolution = 10*grid_search_density_parameter//self.space.num_parametric_dimensions + 1
            # grid_search_resolution = 100

            # Generate parametric grid
            parametric_grid_search = self.space.generate_parametric_grid(grid_search_resolution)
            # Evaluate grid of points
            grid_search_values = self.evaluate(parametric_coordinates=parametric_grid_search, coefficients=self.coefficients.value)
            expanded_points_size = points.shape[0]*grid_search_values.shape[0]
            self._grid_searches[grid_search_density_parameter] = (parametric_grid_search, grid_search_values, expanded_points_size)
        else:
            parametric_grid_search, grid_search_values, expanded_points_size = self._grid_searches[grid_search_density_parameter]
        cutoff_size = 1.5e8
        if expanded_points_size > cutoff_size:
            # grid search sections of points at a time
            num_sections = int(np.ceil(expanded_points_size/cutoff_size))
            section_size = int(np.ceil(points.shape[0]/num_sections))
            closest_point_indices = np.zeros((points.shape[0],), dtype=int)
            for i in range(num_sections):
                start_index = i*section_size
                end_index = min((i+1)*section_size, points.shape[0])
                points_expanded = np.repeat(points[start_index:end_index,np.newaxis,:], grid_search_values.shape[0], axis=1)
                grid_search_displacements = grid_search_values - points_expanded
                grid_search_distances = np.linalg.norm(grid_search_displacements, axis=2)

                # Perform a grid search
                if direction is None:
                    closest_point_indices[start_index:end_index] = np.argmin(grid_search_distances, axis=1)
                else:
                    direction = direction/np.linalg.norm(direction)
                    rho = 1e-3
                    grid_search_distances_along_axis = np.dot(grid_search_displacements, direction)
                    grid_search_distances_from_axis_squared = (1 + rho)*grid_search_distances**2 - grid_search_distances_along_axis**2
                    closest_point_indices[start_index:end_index] = np.argmin(grid_search_distances_from_axis_squared, axis=1)
            
        else:
            points_expanded = np.repeat(points[:,np.newaxis,:], grid_search_values.shape[0], axis=1)
            grid_search_displacements = grid_search_values - points_expanded
            grid_search_distances = np.linalg.norm(grid_search_displacements, axis=2)

            # Perform a grid search
            if direction is None:
                # If no direction is provided, the projection will find the points on the function that are closest to the points to project.
                # The grid search will be used to find the initial guess for the Newton iterations
                
                # Find closest point on function to each point to project
                # closest_point_indices = np.argmin(np.linalg.norm(grid_search_values - points, axis=1))
                closest_point_indices = np.argmin(grid_search_distances, axis=1)

            else:
                # If a direction is provided, the projection will find the points on the function that are closest to the axis defined by the direction.
                # The grid search will be used to find the initial guess for the Newton iterations
                direction = direction/np.linalg.norm(direction)
                rho = 1e-3
                grid_search_distances_along_axis = np.dot(grid_search_displacements, direction)
                grid_search_distances_from_axis_squared = (1 + rho)*grid_search_distances**2 - grid_search_distances_along_axis**2
                closest_point_indices = np.argmin(grid_search_distances_from_axis_squared, axis=1)

        # Use the parametric coordinate corresponding to each closest point as the initial guess for the Newton iterations
        initial_guess = parametric_grid_search[closest_point_indices]


        # current_guess = initial_guess.copy()
        # # As a first implementation approach, loop over points to project and perform Newton optimization for each point
        # for i in range(points.shape[0]):
        #     for j in range(max_newton_iterations):
        #         # Perform B-spline evaluations needed for gradient and hessian (0th, 1st, and 2nd order derivatives needed)
        #         function_value = self.evaluate(current_guess[i]).value

        #         displacement = (points[i] - function_value).flatten()
        #         d_displacement_d_parametric = np.zeros((num_physical_dimensions, self.space.num_parametric_dimensions,))
        #         d2_displacement_d_parametric2 = np.zeros((num_physical_dimensions, self.space.num_parametric_dimensions, self.space.num_parametric_dimensions))
        #         for k in range(self.space.num_parametric_dimensions):
        #             parametric_derivative_orders = np.zeros((self.space.num_parametric_dimensions,), dtype=int)
        #             parametric_derivative_orders[k] = 1
        #             d_displacement_d_parametric[:,k] = -self.space.compute_basis_matrix(
        #                 current_guess[i], parametric_derivative_orders=parametric_derivative_orders
        #                 ).dot(self.coefficients.value.reshape((-1,num_physical_dimensions)))
        #             for m in range(self.space.num_parametric_dimensions):
        #                 parametric_derivative_orders = np.zeros((self.space.num_parametric_dimensions,))
        #                 if m == k:
        #                     parametric_derivative_orders[m] = 2
        #                 else:
        #                     parametric_derivative_orders[k] = 1
        #                     parametric_derivative_orders[m] = 1
        #                 d2_displacement_d_parametric2[:,k,m] = -self.space.compute_basis_matrix(
        #                     current_guess[i], parametric_derivative_orders=parametric_derivative_orders
        #                     ).dot(self.coefficients.value.reshape((-1,num_physical_dimensions)))

        #         # Construct the gradient and hessian
        #         gradient = 2*displacement.dot(d_displacement_d_parametric)
        #         hessian = 2*(np.tensordot(d_displacement_d_parametric, d_displacement_d_parametric, axes=[0,0])
        #                      + np.tensordot(displacement, d2_displacement_d_parametric2, axes=[0,0]))
                
        #         # Remove dof that are on constrant boundary and want to leave (active subspace method)
        #         coorinates_to_remove_on_lower_boundary = np.logical_and(current_guess[i] == 0, gradient > 0)
        #         coorinates_to_remove_on_upper_boundary = np.logical_and(current_guess[i] == 1, gradient < 0)
        #         coorinates_to_remove = np.logical_or(coorinates_to_remove_on_lower_boundary, coorinates_to_remove_on_upper_boundary)
        #         coordinates_to_keep = np.arange(self.space.num_parametric_dimensions)[np.logical_not(coorinates_to_remove)]

        #         # coordinates_to_keep = np.setdiff1d(np.arange(self.space.num_parametric_dimensions), coorinates_to_remove)
        #         reduced_gradient = gradient[coordinates_to_keep]
        #         reduced_hessian = hessian[np.ix_(coordinates_to_keep, coordinates_to_keep)]
                
        #         # # Finite difference check gradient
        #         # finite_difference_gradient = np.zeros((self.space.num_parametric_dimensions,))
        #         # for k in range(self.space.num_parametric_dimensions):
        #         #     delta = 1e-6
        #         #     current_guess_plus_delta = current_guess[i].copy()
        #         #     current_guess_plus_delta[k] += delta
        #         #     function_value_plus_delta = self.evaluate(current_guess_plus_delta).value
        #         #     displacement_plus_delta = (points[i] - function_value_plus_delta).flatten()
        #         #     objective = displacement_plus_delta.dot(displacement_plus_delta)
        #         #     finite_difference_gradient[k] = (objective - displacement.dot(displacement))/delta

        #         # Check for convergence
        #         if np.linalg.norm(reduced_gradient) < newton_tolerance:
        #             break

        #         # Solve the linear system
        #         # delta = np.linalg.solve(hessian, -gradient)
        #         delta = np.linalg.solve(reduced_hessian, -reduced_gradient)

        #         # Update the initial guess
        #         current_guess[i,coordinates_to_keep] += delta
        #         # If any of the coordinates are outside the bounds, set them to the bounds
        #         current_guess[i] = np.clip(current_guess[i], 0., 1.)

        # Experimental implementation that does all the Newton optimizations at once to vectorize many of the computations
        current_guess = initial_guess.copy()
        points_left_to_converge = np.arange(points.shape[0])
        for j in range(max_newton_iterations):
            # Perform B-spline evaluations needed for gradient and hessian (0th, 1st, and 2nd order derivatives needed)
            function_values = self.evaluate(parametric_coordinates=current_guess[points_left_to_converge], coefficients=self.coefficients.value)
            displacements = (points[points_left_to_converge] - function_values).reshape(points_left_to_converge.shape[0], num_physical_dimensions)
            
            d_displacement_d_parametric = np.zeros((points_left_to_converge.shape[0], num_physical_dimensions, self.space.num_parametric_dimensions))
            d2_displacement_d_parametric2 = np.zeros((points_left_to_converge.shape[0], num_physical_dimensions, 
                                                      self.space.num_parametric_dimensions, self.space.num_parametric_dimensions))

            for k in range(self.space.num_parametric_dimensions):
                parametric_derivative_orders = np.zeros((self.space.num_parametric_dimensions,), dtype=int)
                parametric_derivative_orders[k] = 1
                # d_displacement_d_parametric[:, :, k] = -np.tensordot(
                #     self.space.compute_basis_matrix(current_guess, parametric_derivative_orders=parametric_derivative_orders),
                #     self.coefficients.value.reshape(-1, num_physical_dimensions), axes=[1,0])
                d_displacement_d_parametric[:, :, k] = -self.space.compute_basis_matrix(current_guess[points_left_to_converge], 
                                                                                        parametric_derivative_orders=parametric_derivative_orders).dot(
                                                                    self.coefficients.value.reshape(-1, num_physical_dimensions))
                    # NOTE on indices: i=points, j=coefficients, k=physical dimensions

                for m in range(self.space.num_parametric_dimensions):
                    parametric_derivative_orders = np.zeros((self.space.num_parametric_dimensions,))
                    if m == k:
                        parametric_derivative_orders[m] = 2
                    else:
                        parametric_derivative_orders[k] = 1
                        parametric_derivative_orders[m] = 1
                    # d2_displacement_d_parametric2[:, :, k, m] = -np.einsum(
                    #     self.space.compute_basis_matrix(current_guess, parametric_derivative_orders=parametric_derivative_orders),
                    #     self.coefficients.value.reshape((-1, num_physical_dimensions)), 'ij,jk->ik')
                    d2_displacement_d_parametric2[:, :, k, m] = -self.space.compute_basis_matrix(current_guess[points_left_to_converge], 
                                                                            parametric_derivative_orders=parametric_derivative_orders).dot(
                                                                        self.coefficients.value.reshape((-1, num_physical_dimensions)))
                        # NOTE on indices: i=points, j=coefficients, k=physical dimensions

            # Construct the gradient and hessian
            if direction is None:
                gradient = 2 * np.einsum('ij,ijk->ik', displacements, d_displacement_d_parametric)
                hessian = 2 * (np.einsum('ijk,ijm->ikm', d_displacement_d_parametric, d_displacement_d_parametric)
                            + np.einsum('ij,ijkm->ikm', displacements, d2_displacement_d_parametric2))
            else:
                displacement_dot_d_displacement_d_parametric = np.einsum('ij,ijk->ik', displacements, d_displacement_d_parametric)
                direction_dot_displacement = np.einsum('j,ij->i', direction, displacements)
                direction_dot_d_displacement_d_parametric = np.einsum('j,ijk->ik', direction, d_displacement_d_parametric)
                direction_dot_d2_displacement_d_parametric2 = np.einsum('j,ijkm->ikm', direction, d2_displacement_d_parametric2)
                gradient = 2 * ((1 + rho)*displacement_dot_d_displacement_d_parametric 
                                - direction_dot_displacement[:, np.newaxis] * direction_dot_d_displacement_d_parametric)
                hessian = 2 * ( (1 + rho)*(
                    np.einsum('ijk,ijm->ikm', d_displacement_d_parametric, d_displacement_d_parametric)
                    + np.einsum('ij,ijkm->ikm', displacements, d2_displacement_d_parametric2))
                    - np.einsum('ik,im->ikm', direction_dot_d_displacement_d_parametric, direction_dot_d_displacement_d_parametric)
                    - np.einsum('i,ikm->ikm', direction_dot_displacement, direction_dot_d2_displacement_d_parametric2)
                )

            # Remove dof that are on constrant boundary and want to leave (active subspace method)
            coordinates_to_remove_on_lower_boundary = np.logical_and(current_guess[points_left_to_converge] == 0, gradient > 0)
            coordinates_to_remove_on_upper_boundary = np.logical_and(current_guess[points_left_to_converge] == 1, gradient < 0)
            coordinates_to_from_zero_hessian_column = np.where(~hessian.any(axis=1))[0] # Axis is 1 because we want to remove the column
            coordinates_to_remove_boolean = np.logical_or(coordinates_to_remove_on_lower_boundary, coordinates_to_remove_on_upper_boundary)
            coordinates_to_remove_boolean[coordinates_to_from_zero_hessian_column] = True

            coordinates_to_keep_boolean = np.logical_not(coordinates_to_remove_boolean)
            indices_to_keep = []
            for i in range(points_left_to_converge.shape[0]):
                indices_to_keep.append(np.arange(self.space.num_parametric_dimensions)[coordinates_to_keep_boolean[i]])

            reduced_gradients = []
            reduced_hessians = []
            total_gradient_norm = 0.
            counter = 0
            for i in range(points_left_to_converge.shape[0]):
                reduced_gradient = gradient[i, indices_to_keep[counter]]

                if np.linalg.norm(reduced_gradient) < newton_tolerance:
                    points_left_to_converge = np.delete(points_left_to_converge, counter)
                    del indices_to_keep[counter]
                    continue

                # This is after check so it doesn't throw error
                reduced_hessian = hessian[np.ix_(np.array([i]), indices_to_keep[counter], indices_to_keep[counter])][0]    

                reduced_gradients.append(reduced_gradient)
                reduced_hessians.append(reduced_hessian)
                total_gradient_norm += np.linalg.norm(reduced_gradient)
                counter += 1

            # Check for convergence
            if np.linalg.norm(total_gradient_norm) < newton_tolerance:
                break

            # Solve the linear systems
            for i, index in enumerate(points_left_to_converge):
                delta = np.linalg.solve(reduced_hessians[i], -reduced_gradients[i])

                # Update the initial guess
                current_guess[index, indices_to_keep[i]] += delta

            # If any of the coordinates are outside the bounds, set them to the bounds
            current_guess[points_left_to_converge] = np.clip(current_guess[points_left_to_converge], 0., 1.)

        if plot:
            projection_results = self.evaluate(current_guess).value
            plotting_elements = []
            plotting_elements.append(lfs.plot_points(points, color='#00629B', size=10, show=False))
            plotting_elements.append(lfs.plot_points(projection_results, color='#C69214', size=10, show=False))
            self.plot(opacity=0.8, additional_plotting_elements=plotting_elements, show=True)

        if do_pickles:
            # Save the projection
            characters = string.ascii_letters + string.digits  # Alphanumeric characters
            # Generate a random string of the specified length
            random_string = ''.join(random.choice(characters) for _ in range(6))
            projections_folder = 'stored_files/projections'
            name_space_file_path = projections_folder + '/name_space_dict.pickle'
            name_space_dict[long_name_space] = random_string
            with open(name_space_file_path, 'wb+') as handle:
                pickle.dump(name_space_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(projections_folder + f'/{random_string}.pickle', 'wb+') as handle:
                pickle.dump(current_guess, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return current_guess


    def _check_whether_to_load_projection(self, points:np.ndarray, direction:np.ndarray=None, grid_search_density_parameter:int=1,
                                         max_newton_iterations:int=100, newton_tolerance:float=1e-6, force_reproject:bool=False) -> bool:
        # name_space = f'{self.name}'

        # name_space = ''
        # for function in self.functions.values():
        #     function_space = function.space

        #     coefficients = function.coefficients.value
        #     degree = function_space.degree
        #     coeff_shape = function_space.coefficients_shape
        #     knot_vectors_norm = round(np.linalg.norm(function_space.knots), 2)

        #     # if f'{target}_{str(degree)}_{str(coeff_shape)}_{str(knot_vectors_norm)}' in name_space:
        #     #     pass
        #     # else:
        #     name_space += f'_{str(coefficients)}_{str(degree)}_{str(coeff_shape)}_{str(knot_vectors_norm)}'

        knot_vectors_norm = round(np.linalg.norm(self.space.knots), 2)
        function_info = f'{self.name}_{self.coefficients.value}_{self.space.degree}_{knot_vectors_norm}'
        projection_info = f'{points}_{direction}_{grid_search_density_parameter}_{max_newton_iterations}_{newton_tolerance}'
        long_name_space = f'{function_info}_{projection_info}'

        projections_folder = 'stored_files/projections'
        name_space_file_path = projections_folder + '/name_space_dict.pickle'
        
        name_space_dict_file_path = Path(name_space_file_path)
        if name_space_dict_file_path.is_file():
            with open(name_space_file_path, 'rb') as handle:
                name_space_dict = pickle.load(handle)
        else:
            Path("stored_files/projections").mkdir(parents=True, exist_ok=True)
            name_space_dict = {}

        if long_name_space in name_space_dict.keys() and not force_reproject:
            short_name_space = name_space_dict[long_name_space]
            saved_projections_file = projections_folder + f'/{short_name_space}.pickle'
            with open(saved_projections_file, 'rb') as handle:
                parametric_coordinates = pickle.load(handle)
                return parametric_coordinates
        else:
            Path("stored_files/projections").mkdir(parents=True, exist_ok=True)

            return name_space_dict, long_name_space


    def plot(self, point_types:list=['evaluated_points'], plot_types:list=['function'],
              opacity:float=1., color:str|Function='#00629B', color_map:str='jet', surface_texture:str="",
              line_width:float=3., additional_plotting_elements:list=[], show:bool=True) -> list:
        '''
        Plots the B-spline Surface.

        Parameters
        -----------
        points_type : list = ['evaluated_points']
            The type of points to be plotted. {evaluated_points, coefficients}
        plot_types : list = ['function']
            The type of plot {function, wireframe, point_cloud}
        opactity : float = 1.
            The opacity of the plot. 0 is fully transparent and 1 is fully opaque.
        color : str = '#00629B'
            The 6 digit color code to plot the B-spline as. If a function is provided, the function will be used to color the B-spline.
        surface_texture : str = "" {"metallic", "glossy", ...}, optional
            The surface texture to determine how light bounces off the surface.
            See https://github.com/marcomusy/vedo/blob/master/examples/basic/lightings.py for options.
        color_map : str = 'jet'
            The color map to use if the color is a function.
        additional_plotting_elemets : list
            Vedo plotting elements that may have been returned from previous plotting functions that should be plotted with this plot.
        show : bool
            A boolean on whether to show the plot or not. If the plot is not shown, the Vedo plotting element is returned.

        Returns
        -------
        plotting_elements : list
            The Vedo plotting elements that were plotted.
        '''
        if self.coefficients is None:
            raise ValueError("The coefficients of the function are not defined.")
        
        plotting_elements = additional_plotting_elements.copy()
        for point_type in point_types:
            if point_type not in ['evaluated_points', 'coefficients']:
                raise ValueError(f"Invalid point type. Must be 'evaluated_points' or 'coefficients'. Got {point_type}.")
            
            if self.space.num_parametric_dimensions == 1:
                # NOTE: Curve plotting not currently implemented for points in 3D space because I don't have a num_physical_dimensions attribute.
                plotting_elements = self.plot_curve(point_type=point_type, opacity=opacity, color=color, color_map=color_map,
                                       line_width=line_width, additional_plotting_elements=plotting_elements, show=show)
            
            elif self.space.num_parametric_dimensions == 2:
                out = self.plot_surface(point_type=point_type, plot_types=plot_types, opacity=opacity, color=color, color_map=color_map,
                                        surface_texture=surface_texture, line_width=line_width,
                                        additional_plotting_elements=plotting_elements, show=show)
                if isinstance(out, tuple):
                    plotting_elements = out[0]
                    cmin = out[1]
                    cmax = out[2]
                else:
                    plotting_elements = out
            elif self.space.num_parametric_dimensions == 3:
                plotting_elements = self.plot_volume(point_type=point_type, plot_types=plot_types, opacity=opacity, color=color, color_map=color_map,
                                        surface_texture=surface_texture, line_width=line_width,
                                        additional_plotting_elements=plotting_elements, show=show)
            else:
                raise ValueError("The number of parametric dimensions must be 1, 2, or 3 in order to plot.")
            # elif isinstance(self.space, lfs.FunctionSetSpace):
            #     # Then there must be a discrete index so loop over subfunctions and plot them
            #     plotting_elements = []
            #     for index, subfunction_space_index in self.space.index_to_space.items():
            #         subfunction_space = self.space.spaces[subfunction_space_index]
            #         subfunction = Function(space=subfunction_space, coefficients=self.coefficients[self.space.index_to_coefficient_indices[index]])
            #         plotting_elements += subfunction.plot(point_types=point_types, plot_types=plot_types, opacity=opacity, color=color, color_map=color_map,
            #                                               surface_texture=surface_texture, line_width=line_width, 
            #                                               additional_plotting_elements=additional_plotting_elements, show=False)
            #     if show:
            #         lfs.show_plot(plotting_elements=plotting_elements, title='B-Spline Set Plot')
            #     return plotting_elements
        if isinstance(color, Function):
            return plotting_elements, cmin, cmax
        return plotting_elements


    def plot_points(self, point_type:str='evaluated_points', opacity:float=1., color:str|lfs.Function='#00629B', color_map:str='jet', 
                    size:float=10., additional_plotting_elements:list=[], show:bool=True) -> list:
        '''
        Plots the points of the function.

        Parameters
        -----------
        points_type : str = 'evaluated_points'
            The type of points to be plotted. {evaluated_points, coefficients}
        opactity : float = 1.
            The opacity of the plot. 0 is fully transparent and 1 is fully opaque.
        color : str = '#00629B'
            The 6 digit color code to plot the points as. If a function is provided, the function will be used to color the points.
        color_map : str = 'jet'
            The color map to use if the color is a function.
        size : float = 10.
            The size of the points.
        additional_plotting_elemets : list = []
            Vedo plotting elements that may have been returned from previous plotting functions that should be plotted with this plot.
        show : bool = True
            A boolean on whether to show the plot or not. If the plot is not shown, the Vedo plotting element is returned.

        Returns
        -------
        plotting_elements : list
            The Vedo plotting elements that were plotted.
        '''
        import lsdo_function_spaces.utils.plotting_functions as pf
        raise NotImplementedError("This function is not implemented yet.")


    def plot_curve(self, point_type:str='evaluated_points', opacity:float=1., color:str|lfs.Function='#00629B', color_map:str='jet',
                   line_width:float=3., additional_plotting_elements:list=[], show:bool=True):
        '''
        Plots the function as a curve. NOTE: This should only be called if the function is a curve!

        Parameters
        -----------
        points_type : str = 'evaluated_points'
            The type of points to be plotted. {evaluated_points, coefficients}
        opactity : float = 1.
            The opacity of the plot. 0 is fully transparent and 1 is fully opaque.
        color : str = '#00629B'
            The 6 digit color code to plot the function as. If a function is provided, the function will be used to color the curve.
        color_map : str = 'jet'
            The color map to use if the color is a function.
        additional_plotting_elemets : list = []
            Vedo plotting elements that may have been returned from previous plotting functions that should be plotted with this plot.
        show : bool = True
            A boolean on whether to show the plot or not. If the plot is not shown, the Vedo plotting element is returned.

        Returns
        -------
        plotting_elements : list
            The Vedo plotting elements that were plotted.
        '''
        import lsdo_function_spaces.utils.plotting_functions as pf
        if self.space.num_parametric_dimensions != 1:
            raise ValueError("This function is not a curve and cannot be plotted as one.")
        
        plotting_elements = additional_plotting_elements.copy()
        
        # region Generate the points to plot
        if point_type == 'evaluated_points':
            num_points = 100
            parametric_coordinates = np.linspace(0., 1., num_points).reshape((-1,1))
            function_values = self.evaluate(parametric_coordinates).value
            if len(function_values.shape) == 1:
                function_values = function_values.reshape((-1,1))   # Here we want the physical dimension separate for vedo so put it back

            # scale u axis to be more visually clear based on scaling of parameter
            if function_values.shape[-1] < 3:   # Plot against u coordinate
                u_axis_scaling = np.max(function_values) - np.min(function_values)
                if u_axis_scaling != 0:
                    parametric_coordinates = parametric_coordinates * u_axis_scaling
                points = np.hstack((parametric_coordinates, function_values))
            else:
                points = function_values

            if isinstance(color, Function):
                if color.space.num_parametric_dimensions != 1:
                    raise ValueError("The color function must be 1D to plot as a curve.")
                
                color = color.evaluate(parametric_coordinates).value
        elif point_type == 'coefficients':
            # NOTE: Check this line below!! I think this should really be the knot vector but I don't want to hardcode the existence of the knot vector.
            parametric_coordinates = np.linspace(0., 1., self.coefficients.shape[0]).reshape((-1,1))

            # scale u axis to be more visually clear based on scaling of parameter
            u_axis_scaling = np.max(self.coefficients.value) - np.min(self.coefficients.value)
            if u_axis_scaling != 0:
                parametric_coordinates = parametric_coordinates * u_axis_scaling

            points = np.hstack((parametric_coordinates, self.coefficients.value))

            if isinstance(color, Function):
                if color.space.num_parametric_dimensions != 1:
                    raise ValueError("The color function must be 1D to plot as a curve.")
                
                color = color.coefficients.value
                if color.size != points.size:
                    # If the number of coefficients are different, just evaluate the color function at the locations of the coefficients of the function.
                    color = color.evaluate(parametric_coordinates).value
        else:
            raise ValueError("Invalid point type. Must be 'evaluated_points' or 'coefficients'.")
        # endregion Generate the points to plot

        # Call general plot curve function to plot the points with the colors
        plotting_elements = pf.plot_curve(points=points, opacity=opacity, color=color, color_map=color_map, line_width=line_width, 
                                          additional_plotting_elements=plotting_elements, show=show)
        return plotting_elements
    

    def plot_surface(self, point_type:str='evaluated_points', plot_types:list=['function'], opacity:float=1., color:str|lfs.Function='#00629B',
                        color_map:str='jet', surface_texture:str="", line_width:float=3., additional_plotting_elements:list=[], show:bool=True):
        '''
        Plots the function as a surface. NOTE: This should only be called if the function is a surface!

        Parameters
        -----------
        points_type : str = 'evaluated_points'
            The type of points to be plotted. {evaluated_points, coefficients}
        plot_types : list = ['function']
            The type of plot {function, wireframe, point_cloud}
        opactity : float = 1.
            The opacity of the plot. 0 is fully transparent and 1 is fully opaque.
        color : str = '#00629B'
            The 6 digit color code to plot the function as. If a function is provided, the function will be used to color the surface.
        color_map : str = 'jet'
            The color map to use if the color is a function.
        surface_texture : str = ""
            The surface texture to determine how light bounces off the surface.
            See https://github.com/marcomusy/vedo/blob/master/examples/basic/lightings.py for options.
        line_width : float = 3.
            The width of the lines if the plot type is wireframe.
        additional_plotting_elemets : list = []
            Vedo plotting elements that may have been returned from previous plotting functions that should be plotted with this plot.
        show : bool = True
            A boolean on whether to show the plot or not. If the plot is not shown, the Vedo plotting element is returned.

        Returns
        -------
        plotting_elements : list
            The Vedo plotting elements that were plotted.
        '''
        import lsdo_function_spaces.utils.plotting_functions as pf
        if self.space.num_parametric_dimensions != 2:
            raise ValueError("This function is not a surface and cannot be plotted as one.")

        plotting_elements = additional_plotting_elements.copy()
        color_is_function = False

        # region Generate the points to plot
        if point_type == 'evaluated_points':
            num_points = 50

            # Generate meshgrid of parametric coordinates
            mesh_grid_input = []
            for dimension_index in range(self.space.num_parametric_dimensions):
                mesh_grid_input.append(np.linspace(0., 1., num_points))
            parametric_coordinates_tuple = np.meshgrid(*mesh_grid_input, indexing='ij')
            for dimensions_index in range(self.space.num_parametric_dimensions):
                parametric_coordinates_tuple[dimensions_index] = parametric_coordinates_tuple[dimensions_index].reshape((-1,1))
            parametric_coordinates = np.hstack(parametric_coordinates_tuple)
            
            function_values = self.evaluate(parametric_coordinates).value.reshape((num_points,num_points,-1))
            points = function_values

            if isinstance(color, Function):
                color_is_function = True
                if color.space.num_parametric_dimensions != 2:
                    raise ValueError("The color function must be 2D to plot as a surface.")
                color = color.evaluate(parametric_coordinates).value
                color_max = np.max(color)
                color_min = np.min(color)
                if len(color.shape) > 1:
                    if color.shape[1] > 1:
                        color = np.linalg.norm(color, axis=1)
        elif point_type == 'coefficients':
            points = self.coefficients.value    # Do I need to reshape this?

            if isinstance(color, Function):
                if color.space.num_parametric_dimensions != 2:
                    raise ValueError("The color function must be 2D to plot as a surface.")
                
                color = color.coefficients.value
                if color.size != points.size:
                    # If the number of coefficients are different, just evaluate the color function at the locations of the coefficients of the function.
                    # Generate meshgrid of parametric coordinates
                    mesh_grid_input = []
                    for dimension_index in range(self.space.num_parametric_dimensions):
                        mesh_grid_input.append(np.linspace(0., 1., self.coefficients.shape[dimension_index]))
                    parametric_coordinates_tuple = np.meshgrid(*mesh_grid_input, indexing='ij')
                    for dimensions_index in range(self.space.num_parametric_dimensions):
                        parametric_coordinates_tuple[dimensions_index] = parametric_coordinates_tuple[dimensions_index].reshape((-1,1))
                    parametric_coordinates = np.hstack(parametric_coordinates_tuple)
                    color = color.evaluate(parametric_coordinates).value
        else:
            raise ValueError("Invalid point type. Must be 'evaluated_points' or 'coefficients'.")
        # endregion Generate the points to plot

        # Call general plot surface function to plot the points with the colors
        for plot_type in plot_types:
            if plot_type not in ['function', 'wireframe', 'point_cloud']:
                raise ValueError("Invalid plot type. Must be 'function', 'wireframe', or 'point_cloud'.")
            if plot_type == 'point_cloud':
                plotting_elements = pf.plot_points(points=points, opacity=opacity, color=color, color_map=color_map, size=10., 
                                                   additional_plotting_elements=plotting_elements, show=False)
            elif plot_type in ['function', 'wireframe']:
                plotting_elements = pf.plot_surface(points=points, plot_types=[plot_type], opacity=opacity, color=color, color_map=color_map, 
                                                    surface_texture=surface_texture, line_width=line_width, 
                                                    additional_plotting_elements=plotting_elements, show=False)
        if show:
            if self.name is not None:
                pf.show_plot(plotting_elements, title=self.name, axes=1, interactive=True)
            else:
                pf.show_plot(plotting_elements, title="Surface", axes=1, interactive=True)
        if color_is_function:
            return plotting_elements, color_min, color_max
        return plotting_elements
    

    def plot_volume(self, point_type:str='evaluated_points', plot_types:list=['function'], opacity:float=1., color:str|lfs.Function='#00629B',
                        color_map:str='jet', surface_texture:str="", line_width:float=3., additional_plotting_elements:list=[], show:bool=True):
        '''
        Plots the function as a volume. NOTE: This should only be called if the function is a volume!

        Parameters
        -----------
        points_type : str = 'evaluated_points'
            The type of points to be plotted. {evaluated_points, coefficients}
        plot_types : list = ['function']
            The type of plot {function}
        opactity : float = 1.
            The opacity of the plot. 0 is fully transparent and 1 is fully opaque.
        color : str = '#00629B'
            The 6 digit color code to plot the function as. If a function is provided, the function will be used to color the volume.
        color_map : str = 'jet'
            The color map to use if the color is a function.
        surface_texture : str = ""
            The surface texture to determine how light bounces off the surface.
            See https://github.com/marcomusy/vedo/blob/master/examples/basic/lightings.py for options.
        line_width : float = 3.
            The width of the lines if the plot type is wireframe.
        additional_plotting_elemets : list = []
            Vedo plotting elements that may have been returned from previous plotting functions that should be plotted with this plot.
        show : bool = True
            A boolean on whether to show the plot or not. If the plot is not shown, the Vedo plotting elements are still returned.
        
        Returns
        -------
        plotting_elements : list
            The Vedo plotting elements that were plotted.
        '''
        import lsdo_function_spaces.utils.plotting_functions as pf
        if self.space.num_parametric_dimensions != 3:
            raise ValueError("This function is not a volume and cannot be plotted as one.")
        
        # region Generate the points to plot
        if point_type == 'evaluated_points':
            num_points = 50

            # Generate meshgrid of parametric coordinates
            linspace_dimension = np.linspace(0., 1., num_points)
            linspace_meshgrid = np.meshgrid(linspace_dimension, linspace_dimension)
            linspace_dimension1 = linspace_meshgrid[1].reshape((-1,1))
            linspace_dimension2 = linspace_meshgrid[0].reshape((-1,1))
            zeros_dimension = np.zeros((num_points**2,)).reshape((-1,1))
            ones_dimension = np.ones((num_points**2,)).reshape((-1,1))

            parametric_coordinates = []
            parametric_coordinates.append(np.column_stack((linspace_dimension1, linspace_dimension2, zeros_dimension)))
            parametric_coordinates.append(np.column_stack((linspace_dimension1, linspace_dimension2, ones_dimension)))
            parametric_coordinates.append(np.column_stack((linspace_dimension1, zeros_dimension, linspace_dimension2)))
            parametric_coordinates.append(np.column_stack((linspace_dimension1, ones_dimension, linspace_dimension2)))
            parametric_coordinates.append(np.column_stack((zeros_dimension, linspace_dimension1, linspace_dimension2)))
            parametric_coordinates.append(np.column_stack((ones_dimension, linspace_dimension1, linspace_dimension2)))
            
            points = []
            for parametric_coordinate_set in parametric_coordinates:
                points.append(self.evaluate(parametric_coordinates=parametric_coordinate_set).value.reshape((num_points,num_points,-1)))

            plotting_colors = []
            if isinstance(color, Function):
                if color.space.num_parametric_dimensions != 3:
                    raise ValueError("The color function must be 3D to plot as a volume.")
                
                for parametric_coordinate_set in parametric_coordinates:
                    plotting_colors.append(color.evaluate(parametric_coordinates=parametric_coordinate_set).value)
                color = plotting_colors

        elif point_type == 'coefficients':
            points = []
            points.append(self.coefficients.value[0,:,:])
            points.append(self.coefficients.value[-1,:,:])
            points.append(self.coefficients.value[:,0,:])
            points.append(self.coefficients.value[:,-1,:])
            points.append(self.coefficients.value[:,:,0])
            points.append(self.coefficients.value[:,:,-1])

            if isinstance(color, Function):
                if color.space.num_parametric_dimensions != 3:
                    raise ValueError("The color function must be 3D to plot as a volume.")
                
                color = color.coefficients.value
                if color.size != points.size:
                    raise NotImplementedError("For volumes, please use evaluated points to plot or "
                                              + "use a color function that has the same structure of coefficients.")
        else:
            raise ValueError("Invalid point type. Must be 'evaluated_points' or 'coefficients'.")
        # endregion Generate the points to plot

        # Call general plot volume function to plot the points with the colors
        plotting_elements = additional_plotting_elements.copy()
        for plot_type in plot_types:
            if plot_type not in ['function', 'wireframe', 'point_cloud']:
                raise ValueError("Invalid plot type. Must be 'function', 'wireframe', or 'point_cloud'.")

            for i in range(6):
                if isinstance(color, list):
                    plotting_color = color[i]
                else:
                    plotting_color = color

                if plot_type == 'point_cloud':
                    plotting_elements = pf.plot_points(points=points[i].reshape((-1,self.coefficients.shape[-1])), color=plotting_color, size=10., 
                                                       additional_plotting_elements=plotting_elements, show=False)
                elif plot_type in ['function', 'wireframe']:
                    plotting_elements = pf.plot_surface(points=points[i], plot_types=[plot_type], opacity=opacity, color=plotting_color,
                                                        color_map=color_map, surface_texture=surface_texture, line_width=line_width,
                                                         additional_plotting_elements=plotting_elements, show=False)

        if show:
            if self.name is not None:
                pf.show_plot(plotting_elements, title=self.name, axes=1, interactive=True)
            else:
                pf.show_plot(plotting_elements, title="Volume", axes=1, interactive=True)
        return plotting_elements
