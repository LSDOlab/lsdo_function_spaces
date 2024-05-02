from __future__ import annotations

from dataclasses import dataclass
import csdl_alpha as csdl
import numpy as np


from lsdo_function_spaces.core.function_space import FunctionSpace



@dataclass
class Function:
    '''
    Function class. This class is used to represent a function in a given function space. The function space is used to evaluate the function at
    given coordinates, refit the function, and project points onto the function.

    Attributes
    ----------
    space : FunctionSpace
        The function space in which the function resides.
    coefficients : csdl.Variable
        The coefficients of the function.
    '''
    space: FunctionSpace
    coefficients: csdl.Variable

    def __post_init__(self):
        pass

    def evaluate(self, parametric_coordinates:np.ndarray, coefficients:csdl.Variable=None, parametric_derivative_order:tuple=None) -> csdl.Variable:
        '''
        Evaluates the function.

        Parameters
        ----------
        parametric_coordinates : np.ndarray
            The coordinates at which to evaluate the function.
        coefficients : csdl.Variable
            The coefficients of the function.
        parametric_derivative_order : tuple = None
            The order of the parametric derivatives to evaluate.

        Returns
        -------
        csdl.Variable
            The function evaluated at the given coordinates.
        '''
        if coefficients is None:
            coefficients = self.coefficients
            
        return self.space.evaluate(
            coefficients=coefficients,
            parametric_coordinates=parametric_coordinates,
            parametric_derivative_order=parametric_derivative_order)
    

    def refit(self, new_function_space:FunctionSpace, grid_resolution:tuple=None, 
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

        '''
        NOTE: TODO: Look at error in L2 sense and think about whether this actually minimizes the error!!
        Additional NOTE: When the order changes, the ideal parametric coordinate corresponding to a value seems like it might change.
        -- To clarify: A point that is at u=0.1 in one function space may actually ideally be at u=0.15 or whatever in another function space.
        '''
        if parametric_coordinates is None and grid_resolution is None:
            raise ValueError("Either grid resolution or parametric coordinates must be provided.")
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

        evaluation_matrix = self.space.compute_evaluation_matrix(parametric_coordinates, parametric_derivative_orders)
        fitting_values = evaluation_matrix.dot(self.coefficients)
        
        coefficients = new_function_space.fit(
            values=fitting_values,
            parametric_coordinates=parametric_coordinates,
            parametric_derivative_orders=parametric_derivative_orders,
            regularization_parameter=regularization_parameter)
        
        new_function = Function(space=new_function_space, coefficients=coefficients)
        return new_function


    def project(self, points_to_project:np.ndarray, direction:np.ndarray=None, grid_search_density_parameter:int=1, 
                max_newton_iterations:int=100, newton_tolerance:float=1e-6, plot:bool=False) -> csdl.Variable:
        '''
        Projects a set of points onto the function. The points to project must be provided. If a direction is provided, the projection will find
        the points on the function that are closest to the axis defined by the direction. If no direction is provided, the projection will find the
        points on the function that are closest to the points to project. The grid search density parameter controls the density of the grid search
        used to find the initial guess for the Newton iterations. The max newton iterations and newton tolerance control the convergence of the
        Newton iterations. If plot is True, a plot of the projection will be displayed.

        NOTE: Distance is measured by the 2-norm.

        Parameters
        ----------
        points_to_project : np.ndarray -- shape=(num_points, num_phyiscal_dimensions)
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
        pass
        # Perform a grid search

        # Perform Newton iterations
