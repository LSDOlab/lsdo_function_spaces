from dataclasses import dataclass
import csdl_alpha as csdl
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import lsdo_function_spaces as lfs
from typing import Union

'''
NOTE: To implement a function space (for instance, B-splines), all that must be implemented is the evaluation of the basis functions 
    and assembly into a basis matrix.
'''


class FunctionSpace:
    '''
    Base class for function spaces. This class is used to evaluate functions at given coordinates, refit functions, and project points onto functions.
    '''
    def __init__(self, num_parametric_dimensions:int, coefficients_shape:tuple):
        """Base class for function spaces. This class is used to evaluate functions at given coordinates, refit functions, and project points onto functions.

        Parameters
        ----------
        num_parametric_dimensions : int
            Number of parametric dimensions of the function space.
        coefficients_shape : tuple
            Shape of the coefficients of the function space (will end up flattened).
        """
        # num_physical_dimensions : int     # I might need this, but not using for now so don't have if not needed
        # coefficients_shape : tuple    # Seems overly restrictive making things like transition elements and other unstructured functions impossible
        #   -- It seems like we really only need the num_parametric_dimensions
        self.num_parametric_dimensions = num_parametric_dimensions # This is really useful for the function methods
        self.coefficients_shape = coefficients_shape

        if isinstance(self.coefficients_shape, int):
            self.coefficients_shape = (self.coefficients_shape,)


    def generate_parametric_grid(self, grid_resolution:tuple) -> np.ndarray:
        '''
        Generates a parametric grid with the given resolution.

        Parameters
        ----------
        grid_resolution : tuple -- shape=(num_parametric_dimensions,)
            The resolution of the grid.

        Returns
        -------
        np.ndarray -- shape=(num_points, num_parametric_dimensions)
            The parametric grid.
        '''
        if isinstance(grid_resolution, int):
            grid_resolution = (grid_resolution,)*self.num_parametric_dimensions
        if len(grid_resolution) == 1 and self.num_parametric_dimensions > 1:
            grid_resolution = grid_resolution * self.num_parametric_dimensions

        mesh_grid_input = []
        for dimension_index in range(self.num_parametric_dimensions):
            mesh_grid_input.append(np.linspace(0., 1., grid_resolution[dimension_index]))

        parametric_coordinates_tuple = np.meshgrid(*mesh_grid_input, indexing='ij')
        for dimensions_index in range(self.num_parametric_dimensions):
            parametric_coordinates_tuple[dimensions_index] = parametric_coordinates_tuple[dimensions_index].reshape((-1,1))

        parametric_coordinates = np.hstack(parametric_coordinates_tuple)

        return parametric_coordinates
    

    # def evaluate(self, coefficients:csdl.Variable, parametric_coordinates:np.ndarray, parametric_derivative_order:tuple=None,
    #              plot:bool=False) -> csdl.Variable:
    #     '''
    #     Picks a function from the function space with the given coefficients and evaluates or its derivative(s) it at the parametric coordinates.

    #     Parameters
    #     ----------
    #     coefficients : csdl.Variable -- shape=coefficients_shape or (num_coefficients,)
    #         The coefficients of the function.
    #     parametric_coordinates : np.ndarray -- shape=(num_points, num_parametric_dimensions)
    #         The coordinates at which to evaluate the function.
    #     parametric_derivative_order : tuple = None -- shape=(num_points, num_parametric_dimensions)
    #         The order of the parametric derivatives to evaluate.
    #     plot : bool = False
    #         Whether or not to plot the function with the points from the result of the evaluation.

    #     Returns
    #     -------
    #     csdl.Variable
    #         The function evaluated at the given coordinates.
    #     '''
    #     basis_matrix = self.compute_basis_matrix(parametric_coordinates, parametric_derivative_order)
    #     if isinstance(coefficients, csdl.Variable) and sps.issparse(basis_matrix):
    #         coefficients_reshaped = coefficients.reshape((basis_matrix.shape[1], coefficients.size//basis_matrix.shape[1]))
    #         # NOTE: TEMPORARY IMPLEMENTATION SINCE CSDL ONLY SUPPORTS SPARSE MATVECS AND NOT MATMATS
    #         values = csdl.Variable(value=np.zeros((basis_matrix.shape[0], coefficients_reshaped.shape[1])))
    #         for i in range(coefficients_reshaped.shape[1]):
    #             coefficients_column = coefficients_reshaped[:,i].reshape((coefficients_reshaped.shape[0],1))
    #             values = values.set(csdl.slice[:,i], csdl.sparse.matvec(basis_matrix, coefficients_column).reshape((basis_matrix.shape[0],)))
    #         # values = csdl.sparse.matvec or matmat(basis_matrix, coefficients_reshaped)
    #     elif isinstance(coefficients, csdl.Variable):
    #         values = csdl.matvec(basis_matrix, coefficients)
    #     else:
    #         values = basis_matrix.dot(coefficients.reshape((basis_matrix.shape[1], -1)))

    #     return values
    #     # raise NotImplementedError(f"Evaluate method must be implemented in {type(self)} class.")
    

    def compute_basis_matrix(self, parametric_coordinates:np.ndarray, parametric_derivative_orders:np.ndarray=None) -> sps.csc_matrix:
        '''
        Evaluates the basis functions in parametric space and assembles the basis matrix (B(u) in P=B(u).dot(C)) where 
        B(u) is the evaluation matrix, P are the evaluated points in physical space, and C is the coefficients.

        Parameters
        ----------
        parametric_coordinates : np.ndarray -- shape=(num_points, num_parametric_dimensions)
            The coordinates at which to evaluate the function.
        parametric_derivative_orders : np.ndarray = None -- shape=(num_points, num_parametric_dimensions)
            The derivative orders to evaluate.

        Returns
        -------
        sps.csc_matrix
            The evaluation matrix.
        '''
        raise NotImplementedError(f"Compute evaluation matrix method must be implemented in {type(self)} class.")
    

    def compute_fitting_map(self, parametric_coordinates):
        raise NotImplementedError(f"Compute fitting map method must be implemented in {type(self)} class.")
    

    # def refit(self, coefficients:csdl.Variable, grid_resolution:tuple=None, parametric_coordinates:np.ndarray=None, 
    #           parametric_derivative_orders:np.ndarray=None, regularization_parameter:float=None) -> csdl.Variable:
    #     '''
    #     Picks a function from the function space with the given coefficients and evaluates or its derivative(s) it at the parametric coordinates.
    #     It then uses these values to refit the function. Either a grid resolution or parametric coordinates must be provided. 
    #     If both are provided, the parametric coordinates will be used. If derivatives are used, the parametric derivative orders must be provided.

    #     Parameters
    #     ----------
    #     coefficients : csdl.Variable -- shape=coefficients_shape
    #         The coefficients of the function to refit.
    #     grid_resolution : tuple = None -- shape=(num_parametric_dimensions,)
    #         The grid resolution to use for refitting.
    #     parametric_coordinates : np.ndarray = None -- shape=(num_points, num_parametric_dimensions)
    #         The parametric coordinates to use for refitting.
    #     parametric_derivative_orders : np.ndarray = None -- shape=(num_points, num_parametric_dimensions)
    #         The derivative orders to use for refitting.
    #     regularization_parameter : float = None
    #         The regularization parameter to use for refitting. If None, no regularization is used.

    #     Returns
    #     -------
    #     csdl.Variable
    #         The refitted coefficients.
    #     '''
        
    #     '''
    #     NOTE: TODO: Look at error in L2 sense and think about whether this actually minimizes the error!!
    #     Additional NOTE: When the order changes, the ideal parametric coordinate corresponding to a value seems like it might change.
    #     -- To clarify: A point that is at u=0.1 in one function space may actually ideally be at u=0.15 or whatever in another function space.
    #     '''
    #     if parametric_coordinates is None and grid_resolution is None:
    #         raise ValueError("Either grid resolution or parametric coordinates must be provided.")
    #     if parametric_coordinates is not None and grid_resolution is not None:
    #         print("Warning: Both grid resolution and parametric coordinates were provided. Using parametric coordinates.")
    #         # raise Warning("Both grid resolution and parametric coordinates were provided. Using parametric coordinates.")
        
    #     if parametric_coordinates is None:
    #         # if grid_resolution is not None: # Don't need this line because we already error checked at the beginning.
    #         mesh_grid_input = []
    #         for dimension_index in range(grid_resolution.shape[0]):  # Grid resolution is a tuple of the number of points in each parametric dimension
    #             mesh_grid_input.append(np.linspace(0., 1., grid_resolution[dimension_index]))

    #         parametric_coordinates_tuple = np.meshgrid(*mesh_grid_input, indexing='ij')
    #         for dimensions_index in range(grid_resolution.shape[0]):
    #             parametric_coordinates_tuple[dimensions_index] = parametric_coordinates_tuple[dimensions_index].reshape((-1,1))

    #         parametric_coordinates = np.hstack(parametric_coordinates_tuple)

    #     basis_matrix = self.compute_basis_matrix(parametric_coordinates, parametric_derivative_orders)
    #     fitting_values = basis_matrix.dot(coefficients)
        
    #     coefficients = self.fit(values=fitting_values, basis_matrix=basis_matrix, regularization_parameter=regularization_parameter)
        
    #     return coefficients
            
    #     # raise NotImplementedError(f"Refit method must be implemented in {type(self)} class.")
    #     # NOTE: This doesn't just call fit so we don't need to construct the evaluation matrix multiplie times.
    #     #   - Maybe it would be easier to have the fit function optionally take in the evaluation matrix?


    
    def fit(self, values:Union[csdl.Variable, np.ndarray], parametric_coordinates:np.ndarray=None, parametric_derivative_orders:np.ndarray=None,
            basis_matrix:Union[sps.csc_matrix, np.ndarray]=None, regularization_parameter:float=None) -> csdl.Variable:
        '''
        Fits the function to the given data. Either parametric coordinates or an evaluation matrix must be provided. If derivatives are used, the
        parametric derivative orders must be provided. If both parametric coordinates and an evaluation matrix are provided, the evaluation matrix
        will be used.

        Parameters
        ----------
        values : csdl.Variable|np.ndarray -- shape=(num_points,num_physical_dimensions)
            The values of the data.
        parametric_coordinates : np.ndarray -- shape=(num_points, num_parametric_dimensions)
            The parametric coordinates of the data.
        parametric_derivative_orders : np.ndarray = None -- shape=(num_points, num_parametric_dimensions)
            The derivative orders to fit.
        basis_matrix : sps.csc_matrix|np.ndarray = None -- shape=(num_points, num_coefficients)
            The evaluation matrix to use for fitting.
        regularization_parameter : float = None
            The regularization parameter to use for fitting. If None, no regularization is used.

        Returns
        -------
        csdl.Variable
            The coefficients of the fitted function.
        '''
        if parametric_coordinates is None and basis_matrix is None:
            raise ValueError("Either parametric coordinates or an evaluation matrix must be provided.")
        if parametric_coordinates is not None and basis_matrix is not None:
            print("Warning: Both parametric coordinates and an evaluation matrix were provided. Using the evaluation matrix.")
            # raise Warning("Both parametric coordinates and an evaluation matrix were provided. Using the evaluation matrix.")

        if len(values.shape) > 2:
            values = values.reshape((-1, values.shape[-1]))
        elif len(values.shape) == 1:
            values = values.reshape((1, -1))

        if parametric_coordinates is not None:
            try:
                fitting_map = self.compute_fitting_map(parametric_coordinates)
                return fitting_map @ values
            except NotImplementedError:
                basis_matrix = self.compute_basis_matrix(parametric_coordinates, parametric_derivative_orders)
                fitting_matrix = basis_matrix.T.dot(basis_matrix)
                
        if regularization_parameter is not None:
            if sps.issparse(fitting_matrix):
                fitting_matrix += regularization_parameter * sps.eye(fitting_matrix.shape[0]).tocsc()
            else:
                fitting_matrix += regularization_parameter * np.eye(fitting_matrix.shape[0])
        
        if isinstance(values, csdl.Variable) and sps.issparse(fitting_matrix):
            if len(values.shape) > 1:
                coefficients = csdl.Variable(value=np.zeros((fitting_matrix.shape[0], values.shape[1])))
                for i in range(values.shape[1]):
                    fitting_rhs = csdl.sparse.matvec(basis_matrix.T, values[:,i].reshape((values.shape[0],1)))
                    coefficients = coefficients.set(csdl.slice[:,i], csdl.solve_linear(fitting_matrix.toarray(), fitting_rhs).flatten())
                    # NOTE:  # CASTING FITTING MATRIX TO DENSE BECAUSE CSDL DOESN'T HAVE SPARSE SOLVE YET
            else:
                fitting_rhs = csdl.sparse.matvec(basis_matrix.T, values)
                coefficients = csdl.solve_linear(fitting_matrix.toarray(), fitting_rhs)
        else:
            if isinstance(values, csdl.Variable):
                if len(values.shape) > 1:
                    coefficients = csdl.Variable(value=np.zeros((fitting_matrix.shape[0], values.shape[1])))
                    for i in csdl.frange(values.shape[1]):
                        fitting_rhs = basis_matrix.T @ values[:,i]
                        coefficients = coefficients.set(csdl.slice[:,i], csdl.solve_linear(fitting_matrix, fitting_rhs).flatten())
                else:
                    fitting_rhs = basis_matrix.T @ values
                    coefficients = csdl.solve_linear(fitting_matrix, fitting_rhs)
            else:
                fitting_rhs = basis_matrix.T.dot(values)
                if sps.issparse(fitting_matrix):
                    coefficients = np.zeros((fitting_matrix.shape[0], fitting_rhs.shape[1]))
                    if len(fitting_rhs.shape) > 1:
                        for i in range(fitting_rhs.shape[1]):
                            coefficients[:,i] = spsl.spsolve(fitting_matrix, fitting_rhs[:,i])
                    else:
                        coefficients = spsl.spsolve(fitting_matrix, fitting_rhs)
                else:
                    coefficients = np.linalg.solve(fitting_matrix, fitting_rhs)

        coefficients = coefficients.reshape(self.coefficients_shape + (values.shape[-1],))

        return coefficients
        # raise NotImplementedError(f"Fit method must be implemented in {type(self)} class.")

    def fit_function(self, values:np.ndarray, parametric_coordinates:np.ndarray=None, parametric_derivative_orders:np.ndarray=None,
            basis_matrix:Union[sps.csc_matrix, np.ndarray]=None, regularization_parameter:float=None) -> lfs.Function:
        '''
        Fits the function to the given data. Either parametric coordinates or an evaluation matrix must be provided. If derivatives are used, the
        parametric derivative orders must be provided. If both parametric coordinates and an evaluation matrix are provided, the evaluation matrix
        will be used.

        Parameters
        ----------
        parametric_coordinates : np.ndarray -- shape=(num_points, num_parametric_dimensions)
            The parametric coordinates of the data.
        values : np.ndarray -- shape=(num_points,num_physical_dimensions)
            The values of the data.
        parametric_derivative_orders : np.ndarray = None -- shape=(num_points, num_parametric_dimensions)
            The derivative orders to fit.
        basis_matrix : sps.csc_matrix|np.ndarray = None -- shape=(num_points, num_coefficients)
            The evaluation matrix to use for fitting.
        regularization_parameter : float = None
            The regularization parameter to use for fitting. If None, no regularization is used.

        Returns
        -------
        lfs.Function
        '''
        coefficients = self.fit(values=values, parametric_coordinates=parametric_coordinates, parametric_derivative_orders=parametric_derivative_orders,
                                basis_matrix=basis_matrix, regularization_parameter=regularization_parameter)
        function = lfs.Function(space=self, coefficients=coefficients)
        return function
        # raise NotImplementedError(f"Fit function method must be implemented in {type(self)} class.")

    
    def _compute_distance_bounds(self, point, function):
        raise NotImplementedError(f"Compute distance bounds method must be implemented in {type(self)} class.")
    
    def _generate_projection_grid_search_resolution(self, grid_search_density_parameter):
        pass    # NOTE: Don't want this to throw an error because thetr is a default is built in to the projection method.


    # NOTE: Do I want a plot function on the space? I would also have to pass in the coefficients to plot the function. What's the point?
    # Additional NOTE: Type hinting leads to cyclic imports this way. I could just not type hint, but that's not ideal.
    # def plot(self, point_types:list=['evaluated_points'], plot_types:list=['surface'],
    #           opacity:float=1., color:Union[str,Function]='#00629B', surface_texture:str="", additional_plotting_elements:list=[], show:bool=True):
    #     '''
    #     Plots the B-spline Surface.

    #     Parameters
    #     -----------
    #     points_type : list
    #         The type of points to be plotted. {evaluated_points, coefficients}
    #     plot_types : list
    #         The type of plot {surface, wireframe, point_cloud}
    #     opactity : float
    #         The opacity of the plot. 0 is fully transparent and 1 is fully opaque.
    #     color : str
    #         The 6 digit color code to plot the B-spline as.
    #     surface_texture : str = "" {"metallic", "glossy", ...}, optional
    #         The surface texture to determine how light bounces off the surface.
    #         See https://github.com/marcomusy/vedo/blob/master/examples/basic/lightings.py for options.
    #     additional_plotting_elemets : list
    #         Vedo plotting elements that may have been returned from previous plotting functions that should be plotted with this plot.
    #     show : bool
    #         A boolean on whether to show the plot or not. If the plot is not shown, the Vedo plotting element is returned.
    #     '''
    #     if self.space.num_parametric_dimensions == 1:
    #         return self.plot_curve(point_types=point_types, plot_types=plot_types, opacity=opacity, color=color,
    #                                 additional_plotting_elements=additional_plotting_elements, show=show)
    #     elif self.space.num_parametric_dimensions == 2:
    #         return self.plot_surface(point_types=point_types, plot_types=plot_types, opacity=opacity, color=color, 
    #                                  surface_texture=surface_texture, additional_plotting_elements=additional_plotting_elements, show=show)
    #     elif self.space.num_parametric_dimensions == 3:
    #         return self.plot_volume(point_types=point_types, plot_types=plot_types, opacity=opacity, color=color,
    #                                 surface_texture=surface_texture, additional_plotting_elements=additional_plotting_elements, show=show)
        raise NotImplementedError("I still need to implement this :(")
        raise NotImplementedError(f"Plot method must be implemented in {type(self)} class?")

