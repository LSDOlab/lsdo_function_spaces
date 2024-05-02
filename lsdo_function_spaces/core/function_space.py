from dataclasses import dataclass
import csdl_alpha as csdl
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl

'''
NOTE: To implement a function space (for instance, B-splines), all that must be implemented is the evaluation of the basis functions 
    and assembly into an evaluation matrix.
'''


@dataclass
class FunctionSpace:
    '''
    Base class for function spaces. This class is used to evaluate functions at given coordinates, refit functions, and project points onto functions.
    '''
    # num_physical_dimensions : int     # I might need this, but not using for now so don't have if not needed
    # coefficients_shape : tuple    # Seems overly restrictive making things like transition elements and other unstructured functions impossible
    #   -- It seems like we really only need the num_parametric_dimensions
    num_parametric_dimensions : int

    def __post_init__(self):
        '''
        Initializes the function space.
        '''
        pass

    def evaluate(self, coefficients:csdl.Variable, parametric_coordinates:np.ndarray, parametric_derivative_order:tuple=None) -> csdl.Variable:
        '''
        Picks a function from the function space with the given coefficients and evaluates or its derivative(s) it at the parametric coordinates.

        Parameters
        ----------
        coefficients : csdl.Variable -- shape=coefficients_shape or (num_coefficients,)
            The coefficients of the function.
        parametric_coordinates : np.ndarray -- shape=(num_points, num_parametric_dimensions)
            The coordinates at which to evaluate the function.
        parametric_derivative_order : tuple = None -- shape=(num_points, num_parametric_dimensions)
            The order of the parametric derivatives to evaluate.

        Returns
        -------
        csdl.Variable
            The function evaluated at the given coordinates.
        '''
        evaluation_matrix = self.compute_evaluation_matrix(parametric_coordinates, parametric_derivative_order)
        if isinstance(coefficients, csdl.Variable) and sps.issparse(evaluation_matrix):
            values = csdl.sparse.matvec(evaluation_matrix, coefficients)
        elif isinstance(coefficients, csdl.Variable):
            values = csdl.matvec(evaluation_matrix, coefficients)
        else:
            values = evaluation_matrix.dot(coefficients)

        return values
        # raise NotImplementedError(f"Evaluate method must be implemented in {type(self)} class.")
    

    def compute_evaluation_matrix(self, parametric_coordinates:np.ndarray, parametric_derivative_orders:np.ndarray=None) -> sps.csc_matrix:
        '''
        Evaluates the basis functions in parametric space and assembles the evaluation matrix (B(u) in P=B(u).dot(C)) where 
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
    

    def refit(self, coefficients:csdl.Variable, grid_resolution:tuple=None, parametric_coordinates:np.ndarray=None, 
              parametric_derivative_orders:np.ndarray=None, regularization_parameter:float=None) -> csdl.Variable:
        '''
        Picks a function from the function space with the given coefficients and evaluates or its derivative(s) it at the parametric coordinates.
        It then uses these values to refit the function. Either a grid resolution or parametric coordinates must be provided. 
        If both are provided, the parametric coordinates will be used. If derivatives are used, the parametric derivative orders must be provided.

        Parameters
        ----------
        coefficients : csdl.Variable -- shape=coefficients_shape or (num_coefficients,)
            The coefficients of the function to refit.
        grid_resolution : tuple = None -- shape=(num_parametric_dimensions,)
            The grid resolution to use for refitting.
        parametric_coordinates : np.ndarray = None -- shape=(num_points, num_parametric_dimensions)
            The parametric coordinates to use for refitting.
        parametric_derivative_orders : np.ndarray = None -- shape=(num_points, num_parametric_dimensions)
            The derivative orders to use for refitting.
        regularization_parameter : float = None
            The regularization parameter to use for refitting. If None, no regularization is used.

        Returns
        -------
        csdl.Variable
            The refitted coefficients.
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
            for dimension_index in range(self.num_parametric_dimensions):
                mesh_grid_input.append(np.linspace(0., 1., grid_resolution[dimension_index]))

            parametric_coordinates_tuple = np.meshgrid(*mesh_grid_input, indexing='ij')
            for dimensions_index in range(self.num_parametric_dimensions):
                parametric_coordinates_tuple[dimensions_index] = parametric_coordinates_tuple[dimensions_index].reshape((-1,1))

            parametric_coordinates = np.hstack(parametric_coordinates_tuple)

        evaluation_matrix = self.compute_evaluation_matrix(parametric_coordinates, parametric_derivative_orders)
        fitting_values = evaluation_matrix.dot(coefficients)
        
        coefficients = self.fit(values=fitting_values, evaluation_matrix=evaluation_matrix, regularization_parameter=regularization_parameter)
        
        return coefficients
            
        # raise NotImplementedError(f"Refit method must be implemented in {type(self)} class.")
        # NOTE: This doesn't just call fit so we don't need to construct the evaluation matrix multiplie times.
        #   - Maybe it would be easier to have the fit function optionally take in the evaluation matrix?


    
    def fit(self, values:np.ndarray, parametric_coordinates:np.ndarray=None, parametric_derivative_orders:np.ndarray=None,
            evaluation_matrix:sps.csc_matrix|np.ndarray=None, regularization_parameter:float=None) -> csdl.Variable:
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
        evaluation_matrix : sps.csc_matrix|np.ndarray = None -- shape=(num_points, num_coefficients)
            The evaluation matrix to use for fitting.
        regularization_parameter : float = None
            The regularization parameter to use for fitting. If None, no regularization is used.

        Returns
        -------
        csdl.Variable
            The coefficients of the fitted function.
        '''
        if parametric_coordinates is None and evaluation_matrix is None:
            raise ValueError("Either parametric coordinates or an evaluation matrix must be provided.")
        if parametric_coordinates is not None and evaluation_matrix is not None:
            print("Warning: Both parametric coordinates and an evaluation matrix were provided. Using the evaluation matrix.")
            # raise Warning("Both parametric coordinates and an evaluation matrix were provided. Using the evaluation matrix.")

        if parametric_coordinates is not None:
            evaluation_matrix = self.compute_evaluation_matrix(parametric_coordinates, parametric_derivative_orders)
        
        fitting_matrix = evaluation_matrix.T.dot(evaluation_matrix)
        if regularization_parameter is not None:
            if sps.issparse(fitting_matrix):
                fitting_matrix += regularization_parameter * sps.eye(fitting_matrix.shape[0]).tocsc()
            else:
                fitting_matrix += regularization_parameter * np.eye(fitting_matrix.shape[0])
        
        if isinstance(values, csdl.Variable) and sps.issparse(fitting_matrix):
            fitting_rhs = csdl.sparse.matvec(evaluation_matrix.T, values)
            coefficients = csdl.solve_linear(fitting_matrix, fitting_rhs)
        else:
            fitting_rhs = evaluation_matrix.T.dot(values)
            if sps.issparse(fitting_matrix):
                coefficients = np.zeros((fitting_matrix.shape[0], fitting_rhs.shape[1]))
                if len(fitting_rhs.shape) > 1:
                    for i in range(fitting_rhs.shape[1]):
                        coefficients[:,i] = spsl.spsolve(fitting_matrix, fitting_rhs[:,i])
                else:
                    coefficients = spsl.spsolve(fitting_matrix, fitting_rhs)
            else:
                coefficients = np.linalg.solve(fitting_matrix, fitting_rhs)

        return coefficients
        # raise NotImplementedError(f"Fit method must be implemented in {type(self)} class.")
    
    # hey github co-pilot. I don't want to make a project method in the function space class. I want it in the function class.

    # # NOTE: Do I want this?    
    # def create

