import numpy as np
import scipy.sparse as sps
import lsdo_function_spaces as lfs
import csdl_alpha as csdl
from typing import Union

class ParametricMapping:
    def __init__(self, mappling_function:callable, indices:list):
        self.map = mappling_function
        self.indices = indices

# TODO: switch to this implementation
# class ParametricMapping:
#     def __init__(self, forward_map:callable, inverse_map:callable, indices_in:list, indices_out:list):
#         self.map = forward_map
#         self.inverse = inverse_map
#         self.indices_in = indices_in
#         self.indices_out = indices_out


class FunctionSetSpace:
    '''
    Class for representing the space of a particular set of functions.

    Attributes
    ----------
    num_parametric_dimensions : dict[int]
        The number of parametric dimensions/variables for each B-spline that is a part of this B-spline set.
    spaces : dict[lfs.FuncctionSpace] -- list length = number of functions in the set
        The function spaces that make up this FunctionSet.
    connections : dict[list[int]] = None
        The connections between the B-splines in the set. If None, the B-splines are assumed to be independent.

    Methods
    -------
    compute_basis_matrix(parametric_coordinates: np.ndarray, parametric_derivative_orders: np.ndarray = None) -> sps.csc_matrix:
        Computes the basis matrix for the given parametric coordinates and derivative orders.
    '''
    
    def __init__(self, num_parametric_dimensions:dict[int], spaces:dict[lfs.FunctionSpace], connections:dict[list[int]]=None):
        self.num_parametric_dimensions = num_parametric_dimensions
        self.spaces = spaces
        self.connections = connections
        self.parametric_maps = {}

        if isinstance(self.spaces, list):
            self.spaces = {i:space for i, space in enumerate(self.spaces)}

    def add_parametric_map(self, parametric_map:ParametricMapping, function_space:lfs.FunctionSpace):
        '''
        EXPERIMENTAL Applies a parametric map to the function space.

        Parameters
        ----------
        parametric_map : ParametricMapping
            The parametric map to apply.
        function_space : FunctionSpace
            The function space to apply the parametric map to.
        '''
        key = -len(self.parametric_maps)-1
        self.parametric_maps[key] = parametric_map
        inds = parametric_map.indices
        for ind in inds:
            if ind in self.spaces:
                del self.spaces[ind]

        self.spaces[key] = function_space

    def _apply_parametric_maps(self, parametric_coordinates:np.ndarray) -> np.ndarray:
        '''
        EXPERIMENTAL Applies the parametric map to the given parametric coordinates.

        Parameters
        ----------
        parametric_coordinates : np.ndarray
            The parametric coordinates to apply the map to.

        Returns
        -------
        mapped_parametric_coordinates : np.ndarray
            The mapped parametric coordinates.
        '''
        for i, parametric_coord in enumerate(parametric_coordinates):
            index_old, coord = parametric_coord
            for index, parametric_map in self.parametric_maps.items():
                if index_old in parametric_map.indices:
                    coords = parametric_map.map([(index_old, coord)])
                    parametric_coordinates[i] = (index, coords)
                    break

        return parametric_coordinates

    def initialize_function(self, num_physical_dimensions:int, value:float=0, implicit:bool=False) -> tuple[csdl.Variable, lfs.FunctionSet]:
        '''
        Generates a function set with the given number of physical dimensions and initializes the coefficients to the given value.

        Parameters
        ----------
        num_physical_dimensions : int
            The number of physical dimensions for the function set.
        value : float = 0
            The value to initialize the coefficients to.
        implicit : bool = False
            If True, the coefficients are stored as an ImplicitVariable.

        Returns
        -------
        coeff_var : csdl.Variable
            Variable consisting of the stacked coefficients of each function in the set.
            If implicit is True, the variable is an ImplicitVariable.
        function_set : lfs.FunctionSet
            The function set with the initialized coefficients.
        '''
        
        size = 0
        for space in self.spaces.values():
            size += np.prod(space.coefficients_shape)

        if implicit:
            coeff_var = csdl.ImplicitVariable((size, num_physical_dimensions), value=value)
        else:
            coeff_var = csdl.Variable((size, num_physical_dimensions), value=value)

        functions = {}
        start = 0
        for i, space in self.spaces.items():
            end = start + np.prod(space.coefficients_shape)
            function = lfs.Function(space=space, coefficients=coeff_var[start:end,:].reshape(space.coefficients_shape + (num_physical_dimensions,)))
            functions[i] = function
            start = end

        function_set = lfs.FunctionSet(functions=functions, space=self)

        return coeff_var, function_set

    def generate_parametric_grid(self, grid_resolution:tuple) -> list[tuple[int, np.ndarray]]:
        '''
        Generates a parametric grid for the function set space.

        Parameters
        ----------
        grid_resolution : tuple
            The resolution of the grid in each parametric dimension.

        Returns
        -------
        parametric_grid : list[tuple[int, np.ndarray]]
            The grid of parametric coordinates for the FunctionSet (makes a grid of the specified resolution over each function in the set).
        '''

        parametric_grid = []
        for i, space in self.spaces.items():
            space_parametric_grid = space.generate_parametric_grid(grid_resolution=grid_resolution)
            for j in range(space_parametric_grid.shape[0]):
                parametric_grid.append((i, space_parametric_grid[j,:]))

        return parametric_grid

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
            space = self.spaces[index]
            basis_matrix = space.compute_basis_matrix(parametric_coordinates=parametric_coordinate,
                                                      parametric_derivative_orders=parametric_derivative_orders[i],
                                                      expansion_factor=expansion_factor)
            basis_matrix_rows.append(basis_matrix)

        basis_matrix = sps.vstack(basis_matrix_rows, format='csc')
        return basis_matrix
    
    def fit(self, values:Union[csdl.Variable, np.ndarray], parametric_coordinates:list[tuple[int,np.ndarray]]=None,
            parametric_derivative_orders:list[tuple]=None, basis_matrix:Union[sps.csc_matrix, np.ndarray]=None,
            regularization_parameter:float=None) -> list[csdl.Variable]:
        '''
        Fits the function to the given data. Either parametric coordinates or an evaluation matrix must be provided. If derivatives are used, the
        parametric derivative orders must be provided. If both parametric coordinates and an evaluation matrix are provided, the evaluation matrix
        will be used.

        Parameters
        ----------
        values : csdl.Variable|np.ndarray -- shape=(num_points,num_physical_dimensions)
            The values of the data.
        parametric_coordinates : list[tuple[int,np.ndarray]] -- list of tuples of the form (index, parametric_coordinates) for each value
            The parametric coordinates of the data.
        parametric_derivative_orders : np.ndarray = None -- list of tuples of shape=(num_parametric_dimensions,) for each value
            The derivative orders to fit.
        basis_matrix : sps.csc_matrix|np.ndarray = None -- shape=(num_points, num_coefficients)
            The basis matrix to use for fitting.
        regularization_parameter : float = None
            The regularization parameter to use for fitting. If None, no regularization is used.

        Returns
        -------
        coefficients : list[csdl.Variable]
            The fitted coefficients for each function in the set.
        '''
        if parametric_coordinates is None and basis_matrix is None:
            raise ValueError("Either parametric coordinates or an basis matrix must be provided.")
        
        if parametric_coordinates is not None and basis_matrix is not None:
            print("Both parametric coordinates and an basis matrix were provided. The basis matrix will be used.")
            # raise Warning("Both parametric coordinates and an basis matrix were provided. The basis matrix will be used.")
        
        if basis_matrix is not None:
            # Just perform fitting using the basis matrix
            raise NotImplementedError("Fitting using a basis matrix is not yet implemented.")
        else:   # I only have this else statement here because pylance is graying out everything below if I don't have it
            # Perform fitting using the parametric coordinates
            pass

        if self.parametric_maps:
            parametric_coordinates = self._apply_parametric_maps(parametric_coordinates)

        # Current implementation: Perform fitting on each individual function in the set
        num_physical_dimensions = values.shape[-1]

        # Organize values into a list of values for each function in the set
        values_per_function = {}
        parametric_coordinates_per_function = {}
        parametric_derivative_orders_per_function = {}
        for i, space in self.spaces.items():
            values_per_function[i] = []
            parametric_coordinates_per_function[i] = []
            parametric_derivative_orders_per_function[i] = None

        for i, parametric_coordinate in enumerate(parametric_coordinates):
            index, parametric_coordinate = parametric_coordinate
            # values_per_function[index].append(values[i,:])
            values_per_function[index].append(i)
            parametric_coordinates_per_function[index].append(parametric_coordinate)
            if parametric_derivative_orders is not None:
                parametric_derivative_orders_per_function[index].append(parametric_derivative_orders[i])

        for i, space in self.spaces.items():
            if len(values_per_function[i]) > 0:
                parametric_coordinates_per_function[i] = np.vstack(parametric_coordinates_per_function[i])

        # Fit each function in the set
        coefficients = {}
        for i, space in self.spaces.items():
            if len(values_per_function[i]) > 0:
                # if isinstance(values, csdl.Variable):
                #     function_values = csdl.blockmat([[value.reshape((1, value.shape[0]))] for value in values_per_function[i]])
                function_values = values[values_per_function[i]]
                coefficients[i] = space.fit(values=function_values, parametric_coordinates=parametric_coordinates_per_function[i],
                                            parametric_derivative_orders=None, regularization_parameter=regularization_parameter)
            else:
                function_coefficients = csdl.Variable(value=np.zeros(space.coefficients_shape + (num_physical_dimensions, )))
                coefficients[i] = function_coefficients

        return coefficients
    
    def fit_monolithic(self, values:Union[csdl.Variable, np.ndarray], parametric_coordinates:list[tuple[int,np.ndarray]]=None,
        parametric_derivative_orders:list[tuple]=None, basis_matrix:Union[sps.csc_matrix, np.ndarray]=None,
        regularization_parameter:float=None) -> list[csdl.Variable]:
        '''
        Fits the function to the given data. Either parametric coordinates or an evaluation matrix must be provided. If derivatives are used, the
        parametric derivative orders must be provided. If both parametric coordinates and an evaluation matrix are provided, the evaluation matrix
        will be used.

        Parameters
        ----------
        values : csdl.Variable|np.ndarray -- shape=(num_points,num_physical_dimensions)
            The values of the data.
        parametric_coordinates : list[tuple[int,np.ndarray]] -- list of tuples of the form (index, parametric_coordinates) for each value
            The parametric coordinates of the data.
        parametric_derivative_orders : np.ndarray = None -- list of tuples of shape=(num_parametric_dimensions,) for each value
            The derivative orders to fit.
        basis_matrix : sps.csc_matrix|np.ndarray = None -- shape=(num_points, num_coefficients)
            The basis matrix to use for fitting.
        regularization_parameter : float = None
            The regularization parameter to use for fitting. If None, no regularization is used.

        Returns
        -------
        coefficients : list[csdl.Variable]
            The fitted coefficients for each function in the set.
        '''
        # Current implementation: Perform fitting on each individual function in the set
        num_physical_dimensions = values.shape[-1]

        if self.parametric_maps:
            parametric_coordinates = self._apply_parametric_maps(parametric_coordinates)

        # Organize values into a list of values for each function in the set
        values_per_function = {}
        parametric_coordinates_per_function = {}
        parametric_derivative_orders_per_function = {}
        for i, space in self.spaces.items():
            values_per_function[i] = []
            parametric_coordinates_per_function[i] = []
            parametric_derivative_orders_per_function[i] = None


        for i, parametric_coordinate in enumerate(parametric_coordinates):
            index, parametric_coordinate = parametric_coordinate
            # values_per_function[index].append(values[i,:])
            values_per_function[index].append(i)
            parametric_coordinates_per_function[index].append(parametric_coordinate)
            if parametric_derivative_orders is not None:
                parametric_derivative_orders_per_function[index].append(parametric_derivative_orders[i])

        for i, space in self.spaces.items():
            if len(values_per_function[i]) > 0:
                parametric_coordinates_per_function[i] = np.vstack(parametric_coordinates_per_function[i])

        # Fit each function in the set
        coefficients = {}
        for i, space in self.spaces.items():
            if len(values_per_function[i]) > 0:
                # if isinstance(values, csdl.Variable):
                #     function_values = csdl.blockmat([[value.reshape((1, value.shape[0]))] for value in values_per_function[i]])
                function_values = values[values_per_function[i]]
                coefficients[i] = space.fit(values=function_values, parametric_coordinates=parametric_coordinates_per_function[i],
                                            parametric_derivative_orders=None, regularization_parameter=regularization_parameter)
            else:
                # print(f"No data was provided for function {i}.")
                # Kind of hacky way to get size of coefficients
                parametric_coordinate = space.generate_parametric_grid(grid_resolution=(1,1))[0]
                basis_vector = space.compute_basis_matrix(parametric_coordinates=parametric_coordinate)
                num_coefficients = basis_vector.shape[1]
                function_coefficients = csdl.Variable(value=np.zeros((num_coefficients,num_physical_dimensions)))
                coefficients[i] = function_coefficients

        return coefficients

    def fit_function_set(self, values:Union[csdl.Variable,np.ndarray], parametric_coordinates:list[tuple[int,np.ndarray]]=None,
            parametric_derivative_orders:list[tuple]=None, basis_matrix:Union[sps.csc_matrix, np.ndarray]=None,
            regularization_parameter:float=None) -> lfs.FunctionSet:
        '''
        Fits the function to the given data. Either parametric coordinates or an evaluation matrix must be provided. If derivatives are used, the
        parametric derivative orders must be provided. If both parametric coordinates and an evaluation matrix are provided, the evaluation matrix
        will be used.

        Parameters
        ----------
        values : csdl.Variable|np.ndarray -- shape=(num_points,num_physical_dimensions)
            The values of the data.
        parametric_coordinates : list[tuple[int,np.ndarray]] -- list of tuples of the form (index, parametric_coordinates) for each value
            The parametric coordinates of the data.
        parametric_derivative_orders : np.ndarray = None -- list of tuples of shape=(num_parametric_dimensions,) for each value
            The derivative orders to fit.
        basis_matrix : sps.csc_matrix|np.ndarray = None -- shape=(num_points, num_coefficients)
            The basis matrix to use for fitting.
        regularization_parameter : float = None
            The regularization parameter to use for fitting. If None, no regularization is used.

            
        Returns
        -------
        function_set : lfs.FunctionSet
            The fitted function set.
        '''
        coefficients = self.fit(values=values, parametric_coordinates=parametric_coordinates,
                                parametric_derivative_orders=parametric_derivative_orders, basis_matrix=basis_matrix,
                                regularization_parameter=regularization_parameter)
        
        functions = {}
        for i, function_coefficients in coefficients.items():
            functions[i] = lfs.Function(space=self.spaces[i], coefficients=function_coefficients)

        function_set = lfs.FunctionSet(functions=functions, space=self)

        return function_set


# if __name__ == "__main__":
#     import csdl_alpha as csdl
#     recorder = csdl.Recorder(inline=True)
#     recorder.start()

#     num_coefficients1 = 10
#     num_coefficients2 = 5
#     degree1 = 4
#     degree2 = 3
    
#     space_of_cubic_b_spline_surfaces_with_10_cp = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(degree1,degree1),
#                                                               coefficients_shape=(num_coefficients1,num_coefficients1, 3))
#     space_of_quadratic_b_spline_surfaces_with_5_cp = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(degree2,degree2),
#                                                               coefficients_shape=(num_coefficients2,num_coefficients2, 3))
#     b_spline_spaces = [space_of_cubic_b_spline_surfaces_with_10_cp, space_of_quadratic_b_spline_surfaces_with_5_cp]
#     index_to_space = {0:0, 1:1}
#     name_to_index = {'space_of_cubic_b_spline_surfaces_with_10_cp':0, 'quadratic_b_spline_surfaces_5_cp':1}
#     num_parametric_dimensions = {0:2, 1:2}

#     b_spline_set_space = lfs.FunctionSetSpace(num_parametric_dimensions=num_parametric_dimensions, spaces=b_spline_spaces,
#                                             index_to_space=index_to_space, name_to_index=name_to_index)

#     coefficients_line = np.linspace(0., 1., num_coefficients1)
#     coefficients_y, coefficients_x = np.meshgrid(coefficients_line,coefficients_line)
#     coefficients1 = np.stack((coefficients_x, coefficients_y, 0.1*np.random.rand(num_coefficients1,num_coefficients1)), axis=-1)

#     coefficients_line = np.linspace(0., 1., num_coefficients2)
#     coefficients_y, coefficients_x = np.meshgrid(coefficients_line,coefficients_line)
#     coefficients_y += 1.5
#     coefficients2 = np.stack((coefficients_x, coefficients_y, 0.1*np.random.rand(num_coefficients2,num_coefficients2)), axis=-1)

#     coefficients = np.vstack((coefficients1.reshape((-1,3)), coefficients2.reshape((-1,3))))
#     coefficients = csdl.Variable(value=coefficients)

#     my_b_spline_surface_set = lfs.Function(space=b_spline_set_space, coefficients=coefficients)
#     my_b_spline_surface_set.plot()