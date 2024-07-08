import numpy as np
import scipy.sparse as sps
from lsdo_function_spaces import FunctionSpace, Function
from scipy.spatial.distance import cdist
from dataclasses import dataclass
from typing import Union

class IDWFunctionSpace(FunctionSpace):
    """
    Inverse Distance Weighting (IDW) Function Space.

    This function space represents a grid of points in a parametric space using the Inverse Distance Weighting method.
    It provides methods to compute the basis matrix and the fitting map.

    Parameters
    ----------
    num_parametric_dimensions : int
        The number of parametric dimensions.
    order : float
        The order of the inverse distance weighting function.
    conserve : bool, optional
        If True, the weights will be normalized to conserve the sum of the values. Default is True.
    grid_size : tuple, optional
        The size of the grid in each parametric dimension. Default is (10,).
    """

    def __init__(self, num_parametric_dimensions:int, order:float, points:np.ndarray=None, conserve:bool=True, grid_size:Union[int, tuple]=10):
        # TODO: replace num_parametric_dimensions with points.shape[1]

        self.order = order
        self.conserve = conserve
        self.grid_size = grid_size
        self.points = points


    # def __post_init__(self):
        """
        Initialize an IDW function space.

        Parameters
        ----------
        order : float
            The order of the inverse distance weighting function.
        conserve : bool, optional
            If True, the weights will be normalized to conserve the sum of the values. Default is True.

        """
        if self.points is None:
            if isinstance(self.grid_size, int):
                self.grid_size = (self.grid_size,)*num_parametric_dimensions
            linspaces = [np.linspace(0, 1, n) for n in self.grid_size]
            self.points = np.array(np.meshgrid(*linspaces)).T.reshape(-1, num_parametric_dimensions)

        super().__init__(num_parametric_dimensions, (self.points.shape[0],))
        

    def compute_basis_matrix(self, parametric_coordinates:np.ndarray, parametric_derivative_orders: np.ndarray=None, expansion_factor:int=None) -> np.ndarray:
        """
        Compute the basis matrix for the given parametric coordinates.

        Parameters
        ----------
        parametric_coordinates : np.ndarray
            The parametric coordinates for which to compute the basis matrix.
        parametric_derivative_orders : np.ndarray, optional
            The derivative orders of the parametric coordinates. Default is None.
        expansion_factor : int, optional
            The expansion factor. Default is None.

        Returns
        -------
        np.ndarray
            The computed basis matrix.

        Raises
        ------
        NotImplementedError
            If parametric_derivative_orders or expansion_factor is not None.

        """
        if parametric_derivative_orders is not None:
            raise NotImplementedError('IDWFunctionSpace does not support derivatives')
        if expansion_factor is not None:
            raise NotImplementedError('IDWFunctionSpace does not support expansion factors')

        if len(parametric_coordinates.shape) == 1:
            parametric_coordinates = parametric_coordinates.reshape(1, -1)

        dist = cdist(self.points, parametric_coordinates)
        with np.errstate(divide='ignore', invalid='ignore'):
            weights = 1.0/dist**self.order
            if self.conserve:
                weights = weights.T
                weights /= weights.sum(axis=0)
            else:
                weights /= weights.sum(axis=0)
                weights = weights.T
        np.nan_to_num(weights, copy=False, nan=1.)
        return weights

    def compute_fitting_map(self, parametric_coordinates:np.ndarray, parametric_derivative_orders: np.ndarray=None) -> np.ndarray:
        """
        Compute the fitting map for the given parametric coordinates.

        Parameters
        ----------
        parametric_coordinates : np.ndarray
            The parametric coordinates for which to compute the fitting map.
        parametric_derivative_orders : np.ndarray, optional
            The derivative orders of the parametric coordinates. Default is None.

        Returns
        -------
        np.ndarray
            The computed fitting map.

        Raises
        ------
        NotImplementedError
            If parametric_derivative_orders is not None.

        """
        if parametric_derivative_orders is not None:
            raise NotImplementedError('IDWFunctionSpace does not support derivatives')
        parametric_coordinates = parametric_coordinates.reshape(-1, self.num_parametric_dimensions)
        if self.conserve:
            dist = cdist(parametric_coordinates, self.points)
            with np.errstate(divide='ignore', invalid='ignore'):
                weights = 1.0/dist**self.order
                weights = weights.T
                weights /= weights.sum(axis=0)
            np.nan_to_num(weights, copy=False, nan=1.)
        else:
            dist = cdist(parametric_coordinates, self.points)
            with np.errstate(divide='ignore', invalid='ignore'):
                weights = 1.0/dist**self.order
                weights /= weights.sum(axis=0)
                weights = weights.T
            np.nan_to_num(weights, copy=False, nan=1.)
            # raise NotImplementedError()
        return weights
    

def test_idw_space():
    import numpy as np
    import csdl_alpha as csdl
    
    rec = csdl.Recorder(inline=True)
    rec.start()

    space = IDWFunctionSpace(2, 2, grid_size=4)
    parametric_coordinates = np.random.rand(10, 2)
    data = 10*np.random.rand(10, 1)
    function = space.fit_function(data, parametric_coordinates)
    eval_data = function.evaluate(parametric_coordinates)

    space = IDWFunctionSpace(2, 2, grid_size=4, conserve=False)
    parametric_coordinates = np.random.rand(10, 2)
    data = 10*np.random.rand(10, 1)
    function = space.fit_function(data, parametric_coordinates)
    eval_data = function.evaluate(parametric_coordinates)
    # print(eval_data.value - data)

    # print(function.coefficients.value)

if __name__ == '__main__':
    test_idw_space()
