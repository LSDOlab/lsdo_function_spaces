import numpy as np
import scipy.sparse as sps
from lsdo_function_spaces import LinearFunctionSpace, Function
from scipy.spatial.distance import cdist
from dataclasses import dataclass
from typing import Union
import csdl_alpha as csdl

class RBFFunctionSpace(LinearFunctionSpace):
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

    def __init__(self, num_parametric_dimensions:int, radial_function:str='gaussian', points:np.ndarray=None, grid_size:Union[int, tuple]=10, epsilon:float=1, k:int=2):
        """
        Initialize an IDW function space.

        Parameters
        ----------
        order : float
            The order of the inverse distance weighting function.
        conserve : bool, optional
            If True, the weights will be normalized to conserve the sum of the values. Default is True.

        """

        self.grid_size = grid_size
        self.points = points
        self.radial_function = radial_function
        self.epsilon = epsilon
        self.k = k
        

        if self.points is None:
            if isinstance(self.grid_size, int):
                self.grid_size = (self.grid_size,)*num_parametric_dimensions
            linspaces = [np.linspace(0, 1, n) for n in self.grid_size]
            self.points = np.array(np.meshgrid(*linspaces)).T.reshape(-1, num_parametric_dimensions)

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
        # if parametric_derivative_orders is not None:
        #     raise NotImplementedError('IDWFunctionSpace does not support derivatives')
        # if expansion_factor is not None:
        #     raise NotImplementedError('IDWFunctionSpace does not support expansion factors')

        if len(parametric_coordinates.shape) == 1:
            parametric_coordinates = parametric_coordinates.reshape(1, -1)

        dist = cdist(parametric_coordinates, self.points)
        phi = getattr(self, f'_{self.radial_function}')(dist, self.epsilon)

        return phi

    def _gaussian(x, epsilon=1):
        return np.exp(-(epsilon*x)**2)

    def _polyharmonic_spline(x, k=2):
        if k % 2 == 0:
            return x**(k-1) * np.log(x**x)
        else:
            return x**k

    def _inverse_quadratic(x, epsilon=1):
        return 1/(1 + (epsilon*x)**2)

    def _inverse_multiquadric(x, epsilon=1):
        return 1/np.sqrt(1 + (epsilon*x)**2)

    def _bump(x, epsilon=1):
        if x < 1/epsilon:
            return np.exp(1/(1-(epsilon*x)**2))
        else:
            return 0



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
    sparse_space = IDWFunctionSpace(2, 2, grid_size=4, conserve=False, n_neighbors=3)
    parametric_coordinates = np.random.rand(10, 2)
    data = 10*np.random.rand(10, 1)
    function = space.fit_function(data, parametric_coordinates)
    eval_data = function.evaluate(parametric_coordinates)
    sparse_function = sparse_space.fit_function(data, parametric_coordinates)
    sparse_eval_data = sparse_function.evaluate(parametric_coordinates)
    print('eval_data:', eval_data.value)
    print('sparse_eval_data:', sparse_eval_data.value)
    # print(eval_data.value - data)

    # print(function.coefficients.value)

if __name__ == '__main__':
    test_idw_space()
