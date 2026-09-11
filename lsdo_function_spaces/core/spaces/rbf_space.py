import numpy as np
import scipy.sparse as sps
from ..function_space import LinearFunctionSpace
from scipy.spatial.distance import cdist
from dataclasses import dataclass
from typing import Union
import csdl_alpha as csdl

class RBFFunctionSpace(LinearFunctionSpace):
    """
    Radial Basis Function (RBF) Function Space.

    This function space evaluates basis functions centered at support points
    using radial basis kernels such as Gaussian, Polyharmonic, and Multiquadrics.

    Parameters
    ----------
    num_parametric_dimensions : int
        The number of parametric dimensions.
    radial_function : str, optional
        The type of radial basis function kernel ('gaussian', 'polyharmonic_spline',
        'inverse_quadratic', 'inverse_multiquadric', 'bump'). Default is 'gaussian'.
    points : np.ndarray, optional
        Explicit support center points. If None, a uniform grid is generated based on grid_size.
    grid_size : Union[int, tuple], optional
        The size of the center point grid in each dimension. Default is 10.
    epsilon : float, optional
        Shape parameter for Gaussian, inverse quadratic, and multiquadric kernels. Default is 1.
    k : int, optional
        Power parameter for polyharmonic splines. Default is 2.
    """

    def __init__(self, num_parametric_dimensions:int, radial_function:str='gaussian', points:np.ndarray=None, grid_size:Union[int, tuple]=10, epsilon:float=1, k:int=2):
        """
        Initialize an RBF function space.
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

        super().__init__(num_parametric_dimensions, self.points.shape[0])

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

        dist = cdist(parametric_coordinates, self.points, 'euclidean')
        if not hasattr(self, f'_{self.radial_function}'):
            raise ValueError(f"Radial function '{self.radial_function}' is not supported.")
        phi = getattr(self, f'_{self.radial_function}')(dist)

        # sum the basis functions so the total influence per evaluation point is 1
        phi = phi / np.sum(phi, axis=1, keepdims=True)

        return phi

    def _gaussian(self, x):
        return np.exp(-(self.epsilon*x)**2)

    def _polyharmonic_spline(self, x):
        if self.k % 2 == 0:
            return x**(self.k-1) * np.log(x**x)
        else:
            return x**self.k

    def _inverse_quadratic(self, x):
        return 1/(1 + (self.epsilon*x)**2)

    def _inverse_multiquadric(self, x):
        return 1/np.sqrt(1 + (self.epsilon*x)**2)

    def _bump(self, x):
        return np.piecewise(x, [x < 1/self.epsilon], [lambda x: np.exp(1/((self.epsilon*x)**2-1)), 0])




def test_rbf_space():
    import numpy as np
    import csdl_alpha as csdl
    
    rec = csdl.Recorder(inline=True)
    rec.start()

    space = RBFFunctionSpace(num_parametric_dimensions=2, 
                             radial_function='bump',
                             grid_size=20)
    parametric_coordinates = np.random.rand(10, 2)
    data = 10*np.random.rand(10, 1)
    function = space.fit_function(data, parametric_coordinates)
    eval_data = function.evaluate(parametric_coordinates)

    print(eval_data.value - data)

    # print(function.coefficients.value)

if __name__ == '__main__':
    test_rbf_space()
