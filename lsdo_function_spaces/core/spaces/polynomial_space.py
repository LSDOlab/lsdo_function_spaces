import numpy as np
import scipy.sparse as sps
from lsdo_function_spaces import FunctionSpace, Function
from scipy.spatial.distance import cdist
from dataclasses import dataclass
from typing import Union
from numpy.polynomial import polynomial as poly

class PolynomialSpace(FunctionSpace):
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

    def __init__(self, num_parametric_dimensions:int, order:Union[int, tuple[int]]):
        self.order = order


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
        if isinstance(self.order, int):
            self.order = (self.order,)*self.num_parametric_dimensions

        self.size = np.prod([order + 1 for order in self.order])
        super().__init__(num_parametric_dimensions, (self.size,))

    def compute_basis_matrix(self, parametric_coordinates:np.ndarray, parametric_derivative_orders:np.ndarray=None, expansion_factor:int=None) -> np.ndarray:
        """
        Compute the basis matrix for the given parametric coordinates.
        """
        if parametric_derivative_orders is not None:
            raise NotImplementedError('PolynomialSpace does not support derivatives')
        if self.num_parametric_dimensions == 1:
            weights = poly.polyvander(parametric_coordinates, self.order)
        elif self.num_parametric_dimensions == 2:
            if len(parametric_coordinates.shape) == 1:
                parametric_coordinates = parametric_coordinates.reshape(-1, 2)
            weights = poly.polyvander2d(parametric_coordinates[:, 0], parametric_coordinates[:, 1], self.order)
        elif self.num_parametric_dimensions == 3:
            weights = poly.polyvander3d(parametric_coordinates[:, 0], parametric_coordinates[:, 1], parametric_coordinates[:, 2], self.order)
        else:
            raise NotImplementedError('PolynomialSpace only supports up to 3 dimensions')

        return weights
    


# def test_polynomial_space():
#     num_parametric_dimensions = 2
#     order = 2
#     space = PolynomialSpace(num_parametric_dimensions=num_parametric_dimensions, order=order)
#     parametric_coordinates = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
#     basis_matrix = space.compute_basis_matrix(parametric_coordinates)
#     assert basis_matrix.shape == (3, 9)
#     assert np.allclose(basis_matrix, [[1., 0.1, 0.01, 0.2, 0.02, 0.01, 0.04, 0.008, 0.004],
#                                       [1., 0.3, 0.09, 0.4, 0.12, 0.04, 0.16, 0.048, 0.016],
#                                       [1., 0.5, 0.25, 0.6, 0.3, 0.15, 0.36, 0.18, 0.09]])

#     order = (2, 3)
#     space = PolynomialSpace(num_parametric_dimensions=num_parametric_dimensions, order=order)
#     parametric_coordinates = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
#     basis_matrix = space.compute_basis_matrix(parametric_coordinates)
#     assert basis_matrix.shape == (3, 18)
#     assert np.allclose(basis_matrix, [[1., 0.1, 0.01, 0.2, 0.02, 0.01, 0.04, 0.008, 0.004, 0.008, 0.0016, 0.0008, 0.016, 0.0032, 0.0016, 0.032, 0.0064, 0.0032],
#                                       [1., 0.3, 0.09, 0.4, 0.12, 0.04, 0.16, 0.048, 0.016, 0.024, 0.0072, 0.0024, 0.064, 0.0192, 0, 0.128, 0.0384, 0.0128],
#                                         [1., 0.5, 0.25, 0.6, 0.3, 0.15, 0.36, 0.18, 0.09, 0.04, 0.02, 0.01, 0.1, 0.05, 0.025, 0.2, 0.1, 0.05]])
    

# if __name__ == '__main__':
#     test_polynomial_space()
#     print('PolynomialSpace tests passed.')