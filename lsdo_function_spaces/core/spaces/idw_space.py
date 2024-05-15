import numpy as np
import scipy.sparse as sps
from lsdo_function_spaces import FunctionSpace, Function
from scipy.spatial.distance import cdist
from dataclasses import dataclass

@dataclass
class IDWFunctionSpace(FunctionSpace):
    points : np.ndarray
    order : float
    conserve : bool = True


    def __post_init__(self):
        """
        Initialize an IDW function space.

        Parameters
        ----------
        points : np.ndarray
            An array of points representing the locations of the sample points.
        order : float
            The order of the inverse distance weighting function.
        conserve_values : bool
            If True, the weights will be normalized to conserve the sum of the values.
        """
        pass

    def compute_basis_matrix(self, parametric_coordinates:np.ndarray, parametric_derivative_orders: np.ndarray=None, expansion_factor:int=None) -> np.ndarray:
        if parametric_derivative_orders is not None:
            raise NotImplementedError('IDWFunctionSpace does not support derivatives')
        if expansion_factor is not None:
            raise NotImplementedError('IDWFunctionSpace does not support expansion factors')
        
        if len(parametric_coordinates.shape) == 1:
            parametric_coordinates = parametric_coordinates.reshape(1, -1)

        print(parametric_coordinates.shape)
        print(self.points.shape)

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
        if parametric_derivative_orders is not None:
            raise NotImplementedError('IDWFunctionSpace does not support derivatives')
        if self.conserve:
            print(parametric_coordinates)
            print(self.points)
            dist = cdist(parametric_coordinates, self.points)
            with np.errstate(divide='ignore', invalid='ignore'):
                weights = 1.0/dist**self.order
                weights = weights.T
                weights /= weights.sum(axis=0)
            np.nan_to_num(weights, copy=False, nan=1.)
        else:
            raise NotImplementedError()
        return weights