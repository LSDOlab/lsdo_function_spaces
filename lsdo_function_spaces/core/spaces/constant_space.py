import numpy as np
import scipy.sparse as sps
from lsdo_function_spaces import FunctionSpace, Function
from scipy.spatial.distance import cdist
from dataclasses import dataclass
from typing import Union

class ConstantSpace(FunctionSpace):
    """
    Constant Function Space.

    This function space represents a constant value in a parametric space.

    Parameters
    ----------
    num_parametric_dimensions : int
        The number of parametric dimensions.
    """

    def __init__(self, num_parametric_dimensions:int):
        super().__init__(num_parametric_dimensions, (1,))
        

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

        weights = np.ones((parametric_coordinates.shape[0], 1))
        return weights
