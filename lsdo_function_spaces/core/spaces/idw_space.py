import numpy as np
import scipy.sparse as sps
from lsdo_function_spaces import LinearFunctionSpace, Function
from scipy.spatial.distance import cdist
from dataclasses import dataclass
from typing import Union
import csdl_alpha as csdl

class IDWFunctionSpace(LinearFunctionSpace):
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

    def __init__(self, num_parametric_dimensions:int, order:float, points:np.ndarray=None, conserve:bool=True, grid_size:Union[int, tuple]=10, n_neighbors:int=None):
        """
        Initialize an IDW function space.

        Parameters
        ----------
        order : float
            The order of the inverse distance weighting function.
        conserve : bool, optional
            If True, the weights will be normalized to conserve the sum of the values. Default is True.

        """

        self.order = order
        self.conserve = conserve
        self.grid_size = grid_size
        self.points = points
        self.n_neighbors = n_neighbors

        if n_neighbors is not None and conserve:
            raise ValueError('IDWFunctionSpace does not support n_neighbors and conserve=True simultaneously')
        

        if self.points is None:
            if isinstance(self.grid_size, int):
                self.grid_size = (self.grid_size,)*num_parametric_dimensions
            linspaces = [np.linspace(0, 1, n) for n in self.grid_size]
            self.points = np.array(np.meshgrid(*linspaces)).T.reshape(-1, num_parametric_dimensions)

        if n_neighbors is not None:
            if n_neighbors > self.points.shape[0]:
                raise ValueError('n_neighbors cannot be greater than the number of points')
            if n_neighbors < 1:
                raise ValueError('n_neighbors must be greater than 0')
            
        super().__init__(num_parametric_dimensions, (self.points.shape[0], 1))
        
    def stitch(self, self_face, self_coeffs, other, other_face, other_coeffs):
        """
        Stitch two IDW function spaces together.

        Parameters
        ----------
        self_face : int
            The face of the current function space.
        other : IDWFunctionSpace
            The other function space to stitch.
        other_face : int
            The face of the other function space.

        Returns
        -------
        IDWFunctionSpace
            The stitched function space.

        """
        if self_face == 1:
            self_inds = np.where(self.points[:, 1] == 0)
            self_free_index = 0
        elif self_face == 2:
            self_inds = np.where(self.points[:, 0] == 1)
            self_free_index = 1
        elif self_face == 3:
            self_inds = np.where(self.points[:, 1] == 1)
            self_free_index = 0
        elif self_face == 4:
            self_inds = np.where(self.points[:, 0] == 0)
            self_free_index = 1
        self_inds = [int(ind) for ind in self_inds[0]]

        if other_face == 1:
            other_inds = np.where(other.points[:, 1] == 0)
            other_free_index = 0
        elif other_face == 2:
            other_inds = np.where(other.points[:, 0] == 1)
            other_free_index = 1
        elif other_face == 3:
            other_inds = np.where(other.points[:, 1] == 1)
            other_free_index = 0
        elif other_face == 4:
            other_inds = np.where(other.points[:, 0] == 0)
            other_free_index = 1
        other_inds = [int(ind) for ind in other_inds[0]]

        other_inds_sorted = []
        for i, point in enumerate(self.points[self_inds]):
            for j, other_point in enumerate(other.points[other_inds]):
                if np.allclose(point[self_free_index], other_point[other_free_index]):
                    other_inds_sorted.append(other_inds[j])
                    break

        if len(other_inds_sorted) != len(self_inds):
            raise ValueError('Could not find all corresponding points between the two faces')
        
        for i, j in csdl.frange(vals=(self_inds, other_inds_sorted)):
            self_face_coeffs = self_coeffs[i]
            other_face_coeffs = other_coeffs[j]
            average_coeffs = (self_face_coeffs + other_face_coeffs)/2
            self_coeffs = self_coeffs.set(csdl.slice[i], average_coeffs)
            other_coeffs = other_coeffs.set(csdl.slice[j], average_coeffs)
        
        return self_coeffs, other_coeffs

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

        if self.n_neighbors is None:
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
        else:
            # assemble a sparse matrix with the weights of the n_neighbors closest points
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='ball_tree').fit(self.points)
            distances, indices = nbrs.kneighbors(parametric_coordinates)
            with np.errstate(divide='ignore', invalid='ignore'):
                weights = 1.0/distances**self.order
                weights /= weights.sum(axis=1)[:, np.newaxis]
            np.nan_to_num(weights, copy=False, nan=1.)
            inv_indices = np.repeat(np.arange(indices.shape[0]), indices.shape[1])
            weights = sps.csr_matrix((weights.ravel(), (inv_indices, indices.ravel())), shape=(parametric_coordinates.shape[0], self.points.shape[0]))

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
        # if parametric_derivative_orders is not None:
        #     raise NotImplementedError('IDWFunctionSpace does not support derivatives')

        parametric_coordinates = parametric_coordinates.reshape(-1, self.num_parametric_dimensions)
        if self.n_neighbors is None:
            dist = cdist(parametric_coordinates, self.points)
            with np.errstate(divide='ignore', invalid='ignore'):
                weights = 1.0/dist**self.order
                if self.conserve:
                    weights = weights.T
                    weights /= weights.sum(axis=0)
                else:
                    weights /= weights.sum(axis=0)
                    weights = weights.T
            np.nan_to_num(weights, copy=False, nan=1.)
        else:
            if self.n_neighbors > parametric_coordinates.shape[0]:
                n_neighbors = parametric_coordinates.shape[0]
            else:
                n_neighbors = self.n_neighbors
            # assemble a sparse matrix with the weights of the n_neighbors closest points
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(parametric_coordinates)
            distances, indices = nbrs.kneighbors(self.points)
            with np.errstate(divide='ignore', invalid='ignore'):
                weights = 1.0/distances**self.order
                weights /= weights.sum(axis=1)[:,np.newaxis]
            np.nan_to_num(weights, copy=False, nan=1.)
            inv_indices = np.repeat(np.arange(indices.shape[0]), indices.shape[1])
            weights = sps.csr_matrix((weights.ravel(), (inv_indices, indices.ravel())), shape=(self.points.shape[0], parametric_coordinates.shape[0]))
        return weights
    

def test_idw_space():
    import numpy as np
    import csdl_alpha as csdl
    
    rec = csdl.Recorder(inline=True)
    rec.start()

    space = IDWFunctionSpace(2, 2, grid_size=4)
    parametric_coordinates = np.random.rand(100, 2)
    data = 10*np.random.rand(100, 1)
    function = space.fit_function(data, parametric_coordinates)
    eval_data = function.evaluate(parametric_coordinates)

    space = IDWFunctionSpace(2, 2, grid_size=4, conserve=False)
    sparse_space = IDWFunctionSpace(2, 2, grid_size=4, conserve=False, n_neighbors=3)
    parametric_coordinates = np.random.rand(100, 2)
    data = 10*np.random.rand(100, 1)
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
