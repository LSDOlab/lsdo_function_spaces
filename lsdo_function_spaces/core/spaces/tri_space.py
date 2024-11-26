import numpy as np
import scipy.sparse as sps
from ..function_space import LinearFunctionSpace
from scipy.spatial.distance import cdist
from dataclasses import dataclass
from typing import Union
import csdl_alpha as csdl

class LinearTriangulationSpace(LinearFunctionSpace):
    def __init__(self, nodes=None, elements=None, grid_size=(10,10)):
        """
        Triangulation Function Space.

        This function space represents a mesh of 3-node triangles

        Parameters
        ----------
        nodes : np.ndarray
            The nodes of the mesh. (n_nodes, 2)
        elements : np.ndarray
            The elements of the mesh. (n_elements, 3)
        """
        if nodes is None:
            # Create a grid of nodes based on the grid size
            x = np.linspace(0, 1, grid_size[0])
            y = np.linspace(0, 1, grid_size[1])
            nodes = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)

            # create elements
            elements = []
            for i in range(grid_size[0]-1):
                for j in range(grid_size[1]-1):
                    elements.append([(i+1)*grid_size[0]+j+1, (i+1)*grid_size[0]+j, i*grid_size[0]+j+1])
                    elements.append([i*grid_size[0]+j, i*grid_size[0]+j+1, (i+1)*grid_size[0]+j])


            elements = np.array(elements)

        num_parametric_dimensions = 2

        self.nodes = nodes
        self.elements = elements
        

        super().__init__(num_parametric_dimensions, (self.nodes.shape[0],))

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
        if expansion_factor is not None:
            raise NotImplementedError

        # Compute what element each parametric coordinate is in and the local coordinates within that element
        elemental_indices, elemental_coordinates = self.compute_elemental_coordinates(parametric_coordinates)

        # Compute basis matrix
        basis_matrix = np.zeros((parametric_coordinates.shape[0], self.nodes.shape[0]))
        pdo = tuple(parametric_derivative_orders) if parametric_derivative_orders is not None else None
        
        # TODO: eliminate the loop - a bit confusing
        # TODO: also this is kinda hard-coded for linear elements, consider changing
        for i, element_index in enumerate(elemental_indices):
            if pdo is None or pdo == (0, 0):
                shape_functions = self.compute_shape_functions(elemental_coordinates[i].reshape(1, -1))
                element_coord_indices = self.elements[element_index]
                basis_matrix[i, element_coord_indices] = shape_functions
            elif pdo == (1, 0) or pdo == (0, 1):
                shape_function_jacobian = self.compute_shape_function_jacobian(elemental_indices[i].reshape(1, -1), elemental_coordinates[i].reshape(1, -1))
                element_coord_indices = self.elements[element_index]
                basis_matrix[i, element_coord_indices] = shape_function_jacobian[0, 0, :] if pdo == (1, 0) else shape_function_jacobian[0, 1, :]
            else:
                break
                

        return basis_matrix
    
    def compute_elemental_coordinates(self, parametric_coordinates):
        """
        Compute the element index and local coordinates within the element for the given parametric coordinates.

        Parameters
        ----------
        parametric_coordinates : np.ndarray
            The parametric coordinates for which to compute the element index and local coordinates.

        Returns
        -------
        np.ndarray
            The element index for each parametric coordinate.
        np.ndarray
            The local coordinates within the element for each parametric coordinate.

        """
        # Compute the element local coordinates
        local_coordinates = np.zeros((len(self.elements), parametric_coordinates.shape[0], 2))
        shape_functions = np.zeros((len(self.elements), parametric_coordinates.shape[0], 3))
        for i, element in enumerate(self.elements):
            local_coordinates[i,:,:] = self.compute_local_coordinates(parametric_coordinates, self.nodes[element])
            shape_functions[i,:,:] = self.compute_shape_functions(local_coordinates[i,:,:])

        # Find the element each parametric coordinate is in
        tol = 1e-10     # TODO: consider only applying the tol if no element is found
        elemental_indices = np.all((shape_functions <= 1+tol) & (shape_functions >= 0-tol), axis=2).argmax(axis=0)

        # Compute the local coordinates within the element
        elemental_coordinates = local_coordinates[elemental_indices, np.arange(parametric_coordinates.shape[0])]

        return elemental_indices, elemental_coordinates
    
    def compute_local_coordinates(self, parametric_coordinates, element_nodes):
        """
        Computes the local coordinates within the element for the given parametric coordinates.
        
        Parameters
        ----------
        parametric_coordinates : np.ndarray
            The parametric coordinates for which to compute the local coordinates.

        Returns
        -------
        np.ndarray
            The local coordinates within the element for each parametric coordinate. (n_parametric_coordinates, 3)
        """
        x1, x2, x3 = element_nodes[0,0], element_nodes[1,0], element_nodes[2,0]
        y1, y2, y3 = element_nodes[0,1], element_nodes[1,1], element_nodes[2,1]
        x, y = parametric_coordinates[:,0], parametric_coordinates[:,1]

        area2 = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)

        xi = ((x-x1)*(y3-y1)-(y-y1)*(x3-x1))/area2
        eta = ((x1-x1)*(y-y1)-(y2-y1)*(x-x1))/area2

        return np.vstack((xi, eta)).T
    
    def compute_shape_functions(self, local_coordinates):
        """
        Compute the shape functions for the given local coordinates.

        Parameters
        ----------
        local_coordinates : np.ndarray
            The local coordinates for which to compute the shape functions.

        Returns
        -------
        np.ndarray
            The computed shape functions.
        """

        N1 = 1-local_coordinates[:,0]-local_coordinates[:,1]
        N2 = local_coordinates[:,0]
        N3 = local_coordinates[:,1]

        return np.vstack((N1, N2, N3)).T

    def compute_shape_function_jacobian(self, elemental_indices, local_coordinates):
        """
        Compute the shape function gradients for the given local coordinates.

        Parameters
        ----------
        local_coordinates : np.ndarray
            The local coordinates for which to compute the shape function gradients.

        Returns
        -------
        np.ndarray
            The computed shape function gradients.
        """
        J = np.zeros((local_coordinates.shape[0], 2, 3))
        for i, element_index in enumerate(elemental_indices):
            # shortcut for linear triangles
            element_nodes = self.nodes[self.elements[element_index]]
            x1, x2, x3 = element_nodes[0,0,0], element_nodes[0,1,0], element_nodes[0,2,0]
            y1, y2, y3 = element_nodes[0,0,1], element_nodes[0,1,1], element_nodes[0,2,1]
            
            area2 = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)

            J[i,:,:] = np.array([[y2-y3, y3-y1, y1-y2], [x3-x2, x1-x3, x2-x1]])/area2

        return J


def test_tri():
    from scipy.stats.qmc import LatinHypercube
    import lsdo_function_spaces as lfs

    rec = csdl.Recorder(inline=True)
    rec.start()
    
    space = lfs.LinearTriangulationSpace(grid_size=(10,10))

    np.random.seed(0)
    num_points = 1000
    rand_parametric_coordinates = LatinHypercube(d=2, seed=7).random(num_points)

    parametric_coordinates = space.nodes
    height = (np.sin(2*np.pi*parametric_coordinates[:,0]) + .5*np.cos(2*np.pi*parametric_coordinates[:,1])).reshape(-1,1)
    data = np.hstack((parametric_coordinates*10, height))

    function = lfs.Function(space, data)
    function.evaluate(rand_parametric_coordinates, plot=True)

