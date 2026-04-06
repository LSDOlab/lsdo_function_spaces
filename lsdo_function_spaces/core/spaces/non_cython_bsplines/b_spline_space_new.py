import numpy as np
import lsdo_function_spaces as lfs
from lsdo_function_spaces.core.function_space import LinearFunctionSpace
import scipy.sparse as sps
from typing import Union, Tuple
import csdl_alpha as csdl
from lsdo_function_spaces.core.spaces.non_cython_bsplines.compute_basis_matrix_numpy import compute_basis_matrix_numpy
from lsdo_function_spaces.core.spaces.non_cython_bsplines.b_spline_csdl_custom_ops import BasisMatrixCustomOp, BSplineEvalCustomOp
from lsdo_function_spaces.core.spaces.non_cython_bsplines.b_spline_patch_projection import compute_point_to_bspline_projection
from scipy.spatial import cKDTree
import jax
import jax.numpy as jnp


class BSplineSpaceNew(LinearFunctionSpace):
    """
    A class representing a B-spline space with a new implementation (replacing Cython).
    Inherits from LinearFunctionSpace.

    Attributes
    ----------
    num_parametric_dimensions : int
        The number of parametric dimensions/variables of a function from this function space.
    
    degree : tuple
        The degree of the B-spline in each parametric dimension.
    
    coefficients_shape : tuple
        The shape/structure that the coefficients are arranged in. For a surface (num_parametric_dimensions=2), the shape would be:
          (nu,nv)
    
    knots : tuple(ndarrays) = None (NOTE: this a change from the original code)
        Tuple of knot vectors for each parametric dimension. 
        If None, the knots will be generated from the coefficients_shape and degree.
    
    knot_indices : list[np.ndarray] = None -- shape of list=(num_parametric_dimensions,), shape of inner np.ndarray=(num_knots_in_that_dimension,)
        The indices of the knots for each parametric dimension. If None, the indices will be generated from the knot vector.
        NOTE: The knot indices are not used in the new implementation, but they are kept for compatibility with the old implementation.
    """
    def __init__(
            self, 
            num_parametric_dimensions:int, 
            degree : tuple, 
            coefficients_shape : tuple, 
            knots: tuple=None, 
            knot_indices : list[np.ndarray]=None, 
        ):
        # TODO: replace num_parametric_dimensions with len(coefficients_shape)
        self.degree = degree
        self.knots = knots
        self.knot_indices = knot_indices
        super().__init__(num_parametric_dimensions, coefficients_shape)

        if isinstance(self.degree, int):
            self.degree = (self.degree,)*self.num_parametric_dimensions

        for i in range(self.num_parametric_dimensions):
            if self.degree[i] < 0:
                raise ValueError(f'Degree in axis {i} must be non-negative.')
            if self.degree[i] >= self.coefficients_shape[i]:
                raise ValueError(f'Degree in axis {i} must be less than the number of coefficients in each dimension.')

        if self.knots is None:
            # Create open uniform knot vectors in each parametric dimension
            self.knots = tuple(
                np.concatenate([
                    np.zeros(self.degree[i]),
                    np.linspace(0, 1, self.coefficients_shape[i] - self.degree[i] + 1),
                    np.ones(self.degree[i])
                ])
                for i in range(num_parametric_dimensions)
            )

        elif self.knot_indices is None:
            self.knot_indices = []
            knot_index = 0
            for i in range(self.num_parametric_dimensions):
                num_knots_i = self.coefficients_shape[i] + self.degree[i] + 1
                self.knot_indices.append(np.arange(knot_index, knot_index + num_knots_i))
                knot_index += num_knots_i

    def _evaluate(
            self, 
            coefficients: Union[np.ndarray, csdl.Variable],
            parametric_coordinates: Union[np.ndarray, csdl.Variable],
            parametric_derivative_orders: Tuple[int] = None,
        ) -> csdl.Variable:
        """
        Evaluate the B-spline basis functions at parameter u.
        NOTE: This method is overridden here to use the new implementation.
        """
        if not isinstance(coefficients, (np.ndarray, csdl.Variable)):
            raise TypeError(
                f"coefficients must be a numpy array or a CSDL variable, "
                f"but got type {type(coefficients)}."
            )
        
        if not isinstance(parametric_coordinates, (np.ndarray, csdl.Variable)):
            raise TypeError(
                f"parametric_coordinates must be a numpy array or a CSDL variable, "
                f"but got type {type(parametric_coordinates)}."
            )
        
        try:
            parametric_coordinates = parametric_coordinates.reshape(-1, self.num_parametric_dimensions)
        
        except ValueError:
            raise ValueError(
                f"parametric_coordinates must have shape (num_points, {self.num_parametric_dimensions}), "
                f"but got shape {parametric_coordinates.shape}."
            )

        if isinstance(coefficients, np.ndarray):
            coefficients = csdl.Variable(value=coefficients)

        if isinstance(parametric_coordinates, np.ndarray):
            basis_matrix = compute_basis_matrix_numpy(
                us=parametric_coordinates,
                degrees=self.degree,
                knot_vectors=self.knots,
                der_orders=parametric_derivative_orders,
            )
            if coefficients.shape != (basis_matrix.shape[1], coefficients.size//basis_matrix.shape[1]):
                coefficients = coefficients.reshape(
                    (basis_matrix.shape[1], coefficients.size//basis_matrix.shape[1])
                )
            values = csdl.Variable(value=np.zeros((basis_matrix.shape[0], coefficients.shape[1])))
            for i in csdl.frange(coefficients.shape[1]):
                coefficients_column = coefficients[:,i].reshape((coefficients.shape[0], 1))
                values = values.set(csdl.slice[:,i], csdl.sparse.matvec(basis_matrix, coefficients_column).reshape((basis_matrix.shape[0],)))
            
            values = values.reshape((parametric_coordinates.shape[0], coefficients.shape[-1]))

            return values

        else:
            b_spline_eval_op = BSplineEvalCustomOp(
                knots=self.knots,
                degree=self.degree,
                coefficients_shape=self.coefficients_shape,
                der_orders=parametric_derivative_orders,
            )

            values = b_spline_eval_op.evaluate(
                parametric_coordinates=parametric_coordinates,
                coefficients=coefficients,
            )

            return values

    def _generate_parametric_grid(self, knot_vectors, N):
        """
        Generate a tensor‐grid of parametric sample points.

        Parameters
        ----------
        knot_vectors : sequence of 1D arrays, length d
            Each array is the (non-decreasing) knot vector for one parametric dim.
        N : int
            Number of sample points per knot‐interval, including its left endpoint
            but *excluding* the right endpoint (we’ll add the very last knot at the end).

        Returns
        -------
        grid : ndarray, shape (M, d)
            The cartesian product of the 1D sample arrays in each dimension.
        """
        samples_1d = []
        for U in knot_vectors:
            # take only the unique knots in sorted order
            knots = np.unique(U)
            # for each interval [knots[j], knots[j+1]), sample N points
            pts = []
            for j in range(len(knots) - 1):
                a, b = knots[j], knots[j+1]
                pts.append(np.linspace(a, b, N, endpoint=False))
            # finally include the very last knot
            pts.append(np.array([knots[-1]]))
            samples_1d.append(np.concatenate(pts))

        # build the d‐dimensional tensor grid
        mesh = np.meshgrid(*samples_1d, indexing="ij")
        # flatten each coordinate array and stack into shape (M, d)
        coord_arrays = [m.flatten() for m in mesh]
        grid = np.stack(coord_arrays, axis=-1)
        return grid

    def _project(
        self,
        points_in_space: Union[np.ndarray, csdl.Variable],
        coefficients: Union[np.ndarray, csdl.Variable],
        plot=False,
        grid_search_density=100,
    ):
        """
        Project points in space onto the B-spline surface defined by the function.
        
        Parameters
        ----------
        points_in_space : Union[np.ndarray, csdl.Variable]
            Points in space to project onto the B-spline surface.
        coefficients : Union[np.ndarray, csdl.Variable]
            Coefficients of the B-spline basis functions.
        plot : bool, optional
            If True, plot the points in space and the projected points on the B-spline surface.
            Default is False.
        grid_search_density : int, optional
            Density of the grid used for the initial guess in the projection.
            This is the number of sample points per knot-interval.
            Default is 100.

        Returns
        -------
        Union[np.ndarray, csdl.Variable]
            The parametric coordinates of the projection of the points in space onto the B-spline surface.
        """
        if isinstance(coefficients, csdl.Variable):
            coefficients = coefficients.value

        if not isinstance(points_in_space, (np.ndarray, csdl.Variable)):
            raise TypeError(
                f"points_in_space must be a numpy array or a CSDL variable, "
                f"but got type {type(points_in_space)}."
            )
        if isinstance(points_in_space, csdl.Variable):
            raise NotImplementedError(
                "Projection of CSDL variables is not implemented yet. "
                "Please provide a numpy array of points in space."
            )
        fun = lfs.Function(
                space=self,
                coefficients=coefficients,
            )
        
        para_grid = self._generate_parametric_grid(
            knot_vectors=self.knots,
            N=grid_search_density,
        )
        print("Generating parametric grid with shape:", para_grid.shape)

        basis_mat = compute_basis_matrix_numpy(
            us=para_grid,
            degrees=self.degree,
            knot_vectors=self.knots,
        )

        surface_grid = basis_mat @ coefficients.reshape(-1, coefficients.shape[-1])

        kd_tree = cKDTree(surface_grid)
        nearest_index = kd_tree.query(points_in_space, k=1)[1]
        nearest_para_points = para_grid[nearest_index]

        batched_projection = jax.jit(jax.vmap(
            lambda pt, u0, cps: compute_point_to_bspline_projection(
                point=pt,
                degrees=self.degree,
                coefficients=cps,
                para_coords=u0,
                knots=tuple([jnp.array(knot_vectors_i) for knot_vectors_i in self.knots]),
            ), in_axes=(0, 0, None)
        ))

        para, res, converged, final_i, J, _, _ = batched_projection(
            points_in_space, 
            nearest_para_points,
            coefficients,
        )
        para = np.array(para).reshape(-1, self.num_parametric_dimensions)


        if not converged.all():
            print(f"Warning: {np.sum(~converged)} out of {len(converged)} projection points did not converge.")
            print("Initial guess for these points was:", nearest_para_points[~converged])
            print("Final parameter coordinates for these points were:", para[~converged])
            print("Final residuals for these points were:", res[~converged])
            print("Final iteration counts for these points were:", final_i[~converged])
            print("Jacobian for these points was:", J[~converged])


        if plot:
            point_cloud = lfs.plot_points(
                points=points_in_space,
                color="#00FF1A",
                opacity=0.5,
                size=8,
                show=False
            )

            projected_points = fun.evaluate(
                parametric_coordinates=para,
            ).value
            project_point_cloud = lfs.plot_points(
                points=projected_points,
                color="#FF0000",
                size=4,
                show=False
            )

            fun.plot(additional_plotting_elements=[point_cloud, project_point_cloud])


        return para

    def compute_basis_matrix(
            self, 
            parametric_coordinates: Union[np.ndarray, csdl.Variable], 
            parametric_derivative_orders: Tuple[int] = None,
        ) -> Union[sps.coo_matrix, csdl.Variable]:
        '''
        Evaluates the basis functions of the B-spline at the given parametric coordinates and assembles it into a sparse matrix.

        Parameters
        ----------
        parametric_coordinates : np.ndarray or csdl.Variable
            The parametric coordinates at which to evaluate the basis functions.
            shape=(num_points, num_parametric_dimensions)
        
        parametric_derivative_orders : tuple[int] = None
            The derivative orders for each parametric dimension.


        Returns
        -------
        Union[sps.coo_matrix, csdl.Variable]
            A sparse (sps.coo_matrix) basis matrix or a dense (csdl.Variable) matrix containing the evaluated basis functions.
            The shape of the matrix is (num_points, num_coefficients), where num_coefficients is the product of the coefficients in each parametric dimension.
            If parametric_derivative_orders is provided, the matrix will contain the derivatives of the basis functions.
        '''
        # Check if the input is a CSDL variable or a numpy array
        if isinstance(parametric_coordinates, csdl.Variable):
            basis_mat_custom_op = BasisMatrixCustomOp(
                knots=self.knots,
                degree=self.degree,
                coefficients_shape=self.coefficients_shape,
                der_orders=parametric_derivative_orders,
            )
            try:
                parametric_coordinates = parametric_coordinates.reshape(-1, self.num_parametric_dimensions)
            except ValueError:
                raise ValueError(
                    f"parametric_coordinates must have shape (num_points, {self.num_parametric_dimensions}), "
                    f"but got shape {parametric_coordinates.shape}."
                )
            basis_mat = basis_mat_custom_op.evaluate(parametric_coordinates)

            return basis_mat
        
        elif isinstance(parametric_coordinates, np.ndarray):
            try:
                parametric_coordinates = parametric_coordinates.reshape(-1, self.num_parametric_dimensions)
            except ValueError:
                raise ValueError(
                    f"parametric_coordinates must have shape (num_points, {self.num_parametric_dimensions}), "
                    f"but got shape {parametric_coordinates.shape}."
                )
            
            basis_mat = compute_basis_matrix_numpy(
                us=parametric_coordinates,
                degrees=self.degree,
                knot_vectors=self.knots,
                der_orders=parametric_derivative_orders,
            )

            return basis_mat
            
        else:
            raise TypeError(
                f"parametric_coordinates must be a numpy array or a CSDL variable, "
                f"but got type {type(parametric_coordinates)}."
            )
        
    
    def _compute_distance_bounds(self, point:np.ndarray, function:lfs.Function, direction=None) -> float:
        '''
        Computes the distance bounds for the given point.
        '''
        if not hasattr(function, 'bounding_box'):
            coefficients = function.coefficients.value.reshape((-1, function.num_physical_dimensions))
            function.bounding_box = np.zeros((2, coefficients.shape[-1]))
            if self.num_parametric_dimensions == 1:
                function.bounding_box[0, 0] = np.min(coefficients)
                function.bounding_box[1, 0] = np.max(coefficients)
            else:
                function.bounding_box[0, :] = np.min(coefficients, axis=0)
                function.bounding_box[1, :] = np.max(coefficients, axis=0)

        if direction is None:
            neg = function.bounding_box[0] - point
            pos = point - function.bounding_box[1]
            distance_vector = np.maximum(np.maximum(neg, pos), 0)
            return np.linalg.norm(distance_vector)
        else:
            closest_point = np.zeros((len(point),))
            for i in range(len(point)):
                if point[i] < function.bounding_box[0, i]:
                    closest_point[i] = function.bounding_box[0, i]
                elif point[i] > function.bounding_box[1, i]:
                    closest_point[i] = function.bounding_box[1, i]
                else:
                    closest_point[i] = point[i]
            t = np.dot(direction, (closest_point - point)) / np.dot(direction, direction)
            closest_point_on_line = point + t * direction
            return np.linalg.norm(closest_point_on_line - closest_point)
        

if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    import lsdo_function_spaces as lfs
    import time

    jax.config.update("jax_enable_x64", True)  # Use 64-bit precision for JAX

    # Define the B-spline space parameters
    num_cp_x = 10
    num_cp_y = 8
    nx = num_cp_x - 1  # Number of control points - 1
    ny = num_cp_y - 1  # Number of control points - 1
    px = 3  # Degree of the B-spline
    py = 2  # Degree of the B-spline
    p = (px, py)
    coefficients_shape = (num_cp_x, num_cp_y)
    derivative_orders = (0, 0)  # derivative orders for the evaluation

    # Define the B-spline spaces
    b_spline_space_old = lfs.BSplineSpace(
        num_parametric_dimensions=2,
        degree=p,
        coefficients_shape=coefficients_shape,
    )

    b_spline_space_new = BSplineSpaceNew(
        num_parametric_dimensions=2,
        degree=p,
        coefficients_shape=coefficients_shape,
    )

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    # Define the coefficients
    coeffs_x, coeffs_y = np.meshgrid(np.linspace(0, 5, num_cp_x), np.linspace(0, 2, num_cp_y), indexing='ij')
    coeffs = np.array(np.stack((coeffs_x, coeffs_y, 0.2 * np.random.rand(num_cp_x, num_cp_y)), axis=-1))
    coeffs_csdl = csdl.Variable(value=coeffs, name='coefficients')
    coeffs_csdl.set_as_design_variable()


    num_para_coords = 20 # NOTE: the actual number is squared
    epsilon = 1e-6
    u, v = np.meshgrid(
        np.linspace(epsilon, 1-epsilon, num_para_coords), 
        np.linspace(epsilon, 1-epsilon, num_para_coords), 
        indexing='ij'
    )
    us = np.array(np.stack((u.flatten(), v.flatten()), axis=-1))
    us_csdl = csdl.Variable(value=us, name='parametric_coordinates')
    us_csdl.set_as_design_variable()

    b_spline_fun_old = lfs.Function(
        space=b_spline_space_old,
        coefficients=coeffs_csdl,
    )

    print("\n")
    print("==========Timing and comparing old and new B-spline evaluations==========")
    t1 = time.time()
    b_spline_eval_old = b_spline_fun_old.evaluate(
        parametric_coordinates=us,
        parametric_derivative_orders=derivative_orders,
    ).value
    t2 = time.time()
    print("Old B-spline evaluation time:", t2 - t1)

    b_spline_fun_new = lfs.Function(
        space=b_spline_space_new,
        coefficients=coeffs_csdl,
    )
    t1 = time.time()
    b_spline_eval_new = b_spline_fun_new.evaluate(
        parametric_coordinates=us,
        parametric_derivative_orders=derivative_orders,
    ).value
    t2 = time.time()
    print("New B-spline evaluation time (numpy):", t2 - t1)
    print("Max discrepancy:", np.max(np.abs(b_spline_eval_old - b_spline_eval_new)))
    print("Mean discrepancy:", np.mean(np.abs(b_spline_eval_old - b_spline_eval_new)))


    print("\n")
    print("==========Testing Projections==========")
    random_points_in_space = np.random.rand(100000, 3)  # Random points in space
    para_coords = b_spline_space_new._project(
        points_in_space=random_points_in_space,
        coefficients=coeffs_csdl, 
        plot=True,
    )

    print("\n")
    print("==========Verifying derivatives==========")
    b_spline_eval_new = b_spline_fun_new.evaluate(
        parametric_coordinates=us_csdl,
        parametric_derivative_orders=derivative_orders,
    )
    objective = csdl.sum(b_spline_eval_new)
    objective.set_as_objective()

    jax_sim = csdl.experimental.JaxSimulator(
        recorder=recorder,
    )
    jax_sim.check_optimization_derivatives(step_size=epsilon)



