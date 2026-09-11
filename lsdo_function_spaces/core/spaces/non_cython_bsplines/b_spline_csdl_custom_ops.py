import csdl_alpha as csdl
import numpy as np
import jax
import jax.numpy as jnp
import warnings

from lsdo_function_spaces.core.spaces.non_cython_bsplines.compute_basis_matrix_jax import compute_basis_matrix_jax
try:
    _CustomExplicitOperation = csdl.experimental.CustomExplicitOperationBeta
except AttributeError:
    _CustomExplicitOperation = object


class BasisMatrixCustomOpVJP(_CustomExplicitOperation):
    def __init__(
            self,
            knots: tuple,
            degree: tuple,
            coefficients_shape: tuple,
            der_orders: tuple = None,
        ):
        super().__init__()
        self.knots = knots
        self.degree = degree
        self.coefficients_shape = coefficients_shape
        self.der_orders = der_orders

        def compute_basis_matrix(parametric_coordinates):
            return compute_basis_matrix_jax(
                us=parametric_coordinates,
                degrees=self.degree,
                knot_vectors=self.knots,
                der_orders=self.der_orders,
            ).todense()
        
        def compute_basis_matrix_vjp(parametric_coordinates):
            return jax.vjp(
                compute_basis_matrix,
                parametric_coordinates,
            )[1]

        self.vjp_fun = compute_basis_matrix_vjp

        self._cache = {}

    def evaluate(self, inputs, d_outputs):
        para_coords = inputs["parametric_coordinates"]
        d_basis_mat = d_outputs["basis_matrix"]

        self.declare_input("parametric_coordinates", para_coords)
        self.declare_input("d_basis_matrix", d_basis_mat)

        d_para_coords = self.create_output(
            "d_parametric_coordinates",
            shape=para_coords.shape,
        )
        
        d_inputs = {
            "parametric_coordinates": d_para_coords,
        }

        return d_inputs
    
    def compute(self, inputs, outputs):
        parametric_coordinates = inputs["parametric_coordinates"]
        d_basis_matrix = inputs["d_basis_matrix"]


        save_name = f"para_coords_shape_{parametric_coordinates.shape}"

        if save_name in self._cache:
            basis_matrix_fun = self._cache[save_name]
        else:
            basis_matrix_fun = jax.jit(
                self.vjp_fun,
            )
            self._cache[save_name] = basis_matrix_fun
        d_parametric_coordinates = basis_matrix_fun(jnp.array(parametric_coordinates))(jnp.array(d_basis_matrix))[0]
        outputs["d_parametric_coordinates"] = np.array(d_parametric_coordinates)

class BasisMatrixCustomOp(_CustomExplicitOperation):
    def __init__(
            self,
            knots: tuple,
            degree: tuple,
            coefficients_shape: tuple,
            der_orders: tuple = None,
        ):
        super().__init__()
        self.knots = knots
        self.degree = degree
        self.coefficients_shape = coefficients_shape
        self.der_orders = der_orders

        def compute_basis_matrix(parametric_coordinates):
            result = compute_basis_matrix_jax(
                us=parametric_coordinates,
                degrees=self.degree,
                knot_vectors=self.knots,
                der_orders=self.der_orders,
            ).todense()
            # Only squeeze the last axis if it's size 1 (for scalar case)
            # but preserve the first axis (number of points)
            if result.shape[-1] == 1:
                return result.squeeze(axis=-1)
            return result
        self.basis_matrix_computation_function = compute_basis_matrix

        self._cache = {}

    def evaluate(self, parametric_coordinates):
        """Compute the (DENSE!) basis matrix for the given parametric coordinates.
        CSDL does not support sparse matrices (as CSDL variables), so we return a dense matrix.

        Parameters
        ----------
        parametric_coordinates : csdl.Variable
            The parametric coordinates at which to evaluate the basis functions.

        Returns
        -------
        csdl.Variable
            A dense matrix containing the evaluated basis functions.
        """
        self.declare_input("parametric_coordinates", parametric_coordinates)
        
        basis_matrix = self.create_output(
            "basis_matrix",
            shape=(parametric_coordinates.shape[0], np.prod(self.coefficients_shape)),
        )
        self.basis_matrix_shape = basis_matrix.shape

        self.declare_vjp_function(
            BasisMatrixCustomOpVJP,
            knots=self.knots,
            degree=self.degree,
            coefficients_shape=self.coefficients_shape,
            der_orders=self.der_orders,
        )

        return basis_matrix
    
    def compute(self, inputs, outputs):
        parametric_coordinates = inputs["parametric_coordinates"]

        save_name = f"para_coords_shape_{parametric_coordinates.shape}"

        if save_name in self._cache:
            basis_matrix_fun = self._cache[save_name]
        else:
            basis_matrix_fun = jax.jit(
                self.basis_matrix_computation_function,
            )
            self._cache[save_name] = basis_matrix_fun

        basis_matrix = basis_matrix_fun(jnp.array(parametric_coordinates))
        outputs["basis_matrix"] = np.array(basis_matrix).reshape(self.basis_matrix_shape)

class BSplineEvalCustomOpVJP(_CustomExplicitOperation):
    def __init__(
            self, 
            knots, 
            degree, 
            coefficients_shape, 
            der_orders=None
        ):
        super().__init__()
        self.knots = knots
        self.degree = degree
        self.coefficients_shape = coefficients_shape
        self.der_orders = der_orders

        def evluate_b_spline(us, p, knots_jnp, coeffs, der_orders=None):
            ndim = len(p)
            if der_orders is None:
                der_orders = (0,) * ndim
            num_phys_dims = coeffs.shape[-1]
            basis_matrix = compute_basis_matrix_jax(us, p, knots_jnp, der_orders)
            return basis_matrix @ coeffs.reshape(-1, num_phys_dims)
        
        def evaluate_b_spline_jax_wrapped(us, coeffs):
            return evluate_b_spline(us, self.degree, self.knots, coeffs, self.der_orders)

        def compute_vjp(us, coeffs):
            return jax.vjp(
                evaluate_b_spline_jax_wrapped,
                us,
                coeffs,
            )[1]
        
        self.vjp_fun = compute_vjp

        self._cache = {}

    def evaluate(self, inputs, d_outputs):
        parametric_coordinates = inputs["parametric_coordinates"]
        coefficients = inputs["coefficients"]
        d_b_spline_values = d_outputs["b_spline_values"]

        self.declare_input("parametric_coordinates", parametric_coordinates)
        self.declare_input("coefficients", coefficients)
        self.declare_input("d_b_spline_values", d_b_spline_values)

        d_parametric_coordinates = self.create_output(
            "d_parametric_coordinates",
            shape=parametric_coordinates.shape,
        )
        
        d_coefficients = self.create_output(
            "d_coefficients",
            shape=coefficients.shape,
        )

        d_inputs = {
            "parametric_coordinates": d_parametric_coordinates,
            "coefficients": d_coefficients,
        }

        return d_inputs
    
    def compute(self, inputs, outputs):
        parametric_coordinates = inputs["parametric_coordinates"]
        coefficients = inputs["coefficients"]
        d_b_spline_values = inputs["d_b_spline_values"]

        save_name = f"para_coords_shape_{parametric_coordinates.shape}_coeffs_shape_{coefficients.shape}"

        if save_name in self._cache:
            evaluate_b_spline_jax = self._cache[save_name]
        else:
            evaluate_b_spline_jax = jax.jit(
                self.vjp_fun,
            )
            self._cache[save_name] = evaluate_b_spline_jax

        d_parametric_coordinates, d_coefficients = evaluate_b_spline_jax(
            jnp.array(parametric_coordinates),
            jnp.array(coefficients),
        )(jnp.array(d_b_spline_values))

        outputs["d_parametric_coordinates"] = np.array(d_parametric_coordinates)
        outputs["d_coefficients"] = np.array(d_coefficients)

class BSplineEvalCustomOp(_CustomExplicitOperation):
    def __init__(self, knots, degree, coefficients_shape, der_orders=None):
        super().__init__()
        self.knots = tuple(jnp.array(knots_i) for knots_i in knots)
        self.degree = degree
        self.coefficients_shape = coefficients_shape
        self.der_orders = der_orders

        
        def evluate_b_spline(us, p, knots_jnp, coeffs, der_orders=None):
            ndim = len(p)
            if der_orders is None:
                der_orders = (0,) * ndim
            num_phys_dims = coeffs.shape[-1]
            basis_matrix = compute_basis_matrix_jax(us, p, knots_jnp, der_orders)
            return basis_matrix @ coeffs.reshape(-1, num_phys_dims)
        
        def evaluate_b_spline_jax_wrapped(us, coeffs):
            return evluate_b_spline(us, self.degree, self.knots, coeffs, self.der_orders).squeeze()
        
        self.evaluate_b_spline_jax = evaluate_b_spline_jax_wrapped

        self._cache = {}


    def evaluate(self, parametric_coordinates, coefficients):
        """Evaluate the B-spline basis functions at the given parametric coordinates.

        Parameters
        ----------
        parametric_coordinates : csdl.Variable
            The parametric coordinates at which to evaluate the basis functions.
            shape=(num_points, num_parametric_dimensions)
        
        coefficients : csdl.Variable
            The coefficients of the B-spline basis functions.
            shape=(ncp_x, ncp_y, ..., num_physical_dimensions)

        Returns
        -------
        csdl.Variable
            The evaluated B-spline values at the given parametric coordinates.
            shape=(num_points, num_physical_dimensions)
        """
        self.declare_input("parametric_coordinates", parametric_coordinates)
        self.declare_input("coefficients", coefficients)

        num_eval_points = parametric_coordinates.shape[0]
        num_physical_dimensions = coefficients.shape[-1]
        output_shape = (num_eval_points, num_physical_dimensions)

        b_spline_values = self.create_output("b_spline_values", shape=output_shape)

        self.declare_vjp_function(
            BSplineEvalCustomOpVJP,
            knots=self.knots,
            degree=self.degree,
            coefficients_shape=self.coefficients_shape,
            der_orders=self.der_orders,
        )

        return b_spline_values
    
    def compute(self, inputs, outputs):
        parametric_coordinates = inputs["parametric_coordinates"]
        coefficients = inputs["coefficients"]

        save_name = f"para_coords_shape_{parametric_coordinates.shape}_coeffs_shape_{coefficients.shape}"

        if save_name in self._cache:
            evaluate_b_spline_jax = self._cache[save_name]
        else:
            evaluate_b_spline_jax = jax.jit(
                self.evaluate_b_spline_jax,
            )
            self._cache[save_name] = evaluate_b_spline_jax

        b_spline_values = evaluate_b_spline_jax(
            jnp.array(parametric_coordinates),
            jnp.array(coefficients),
        )
        outputs["b_spline_values"] = np.array(b_spline_values).reshape(
            (parametric_coordinates.shape[0], coefficients.shape[-1])
        )


if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    import lsdo_function_spaces as lfs
    import time

    # Define the B-spline space parameters
    num_cp_x = 10
    num_cp_y = 8
    nx = num_cp_x - 1  # Number of control points - 1
    ny = num_cp_y - 1  # Number of control points - 1
    px = 3  # Degree of the B-spline
    py = 2  # Degree of the B-spline
    p = (px, py)
    coefficients_shape = (num_cp_x, num_cp_y)
    derivative_orders = (1, 0)  # # derivatives for the evaluation

    knots_x = np.concatenate(
        [np.zeros(px), 
         np.linspace(0, 1, num_cp_x - px + 1), 
         np.ones(px)]
    )
    knots_y = np.concatenate(
        [np.zeros(py), 
         np.linspace(0, 1, num_cp_y - py + 1), 
         np.ones(py)]
    )

    knots = (knots_x, knots_y)  # Knot vectors for each dimension
    knots_jnp = (jnp.array(knots_x), jnp.array(knots_y))  # JAX-compatible knot vectors

    coeffs_x, coeffs_y = np.meshgrid(np.linspace(0, 5, num_cp_x), np.linspace(0, 2, num_cp_y), indexing='ij')
    coeffs = np.array(np.stack((coeffs_x, coeffs_y, 0.2 * np.random.rand(num_cp_x, num_cp_y)), axis=-1))

    rec = csdl.Recorder(inline=True)
    rec.start()

    coeffs_csdl = csdl.Variable(
        name="coefficients",
        value=coeffs,
    )
    coeffs_csdl.set_as_design_variable()

    num_para_coords = 20 # NOTE: the actual number is squared
    epsilon = 1e-8
    u, v = np.meshgrid(np.linspace(epsilon, 1-epsilon, num_para_coords), 
                       np.linspace(epsilon, 1-epsilon, num_para_coords), indexing='ij')
    us = np.array(np.stack((u.flatten(), v.flatten()), axis=-1))
    
    us_csdl = csdl.Variable(name="parametric_coordinates", value=us)
    us_csdl.set_as_design_variable()

    b_spline_eval_comp = BSplineEvalCustomOp(
        knots=knots_jnp,
        degree=p,
        coefficients_shape=coefficients_shape,
        der_orders=derivative_orders,
    )
    new_eval_points = b_spline_eval_comp.evaluate(us_csdl, coeffs_csdl)
    objective = csdl.sum((new_eval_points))
    objective.set_as_objective()

    jax_sim = csdl.experimental.JaxSimulator(
        rec,
    )

    jax_sim.check_optimization_derivatives(step_size=epsilon)

    b_spline_space_old = lfs.BSplineSpace(
        num_parametric_dimensions=2,
        degree=p,
        coefficients_shape=coefficients_shape,
    )
    old_b_spline_fun = lfs.Function(
        space=b_spline_space_old,
        coefficients=coeffs_csdl,
    )
    old_eval_points = old_b_spline_fun.evaluate(
        parametric_coordinates=us,
        parametric_derivative_orders=derivative_orders,
    )

    # compare the results
    print("Evaluated points (new):", new_eval_points.value)
    print("Evaluated points (old):", old_eval_points.value)
    # Check if the results are the same
    if np.allclose(new_eval_points.value, old_eval_points.value):
        print("The evaluated points are equal.")
    else:
        print("The evaluated points are NOT equal.")
        print("Max difference:", np.max(np.abs(new_eval_points.value - old_eval_points.value)))

