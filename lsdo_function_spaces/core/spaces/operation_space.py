import numpy as np
import scipy.sparse as sps
from lsdo_function_spaces import FunctionSpace, Function
from scipy.spatial.distance import cdist
from dataclasses import dataclass
from typing import Union
import csdl_alpha as csdl

class OperationFunctionSpace(FunctionSpace):
    def __init__(self, inputs:list, operation:callable, num_parametric_dimensions:int):
        '''
        Function space that applies an operation to a list of inputs.

        Parameters
        ----------
        inputs : list[Function]
            List of functions/variables to apply the operation to.
        operation : callable
            Operation to apply to the functions - usually an internal csdl function, like csdl.add or csdl.cross.
        '''
        self.inputs = inputs
        self.operation = operation
        
        num_coefficients = 0
        for input in inputs:
            if isinstance(input, Function):
                num_coefficients += np.prod(input.space.coefficients_shape)

        super().__init__(num_parametric_dimensions, (0,))

    def _evaluate(self, coefficients, parametric_coordinates, parametric_derivative_orders):
        '''
        Evaluates the function.

        Parameters
        ----------
        parametric_coordinates : np.ndarray -- shape=(num_points, num_parametric_dimensions)
            The coordinates at which to evaluate the function.
        parametric_derivative_order : tuple = None -- shape=(num_points,num_parametric_dimensions)
            The order of the parametric derivatives to evaluate.
        coefficients : csdl.Variable = None -- shape=coefficients_shape
            The coefficients of the function.

        Returns
        -------
        function_values : csdl.Variable
            The function evaluated at the given coordinates.
        '''
        import lsdo_function_spaces as lfs

        if parametric_derivative_orders is not None:
            raise ValueError('Derivatives not supported in operation function spaces')
        # if coefficients is not None:
        #     raise ValueError('Coefficients not supported in operation function spaces')
        
        op_inputs = []
        for input in self.inputs:
            if isinstance(input, lfs.Function):
                op_inputs.append(input.evaluate(parametric_coordinates))
            else:
                op_inputs.append(input)
                
        return self.operation(*op_inputs)
        