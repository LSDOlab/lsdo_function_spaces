import csdl_alpha
from ..spaces.operation_space import OperationFunctionSpace
from ..function import Function
from ..function_set import FunctionSet
import functools
import numpy as np
from typing import Union

def decorate_csdl_op(op):
    def wrapper(*args, **kwargs) -> Union[Function, FunctionSet]:
        is_set = False
        keys = []
        functions = []
        for arg in args:
            if isinstance(arg, Function):
                functions.append(arg)
            elif isinstance(arg, FunctionSet):
                functions.append(arg)
                is_set = True
                keys = arg.functions.keys()

        if len(functions) == 0:
            raise ValueError('No functions found in operation')

        num_parametric_dimensions = functions[0].space.num_parametric_dimensions
        for function in functions:
            if function.space.num_parametric_dimensions != num_parametric_dimensions:
                raise ValueError('All functions must have the same number of parametric dimensions')

            if is_set:
                if not isinstance(function, FunctionSet):
                    raise ValueError('Can\'t mix Function and FunctionSet in operation')
                if function.functions.keys() != keys:
                    raise ValueError('All functions must have the same keys')
        
        def operation(*args):
            return op(*args, **kwargs)
        
        if is_set:
            return FunctionSet({key: Function(space=OperationFunctionSpace([arg.functions[key] if arg in functions else arg for arg in args], operation, num_parametric_dimensions[key]), coefficients=np.zeros(3)) for key in keys})
        else:
            return OperationFunctionSpace(args, operation, num_parametric_dimensions)
    functools.update_wrapper(wrapper, op)
    return wrapper

add = decorate_csdl_op(csdl_alpha.add)
sub = decorate_csdl_op(csdl_alpha.sub)
mult = decorate_csdl_op(csdl_alpha.mult)
div = decorate_csdl_op(csdl_alpha.div)
power = decorate_csdl_op(csdl_alpha.power)
negate = decorate_csdl_op(csdl_alpha.negate)

