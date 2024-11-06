import csdl_alpha
from ..spaces.operation_space import OperationFunctionSpace
from ..function import Function
from ..function_set import FunctionSet
import functools
import numpy as np
from typing import Union

def decorate_csdl_op(op, set_kwargs={}) -> callable:
    def wrapper(*args, **kwargs) -> Union[Function, FunctionSet]:
        kwargs = {**set_kwargs, **kwargs}
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

# Basic operations
add = decorate_csdl_op(csdl_alpha.add)
sub = decorate_csdl_op(csdl_alpha.sub)
mult = decorate_csdl_op(csdl_alpha.mult)
div = decorate_csdl_op(csdl_alpha.div)
power = decorate_csdl_op(csdl_alpha.power)
negate = decorate_csdl_op(csdl_alpha.negate)
sqrt = decorate_csdl_op(csdl_alpha.sqrt)
exp = decorate_csdl_op(csdl_alpha.exp)
log = decorate_csdl_op(csdl_alpha.log)

# min/max
absolute = decorate_csdl_op(csdl_alpha.absolute)
maximum = decorate_csdl_op(csdl_alpha.maximum, set_kwargs={'axes': (1,)})
minimum = decorate_csdl_op(csdl_alpha.minimum, set_kwargs={'axes': (1,)})
average = decorate_csdl_op(csdl_alpha.average, set_kwargs={'axes': (1,)})
sum = decorate_csdl_op(csdl_alpha.sum, set_kwargs={'axes': (1,)})
argsum = decorate_csdl_op(csdl_alpha.sum)
product = decorate_csdl_op(csdl_alpha.product, set_kwargs={'axes': (1,)})

# Vector operations
tensordot = decorate_csdl_op(csdl_alpha.tensordot, set_kwargs={'axes': (1,)})
cross = decorate_csdl_op(csdl_alpha.cross, set_kwargs={'axis': 1})
norm = decorate_csdl_op(csdl_alpha.norm, set_kwargs={'axes': (1,)})

# Trigonometric functions
sin = decorate_csdl_op(csdl_alpha.sin)
cos = decorate_csdl_op(csdl_alpha.cos)
tan = decorate_csdl_op(csdl_alpha.tan)
arcsin = decorate_csdl_op(csdl_alpha.arcsin)
arccos = decorate_csdl_op(csdl_alpha.arccos)
arctan = decorate_csdl_op(csdl_alpha.arctan)
sinh = decorate_csdl_op(csdl_alpha.sinh)
cosh = decorate_csdl_op(csdl_alpha.cosh)
tanh = decorate_csdl_op(csdl_alpha.tanh)

# Other
bessel = decorate_csdl_op(csdl_alpha.bessel)
# concatenate = decorate_csdl_op(csdl_alpha.concatenate, set_kwargs={'axis': 1}) # the fact that the input is a list of functions is a problem