from dataclasses import dataclass
from typing import Union, Optional
import numpy as np
import numpy.typing as npt
import csdl_alpha as csdl


@dataclass
class Sinusoid:
    """
    Sinusoidal function of a single parametric variable (e.g. time t):
    f(t) = amplitude * (sin(2 * pi * frequency * t + phase) + offset)

    Parameters
    ----------
    amplitude : Union[float, csdl.Variable]
        Peak amplitude of the sinusoid.
    frequency : Union[float, csdl.Variable]
        Oscillation frequency in Hz. Default is 1.0.
    phase : Union[float, csdl.Variable]
        Phase shift in radians. Default is 0.0.
    offset : Union[float, csdl.Variable]
        DC amplitude offset. Default is 0.0.
    name : Optional[str]
        Optional identifier name.
    """
    amplitude: Union[float, csdl.Variable] = 1.0
    frequency: Union[float, csdl.Variable] = 1.0
    phase: Union[float, csdl.Variable] = 0.0
    offset: Union[float, csdl.Variable] = 0.0
    amplitude_offset: Optional[Union[float, csdl.Variable]] = None
    name: Optional[str] = None

    def __post_init__(self):
        if self.amplitude_offset is not None:
            self.offset = self.amplitude_offset

    def evaluate(self, parametric_coordinates: Union[csdl.Variable, npt.NDArray[np.float64], float],
                 parametric_derivative_orders: Optional[int] = None) -> Union[csdl.Variable, float, npt.NDArray[np.float64]]:
        """
        Evaluates the sinusoid at the given parametric coordinate(s) (e.g. time t).
        """
        t = parametric_coordinates
        is_csdl = isinstance(t, csdl.Variable) or isinstance(self.amplitude, csdl.Variable) or \
                  isinstance(self.frequency, csdl.Variable) or isinstance(self.phase, csdl.Variable) or \
                  isinstance(self.offset, csdl.Variable)
        if not is_csdl:
            try:
                rec = csdl.get_current_recorder()
                if rec is not None:
                    is_csdl = True
            except ValueError:
                is_csdl = False

        if is_csdl:
            omega = 2.0 * np.pi * self.frequency
            if parametric_derivative_orders is None or parametric_derivative_orders == 0:
                return self.amplitude * (csdl.sin(omega * t + self.phase) + self.offset)
            elif parametric_derivative_orders == 1:
                return self.amplitude * omega * csdl.cos(omega * t + self.phase)
            else:
                raise NotImplementedError("Higher-order derivatives not yet implemented.")
        else:
            omega = 2.0 * np.pi * float(self.frequency)
            if parametric_derivative_orders is None or parametric_derivative_orders == 0:
                return float(self.amplitude) * (np.sin(omega * t + float(self.phase)) + float(self.offset))
            elif parametric_derivative_orders == 1:
                return float(self.amplitude) * omega * np.cos(omega * t + float(self.phase))
            else:
                raise NotImplementedError("Higher-order derivatives not yet implemented.")

    def __call__(
        self, parametric_coordinates: Union[csdl.Variable, npt.NDArray, float]
    ) -> Union[csdl.Variable, float, np.ndarray]:
        return self.evaluate(parametric_coordinates)
