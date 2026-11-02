from __future__ import annotations

from dataclasses import dataclass
from typing import Union, Optional, Sequence, List, Tuple, Set
import numpy as np
import numpy.typing as npt
import csdl_alpha as csdl

from lsdo_function_spaces.core.function import Function
from lsdo_function_spaces.core.function_set import FunctionSet
from lsdo_function_spaces.core.interface import InterfaceDomain


@dataclass(eq=False)
class Marker:
    """
    A semantic marker (datum point / evaluation site) in a Function or FunctionSet.

    Defined in the parametric domain of the function space, allowing values and
    derivatives to be evaluated as the function coefficients change.

    Parameters
    ----------
    entity : Union[Function, FunctionSet]
        The function or function set to which the marker is attached.
    parametric_coordinates : Optional[Union[npt.NDArray[np.float64], Tuple[int, npt.NDArray[np.float64]], List[Tuple[int, npt.NDArray[np.float64]]], csdl.Variable, Function, Tuple[int, Function]]]
        The parametric coordinate(s) on the function or function set.
    name : Optional[str]
        The name of the marker. If None, an auto-generated unique name is assigned.
    initial_point : Optional[npt.NDArray[np.float64]]
        The initial physical coordinates of the marker, if known.
    domain : Optional[InterfaceDomain]
        The intrinsic domain of the marker. Deduce as 0-D point domain if omitted.
    """
    entity: Union[Function, FunctionSet]
    parametric_coordinates: Optional[
        Union[
            npt.NDArray[np.float64],
            Tuple[int, npt.NDArray[np.float64]],
            List[Tuple[int, npt.NDArray[np.float64]]],
            csdl.Variable,
            Function,
            Tuple[int, Function],
        ]
    ] = None
    name: Optional[str] = None
    initial_point: Optional[npt.NDArray[np.float64]] = None
    domain: Optional[InterfaceDomain] = None

    _marker_counter: int = 0

    def __hash__(self):
        return id(self)

    def __post_init__(self):
        if self.name is None:
            Marker._marker_counter += 1
            self.name = f"marker_{Marker._marker_counter}"

        # Deduce domain if omitted
        if self.domain is None:
            from lsdo_function_spaces.core.function import Function
            coords = self.parametric_coordinates
            if isinstance(coords, Function) or (
                isinstance(coords, tuple)
                and len(coords) == 2
                and isinstance(coords[1], Function)
            ):
                fn = coords[1] if isinstance(coords, tuple) else coords
                bounds = getattr(fn, "parametric_bounds", None)
                if bounds is not None:
                    self.domain = InterfaceDomain(bounds=np.asarray(bounds, dtype=np.float64), num_copies=1)
                else:
                    m_dim = getattr(fn.space, "num_parametric_dimensions", 1)
                    self.domain = InterfaceDomain(bounds=np.array([[0.0, 1.0]] * m_dim, dtype=np.float64), num_copies=1)
            elif callable(coords) or (
                isinstance(coords, tuple)
                and len(coords) == 2
                and callable(coords[1])
            ):
                self.domain = InterfaceDomain.interval(0.0, 1.0)
            elif isinstance(coords, list):
                self.domain = InterfaceDomain.point(num_points=len(coords))
            elif isinstance(coords, np.ndarray) and coords.ndim == 2:
                self.domain = InterfaceDomain.point(num_points=coords.shape[0])
            else:
                self.domain = InterfaceDomain.point(num_points=1)

    def copy(
        self,
        entity: Optional[Union[Function, FunctionSet]] = None,
        name: Optional[str] = None,
    ) -> Marker:
        """
        Creates a copy of the marker, optionally rebinding to a new entity.

        Parameters
        ----------
        entity : Optional[Union[Function, FunctionSet]], optional
            New target entity to bind the copied marker to. Default is self.entity.
        name : Optional[str], optional
            New name for the copied marker. Default is self.name.

        Returns
        -------
        Marker
            The copied marker.
        """
        return self.__class__(
            entity=entity if entity is not None else self.entity,
            parametric_coordinates=self.parametric_coordinates,
            name=name if name is not None else self.name,
            initial_point=self.initial_point.copy() if self.initial_point is not None else None,
            domain=self.domain,
        )

    @classmethod
    def from_point(
        cls,
        entity: Union[Function, FunctionSet],
        point: npt.NDArray[np.float64],
        projection_tolerance: float = 1e-6,
        name: Optional[str] = None,
    ) -> Marker:
        """
        Creates a marker by projecting physical 3D Cartesian coordinates onto the entity.
        Automatically resolves patch indices for a FunctionSet.
        """
        pt = np.asarray(point, dtype=np.float64)
        if pt.ndim == 1:
            pt_proj = pt.reshape((1, -1))
        else:
            pt_proj = pt
        proj_coords = entity.project(pt_proj, projection_tolerance=projection_tolerance)
        if isinstance(proj_coords, list) and len(proj_coords) == 1:
            coords = proj_coords[0]
        else:
            coords = proj_coords
        return cls(
            entity=entity,
            parametric_coordinates=coords,
            domain=InterfaceDomain.point(num_points=1),
            initial_point=pt.flatten(),
            name=name,
        )

    @classmethod
    def from_points(
        cls,
        entity: Union[Function, FunctionSet],
        points: npt.NDArray[np.float64],
        projection_tolerance: float = 1e-6,
        name: Optional[str] = None,
    ) -> Marker:
        """
        Creates a multi-point marker (k >= 1) by projecting physical 3D points.
        """
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape((1, -1))
        proj_coords = entity.project(pts, projection_tolerance=projection_tolerance)
        return cls(
            entity=entity,
            parametric_coordinates=proj_coords,
            domain=InterfaceDomain.point(num_points=pts.shape[0]),
            initial_point=pts,
            name=name,
        )

    @classmethod
    def from_parametric(
        cls,
        entity: Union[Function, FunctionSet],
        parametric_coordinates: Union[
            npt.NDArray[np.float64],
            Tuple[int, npt.NDArray[np.float64]],
            List[Tuple[int, npt.NDArray[np.float64]]],
            csdl.Variable,
            Function,
            Tuple[int, Function],
        ],
        name: Optional[str] = None,
    ) -> Marker:
        """Creates a marker directly from parametric coordinates."""
        return cls(entity=entity, parametric_coordinates=parametric_coordinates, name=name)

    @classmethod
    def from_function(
        cls,
        entity: Union[Function, FunctionSet],
        function_map: Union[Function, Tuple[int, Function]],
        domain: Optional[InterfaceDomain] = None,
        name: Optional[str] = None,
    ) -> Marker:
        """
        Creates a continuous marker where alpha(gamma) is an lsdo_function_spaces.Function
        mapping gamma in Gamma to parametric coordinates in the entity's domain.
        Supports 1-D curves, 2-D surfaces, and general m-dimensional parametric maps.

        Parameters
        ----------
        entity : Union[Function, FunctionSet]
            Target function or function set.
        function_map : Union[Function, Tuple[int, Function]]
            Parametric map function (or (patch_idx, function)).
        domain : Optional[InterfaceDomain], optional
            Domain of the marker. If None, deduced from function_map.parametric_bounds.
        name : Optional[str], optional
            Descriptive name for the marker.

        Returns
        -------
        Marker
            The newly created continuous function marker.
        """
        if domain is None:
            fn = function_map[1] if isinstance(function_map, tuple) else function_map
            bounds = getattr(fn, "parametric_bounds", None)
            if bounds is not None:
                domain = InterfaceDomain(bounds=np.asarray(bounds, dtype=np.float64), num_copies=1)
            else:
                m_dim = getattr(fn.space, "num_parametric_dimensions", 1)
                domain = InterfaceDomain(bounds=np.array([[0.0, 1.0]] * m_dim, dtype=np.float64), num_copies=1)

        return cls(
            entity=entity,
            parametric_coordinates=function_map,
            domain=domain,
            name=name,
        )

    @classmethod
    def from_curve(
        cls,
        entity: Union[Function, FunctionSet],
        curve_map: Union[Function, Tuple[int, Function]],
        name: Optional[str] = None,
    ) -> Marker:
        """
        Creates a 1-D curve marker where alpha(s) is an lsdo_function_spaces.Function
        mapping s in [0, 1] to (u, v) in the entity's parametric space.
        """
        return cls.from_function(
            entity=entity,
            function_map=curve_map,
            domain=InterfaceDomain.interval(0.0, 1.0),
            name=name,
        )

    @property
    def function(self) -> Function:
        """Alias for entity when attached to a Function."""
        return self.entity  # type: ignore

    @property
    def function_set(self) -> FunctionSet:
        """Alias for entity when attached to a FunctionSet."""
        return self.entity  # type: ignore

    @property
    def is_constant(self) -> bool:
        """True if coordinates/curve map coefficients are numerical, False if CSDL variable."""
        from lsdo_function_spaces.core.function import Function
        if isinstance(self.parametric_coordinates, csdl.Variable):
            return False
        if isinstance(self.parametric_coordinates, Function):
            return getattr(self.parametric_coordinates, "is_constant", True)
        if isinstance(self.parametric_coordinates, (list, tuple)):
            for item in self.parametric_coordinates:
                if isinstance(item, csdl.Variable):
                    return False
                if isinstance(item, Function) and not getattr(item, "is_constant", True):
                    return False
                if isinstance(item, (list, tuple)):
                    for sub in item:
                        if isinstance(sub, csdl.Variable):
                            return False
                        if isinstance(sub, Function) and not getattr(sub, "is_constant", True):
                            return False
        return True

    @property
    def parametric_bounds(self) -> Union[Tuple[Tuple[float, float], ...], Dict[int, Tuple[Tuple[float, float], ...]]]:
        """Returns the valid parametric bounds for the entity (or patches) to which this marker is attached."""
        return self.entity.parametric_bounds

    def compute_boundary_violation(self) -> Optional[csdl.Variable]:
        """
        Computes the quadratic box boundary violation penalty for variable parametric coordinates:
            sum( relu(u_min - u)^2 + relu(u - u_max)^2 )
        Returns None if coordinates are numeric constant, or a scalar CSDL variable if variable.
        """
        if self.is_constant:
            return None

        from lsdo_function_spaces.core.function import Function
        if isinstance(self.entity, Function):
            bounds = self.entity.parametric_bounds
            u_var = self.parametric_coordinates
            if len(u_var.shape) == 1:
                u_var = u_var.reshape((1, -1))
            violations = []
            for d in range(len(bounds)):
                u_min, u_max = bounds[d]
                u_d = u_var[:, d]
                zeros_d = csdl.Variable(value=np.zeros(u_d.shape))
                v_low = csdl.maximum(zeros_d, u_min - u_d)
                v_up = csdl.maximum(zeros_d, u_d - u_max)
                violations.append(csdl.sum(v_low**2 + v_up**2))
            if len(violations) == 1:
                return violations[0]
            return csdl.sum(csdl.concatenate([v.reshape((1,)) for v in violations]))
        else:
            coords = self.parametric_coordinates
            if isinstance(coords, tuple):
                coords = [coords]
            violations = []
            for p_idx, u_var in coords:
                if isinstance(u_var, csdl.Variable):
                    if len(u_var.shape) == 1:
                        u_var = u_var.reshape((1, -1))
                    p_bounds = self.entity.parametric_bounds[p_idx]
                    for d in range(len(p_bounds)):
                        u_min, u_max = p_bounds[d]
                        u_d = u_var[:, d]
                        zeros_d = csdl.Variable(value=np.zeros(u_d.shape))
                        v_low = csdl.maximum(zeros_d, u_min - u_d)
                        v_up = csdl.maximum(zeros_d, u_d - u_max)
                        violations.append(csdl.sum(v_low**2 + v_up**2))
            if len(violations) == 0:
                return None
            if len(violations) == 1:
                return violations[0]
            return csdl.sum(csdl.concatenate([v.reshape((1,)) for v in violations]))

    @property
    def patches(self) -> Set[Tuple[Union[Function, FunctionSet], Optional[int]]]:
        """Returns the set of (entity, patch_id) nodes touched by this marker."""
        if isinstance(self.entity, FunctionSet):
            coords = self.parametric_coordinates
            if isinstance(coords, tuple) and len(coords) == 2 and isinstance(coords[0], (int, np.integer)):
                return {(self.entity, int(coords[0]))}
            elif isinstance(coords, list):
                return {(self.entity, int(p[0])) for p in coords if isinstance(p, tuple)}
            else:
                return {(self.entity, 0)}
        else:
            return {(self.entity, None)}

    def evaluate(
        self,
        entity: Optional[Union[Function, FunctionSet]] = None,
        plot: bool = False,
        non_csdl: bool = False,
        sample_parameters: Optional[npt.NDArray[np.float64]] = None,
    ) -> Union[csdl.Variable, npt.NDArray[np.float64]]:
        """
        Evaluates the function value at this marker's parametric coordinates.
        Accepts both NumPy arrays and CSDL Variables without destroying CSDL graphs.
        """
        target = entity if entity is not None else self.entity
        from lsdo_function_spaces.core.function import Function

        # Continuous curve or surface marker (dimension m >= 1)
        if self.domain is not None and self.domain.dimension >= 1:
            if sample_parameters is not None:
                s_pts = np.asarray(sample_parameters)
                if s_pts.ndim == 1 and self.domain.dimension == 1:
                    s_pts = s_pts.reshape(-1, 1)
            else:
                from lsdo_function_spaces.core.interface import Discretization
                s_pts, _ = Discretization(num_points=10).generate_samples(self.domain)
            if isinstance(self.parametric_coordinates, tuple):
                patch_idx, alpha_fn = self.parametric_coordinates
                if isinstance(alpha_fn, Function):
                    uv = alpha_fn.evaluate(s_pts, non_csdl=non_csdl)
                elif callable(alpha_fn):
                    uv = np.asarray([alpha_fn(s) for s in s_pts])
                else:
                    uv = np.asarray(alpha_fn)
                coords = [(patch_idx, uv[i, :]) for i in range(uv.shape[0])]
                val = target.evaluate(coords, plot=plot, non_csdl=non_csdl)
            else:
                alpha_fn = self.parametric_coordinates
                if isinstance(alpha_fn, Function):
                    uv = alpha_fn.evaluate(s_pts, non_csdl=non_csdl)
                elif callable(alpha_fn):
                    uv = np.asarray([alpha_fn(s) for s in s_pts])
                else:
                    uv = np.asarray(alpha_fn)
                val = target.evaluate(uv, plot=plot, non_csdl=non_csdl)
            return val

        # 0-D point marker(s)
        if isinstance(target, Function):
            coords = self.parametric_coordinates
            if isinstance(coords, np.ndarray):
                if coords.ndim == 1:
                    coords = coords.reshape((1, -1))
            val = target.evaluate(coords, plot=plot, non_csdl=non_csdl)
        else:
            val = target.evaluate(self.parametric_coordinates, plot=plot, non_csdl=non_csdl)

        # Preserve single-point 1D squeeze for backward compatibility
        if hasattr(val, "shape") and len(val.shape) == 2 and val.shape[0] == 1:
            val = val.reshape((val.shape[1],))
        return val

    def evaluate_derivatives(
        self,
        derivative_orders: Sequence[int],
        entity: Optional[Union[Function, FunctionSet]] = None,
        non_csdl: bool = False,
        sample_parameters: Optional[npt.NDArray[np.float64]] = None,
    ) -> Union[csdl.Variable, npt.NDArray[np.float64]]:
        """
        Evaluates parametric derivatives at this marker's parametric coordinates.
        """
        target = entity if entity is not None else self.entity
        from lsdo_function_spaces.core.function import Function

        # Continuous curve or surface marker (dimension m >= 1)
        if self.domain is not None and self.domain.dimension >= 1:
            if sample_parameters is not None:
                s_pts = np.asarray(sample_parameters)
                if s_pts.ndim == 1 and self.domain.dimension == 1:
                    s_pts = s_pts.reshape(-1, 1)
            else:
                from lsdo_function_spaces.core.interface import Discretization
                s_pts, _ = Discretization(num_points=10).generate_samples(self.domain)
            if isinstance(self.parametric_coordinates, tuple):
                patch_idx, alpha_fn = self.parametric_coordinates
                if isinstance(alpha_fn, Function):
                    uv = alpha_fn.evaluate(s_pts, non_csdl=non_csdl)
                elif callable(alpha_fn):
                    uv = np.asarray([alpha_fn(s) for s in s_pts])
                else:
                    uv = np.asarray(alpha_fn)
                coords = [(patch_idx, uv[i, :]) for i in range(uv.shape[0])]
                val = target.evaluate(coords, parametric_derivative_orders=derivative_orders, non_csdl=non_csdl)
            else:
                alpha_fn = self.parametric_coordinates
                if isinstance(alpha_fn, Function):
                    uv = alpha_fn.evaluate(s_pts, non_csdl=non_csdl)
                elif callable(alpha_fn):
                    uv = np.asarray([alpha_fn(s) for s in s_pts])
                else:
                    uv = np.asarray(alpha_fn)
                val = target.evaluate(uv, parametric_derivative_orders=derivative_orders, non_csdl=non_csdl)
            return val

        if isinstance(target, Function):
            coords = self.parametric_coordinates
            if isinstance(coords, np.ndarray) and coords.ndim == 1:
                coords = coords.reshape((1, -1))
            val = target.evaluate(coords, parametric_derivative_orders=derivative_orders, non_csdl=non_csdl)
        else:
            val = target.evaluate(self.parametric_coordinates, parametric_derivative_orders=derivative_orders, non_csdl=non_csdl)
        if hasattr(val, "shape") and len(val.shape) == 2 and val.shape[0] == 1:
            val = val.reshape((val.shape[1],))
        return val
