from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union, Optional, Sequence, List, Tuple, Dict, Set, Literal, TYPE_CHECKING
import numpy as np
import numpy.typing as npt
import scipy.sparse as sps
import csdl_alpha as csdl

from lsdo_function_spaces.core.function import Function
from lsdo_function_spaces.core.function_set import FunctionSet

if TYPE_CHECKING:
    from lsdo_function_spaces.core.marker import Marker


@dataclass
class InterfaceDomain:
    """
    Parametric box domain $\\Gamma_e = \\coprod_{c=1}^k \\prod_{j=1}^m [a_j, b_j]$.

    A unified domain class representing 0-D point sets (single or multi-point)
    and continuous domains (intervals, surfaces).

    Parameters
    ----------
    bounds : npt.NDArray[np.float64]
        Array of shape (m, 2) defining the box intervals $[a_j, b_j]$ for each of the $m$
        parametric dimensions. For a finite 0-D domain, $m = 0$ (shape (0, 2)).
    num_copies : int, optional
        Number of disjoint copies $k \\ge 1$ of the box domain (e.g., $k$ discrete points
        when $m = 0$). Default is 1.

    Attributes
    ----------
    dimension : int
        Intrinsic dimension $m$ of the domain box.
    is_finite : bool
        True if and only if the domain is discrete/finite ($m = 0$).
    """
    bounds: npt.NDArray[np.float64]  # Shape (m, 2)
    num_copies: int = 1              # k >= 1 copies

    def __post_init__(self):
        if not isinstance(self.bounds, np.ndarray):
            self.bounds = np.asarray(self.bounds, dtype=np.float64)
        if self.bounds.ndim == 1 and self.bounds.size == 0:
            self.bounds = self.bounds.reshape((0, 2))
        elif self.bounds.ndim == 1:
            self.bounds = self.bounds.reshape((1, 2))

    @property
    def dimension(self) -> int:
        """Intrinsic dimension m of the domain box."""
        return int(self.bounds.shape[0])

    @property
    def is_finite(self) -> bool:
        """True iff the domain consists of discrete points (m == 0)."""
        return self.dimension == 0

    @classmethod
    def point(cls, num_points: int = 1) -> InterfaceDomain:
        """
        Constructs a 0-D point domain with k copies.

        Parameters
        ----------
        num_points : int, optional
            Number of discrete points $k \\ge 1$. Default is 1.

        Returns
        -------
        InterfaceDomain
            The 0-D point domain.
        """
        return cls(bounds=np.zeros((0, 2), dtype=np.float64), num_copies=num_points)

    @classmethod
    def interval(cls, a: float = 0.0, b: float = 1.0) -> InterfaceDomain:
        """
        Constructs a 1-D continuous interval domain [a, b].

        Parameters
        ----------
        a : float, optional
            Lower interval bound. Default is 0.0.
        b : float, optional
            Upper interval bound. Default is 1.0.

        Returns
        -------
        InterfaceDomain
            The 1-D continuous interval domain.
        """
        return cls(bounds=np.array([[a, b]], dtype=np.float64), num_copies=1)

    @classmethod
    def surface(
        cls,
        u_bounds: Tuple[float, float] = (0.0, 1.0),
        v_bounds: Tuple[float, float] = (0.0, 1.0),
    ) -> InterfaceDomain:
        """
        Constructs a 2-D continuous surface domain [u_a, u_b] x [v_a, v_b].

        Parameters
        ----------
        u_bounds : Tuple[float, float], optional
            Bounds along the first parametric dimension. Default is (0.0, 1.0).
        v_bounds : Tuple[float, float], optional
            Bounds along the second parametric dimension. Default is (0.0, 1.0).

        Returns
        -------
        InterfaceDomain
            The 2-D continuous surface domain.
        """
        return cls(bounds=np.array([u_bounds, v_bounds], dtype=np.float64), num_copies=1)

    @classmethod
    def product(cls, *domains: InterfaceDomain) -> InterfaceDomain:
        """
        Constructs a product domain $\\Gamma = \\prod_i \\Gamma_i$ with concatenated box bounds.

        Parameters
        ----------
        *domains : InterfaceDomain
            One or more InterfaceDomain instances to take the Cartesian product of.

        Returns
        -------
        InterfaceDomain
            Product domain with concatenated bounds of shape (m_total, 2).
        """
        all_bounds = []
        for d in domains:
            if d.bounds.size > 0:
                all_bounds.append(d.bounds)
        if len(all_bounds) == 0:
            return cls(bounds=np.zeros((0, 2), dtype=np.float64), num_copies=1)
        concatenated = np.vstack(all_bounds)
        return cls(bounds=concatenated, num_copies=1)


def _gauss_lobatto_1d(N: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes 1D Gauss-Legendre-Lobatto (LGL) nodes and weights on [-1, 1].
    Nodes include endpoints -1 and +1. Interior nodes are roots of P'_{N-1}(x).
    """
    if N < 2:
        raise ValueError("Gauss-Lobatto quadrature requires at least 2 points.")
    if N == 2:
        return np.array([-1.0, 1.0]), np.array([1.0, 1.0])

    leg_poly = np.polynomial.legendre.Legendre.basis(N - 1)
    d_leg = leg_poly.deriv()
    interior_nodes = np.sort(d_leg.roots())
    nodes = np.concatenate([[-1.0], interior_nodes, [1.0]])

    p_vals = leg_poly(nodes)
    weights = 2.0 / (N * (N - 1) * (p_vals ** 2))
    return nodes, weights


@dataclass
class Discretization:
    """
    Quadrature or collocation discretization for continuous InterfaceDomain (m >= 1).

    Parameters
    ----------
    num_points : Union[int, Sequence[int]]
        Number of quadrature/collocation points per dimension. If a single int is given,
        it is applied to all m dimensions. Default is 10.
    scheme : Literal["gauss_legendre", "gauss_lobatto", "chebyshev", "uniform"]
        Sampling scheme. "gauss_legendre" computes Legendre-Gauss quadrature nodes and weights,
        "gauss_lobatto" computes Gauss-Legendre-Lobatto nodes (including endpoints -1 and 1),
        "chebyshev" computes Chebyshev-Gauss interpolation nodes, and "uniform" generates
        an evenly spaced grid. Default is "gauss_legendre".
    """
    num_points: Union[int, Sequence[int]] = 10
    scheme: Literal["gauss_legendre", "gauss_lobatto", "chebyshev", "uniform"] = "gauss_legendre"

    def generate_samples(self, domain: InterfaceDomain) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Generates quadrature sample points and integration weights on the domain.

        Parameters
        ----------
        domain : InterfaceDomain
            The domain with bounds of shape (m, 2).

        Returns
        -------
        nodes : npt.NDArray[np.float64]
            Sample nodes of shape (N, m).
        weights : npt.NDArray[np.float64]
            Integration weights of shape (N,).
        """
        m = domain.dimension
        if m == 0:
            k = domain.num_copies
            return np.zeros((k, 0), dtype=np.float64), np.ones((k,), dtype=np.float64)

        if isinstance(self.num_points, (int, np.integer)):
            n_pts = [int(self.num_points)] * m
        else:
            n_pts = [int(p) for p in self.num_points]
            if len(n_pts) != m:
                raise ValueError(f"num_points length {len(n_pts)} must match domain dimension {m}.")

        nodes_1d = []
        weights_1d = []

        for j in range(m):
            a, b = float(domain.bounds[j, 0]), float(domain.bounds[j, 1])
            N = n_pts[j]

            if self.scheme == "gauss_legendre":
                xi, w = np.polynomial.legendre.leggauss(N)
                x = 0.5 * (a + b) + 0.5 * (b - a) * xi
                w_scaled = 0.5 * (b - a) * w
            elif self.scheme == "gauss_lobatto":
                xi, w = _gauss_lobatto_1d(N)
                x = 0.5 * (a + b) + 0.5 * (b - a) * xi
                w_scaled = 0.5 * (b - a) * w
            elif self.scheme == "chebyshev":
                if N == 1:
                    x = np.array([0.5 * (a + b)])
                    w_scaled = np.array([b - a])
                else:
                    j_idx = np.arange(N)
                    xi = np.cos((2 * j_idx + 1) / (2 * N) * np.pi)
                    x = 0.5 * (a + b) + 0.5 * (b - a) * xi
                    w_scaled = np.full(N, (b - a) / N)
            elif self.scheme == "uniform":
                x = np.linspace(a, b, N)
                dx = (b - a) / max(1, N - 1) if N > 1 else (b - a)
                w_scaled = np.full(N, dx)
            else:
                raise ValueError(f"Unknown sampling scheme '{self.scheme}'.")

            nodes_1d.append(x)
            weights_1d.append(w_scaled)

        if m == 1:
            return nodes_1d[0].reshape((-1, 1)), weights_1d[0]

        grids = np.meshgrid(*nodes_1d, indexing="ij")
        nodes = np.column_stack([g.flatten() for g in grids])

        w_grids = np.meshgrid(*weights_1d, indexing="ij")
        weights = np.prod(np.column_stack([wg.flatten() for wg in w_grids]), axis=1)

        return nodes, weights


def generate_element_quadrature(
    space: Any,
    scheme: Literal["gauss_lobatto", "gauss_legendre"] = "gauss_lobatto",
    degree_elevation: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates composite element-wise quadrature nodes and weights over the
    smooth polynomial subdomains (knot spans) of a function space.
    Shared boundary nodes (e.g. for Gauss-Lobatto) are deduplicated with summed weights.

    Parameters
    ----------
    space : Any
        Function space (such as BSplineSpace or PolynomialSpace) with degree and optional knots.
    scheme : Literal["gauss_lobatto", "gauss_legendre"], optional
        Quadrature scheme per element. Default is "gauss_lobatto".
    degree_elevation : int, optional
        Extra quadrature points per element along each axis beyond p + 1. Default is 1.

    Returns
    -------
    nodes : np.ndarray
        Unique quadrature nodes of shape (Q, num_parametric_dimensions).
    weights : np.ndarray
        Integration weights of shape (Q,) summing to total parametric volume.
    """
    n_dim = getattr(space, "num_parametric_dimensions", 2)
    degrees = getattr(space, "degree", (1,) * n_dim)
    if isinstance(degrees, int):
        degrees = (degrees,) * n_dim

    # 1. Extract unique knot intervals per axis
    dim_spans = []
    if hasattr(space, "knots") and space.knots is not None:
        for d in range(n_dim):
            kd = space.knots[d] if isinstance(space.knots, (list, tuple)) else space.knots
            u_knots = np.unique(kd)
            spans = [(float(u_knots[i]), float(u_knots[i + 1])) for i in range(len(u_knots) - 1) if u_knots[i + 1] > u_knots[i] + 1e-12]
            if len(spans) == 0:
                spans = [(0.0, 1.0)]
            dim_spans.append(spans)
    else:
        dim_spans = [[(0.0, 1.0)] for _ in range(n_dim)]

    # 2. Quadrature points per element along each axis
    n_pts_per_elem = [max(2, int(deg + 1 + degree_elevation)) for deg in degrees]

    # 3. Generate 1D rules per axis and tensor-product over subdomains
    import itertools
    all_elem_nodes = []
    all_elem_weights = []

    for elem_boxes in itertools.product(*dim_spans):
        nodes_1d = []
        weights_1d = []
        for d in range(n_dim):
            a, b = elem_boxes[d]
            N = n_pts_per_elem[d]
            disc = Discretization(num_points=N, scheme=scheme)
            dom_1d = InterfaceDomain.interval(a, b)
            x_1d, w_1d = disc.generate_samples(dom_1d)
            nodes_1d.append(x_1d.flatten())
            weights_1d.append(w_1d)

        grids = np.meshgrid(*nodes_1d, indexing="ij")
        elem_nodes = np.column_stack([g.flatten() for g in grids])
        w_grids = np.meshgrid(*weights_1d, indexing="ij")
        elem_w = np.prod(np.column_stack([wg.flatten() for wg in w_grids]), axis=1)

        all_elem_nodes.append(elem_nodes)
        all_elem_weights.append(elem_w)

    raw_nodes = np.vstack(all_elem_nodes)
    raw_weights = np.concatenate(all_elem_weights)

    # 4. Deduplicate coincident nodes at element boundaries
    rounded = np.round(raw_nodes, decimals=10)
    unique_nodes, inverse_indices = np.unique(rounded, axis=0, return_inverse=True)

    unique_weights = np.zeros(len(unique_nodes), dtype=np.float64)
    np.add.at(unique_weights, inverse_indices, raw_weights)

    return unique_nodes, unique_weights


@dataclass
class Provenance:
    """
    Metadata tracking the origin, detection tolerances, and state of an Interface.

    Parameters
    ----------
    source_type : Literal["declared", "read_brep", "detected"], optional
        Origin of the interface. "declared" indicates user-specified; "read_brep" indicates
        extracted from STEP topological adjacency; "detected" indicates geometric proximity
        discovery. Default is "declared".
    tolerance : float, optional
        Tolerance used during discovery or verification. Default is 1e-6.
    measured_gap : float, optional
        Measured maximum Euclidean distance between paired markers across the interface.
        Default is 0.0.
    relative_orientation : int, optional
        Relative orientation of adjacent patches (+1 for aligned, -1 for opposed). Default is 1.
    is_locked : bool, optional
        Whether the interface definition is locked against automated updates. Default is False.
    """
    source_type: Literal["declared", "read_brep", "detected"] = "declared"
    tolerance: float = 1e-6
    measured_gap: float = 0.0
    relative_orientation: int = 1
    is_locked: bool = False


class Interface:
    """
    Explicit hyperedge coupling an ordered sequence of Marker objects.

    An Interface represents the topological connection $\\Gamma_e$ between two or more
    function entities via their respective markers. It carries NO equations, NO residuals,
    and NO function values, maintaining a strict separation between topology and mechanics.

    Parameters
    ----------
    markers : Sequence[Marker]
        Ordered sequence of markers coupled by this interface. All markers must share
        the same interface domain $(\\dim \\Gamma_e, k)$.
    provenance : Optional[Provenance], optional
        Provenance metadata for the interface. If None, default declared provenance is used.
    name : Optional[str], optional
        Optional descriptive name for the interface.
    """
    def __init__(
        self,
        markers: Sequence[Marker],
        discretization: Optional[Discretization] = None,
        provenance: Optional[Provenance] = None,
        name: Optional[str] = None,
    ):
        if len(markers) == 0:
            raise ValueError("An Interface must contain at least one Marker.")

        ref_domain = markers[0].domain
        for m in markers[1:]:
            if m.domain.dimension != ref_domain.dimension or m.domain.num_copies != ref_domain.num_copies:
                raise ValueError(
                    f"Domain mismatch between markers '{markers[0].name}' and '{m.name}': "
                    f"({ref_domain.dimension}D, {ref_domain.num_copies} copies) vs "
                    f"({m.domain.dimension}D, {m.domain.num_copies} copies)."
                )

        self.markers = list(markers)
        self.domain = ref_domain
        self.discretization = discretization
        if self.domain.dimension > 0 and self.discretization is None:
            self.discretization = Discretization(num_points=10, scheme="gauss_legendre")

        self.provenance = provenance if provenance is not None else Provenance(source_type="declared")
        self.name = name
        self._cached_traces: Dict[int, sps.csc_matrix] = {}

    def get_samples(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Returns quadrature/collocation sample nodes and integration weights on this interface domain.

        Returns
        -------
        nodes : npt.NDArray[np.float64]
            Sample nodes of shape (N, m).
        weights : npt.NDArray[np.float64]
            Integration weights of shape (N,).
        """
        if self.discretization is not None:
            return self.discretization.generate_samples(self.domain)
        elif self.domain.dimension == 0:
            k = self.domain.num_copies
            return np.zeros((k, 0), dtype=np.float64), np.ones((k,), dtype=np.float64)
        else:
            return Discretization().generate_samples(self.domain)

    @property
    def patches(self) -> Set[Tuple[Union[Function, FunctionSet], Optional[int]]]:
        """Returns the set of all (entity, patch_id) nodes touched by this interface."""
        p_set: Set[Tuple[Union[Function, FunctionSet], Optional[int]]] = set()
        for m in self.markers:
            p_set.update(m.patches)
        return p_set

    def copy(
        self,
        target_map: Optional[Dict[Any, Any]] = None,
        name: Optional[str] = None,
    ) -> Interface:
        """
        Creates a copy of the interface, optionally rebinding markers to new target entities.

        Parameters
        ----------
        target_map : Optional[Dict[Any, Any]], optional
            Mapping from existing entities or markers to new entities or markers.
        name : Optional[str], optional
            New name for the copied interface. Default is self.name.

        Returns
        -------
        Interface
            Copied interface.
        """
        target_map = target_map or {}
        new_markers = []
        for m in self.markers:
            if m in target_map:
                new_markers.append(target_map[m])
            elif m.entity in target_map:
                new_markers.append(m.copy(entity=target_map[m.entity]))
            else:
                new_markers.append(m.copy())
        return self.__class__(
            markers=new_markers,
            discretization=self.discretization,
            provenance=self.provenance,
            name=name if name is not None else self.name,
        )

    def rebind(self, target_map: Dict[Any, Any]) -> None:
        """
        Rebinds marker targets to cloned entity objects.

        Used when parameterization or FFD solvers instantiate deformed geometry instances
        that clone the original network structure.

        Parameters
        ----------
        target_map : Dict[Any, Any]
            Mapping from existing entities or markers to newly instantiated target entities.
        """
        new_markers = []
        for m in self.markers:
            if m in target_map:
                new_markers.append(target_map[m])
            elif m.entity in target_map:
                m.entity = target_map[m.entity]
                new_markers.append(m)
            else:
                new_markers.append(m)
        self.markers = new_markers
        self._cached_traces.clear()

    def get_trace_operator(self, marker_idx: int) -> sps.csc_matrix:
        """
        Returns a cached sparse trace matrix $T_k$ such that the marker trace is $T_k a$.

        For a single Function, $T_k = B(\\alpha_k)$.
        For a multi-patch FunctionSet, $T_k$ is assembled via block-diagonal patch basis
        matrices with row permutation to match marker point ordering:
        $T_k = P \\cdot \\mathrm{diag}(B_0, \\dots, B_p)$.

        Valid for constant markers on Function or FunctionSet entities.

        Parameters
        ----------
        marker_idx : int
            Index of the marker in `self.markers`.

        Returns
        -------
        sps.csc_matrix
            Sparse CSC trace matrix of shape (num_points, num_coefficients).
        """
        marker = self.markers[marker_idx]
        if not marker.is_constant:
            raise ValueError("Trace operator caching is valid only for markers with constant parametric coordinates/maps.")

        if marker_idx not in self._cached_traces:
            from lsdo_function_spaces.core.function import Function
            from lsdo_function_spaces.core.function_set import FunctionSet

            if marker.domain.dimension == 0:
                # Finite 0-D point marker(s)
                if isinstance(marker.entity, Function):
                    coords = np.asarray(marker.parametric_coordinates)
                    if coords.ndim == 1:
                        coords = coords.reshape((1, -1))
                    B = marker.entity.space.compute_basis_matrix(coords)
                    if not sps.issparse(B):
                        B = sps.csc_matrix(B)
                    else:
                        B = B.tocsc()
                    self._cached_traces[marker_idx] = B

                elif isinstance(marker.entity, FunctionSet):
                    fset: FunctionSet = marker.entity
                    coords_list = marker.parametric_coordinates
                    if isinstance(coords_list, tuple):
                        coords_list = [coords_list]

                    patch_map: Dict[int, List[Tuple[int, npt.NDArray]]] = {}
                    for orig_idx, (pidx, pt) in enumerate(coords_list):
                        patch_map.setdefault(pidx, []).append((orig_idx, np.asarray(pt)))

                    block_mats: List[sps.csr_matrix] = []
                    reorder_indices: List[int] = []

                    for pidx in sorted(fset.functions.keys()):
                        fn = fset.functions[pidx]
                        if pidx in patch_map:
                            pts = np.vstack([item[1] for item in patch_map[pidx]])
                            reorder_indices.extend([item[0] for item in patch_map[pidx]])
                            B_p = fn.space.compute_basis_matrix(pts)
                            if not sps.issparse(B_p):
                                B_p = sps.csr_matrix(B_p)
                            else:
                                B_p = B_p.tocsr()
                            block_mats.append(B_p)
                        else:
                            n_ctrl = int(np.prod(fn.space.coefficients_shape))
                            block_mats.append(sps.csr_matrix((0, n_ctrl)))

                    B_diag = sps.block_diag(block_mats, format="csr")
                    inv_perm = np.empty_like(reorder_indices)
                    inv_perm[reorder_indices] = np.arange(len(reorder_indices))
                    self._cached_traces[marker_idx] = B_diag[inv_perm, :].tocsc()

            else:
                # Continuous interface (m >= 1) evaluated at discretization sample points
                nodes, _ = self.get_samples()
                if isinstance(marker.parametric_coordinates, tuple):
                    patch_idx, alpha_fn = marker.parametric_coordinates
                    if isinstance(alpha_fn, Function):
                        pts = alpha_fn.evaluate(nodes, non_csdl=True)
                    elif callable(alpha_fn):
                        pts = np.asarray([alpha_fn(s) for s in nodes])
                    else:
                        pts = np.asarray(alpha_fn)

                    if isinstance(marker.entity, Function):
                        B = marker.entity.space.compute_basis_matrix(pts)
                        self._cached_traces[marker_idx] = sps.csc_matrix(B) if not sps.issparse(B) else B.tocsc()
                    else:
                        fset = marker.entity
                        block_mats = []
                        for pidx in sorted(fset.functions.keys()):
                            fn = fset.functions[pidx]
                            if pidx == patch_idx:
                                B_p = fn.space.compute_basis_matrix(pts)
                                block_mats.append(sps.csr_matrix(B_p) if not sps.issparse(B_p) else B_p.tocsr())
                            else:
                                n_ctrl = int(np.prod(fn.space.coefficients_shape))
                                block_mats.append(sps.csr_matrix((0, n_ctrl)))
                        self._cached_traces[marker_idx] = sps.block_diag(block_mats, format="csr").tocsc()

                else:
                    alpha_fn = marker.parametric_coordinates
                    if isinstance(alpha_fn, Function):
                        pts = alpha_fn.evaluate(nodes, non_csdl=True)
                    elif callable(alpha_fn):
                        pts = np.asarray([alpha_fn(s) for s in nodes])
                    else:
                        pts = np.asarray(alpha_fn)

                    if isinstance(marker.entity, Function):
                        B = marker.entity.space.compute_basis_matrix(pts)
                        self._cached_traces[marker_idx] = sps.csc_matrix(B) if not sps.issparse(B) else B.tocsc()
                    else:
                        fset = marker.entity
                        first_idx = list(fset.functions.keys())[0]
                        block_mats = []
                        for pidx in sorted(fset.functions.keys()):
                            fn = fset.functions[pidx]
                            if pidx == first_idx:
                                B_p = fn.space.compute_basis_matrix(pts)
                                block_mats.append(sps.csr_matrix(B_p) if not sps.issparse(B_p) else B_p.tocsr())
                            else:
                                n_ctrl = int(np.prod(fn.space.coefficients_shape))
                                block_mats.append(sps.csr_matrix((0, n_ctrl)))
                        self._cached_traces[marker_idx] = sps.block_diag(block_mats, format="csr").tocsc()

        return self._cached_traces[marker_idx]


class InterfaceCondition:
    """
    Abstract algebraic condition attached to an Interface.

    Represents a mathematical condition $r_e(\\gamma) = 0$ on the interface domain $\\Gamma_e$.
    Subclasses implement specific evaluation residuals (collocation, contact, kinematic joints).

    Parameters
    ----------
    interface : Optional[Interface], optional
        The interface hyperedge on which this condition is defined.
    name : Optional[str], optional
        Optional descriptive name for the condition.
    """
    def __init__(self, interface: Optional[Interface] = None, name: Optional[str] = None):
        self.interface = interface
        self.name = name

    def evaluate_residual(self) -> csdl.Variable:
        """
        Evaluates the algebraic constraint residual vector.

        Returns
        -------
        csdl.Variable
            CSDL variable representing the residual that must evaluate to zero.
        """
        raise NotImplementedError

    def rebind(self, target_map: Dict[Any, Any]) -> InterfaceCondition:
        """
        Rebinds the condition to target entities or cloned markers.

        Parameters
        ----------
        target_map : Dict[Any, Any]
            Mapping from existing entities/markers to new entities/markers.

        Returns
        -------
        InterfaceCondition
            New condition instance bound to target entities.
        """
        raise NotImplementedError


class CollocationCondition(InterfaceCondition):
    r"""
    Collocation condition enforcing trace equality across markers, unary target planes, or prescribed offsets.

    Supports component selection per marker (e.g. constraining only x and z components)
    and derivative matching for :math:`C^1` continuity.

    Formulation:

    - Multi-marker (star scheme against marker 0):
      :math:`r_k(\gamma) = (f_k \circ \alpha_k)(\gamma) - (f_0 \circ \alpha_0)(\gamma) - \mathrm{offset}`
    - Unary target:
      :math:`r_0(\gamma) = (f_0 \circ \alpha_0)(\gamma) - \mathrm{target}`

    Parameters
    ----------
    interface : Interface
        The interface hyperedge on which the condition is enforced.
    selectors : Optional[Dict[int, Union[Sequence[int], slice]]], optional
        Mapping from marker index in `interface.markers` to component indices (e.g., [0, 2] for x and z).
    offset : Optional[Union[csdl.Variable, npt.NDArray[np.float64], float]], optional
        Prescribed offset vector or target value.
    derivative_orders : Optional[Sequence[int]], optional
        Multi-index specifying derivative orders for Hermite/derivative matching.
    name : Optional[str], optional
        Optional descriptive name for the condition.
    """
    def __init__(
        self,
        interface: Interface,
        selectors: Optional[Dict[int, Union[Sequence[int], slice]]] = None,
        offset: Optional[Union[csdl.Variable, npt.NDArray[np.float64], float]] = None,
        derivative_orders: Optional[Sequence[int]] = None,
        discretization: Optional[Discretization] = None,
        weighted: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(interface=interface, name=name)
        self.selectors = selectors if selectors is not None else {}
        self.offset = offset
        self.derivative_orders = derivative_orders
        self.discretization = discretization
        self.weighted = weighted

    @property
    def is_linear(self) -> bool:
        """Read-only: True iff all markers have constant coordinates."""
        return all(m.is_constant for m in self.interface.markers)

    def evaluate_residual(self) -> csdl.Variable:
        sample_nodes = None
        weights = None
        if self.interface.domain.dimension > 0:
            if self.discretization is not None:
                sample_nodes, weights = self.discretization.generate_samples(self.interface.domain)
            else:
                sample_nodes, weights = self.interface.get_samples()

        traces: List[csdl.Variable] = []
        for idx, marker in enumerate(self.interface.markers):
            if self.derivative_orders is None:
                val = marker.evaluate(sample_parameters=sample_nodes)
            else:
                val = marker.evaluate_derivatives(self.derivative_orders, sample_parameters=sample_nodes)

            # Ensure 2D tensor layout (num_points, num_physical_dimensions)
            if hasattr(val, "shape") and len(val.shape) == 1:
                val = val.reshape((1, -1))

            # Component selection: val[:, sel] slices physical columns, preserving point rows
            if idx in self.selectors:
                sel = self.selectors[idx]
                val = val[:, sel]
            traces.append(val)

        # Unary target condition (e.g. ground plane)
        if len(traces) == 1:
            target = self.offset if self.offset is not None else 0.0
            res = traces[0] - target
            if self.weighted and weights is not None:
                sqrt_w = np.tile(np.sqrt(weights).reshape((-1, 1)), (1, res.shape[1]))
                res = res * sqrt_w
            return res

        # Multi-marker star scheme against marker 0
        residuals: List[csdl.Variable] = []
        for k in range(1, len(traces)):
            diff = traces[k] - traces[0]
            if self.offset is not None:
                diff = diff - self.offset
            if self.weighted and weights is not None:
                sqrt_w = np.tile(np.sqrt(weights).reshape((-1, 1)), (1, diff.shape[1]))
                diff = diff * sqrt_w
            residuals.append(diff)

        if len(residuals) == 1:
            return residuals[0]
        return csdl.concatenate(residuals)

    def rebind(self, target_map: Dict[Any, Any]) -> CollocationCondition:
        """
        Rebinds the collocation condition to target entities or cloned markers.
        """
        new_interface = self.interface.copy(target_map)
        return self.__class__(
            interface=new_interface,
            selectors=self.selectors,
            offset=self.offset,
            derivative_orders=self.derivative_orders,
            discretization=self.discretization,
            weighted=self.weighted,
            name=self.name,
        )


class TransversalDerivativeCondition(InterfaceCondition):
    r"""
    Directional / transversal derivative matching condition across an Interface.

    Enforces :math:`C^1` parametric derivative matching across patch boundaries:

    - Multi-marker (star scheme against marker 0):
      :math:`r_k(\gamma) = D_{d_k}(f_k \circ \alpha_k)(\gamma) - \mu_k D_{d_0}(f_0 \circ \alpha_0)(\gamma)`

    Parameters
    ----------
    interface : Interface
        The interface hyperedge on which the condition is enforced.
    derivative_orders : Union[Sequence[int], Dict[int, Sequence[int]]]
        Multi-index specifying derivative orders for each marker (e.g. {0: (1, 0), 1: (1, 0)}).
        If a single sequence is given, it is applied to all markers.
    ratio : Union[float, Dict[int, float]], optional
        Scaling ratio $\\mu_k$ for derivative magnitude matching. Default is 1.0.
    discretization : Optional[Discretization], optional
        Discretization scheme for continuous interfaces. If None, uses interface.discretization.
    weighted : bool, optional
        Whether to scale residuals by square root of quadrature weights. Default is False.
    name : Optional[str], optional
        Descriptive name for the condition.
    """
    def __init__(
        self,
        interface: Interface,
        derivative_orders: Union[Sequence[int], Dict[int, Sequence[int]]],
        ratio: Union[float, Dict[int, float]] = 1.0,
        discretization: Optional[Discretization] = None,
        weighted: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(interface=interface, name=name)
        if isinstance(derivative_orders, dict):
            self.derivative_orders = derivative_orders
        else:
            self.derivative_orders = {i: tuple(derivative_orders) for i in range(len(interface.markers))}
        self.ratio = ratio
        self.discretization = discretization
        self.weighted = weighted

    @property
    def is_linear(self) -> bool:
        """Read-only: True iff all markers have constant coordinates."""
        return all(m.is_constant for m in self.interface.markers)

    def evaluate_residual(self) -> csdl.Variable:
        sample_nodes = None
        weights = None
        if self.interface.domain.dimension > 0:
            if self.discretization is not None:
                sample_nodes, weights = self.discretization.generate_samples(self.interface.domain)
            else:
                sample_nodes, weights = self.interface.get_samples()

        deriv_traces: List[csdl.Variable] = []
        for idx, marker in enumerate(self.interface.markers):
            d_order = self.derivative_orders[idx]
            val = marker.evaluate_derivatives(d_order, sample_parameters=sample_nodes)
            if hasattr(val, "shape") and len(val.shape) == 1:
                val = val.reshape((1, -1))
            deriv_traces.append(val)

        residuals: List[csdl.Variable] = []
        for k in range(1, len(deriv_traces)):
            mu = self.ratio[k] if isinstance(self.ratio, dict) else self.ratio
            diff = deriv_traces[k] - mu * deriv_traces[0]
            if self.weighted and weights is not None:
                sqrt_w = np.tile(np.sqrt(weights).reshape((-1, 1)), (1, diff.shape[1]))
                diff = diff * sqrt_w
            residuals.append(diff)

        if len(residuals) == 1:
            return residuals[0]
        return csdl.concatenate(residuals)

    def rebind(self, target_map: Dict[Any, Any]) -> TransversalDerivativeCondition:
        """
        Rebinds the transversal derivative condition to target entities or cloned markers.
        """
        new_interface = self.interface.copy(target_map)
        return self.__class__(
            interface=new_interface,
            derivative_orders=self.derivative_orders,
            ratio=self.ratio,
            discretization=self.discretization,
            weighted=self.weighted,
            name=self.name,
        )


class NormalAlignmentCondition(InterfaceCondition):
    r"""
    Surface normal alignment condition enforcing :math:`G^1` / tangent-plane continuity across an Interface.

    Enforces that unit surface normal vectors align across patch boundaries:

    - Multi-marker (star scheme against marker 0):
      :math:`r_k(\gamma) = n_k(\gamma) - \sigma n_0(\gamma)`

    where :math:`\sigma = \pm 1` represents relative surface orientation.

    Parameters
    ----------
    interface : Interface
        The interface hyperedge on which the condition is enforced.
    discretization : Optional[Discretization], optional
        Discretization scheme for continuous interfaces. If None, uses interface.discretization.
    weighted : bool, optional
        Whether to scale residuals by square root of quadrature weights. Default is False.
    name : Optional[str], optional
        Descriptive name for the condition.
    """
    def __init__(
        self,
        interface: Interface,
        discretization: Optional[Discretization] = None,
        weighted: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(interface=interface, name=name)
        self.discretization = discretization
        self.weighted = weighted

    @property
    def is_linear(self) -> bool:
        """Surface normal calculation involves cross product and normalization, which is nonlinear."""
        return False

    def _compute_normals(self, marker: Marker, sample_nodes: Optional[npt.NDArray[np.float64]]) -> csdl.Variable:
        tu = marker.evaluate_derivatives((1, 0), sample_parameters=sample_nodes)
        tv = marker.evaluate_derivatives((0, 1), sample_parameters=sample_nodes)
        if len(tu.shape) == 1:
            tu = tu.reshape((1, -1))
        if len(tv.shape) == 1:
            tv = tv.reshape((1, -1))

        n = csdl.cross(tu, tv, axis=1)
        norm_1d = csdl.norm(n, axes=(1,))
        norm_exp = csdl.expand(norm_1d, n.shape, action="i->ij")
        n_unit = n / (norm_exp + 1e-12)
        return n_unit

    def evaluate_residual(self) -> csdl.Variable:
        sample_nodes = None
        weights = None
        if self.interface.domain.dimension > 0:
            if self.discretization is not None:
                sample_nodes, weights = self.discretization.generate_samples(self.interface.domain)
            else:
                sample_nodes, weights = self.interface.get_samples()

        normals = [self._compute_normals(m, sample_nodes) for m in self.interface.markers]
        sigma = float(self.interface.provenance.relative_orientation)

        residuals: List[csdl.Variable] = []
        for k in range(1, len(normals)):
            diff = normals[k] - sigma * normals[0]
            if self.weighted and weights is not None:
                sqrt_w = np.tile(np.sqrt(weights).reshape((-1, 1)), (1, diff.shape[1]))
                diff = diff * sqrt_w
            residuals.append(diff)

        if len(residuals) == 1:
            return residuals[0]
        return csdl.concatenate(residuals)

    def rebind(self, target_map: Dict[Any, Any]) -> NormalAlignmentCondition:
        """
        Rebinds the normal alignment condition to target entities or cloned markers.
        """
        new_interface = self.interface.copy(target_map)
        return self.__class__(
            interface=new_interface,
            discretization=self.discretization,
            weighted=self.weighted,
            name=self.name,
        )


class ClearanceCondition(InterfaceCondition):
    r"""
    Clearance / non-penetration inequality condition across an Interface.

    Enforces that the Euclidean distance between pairs of markers satisfies a minimum clearance::

        ||f_k(gamma_k) - f_0(gamma_0)|| >= min_clearance

    Parameters
    ----------
    interface : Interface
        The interface hyperedge containing the coupled markers (must have at least 2 markers).
    min_clearance : float
        Minimum required physical clearance distance :math:`d_{\min} \ge 0`. Default is 0.0.
    mode : Literal["violation", "margin"], optional
        Evaluation mode:

        - "violation": evaluates :math:`\max(0, d_{\min}^2 - \|p_k - p_0\|^2)`, which is zero when satisfied.
        - "margin": evaluates :math:`\|p_k - p_0\|^2 - d_{\min}^2`, where :math:`\ge 0` indicates satisfaction.

        Default is "violation".
    discretization : Optional[Discretization], optional
        Discretization scheme for continuous interfaces. If None, uses interface.discretization.
    name : Optional[str], optional
        Descriptive name for the condition.
    """
    def __init__(
        self,
        interface: Interface,
        min_clearance: float = 0.0,
        mode: Literal["violation", "margin"] = "violation",
        discretization: Optional[Discretization] = None,
        name: Optional[str] = None,
    ):
        super().__init__(interface=interface, name=name)
        if len(interface.markers) < 2:
            raise ValueError("ClearanceCondition requires an Interface with at least 2 markers.")
        self.min_clearance = float(min_clearance)
        self.mode = mode
        self.discretization = discretization

    @property
    def is_linear(self) -> bool:
        """Clearance conditions involve Euclidean distance squared, which is nonlinear."""
        return False

    def evaluate_residual(self) -> csdl.Variable:
        sample_nodes = None
        if self.interface.domain.dimension > 0:
            if self.discretization is not None:
                sample_nodes, _ = self.discretization.generate_samples(self.interface.domain)
            else:
                sample_nodes, _ = self.interface.get_samples()

        p0 = self.interface.markers[0].evaluate(sample_parameters=sample_nodes)
        if hasattr(p0, "shape") and len(p0.shape) == 1:
            p0 = p0.reshape((1, -1))

        residuals: List[csdl.Variable] = []
        d_sq = self.min_clearance**2

        for k in range(1, len(self.interface.markers)):
            pk = self.interface.markers[k].evaluate(sample_parameters=sample_nodes)
            if hasattr(pk, "shape") and len(pk.shape) == 1:
                pk = pk.reshape((1, -1))

            diff = pk - p0
            dist_sq = csdl.sum(diff**2, axes=(1,))

            if self.mode == "margin":
                margin = dist_sq - d_sq
                residuals.append(margin)
            else:
                zeros = csdl.Variable(value=np.zeros(dist_sq.shape))
                viol = csdl.maximum(zeros, d_sq - dist_sq)
                residuals.append(viol)

        if len(residuals) == 1:
            return residuals[0]
        return csdl.concatenate(residuals)

    def rebind(self, target_map: Dict[Any, Any]) -> ClearanceCondition:
        """
        Rebinds the clearance condition to target entities or cloned markers.
        """
        new_interface = self.interface.copy(target_map)
        return self.__class__(
            interface=new_interface,
            min_clearance=self.min_clearance,
            mode=self.mode,
            discretization=self.discretization,
            name=self.name,
        )


class VariationalInequalityCondition(InterfaceCondition):
    r"""
    Enforces a continuous functional inequality :math:`c(x, \xi) \le 0` over a parametric domain
    via Galerkin projection onto a non-negative function space basis.

    Supports two selectable variational formulations:
    - `"basis_weighted_ks"` (default): Basis-weighted Kreisselmeier-Steinhauser aggregation
      with stabilized log-sum-exp shift. Preserves uniform violations :math:`c \equiv C \implies R \equiv C`
      to machine precision.
    - `"rectified_galerkin"`: Classical weak variational projection integrating the squared
      rectified violation :math:`\max(0, c)^2` against the test basis functions.

    Parameters
    ----------
    interface : Interface
        The interface hyperedge on which the condition is defined.
    space : Optional[Any], optional
        The test function space (e.g. BSplineSpace) matching the boundary geometry.
        If None, automatically inferred from the interface markers.
    method : Literal["basis_weighted_ks", "rectified_galerkin"], optional
        Variational aggregation formulation. Default is "basis_weighted_ks".
    rho : float, optional
        Sharpness factor for Kreisselmeier-Steinhauser aggregation. Default is 50.0.
    tolerance : float, optional
        Residual upper bound tolerance for rectified Galerkin. Default is 1e-4.
    discretization_scheme : Literal["gauss_lobatto", "gauss_legendre"], optional
        Quadrature rule for integration. Default is "gauss_lobatto".
    degree_elevation : int, optional
        Extra quadrature points per element along each axis beyond p + 1. Default is 1.
    name : Optional[str], optional
        Descriptive name for the condition.
    """
    def __init__(
        self,
        interface: Optional[Interface] = None,
        space: Optional[Any] = None,
        method: Literal["basis_weighted_ks", "rectified_galerkin"] = "basis_weighted_ks",
        rho: float = 50.0,
        tolerance: float = 1e-4,
        discretization_scheme: Literal["gauss_lobatto", "gauss_legendre"] = "gauss_lobatto",
        degree_elevation: int = 1,
        name: Optional[str] = None,
    ):
        super().__init__(interface=interface, name=name)
        self.method = method
        self.rho = float(rho)
        self.tolerance = float(tolerance)
        self.discretization_scheme = discretization_scheme
        self.degree_elevation = int(degree_elevation)

        if space is None and interface is not None:
            if len(interface.markers) > 0:
                ent = interface.markers[0].entity
                if hasattr(ent, "space") and ent.space is not None:
                    space = ent.space
                elif hasattr(ent, "functions") and len(ent.functions) > 0:
                    first_fn = list(ent.functions.values())[0]
                    space = getattr(first_fn, "space", None)

        if space is None:
            raise ValueError(
                "Could not infer test function space. "
                "Please pass `space` explicitly or provide an interface with bound markers."
            )
        self.space = space

        # Precompute quadrature nodes and weights
        self.nodes, self.weights = generate_element_quadrature(
            space=self.space,
            scheme=self.discretization_scheme,
            degree_elevation=self.degree_elevation,
        )

        # Precompute basis matrix and weights outside optimization loop
        B = self.space.compute_basis_matrix(self.nodes)
        if not sps.issparse(B):
            B = sps.csc_matrix(B)
        else:
            B = B.tocsc()

        # Raw integration weights: W_qm = w_q * psi_m(xi_q)
        W = sps.diags(self.weights) @ B

        # Normalized integration weights: W_tilde_qm = W_qm / sum_{q'} W_{q'm}
        col_sums = np.array(W.sum(axis=0)).flatten()
        col_sums[col_sums < 1e-14] = 1.0
        W_tilde = W @ sps.diags(1.0 / col_sums)

        self.W_T = W.T.tocsr()
        self.W_tilde_T = W_tilde.T.tocsr()
        self.num_basis_functions = int(B.shape[1])
        self.num_quad_points = int(self.nodes.shape[0])

    @property
    def is_linear(self) -> bool:
        return False

    def evaluate_projection(self, c_values: csdl.Variable) -> csdl.Variable:
        """
        Projects pointwise constraint values c_values of shape (Q,) or (Q, 1)
        onto the test basis space, returning residual vector R of shape (N_psi,).
        """
        if hasattr(c_values, "shape") and len(c_values.shape) > 1 and c_values.shape[-1] == 1:
            c_flat = csdl.reshape(c_values, (c_values.shape[0],))
        else:
            c_flat = c_values

        if self.method == "basis_weighted_ks":
            # Stabilized Log-Sum-Exp Trick:
            # c_shift = max(c_flat)
            c_shift = csdl.maximum(c_flat)
            c_diff = c_flat - c_shift
            exp_terms = csdl.exp(self.rho * c_diff)

            # W_tilde_T @ exp_terms
            weighted_sum = csdl.sparse.matvec(self.W_tilde_T, exp_terms)
            R = (1.0 / self.rho) * csdl.log(weighted_sum) + c_shift
            return R

        elif self.method == "rectified_galerkin":
            # phi(c) = max(0, c)^2
            zeros = csdl.Variable(shape=c_flat.shape, value=0.0)
            rectified = (csdl.maximum(c_flat, zeros)) ** 2
            R = csdl.sparse.matvec(self.W_T, rectified)
            return R
        else:
            raise ValueError(f"Unknown variational aggregation method '{self.method}'")

    def evaluate_residual(self, field_values: Optional[csdl.Variable] = None) -> csdl.Variable:
        """
        Evaluates the variational inequality residual vector.
        If field_values is provided, projects field_values directly onto the test space.
        """
        if field_values is not None:
            return self.evaluate_projection(field_values)

        if len(self.interface.markers) >= 2:
            p0 = self.interface.markers[0].evaluate(sample_parameters=self.nodes)
            p1 = self.interface.markers[1].evaluate(sample_parameters=self.nodes)
            if hasattr(p0, "shape") and len(p0.shape) == 1:
                p0 = p0.reshape((1, -1))
            if hasattr(p1, "shape") and len(p1.shape) == 1:
                p1 = p1.reshape((1, -1))

            diff = p1 - p0
            dist_sq = csdl.sum(diff**2, axes=(1,))
            # Default non-penetration: c = -dist_sq <= 0
            return self.evaluate_projection(-dist_sq)

        raise ValueError("VariationalInequalityCondition requires field_values or at least 2 markers.")

    def rebind(self, target_map: Dict[Any, Any]) -> VariationalInequalityCondition:
        """
        Rebinds the variational condition to target entities or cloned markers.
        """
        new_interface = self.interface.copy(target_map) if self.interface is not None else None
        new_space = target_map.get(self.space, self.space)
        return self.__class__(
            interface=new_interface,
            space=new_space,
            method=self.method,
            rho=self.rho,
            tolerance=self.tolerance,
            discretization_scheme=self.discretization_scheme,
            degree_elevation=self.degree_elevation,
            name=self.name,
        )
