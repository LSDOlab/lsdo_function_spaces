from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union, Optional, Sequence, List, Tuple, Dict, Set
import numpy as np
import numpy.typing as npt
import scipy.sparse as sps
import csdl_alpha as csdl

from lsdo_function_spaces.core.function import Function
from lsdo_function_spaces.core.function_set import FunctionSet
from lsdo_function_spaces.core.interface import Interface, InterfaceCondition, CollocationCondition


@dataclass
class RankNullityReport:
    """
    Diagnostic report of constraint matrix rank, redundancy, and nullity.

    Parameters
    ----------
    num_variables : int
        Total number of function coefficients / DOFs in the network.
    num_constraints : int
        Total number of scalar constraint equations evaluated.
    rank : int
        Numerical rank $r = \\mathrm{rank}(J_c)$ computed via singular value decomposition.
    redundancy : int
        Number of redundant constraint equations: $p - r$.
    nullity : int
        Dimension of the constraint null space: $n - r$.
    gauge_modes : int
        Number of rigid-body or unconstrained gauge degrees of freedom accounted for.
    relative_freedom : int
        Effective design degrees of freedom: $\\max(0, \\mathrm{nullity} - \\mathrm{gauge\\_modes})$.
    rank_tolerance : float
        Singular value cutoff threshold used to determine rank.
    is_licq_satisfied : bool
        True if the Linear Independence Constraint Qualification (LICQ) holds (redundancy == 0).
    """
    num_variables: int
    num_constraints: int
    rank: int
    redundancy: int
    nullity: int
    gauge_modes: int
    relative_freedom: int
    rank_tolerance: float
    is_licq_satisfied: bool


class FunctionNetwork:
    """
    Coupled network of functions, multi-patch function sets, interfaces, and conditions.

    A FunctionNetwork represents the overall topological and algebraic assembly:
    - Nodes: Function or FunctionSet patches $(f_i, \\mathrm{patch\\_id})$.
    - Hyperedges: Interface objects coupling markers across patches.
    - Mechanics: InterfaceCondition objects enforcing algebraic residuals.

    Parameters
    ----------
    name : str, optional
        Descriptive name for the network. Default is "network".
    functions : Optional[Sequence[Union[Function, FunctionSet]]], optional
        Optional initial list of functions or function sets to register. Default is None.
    interfaces : Optional[Sequence[Interface]], optional
        Optional initial list of interfaces to register. Default is None.
    conditions : Optional[Sequence[InterfaceCondition]], optional
        Optional initial list of interface conditions to register. Default is None.
    """
    def __init__(
        self,
        name: str = "network",
        functions: Optional[Sequence[Union[Function, FunctionSet]]] = None,
        interfaces: Optional[Sequence[Interface]] = None,
        conditions: Optional[Sequence[InterfaceCondition]] = None,
    ):
        self.name = name
        self.functions: List[Union[Function, FunctionSet]] = []
        self.interfaces: List[Interface] = []
        self.conditions: List[InterfaceCondition] = []
        self.component_groupings: Dict[str, List[Tuple[Union[Function, FunctionSet], Optional[int]]]] = {}

        if functions is not None:
            for f in functions:
                self.add_function(f)
        if interfaces is not None:
            for iface in interfaces:
                self.add_interface(iface)
        if conditions is not None:
            for cond in conditions:
                self.add_condition(cond)

    def add_function(self, function: Union[Function, FunctionSet]) -> Union[Function, FunctionSet]:
        """
        Registers a Function or FunctionSet entity in the network.

        Parameters
        ----------
        function : Union[Function, FunctionSet]
            The function entity to register.

        Returns
        -------
        Union[Function, FunctionSet]
            The registered entity.
        """
        if function not in self.functions:
            self.functions.append(function)
        return function

    def add_interface(self, interface: Interface) -> Interface:
        """
        Registers an Interface in the network and automatically registers all coupled entities.

        Parameters
        ----------
        interface : Interface
            The interface to register.

        Returns
        -------
        Interface
            The registered interface.
        """
        if interface not in self.interfaces:
            self.interfaces.append(interface)
        # Ensure all marker entities are registered
        for m in interface.markers:
            self.add_function(m.entity)
        return interface

    def add_condition(self, condition: InterfaceCondition) -> InterfaceCondition:
        """
        Registers an InterfaceCondition and its underlying Interface in the network.

        Parameters
        ----------
        condition : InterfaceCondition
            The condition to register.

        Returns
        -------
        InterfaceCondition
            The registered condition.
        """
        if condition not in self.conditions:
            self.conditions.append(condition)
        self.add_interface(condition.interface)
        return condition

    def discover_internal_seams(
        self,
        entity: Optional[FunctionSet] = None,
        tolerance: float = 1e-5,
        search_n: int = 7,
        create_conditions: bool = True,
    ) -> List[Interface]:
        """
        Discovers internal seams across patches of FunctionSet entities in the network,
        registers the interfaces, and optionally creates CollocationCondition instances.

        Parameters
        ----------
        entity : Optional[FunctionSet], optional
            Target FunctionSet to inspect. If None, inspects all registered FunctionSets.
        tolerance : float, optional
            Detection gap tolerance. Default is 1e-5.
        search_n : int, optional
            Number of sample points along each edge. Default is 7.
        create_conditions : bool, optional
            Whether to automatically add CollocationCondition for each discovered seam. Default is True.

        Returns
        -------
        List[Interface]
            List of newly discovered and registered interfaces.
        """
        from lsdo_function_spaces.core.function_set import FunctionSet
        targets = [entity] if entity is not None else [f for f in self.functions if isinstance(f, FunctionSet)]
        all_discovered = []
        for target in targets:
            self.add_function(target)
            interfaces = target.discover_interfaces(tolerance=tolerance, search_n=search_n)
            for inter in interfaces:
                self.add_interface(inter)
                if create_conditions:
                    self.add_condition(CollocationCondition(interface=inter, name=f"colloc_{inter.name}"))
                all_discovered.append(inter)
        return all_discovered

    def _get_all_patch_nodes(self) -> Set[Tuple[Union[Function, FunctionSet], Optional[int]]]:
        """Resolves all patch-level nodes (entity, patch_id) across registered entities."""
        nodes: Set[Tuple[Union[Function, FunctionSet], Optional[int]]] = set()
        for ent in self.functions:
            if isinstance(ent, FunctionSet):
                for p in ent.functions.keys():
                    nodes.add((ent, p))
            else:
                nodes.add((ent, None))
        return nodes

    def compute_connected_components(self) -> List[List[Tuple[Union[Function, FunctionSet], Optional[int]]]]:
        """
        Computes connected components of the network at the (entity, patch) level.
        Patches touched by the same interface belong to the same component.

        Returns
        -------
        List[List[Tuple[Union[Function, FunctionSet], Optional[int]]]]
            List of connected components, each represented as a list of (entity, patch_id) nodes.
        """
        all_nodes = self._get_all_patch_nodes()
        adj: Dict[Tuple[Union[Function, FunctionSet], Optional[int]], Set[Tuple[Union[Function, FunctionSet], Optional[int]]]] = {node: set() for node in all_nodes}

        for inter in self.interfaces:
            p_list = list(inter.patches)
            for i in range(len(p_list)):
                for j in range(i + 1, len(p_list)):
                    adj[p_list[i]].add(p_list[j])
                    adj[p_list[j]].add(p_list[i])

        visited: Set[Tuple[Union[Function, FunctionSet], Optional[int]]] = set()
        components: List[List[Tuple[Union[Function, FunctionSet], Optional[int]]]] = []
        for node in all_nodes:
            if node not in visited:
                comp: List[Tuple[Union[Function, FunctionSet], Optional[int]]] = []
                queue = [node]
                visited.add(node)
                while queue:
                    curr = queue.pop(0)
                    comp.append(curr)
                    for nbr in adj[curr]:
                        if nbr not in visited:
                            visited.add(nbr)
                            queue.append(nbr)
                components.append(comp)
        return components

    def classify_interfaces(self) -> Tuple[List[Interface], List[Interface]]:
        """
        Classifies interfaces into internal-to-component vs between-component at the patch level.
        Patches not in any component grouping are treated as singleton components.

        Returns
        -------
        Tuple[List[Interface], List[Interface]]
            (internal_interfaces, between_interfaces)
        """
        all_nodes = self._get_all_patch_nodes()
        node2comp: Dict[Tuple[Union[Function, FunctionSet], Optional[int]], str] = {}
        for cname, node_list in self.component_groupings.items():
            for node in node_list:
                node2comp[node] = cname
        for node in all_nodes:
            if node not in node2comp:
                node2comp[node] = f"singleton_{id(node[0])}_{node[1]}"

        internal: List[Interface] = []
        between: List[Interface] = []

        for inter in self.interfaces:
            comp_ids = {node2comp[p] for p in inter.patches if p in node2comp}
            if len(comp_ids) <= 1:
                internal.append(inter)
            else:
                between.append(inter)
        return internal, between

    def compute_cycle_rank(self) -> Tuple[int, int]:
        """
        Computes the first Betti number $b_1$ on the patch-level incidence multigraph:
            $b_{1,\\mathrm{multigraph}} = |E_{\\mathrm{inc}}| - (|V_{\\mathrm{patches}}| + |V_{\\mathrm{interfaces}}|) + C$
        and the macroscopic loop count after collapsing parallel edges between identical patch sets.

        Returns
        -------
        Tuple[int, int]
            (b1_multigraph, b1_macroscopic)
            - b1_multigraph counts all independent topological cycles including multi-point parallel attachments.
            - b1_macroscopic counts global topological loops between components.
        """
        all_nodes = self._get_all_patch_nodes()
        V_patches = len(all_nodes)
        V_interfaces = len(self.interfaces)
        E_inc = sum(len(inter.patches) for inter in self.interfaces)
        C = len(self.compute_connected_components()) if all_nodes else 0

        b1_multigraph = max(0, E_inc - (V_patches + V_interfaces) + C)

        # Collapse parallel edges between the exact same set of patch nodes
        unique_edges: Set[frozenset[Tuple[Union[Function, FunctionSet], Optional[int]]]] = set()
        for inter in self.interfaces:
            if len(inter.patches) > 1:
                unique_edges.add(frozenset(inter.patches))

        E_collapsed = len(unique_edges)
        b1_macro = max(0, E_collapsed - V_patches + C)
        return b1_multigraph, b1_macro

    def evaluate_residuals(self) -> csdl.Variable:
        """
        Evaluates and concatenates all registered interface condition residuals.

        Returns
        -------
        csdl.Variable
            Concatenated vector of all constraint residuals $r(a) = 0$.
        """
        if len(self.conditions) == 0:
            raise ValueError("No conditions registered in FunctionNetwork to evaluate.")
        residuals = [cond.evaluate_residual() for cond in self.conditions]
        if len(residuals) == 1:
            return residuals[0]
        # Flatten each residual to 1D vector to ensure robust concatenation
        # across vector conditions (N, d) and scalar conditions (1,)
        flat_residuals = [r.reshape((-1,)) if len(r.shape) > 1 else r for r in residuals]
        return csdl.concatenate(flat_residuals)

    def analyze_freedom(
        self,
        rank_tolerance: float = 1e-10,
        gauge_modes: int = 0,
    ) -> RankNullityReport:
        """
        Assembles the linear constraint Jacobian from constant collocation conditions
        and computes numerical rank, redundancy, nullity, and LICQ satisfaction via SVD.

        Parameters
        ----------
        rank_tolerance : float, optional
            Singular value cutoff threshold used to determine numerical rank. Default is 1e-10.
        gauge_modes : int, optional
            Number of expected rigid-body or unconstrained gauge degrees of freedom. Default is 0.

        Returns
        -------
        RankNullityReport
            Detailed diagnostic report containing variable count, constraint count, rank,
            redundancy, nullity, relative freedom, and LICQ boolean flag.
        """
        from lsdo_function_spaces.core.function import Function
        from lsdo_function_spaces.core.function_set import FunctionSet

        # Build column offsets for each function entity
        col_offsets: Dict[int, int] = {}
        total_cols = 0
        phys_dim = None

        for f in self.functions:
            col_offsets[id(f)] = total_cols
            if isinstance(f, Function):
                n_ctrl = int(np.prod(f.space.coefficients_shape))
                p_dim = f.num_physical_dimensions
            else:
                n_ctrl = sum(int(np.prod(fn.space.coefficients_shape)) for fn in f.functions.values())
                p_dim = list(f.functions.values())[0].num_physical_dimensions
            if phys_dim is None:
                phys_dim = p_dim
            total_cols += n_ctrl

        row_blocks: List[sps.csr_matrix] = []
        total_constraints = 0

        for cond in self.conditions:
            if not isinstance(cond, CollocationCondition) or not cond.is_linear:
                continue
            inter = cond.interface
            markers = inter.markers
            k_points = inter.domain.num_copies

            # Trace matrix for each marker
            traces = [inter.get_trace_operator(m_idx) for m_idx in range(len(markers))]

            if len(markers) == 1:
                # Unary target: T_0 @ a_0 = target
                T0 = traces[0]
                sel = cond.selectors.get(0, None)
                n_sel = len(sel) if sel is not None else phys_dim
                block = sps.lil_matrix((T0.shape[0] * n_sel, total_cols * phys_dim))
                # Populate entries
                start_c = col_offsets[id(markers[0].entity)] * phys_dim
                # Assemble per-component rows
                row_idx = 0
                components = sel if sel is not None else list(range(phys_dim))
                for pt_i in range(T0.shape[0]):
                    for c_idx in components:
                        # Row corresponds to (pt_i, c_idx)
                        row_entries = T0.getrow(pt_i).tocoo()
                        for c_col, val in zip(row_entries.col, row_entries.data):
                            block[row_idx, start_c + c_col * phys_dim + c_idx] = val
                        row_idx += 1
                row_blocks.append(block.tocsr())
                total_constraints += row_idx

            else:
                # Star scheme: T_k @ a_k - T_0 @ a_0 = 0
                T0 = traces[0]
                start_c0 = col_offsets[id(markers[0].entity)] * phys_dim
                sel0 = cond.selectors.get(0, None)
                comp0 = sel0 if sel0 is not None else list(range(phys_dim))

                for k in range(1, len(markers)):
                    Tk = traces[k]
                    start_ck = col_offsets[id(markers[k].entity)] * phys_dim
                    sel_k = cond.selectors.get(k, None)
                    comp_k = sel_k if sel_k is not None else list(range(phys_dim))
                    n_comp = min(len(comp0), len(comp_k))

                    block = sps.lil_matrix((Tk.shape[0] * n_comp, total_cols * phys_dim))
                    row_idx = 0
                    for pt_i in range(Tk.shape[0]):
                        for c_sub in range(n_comp):
                            c0 = comp0[c_sub]
                            ck = comp_k[c_sub]
                            # +Tk entry
                            row_entries_k = Tk.getrow(pt_i).tocoo()
                            for col_idx, val in zip(row_entries_k.col, row_entries_k.data):
                                block[row_idx, start_ck + col_idx * phys_dim + ck] += val
                            # -T0 entry
                            row_entries_0 = T0.getrow(pt_i).tocoo()
                            for col_idx, val in zip(row_entries_0.col, row_entries_0.data):
                                block[row_idx, start_c0 + col_idx * phys_dim + c0] -= val
                            row_idx += 1
                    row_blocks.append(block.tocsr())
                    total_constraints += row_idx

        num_vars = total_cols * (phys_dim if phys_dim is not None else 1)

        if len(row_blocks) == 0:
            rank = 0
        else:
            J_c = sps.vstack(row_blocks).toarray()
            s = np.linalg.svd(J_c, compute_uv=False)
            rank = int(np.sum(s > rank_tolerance))

        redundancy = max(0, total_constraints - rank)
        nullity = max(0, num_vars - rank)
        rel_freedom = max(0, nullity - gauge_modes)

        return RankNullityReport(
            num_variables=num_vars,
            num_constraints=total_constraints,
            rank=rank,
            redundancy=redundancy,
            nullity=nullity,
            gauge_modes=gauge_modes,
            relative_freedom=rel_freedom,
            rank_tolerance=rank_tolerance,
            is_licq_satisfied=(redundancy == 0),
        )

    def rebind(self, target_map: Dict[Any, Any], name: Optional[str] = None) -> FunctionNetwork:
        """
        Rebinds the network's functions, interfaces, conditions, and component groupings
        according to target_map.

        Parameters
        ----------
        target_map : Dict[Any, Any]
            Mapping from existing entities/markers to newly instantiated target entities/markers.
        name : Optional[str], optional
            New name for the rebound network. Default is self.name.

        Returns
        -------
        FunctionNetwork
            A new network instance bound to the target entities.
        """
        new_network = self.__class__(name=name if name is not None else self.name)

        # Rebind registered functions
        for fn in self.functions:
            new_fn = target_map.get(fn, fn)
            new_network.add_function(new_fn)

        # Rebind conditions (which automatically registers rebound interfaces)
        for cond in self.conditions:
            new_cond = cond.rebind(target_map)
            new_network.add_condition(new_cond)

        # Rebind standalone interfaces that have no attached conditions
        cond_interfaces = {c.interface for c in self.conditions}
        for iface in self.interfaces:
            if iface not in cond_interfaces:
                new_iface = iface.copy(target_map)
                new_network.add_interface(new_iface)

        # Rebind component groupings
        new_groupings = {}
        for comp_name, patches in self.component_groupings.items():
            new_patches = []
            for target, patch_id in patches:
                new_target = target_map.get(target, target)
                new_patches.append((new_target, patch_id))
            new_groupings[comp_name] = new_patches
        new_network.component_groupings = new_groupings

        return new_network

    def copy(self, name: Optional[str] = None) -> FunctionNetwork:
        """
        Creates a deep copy of the network, copying all registered functions/function sets
        and rebinding interfaces and conditions to the copied entities.

        Parameters
        ----------
        name : Optional[str], optional
            New name for the copied network. Default is self.name.

        Returns
        -------
        FunctionNetwork
            A deep copy of the function network.
        """
        target_map = {}
        for fn in self.functions:
            if hasattr(fn, "copy"):
                target_map[fn] = fn.copy()
            else:
                target_map[fn] = fn
        return self.rebind(target_map, name=name)
