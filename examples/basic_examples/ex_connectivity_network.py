"""
Example: Topological Connectivity, Semantic Markers, and Function Networks
=============================================================================
This example demonstrates the core connectivity layer of lsdo_function_spaces:

1. Synthetic Multi-Patch Assembly:
   - Defining two adjacent 2D B-spline surface patches meeting along a boundary seam.
   - Defining an internal 1D B-spline curve representing a structural spar.
   - Attaching semantic Markers:
     * Continuous 1D curve markers along the seam (Marker.from_function).
     * Discrete 0-D point markers for structural attachments (Marker.from_parametric).
     * Sliding / parameterized markers driven by CSDL design variables.
   - Establishing Interfaces with Gauss-Legendre quadrature discretization.
   - Formulating InterfaceConditions:
     * C^0 collocation (position trace matching).
     * C^1 transversal derivative matching (parametric slope continuity).
     * Structural joint collocation.
     * Clearance condition over product domains.
   - Assembling a FunctionNetwork:
     * Topological graph analysis (connected components, cycle rank / Betti numbers).
     * Linear freedom analysis via SVD (variable count, rank, nullity, LICQ, redundant rows).
     * Evaluating concatenated residuals in NumPy and CSDL computational graph modes.
     * Rebinding network topology to cloned/perturbed geometry instances.

2. CAD-Based Automated Seam Discovery:
   - Importing a multi-patch aircraft STEP geometry (rectangular_wing.stp).
   - Running automated geometric seam detection (discover_internal_seams).
   - Performing full rank-nullity analysis on the multi-patch CAD network.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import csdl_alpha as csdl
import lsdo_function_spaces as lfs


def run_synthetic_connectivity_showcase(plot: bool = False) -> lfs.FunctionNetwork:
    """
    Demonstrates synthetic multi-patch assembly, markers, interfaces, conditions,
    and freedom analysis.

    Parameters
    ----------
    plot : bool, optional
        Whether to open interactive PyVista 3D visualization. Default is False.

    Returns
    -------
    FunctionNetwork
        The assembled and analyzed FunctionNetwork.
    """
    print("\n" + "=" * 76)
    print("PART 1: Synthetic Multi-Patch Assembly & FunctionNetwork Architecture")
    print("=" * 76)

    # -------------------------------------------------------------------------
    # 1. Define Geometric Entities (B-spline surfaces and curve)
    # -------------------------------------------------------------------------
    # Surface space: bicubic (degree 2x2) with 4x4 control points
    space_surf = lfs.BSplineSpace(
        num_parametric_dimensions=2,
        degree=(2, 2),
        coefficients_shape=(4, 4),
    )

    # Patch 1: Left aerodynamic panel spanning x in [0, 1], y in [0, 1]
    u1, v1 = np.meshgrid(np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 4), indexing="ij")
    pts1 = np.zeros((4, 4, 3))
    pts1[:, :, 0] = u1
    pts1[:, :, 1] = v1
    pts1[:, :, 2] = 0.08 * np.sin(np.pi * u1) * np.sin(np.pi * v1)
    panel_left = lfs.Function(space=space_surf, coefficients=pts1, name="panel_left")

    # Patch 2: Right aerodynamic panel spanning x in [1, 2], y in [0, 1]
    u2, v2 = np.meshgrid(np.linspace(1.0, 2.0, 4), np.linspace(0.0, 1.0, 4), indexing="ij")
    pts2 = np.zeros((4, 4, 3))
    pts2[:, :, 0] = u2
    pts2[:, :, 1] = v2
    pts2[:, :, 2] = 0.08 * np.sin(np.pi * u2) * np.sin(np.pi * v2)
    panel_right = lfs.Function(space=space_surf, coefficients=pts2, name="panel_right")

    # Internal vertical spar curve: s in [0, 1] at (x=0.5, y=0.5, z in [-0.5, 0.0])
    spar_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=2, coefficients_shape=(4,))
    spar_pts = np.zeros((4, 3))
    spar_pts[:, 0] = 0.5
    spar_pts[:, 1] = 0.5
    spar_pts[:, 2] = np.linspace(-0.5, 0.0, 4)
    spar_curve = lfs.Function(space=spar_space, coefficients=spar_pts, name="internal_spar")

    print(f"Created surface '{panel_left.name}' (coefficients: {panel_left.coefficients.shape})")
    print(f"Created surface '{panel_right.name}' (coefficients: {panel_right.coefficients.shape})")
    print(f"Created curve '{spar_curve.name}' (coefficients: {spar_curve.coefficients.shape})")

    # -------------------------------------------------------------------------
    # 2. Define Semantic Markers
    # -------------------------------------------------------------------------
    # Continuous 1D seam mapping functions from Gamma = [0, 1] to Omega = [0, 1]^2:
    # On panel_left, seam is at u=1: alpha1(gamma) = (1.0, gamma)
    # On panel_right, seam is at u=0: alpha2(gamma) = (0.0, gamma)
    map_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(2,))
    
    alpha1_fn = lfs.Function(
        space=map_space,
        coefficients=np.array([[1.0, 0.0], [1.0, 1.0]]),
        name="alpha_seam_left",
    )
    seam_marker_left = lfs.Marker.from_function(panel_left, alpha1_fn, name="seam_left")

    alpha2_fn = lfs.Function(
        space=map_space,
        coefficients=np.array([[0.0, 0.0], [0.0, 1.0]]),
        name="alpha_seam_right",
    )
    seam_marker_right = lfs.Marker.from_function(panel_right, alpha2_fn, name="seam_right")

    # Discrete 0-D point markers for structural joint
    spar_tip_marker = lfs.Marker.from_parametric(spar_curve, np.array([1.0]), name="spar_tip")
    panel_joint_marker = lfs.Marker.from_parametric(panel_left, np.array([0.5, 0.5]), name="panel_joint")

    # Sliding / Parameterized marker whose parametric coordinate is a CSDL design variable
    u_slider = csdl.Variable(value=np.array([0.3, 0.7]), name="slider_uv")
    sliding_marker = lfs.Marker.from_parametric(panel_left, u_slider, name="sliding_sensor")

    print("\nSemantic Markers:")
    print(f"  * {seam_marker_left.name}: continuous 1D map (dim {seam_marker_left.domain.dimension})")
    print(f"  * {seam_marker_right.name}: continuous 1D map (dim {seam_marker_right.domain.dimension})")
    print(f"  * {spar_tip_marker.name}: discrete 0-D marker")
    print(f"  * {sliding_marker.name}: parameterized marker (is_constant={sliding_marker.is_constant})")

    # -------------------------------------------------------------------------
    # 3. Interfaces and Discretization
    # -------------------------------------------------------------------------
    # Continuous seam interface discretized using 8-point Gauss-Legendre quadrature
    quadrature = lfs.Discretization(num_points=8, scheme="gauss_legendre")
    seam_interface = lfs.Interface(
        markers=[seam_marker_left, seam_marker_right],
        discretization=quadrature,
        name="panel_seam_interface",
    )
    sample_nodes, sample_weights = seam_interface.get_samples()
    print(f"\nInterface '{seam_interface.name}': {len(sample_nodes)} Gauss-Legendre quadrature nodes")

    # Discrete structural interface coupling the spar tip to the panel
    joint_interface = lfs.Interface(
        markers=[spar_tip_marker, panel_joint_marker],
        name="spar_joint_interface",
    )

    # -------------------------------------------------------------------------
    # 4. Interface Conditions (Mechanics / Constraints)
    # -------------------------------------------------------------------------
    # A. C^0 position trace matching: Tr_{alpha1}(panel_left) - Tr_{alpha2}(panel_right) = 0
    c0_condition = lfs.CollocationCondition(
        interface=seam_interface,
        weighted=True,  # Scales rows by sqrt(w_p) for exact L2 energy residual
        name="c0_seam_collocation",
    )

    # B. C^1 transversal derivative matching across the seam (d/du slope continuity)
    c1_condition = lfs.TransversalDerivativeCondition(
        interface=seam_interface,
        derivative_orders=(1, 0),
        weighted=True,
        name="c1_transversal_slope",
    )

    # C. Discrete joint collocation: x_spar(1) - x_panel(0.5, 0.5) = 0
    joint_condition = lfs.CollocationCondition(
        interface=joint_interface,
        name="spar_joint_collocation",
    )

    # D. Clearance condition between panel_left and an upper ceiling surface
    clearance_probe_fn = lfs.Function(
        space=space_surf,
        coefficients=pts1 + np.array([0.0, 0.0, 0.25]),
        name="clearance_ceiling",
    )
    probe_marker = lfs.Marker.from_parametric(clearance_probe_fn, np.array([0.5, 0.5]), name="ceiling_marker")
    clearance_interface = lfs.Interface(
        markers=[panel_joint_marker, probe_marker],
        name="clearance_interface",
    )
    clearance_condition = lfs.ClearanceCondition(
        interface=clearance_interface,
        min_clearance=0.15,
        name="vertical_clearance",
    )

    # -------------------------------------------------------------------------
    # 5. Assemble FunctionNetwork
    # -------------------------------------------------------------------------
    network = lfs.FunctionNetwork(
        name="aircraft_panel_assembly",
        conditions=[c0_condition, c1_condition, joint_condition, clearance_condition],
    )
    network.add_function(clearance_probe_fn)

    print(f"\nFunctionNetwork '{network.name}' Assembled:")
    print(f"  * Functions: {[f.name for f in network.functions]}")
    print(f"  * Interfaces: {[i.name for i in network.interfaces]}")
    print(f"  * Conditions: {[c.name for c in network.conditions]}")

    # Topological graph properties
    components = network.compute_connected_components()
    b1_multi, b1_macro = network.compute_cycle_rank()
    print(f"  * Connected Components: {len(components)}")
    print(f"  * Cycle Rank (Betti number b1): multigraph={b1_multi}, macroscopic={b1_macro}")

    # -------------------------------------------------------------------------
    # 6. Linear Freedom & SVD Rank-Nullity Analysis
    # -------------------------------------------------------------------------
    freedom = network.analyze_freedom(rank_tolerance=1e-10)
    print("\n" + "-" * 50)
    print("Linear Freedom Analysis (Algebraic SVD)")
    print("-" * 50)
    print(f"Total Unknown Coefficients (n): {freedom.num_variables}")
    print(f"Total Linear Constraint Rows (m): {freedom.num_constraints}")
    print(f"Constraint Jacobian Rank (r):    {freedom.rank}")
    print(f"Redundant Constraints (m - r):   {freedom.redundancy}")
    print(f"Remaining Free DOFs (Nullity):   {freedom.nullity}")
    print(f"LICQ Satisfied (Full Row Rank):  {freedom.is_licq_satisfied}")

    # -------------------------------------------------------------------------
    # 7. Evaluate Residuals (CSDL Graph & NumPy Values)
    # -------------------------------------------------------------------------
    residuals_csdl = network.evaluate_residuals()
    print(f"\nEvaluated Residuals Vector: shape={residuals_csdl.shape}, L2-norm={np.linalg.norm(residuals_csdl.value):.4e}")

    # Sliding marker evaluation
    slider_position = sliding_marker.evaluate()
    print(f"Sliding Sensor Position at (u, v)={u_slider.value}: {slider_position.value}")

    # -------------------------------------------------------------------------
    # 8. Rebinding Topology to Cloned Geometry Instances
    # -------------------------------------------------------------------------
    # Optimization/FFD routines clone original geometries upon deformation.
    # network.rebind() retargets all markers and conditions to the new instances.
    panel_left_new = panel_left.copy()
    panel_right_new = panel_right.copy()
    spar_curve_new = spar_curve.copy()
    clearance_probe_new = clearance_probe_fn.copy()

    rebound_network = network.rebind({
        panel_left: panel_left_new,
        panel_right: panel_right_new,
        spar_curve: spar_curve_new,
        clearance_probe_fn: clearance_probe_new,
    })
    print(f"\nSuccessfully rebound network '{rebound_network.name}' to new geometry instances.")
    assert rebound_network.functions[0] is panel_left_new

    # -------------------------------------------------------------------------
    # 9. 3D PyVista Visualization
    # -------------------------------------------------------------------------
    if plot:
        print("\nDisplaying 3D assembly plot...")
        elements = panel_left.plot(opacity=0.7, color="#3498db", show=False)
        elements = panel_right.plot(opacity=0.7, color="#2ecc71", additional_plotting_elements=elements, show=False)
        elements = spar_curve.plot(color="#e74c3c", line_width=4, additional_plotting_elements=elements, show=False)
        seam_pts = seam_marker_left.evaluate(sample_parameters=sample_nodes, non_csdl=True)
        elements = lfs.plot_points(seam_pts, color="#f39c12", size=10, additional_plotting_elements=elements, show=False)
        lfs.show_plot(elements, title="Topological Connectivity Assembly")

    return network


def run_cad_seam_discovery_showcase(stp_path: Optional[str] = None) -> lfs.FunctionNetwork:
    """
    Demonstrates automated geometric seam detection and condition synthesis on
    an imported multi-patch STEP CAD geometry.

    Parameters
    ----------
    stp_path : Optional[str], optional
        Path to the STEP file. If None, uses rectangular_wing.stp.

    Returns
    -------
    FunctionNetwork
        The network generated from the CAD geometry.
    """
    print("\n" + "=" * 76)
    print("PART 2: CAD-Based Automated Seam Discovery & Multi-Patch Network")
    print("=" * 76)

    if stp_path is None:
        repo_root = Path(__file__).resolve().parent.parent
        stp_path = str(repo_root / "import_files_for_examples" / "rectangular_wing.stp")

    print(f"Importing CAD geometry from: {stp_path}")
    wing = lfs.import_file(stp_path, parallelize=False)
    print(f"Imported FunctionSet '{wing.name}' containing {len(wing.functions)} B-spline patches.")

    # Create network and discover internal boundary seams across patches
    cad_network = lfs.FunctionNetwork(name="wing_cad_network")
    discovered_seams = cad_network.discover_internal_seams(
        entity=wing,
        tolerance=1e-3,
        search_n=7,
        create_conditions=True,
    )
    print(f"Discovered {len(discovered_seams)} internal boundary seams with tolerance 1e-3.")
    print(f"Automatically generated {len(cad_network.conditions)} CollocationConditions.")

    # Perform freedom analysis on the multi-patch CAD network
    freedom_cad = cad_network.analyze_freedom()
    print("\n" + "-" * 50)
    print("CAD Wing Freedom Analysis Report")
    print("-" * 50)
    print(f"Total Patch Control Points:      {freedom_cad.num_variables}")
    print(f"Collocation Constraint Equations: {freedom_cad.num_constraints}")
    print(f"Linear Rank:                     {freedom_cad.rank}")
    print(f"Redundancy:                      {freedom_cad.redundancy}")
    print(f"Design Nullity:                  {freedom_cad.nullity}")
    print(f"LICQ Satisfied:                  {freedom_cad.is_licq_satisfied}")

    # Evaluate all seam residuals
    cad_residuals = cad_network.evaluate_residuals()
    print(f"\nEvaluated CAD Seam Residuals: shape={cad_residuals.shape}, max gap={np.max(np.abs(cad_residuals.value)):.4e}")

    return cad_network


if __name__ == "__main__":
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    run_synthetic_connectivity_showcase(plot=True)
    run_cad_seam_discovery_showcase()
    print("\n" + "=" * 76)
    print("All connectivity showcases completed successfully!")
    print("=" * 76)

