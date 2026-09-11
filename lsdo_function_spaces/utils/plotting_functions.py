


import numpy as np
import pyvista as pv
from typing import Union


def _normalize_points(points: np.ndarray) -> np.ndarray:
    if points.shape[-1] > 3:
        raise ValueError(
            'The points must have 3 or fewer physical dimensions (the size of the last axis).'
            f' The provided points have {points.shape[-1]} physical dimensions. You probably want to reshape.'
        )
    points = points.reshape((points.size // points.shape[-1], points.shape[-1]))
    if points.shape[-1] == 1:
        points = np.hstack((points, np.zeros((points.shape[0], 2))))
    elif points.shape[-1] == 2:
        points = np.hstack((points, np.zeros((points.shape[0], 1))))
    return points


def _normalize_grid(points: np.ndarray) -> np.ndarray:
    if points.shape[-1] > 3:
        raise ValueError(
            'The points must have 3 or fewer physical dimensions (the size of the last axis).'
            f' The provided points have {points.shape[-1]} physical dimensions. You probably want to reshape.'
        )
    if points.shape[-1] == 1:
        zeros = np.zeros(points.shape[:-1] + (2,))
        points = np.concatenate((points, zeros), axis=-1)
    elif points.shape[-1] == 2:
        zeros = np.zeros(points.shape[:-1] + (1,))
        points = np.concatenate((points, zeros), axis=-1)
    return points


def _extract_scalars(color: np.ndarray, num_points: int):
    if color.ndim == 1:
        if color.shape[0] != num_points:
            raise ValueError("Color array length must match number of points.")
        return color, False
    if color.ndim == 2 and color.shape[0] == num_points:
        if color.shape[1] == 1:
            return color.reshape(-1), False
        if color.shape[1] == 3:
            return color, True
    flattened = color.reshape(-1)
    if flattened.size == num_points:
        return flattened, False
    if flattened.size == num_points * 3:
        return flattened.reshape((num_points, 3)), True
    raise ValueError("Color array size does not match number of points.")


def _make_plot_element(mesh, **kwargs) -> dict:
    return {"mesh": mesh, "kwargs": kwargs}


def _flatten_plotting_elements(elements: list) -> list:
    """Recursively flatten nested lists of plotting elements."""
    flattened = []
    for item in elements:
        if isinstance(item, list):
            flattened.extend(_flatten_plotting_elements(item))
        else:
            flattened.append(item)
    return flattened


def show_plot(plotting_elements:list, title:str,  axes:bool=True, view_up:str="z", interactive:bool=True, camera:dict={}, screenshot:str=""):
    '''
    Shows the plot.

    Parameters
    -----------
    plotting_elements : list
        The list of PyVista plotting elements to plot.
    title : str
        The title of the plot.
    axes : bool = True
        A boolean on whether to show the axes or not.
    viewup : str = "z"
        The direction of the view up.
    interactive : bool = True
        A boolean on whether the plot is interactive or not.
    '''
    plotter = pv.Plotter()
    if axes:
        plotter.show_axes()

    # Flatten nested lists to handle cases where users pass [plot_points_result]
    plotting_elements = _flatten_plotting_elements(plotting_elements)

    for element in plotting_elements:
        if isinstance(element, dict) and "mesh" in element:
            mesh = element["mesh"]
            kwargs = element.get("kwargs", {})
            plotter.add_mesh(mesh, **kwargs)
        elif isinstance(element, tuple) and len(element) == 2:
            mesh, kwargs = element
            plotter.add_mesh(mesh, **kwargs)
        elif isinstance(element, pv.Actor):
            plotter.add_actor(element)
        elif isinstance(element, pv.DataSet):
            plotter.add_mesh(element)

    if view_up:
        view_map = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}
        if isinstance(view_up, str):
            view_up = view_map.get(view_up)
        if view_up is not None:
            plotter.camera.SetViewUp(*view_up)

    if camera:
        camera_obj = plotter.camera
        if "position" in camera:
            camera_obj.position = camera["position"]
        if "focal_point" in camera:
            camera_obj.focal_point = camera["focal_point"]
        if "viewup" in camera:
            viewup = camera["viewup"]
            if isinstance(viewup, str):
                viewup = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}.get(viewup)
            if viewup is not None:
                camera_obj.SetViewUp(*viewup)
        if "distance" in camera:
            camera_obj.distance = camera["distance"]

    if screenshot:
        plotter.show(title=title, interactive=interactive, screenshot=screenshot)
    else:
        plotter.show(title=title, interactive=interactive)


def plot_points(points:np.ndarray, opacity:float=1., color:Union[str, np.ndarray]='#00629B', color_map:str='jet', size=6.,
                additional_plotting_elements:list=[], show:bool=True):
    '''
    Plots a point cloud.

    Parameters
    -----------
    points : np.ndarray
        The points to plot.
    opactity : float = 1.
        The opacity of the plot. 0 is fully transparent and 1 is fully opaque.
    color : str | np.ndarray = '#00629B'
        The 6 digit color code to plot the points as. A numpy array of colors can be provided to color the points individually according to a cmap.
    color_map : str = 'jet'
        The color map to use if the color is a numpy array.
    size : float = 6.
        The size (radius) of the points.
    additional_plotting_elemets : list = []
        PyVista plotting elements that may have been returned from previous plotting functions that should be plotted with this plot.
    show : bool = True
        A boolean on whether to show the plot or not. If the plot is not shown, the plotting element is still returned.
    '''
    plotting_elements = _flatten_plotting_elements(additional_plotting_elements.copy())

    original_dim = points.shape[-1]
    original_dim = points.shape[-1]
    points = _normalize_points(points)
    plotting_points = pv.PolyData(points)
    kwargs = dict(opacity=opacity, point_size=size, render_points_as_spheres=True)
    if isinstance(color, str):
        kwargs["color"] = color
    elif isinstance(color, np.ndarray):
        scalars, rgb = _extract_scalars(color, points.shape[0])
        kwargs["scalars"] = scalars
        kwargs["cmap"] = color_map
        kwargs["rgb"] = rgb

    plotting_elements.append(_make_plot_element(plotting_points, **kwargs))

    if points.shape[-1] == 3:
        view_up = "z"
    else:
        view_up = "y"

    if show:
        view_up = "z" if original_dim == 3 else "y"
        show_plot(plotting_elements, 'Points', axes=1, view_up=view_up, interactive=True)
    return plotting_elements



def plot_curve(points:np.ndarray, opacity:float=1., color:Union[str, np.ndarray]='#00629B', color_map:str='jet', line_width:float=3.,
              additional_plotting_elements:list=[], show:bool=True):
    '''
    Plots the B-spline Surface.

    Parameters
    -----------
    points : np.ndarray -- shape=(num_points, num_physical_dimensions)
        The points of the curve to be plotted.
    opactity : float
        The opacity of the plot. 0 is fully transparent and 1 is fully opaque.
    color : str = '#00629B'
        The 6 digit color code to plot the curve as. A numpy array of colors can be provided to color the points individually according to a cmap.
    color_map : str = 'jet
        The color map to use if the color is a numpy array.
    additional_plotting_elemets : list
        PyVista plotting elements that may have been returned from previous plotting functions that should be plotted with this plot.
    show : bool
        A boolean on whether to show the plot or not. If the plot is not shown, the plotting element is returned.
    '''
    # NOTE: The function object performs the evaluation(s) to get the points (and colors if applicable) and then these functions do the plotting.
    
    plotting_elements = _flatten_plotting_elements(additional_plotting_elements.copy())

    points = _normalize_points(points)
    plotting_line = pv.lines_from_points(points, close=False)
    kwargs = dict(opacity=opacity, line_width=line_width, render_lines_as_tubes=True)
    
    
    if isinstance(color, str):
        kwargs["color"] = color
    elif isinstance(color, np.ndarray):
        scalars, rgb = _extract_scalars(color, points.shape[0])
        kwargs["scalars"] = scalars
        kwargs["cmap"] = color_map
        kwargs["rgb"] = rgb

    plotting_elements.append(_make_plot_element(plotting_line, **kwargs))

    if show:
        if original_dim < 3:
            view_up = "y"
        else:
            view_up = "z"
        # plotter.show(plotting_elements, f'B-spline Curve', axes=1, view_up=view_up, interactive=True)
        show_plot(plotting_elements, 'Curve', axes=1, view_up=view_up, interactive=True)
        return plotting_elements
        
    return plotting_elements


def plot_surface(points:np.ndarray, plot_types:list=['function'], opacity:float=1., 
                 color:Union[str, np.ndarray]='#00629B', color_map:str='jet', surface_texture:str="", 
                 line_width:float=3., additional_plotting_elements:list=[], show:bool=True):
    '''
    Plots the B-spline Surface.

    Parameters
    -----------
    points : np.ndarray -- shape=(num_points_u, num_points_v, num_physical_dimensions)
        The type of points to be plotted. {evaluated_points, coefficients}
    plot_types : list
        The type of plot {function, wireframe}
    opactity : float
        The opacity of the plot. 0 is fully transparent and 1 is fully opaque.
    color : str = '#00629B'
        The 6 digit color code to plot the surface as. A numpy array of colors can be provided to color the points individually according to a cmap.
    color_map : str = 'jet'
        The color map to use if the color is a numpy array.
    surface_texture : str = "" {"metallic", "glossy", ...}, optional
        The surface texture to determine how light bounces off the surface.
        This is kept for API compatibility.
    additional_plotting_elemets : list
        PyVista plotting elements that may have been returned from previous plotting functions that should be plotted with this plot.
    show : bool
        A boolean on whether to show the plot or not. If the plot is not shown, the plotting element is returned.
    '''
    plotting_elements = _flatten_plotting_elements(additional_plotting_elements.copy())

    num_plot_u = points.shape[0]
    num_plot_v = points.shape[1]
    original_dim = points.shape[-1]

    import csdl_alpha as csdl
    if isinstance(points, csdl.Variable):
        points = points.value

    points = _normalize_grid(points)
    x = points[:, :, 0]
    y = points[:, :, 1]
    z = points[:, :, 2]
    mesh = pv.StructuredGrid(x, y, z)

    num_points = num_plot_u * num_plot_v
    scalar_kwargs = {}
    if isinstance(color, np.ndarray):
        color_values = color
        if color_values.shape[:2] == (num_plot_u, num_plot_v):
            color_values = color_values.reshape((num_points, -1)) if color_values.ndim > 2 else color_values.reshape(-1)
        scalars, rgb = _extract_scalars(color_values, num_points)
        scalar_kwargs = {"scalars": scalars, "cmap": color_map, "rgb": rgb}

    if 'function' in plot_types:
        kwargs = dict(opacity=opacity)
        if isinstance(color, str):
            kwargs["color"] = color
        else:
            kwargs.update(scalar_kwargs)
        plotting_elements.append(_make_plot_element(mesh, **kwargs))
    if 'wireframe' in plot_types:
        kwargs = dict(opacity=opacity, style="wireframe", line_width=line_width)
        if isinstance(color, str):
            kwargs["color"] = color
        else:
            kwargs.update(scalar_kwargs)
        plotting_elements.append(_make_plot_element(mesh, **kwargs))

    if show:
        if original_dim < 3:
            view_up = "y"
        else:
            view_up = "z"
        show_plot(plotting_elements, 'Surface', axes=1, view_up=view_up, interactive=True)

    return plotting_elements

def get_surface_mesh(surface, color=None, grid_n=50, offset=0):
    import lsdo_function_spaces as fs
    surface:fs.Function = surface

    # Generate meshgrid of parametric coordinates
    mesh_grid_input = []
    for dimension_index in range(2):
        mesh_grid_input.append(np.linspace(0., 1., grid_n))
    parametric_coordinates_tuple = np.meshgrid(*mesh_grid_input, indexing='ij')
    for dimensions_index in range(2):
        parametric_coordinates_tuple[dimensions_index] = parametric_coordinates_tuple[dimensions_index].reshape((-1,1))
    grid = np.hstack(parametric_coordinates_tuple)

    # grid = surface.space.generate_parametric_grid(grid_n)
    points = surface.evaluate(grid, non_csdl=True).reshape((grid_n, grid_n, surface.num_physical_dimensions))
    vertices = []
    faces = []
    for u_index in range(grid_n):
        for v_index in range(grid_n):
            vertex = tuple(points[u_index, v_index, :])
            vertices.append(vertex)
            if u_index != 0 and v_index != 0:
                face = tuple((
                    (u_index-1)*grid_n+(v_index-1)+offset,
                    (u_index-1)*grid_n+(v_index)+offset,
                    (u_index)*grid_n+(v_index)+offset,
                    (u_index)*grid_n+(v_index-1)+offset,
                ))
                faces.append(face)
    if color is not None:
        c_points = color.evaluate(grid, non_csdl=True)
        if len(c_points.shape) > 1:
            if c_points.shape[1] > 1:
                c_points = np.linalg.norm(c_points, axis=1)
        return vertices, faces, c_points
    return vertices, faces


def make_scalar_bar_element(color_min: float, color_max: float, color_map: str = "jet"):
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mesh = pv.PolyData(points)
    scalars = np.array([color_min, color_max])
    kwargs = dict(scalars=scalars, cmap=color_map, show_scalar_bar=True, opacity=0.0)
    return _make_plot_element(mesh, **kwargs)

