


import numpy as np
from typing import Union

def show_plot(plotting_elements:list, title:str, axes:bool=False, view_up:str="z", interactive:bool=True):
    '''
    Shows the plot.

    Parameters
    -----------
    plotting_elements : list
        The list of Vedo plotting elements to plot.
    title : str
        The title of the plot.
    axes : bool = True
        A boolean on whether to show the axes or not.
    viewup : str = "z"
        The direction of the view up.
    interactive : bool = True
        A boolean on whether the plot is interactive or not.
    '''
    import vedo
    plotter = vedo.Plotter()
    plotter.show(plotting_elements, title, axes=axes, viewup=view_up, interactive=interactive)


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
        Vedo plotting elements that may have been returned from previous plotting functions that should be plotted with this plot.
    show : bool = True
        A boolean on whether to show the plot or not. If the plot is not shown, the Vedo plotting element is still returned.
    '''
    import vedo

    plotting_elements = additional_plotting_elements.copy()

    if points.shape[-1] > 3:
        raise ValueError('The points must have 3 or fewer physical dimensions (the size of the last axis).' +  
                         f'The provided points have {points.shape[-1]} physical dimensions. You probably want to reshape.')

    points = points.reshape((points.size//points.shape[-1],points.shape[-1]))

    plotting_points = vedo.Points(points, r=size).opacity(opacity)
    if isinstance(color, str):
        plotting_points.color(color)
    elif isinstance(color, np.ndarray):
        plotting_points.cmap(color_map, color)

    plotting_elements.append(plotting_points)

    if points.shape[-1] == 3:
        view_up = "z"
    else:
        view_up = "y"

    if show:
        show_plot(plotting_elements, 'Points', axes=1, interactive=True)
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
        Vedo plotting elements that may have been returned from previous plotting functions that should be plotted with this plot.
    show : bool
        A boolean on whether to show the plot or not. If the plot is not shown, the Vedo plotting element is returned.
    '''
    import vedo
    # NOTE: The function object performs the evaluation(s) to get the points (and colors if applicable) and then these functions do the vedo.
    
    plotting_elements = additional_plotting_elements.copy()

    if points.shape[-1] > 3:
        raise ValueError('The points must have 3 or fewer physical dimensions (the size of the last axis).' +  
                         f'The provided points have {points.shape[-1]} physical dimensions. You probably want to reshape.')

    points = points.reshape((points.size//points.shape[-1],points.shape[-1]))

    plotting_line = vedo.Line(points).linewidth(line_width).opacity(opacity)
    
    # if 'wireframe' in plot_types:
    #     num_points = np.cumprod(points.shape[:-1])[-1]
    #     plotting_elements.append(vedo.Points(points.reshape((num_points,-1)), r=12).color(color))
    
    if isinstance(color, str):
        plotting_line.color(color)
    elif isinstance(color, np.ndarray):
        plotting_line.cmap(color_map, color)

    plotting_elements.append(plotting_line)

    if show:
        if points.shape[-1] < 3:
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
        See https://github.com/marcomusy/vedo/blob/master/examples/basic/lightings.py for options.
    additional_plotting_elemets : list
        Vedo plotting elements that may have been returned from previous plotting functions that should be plotted with this plot.
    show : bool
        A boolean on whether to show the plot or not. If the plot is not shown, the Vedo plotting element is returned.
    '''
    import vedo
    plotting_elements = additional_plotting_elements.copy()

    num_plot_u = points.shape[0]
    num_plot_v = points.shape[1]

    # NOTE: When doing triangles, this will have to be generalized. 
    # -- Maybe the input should be points and element_map instead of just points in a structured shape/grid?
    vertices = []
    faces = []
    for u_index in range(num_plot_u):
        for v_index in range(num_plot_v):
            vertex = tuple(points[u_index, v_index, :])
            vertices.append(vertex)
            if u_index != 0 and v_index != 0:
                face = tuple((
                    (u_index-1)*num_plot_v+(v_index-1),
                    (u_index-1)*num_plot_v+(v_index),
                    (u_index)*num_plot_v+(v_index),
                    (u_index)*num_plot_v+(v_index-1),
                ))
                faces.append(face)

    mesh = vedo.Mesh([vertices, faces]).opacity(opacity).lighting(surface_texture)
    if isinstance(color, str):
        mesh.color(color)
    elif isinstance(color, np.ndarray):
        mesh.cmap(color_map, color)
    if 'function' in plot_types:
        plotting_elements.append(mesh)
    if 'wireframe' in plot_types:
        mesh = vedo.Mesh([vertices, faces]).opacity(opacity)
        plotting_elements.append(mesh.wireframe().linewidth(line_width))

    if show:
        # plotter = vedo.Plotter()
        # plotter.show(plotting_elements, 'B-spline Surface', axes=1, viewup="z", interactive=True)
        if points.shape[-1] < 3:
            view_up = "y"
        else:
            view_up = "z"
        show_plot(plotting_elements, 'Surface', axes=1, view_up=view_up, interactive=True)

    return plotting_elements

def get_surface_mesh(surface, color=None, grid_n=25, offset=0):
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

# NOTE: Vedo doesn't seem to have volume plotting? So just have the volumes call the plot_surface function for the outer surfaces.
# def plot_volume(self, points:np.ndarray, plot_types:list=['surfaces'],
#             opacity:float=1., color:str='#00629B', surface_texture:str="", additional_plotting_elements:list=[], show:bool=True):
#     '''
#     Plots the outer surfaces of the volume.

#     Parameters
#     -----------
#     points_type : list
#         The type of points to be plotted. {evaluated_points, coefficients}
#     plot_types : list
#         The type of plot {surface, wireframe, point_cloud}
#     opactity : float
#         The opacity of the plot. 0 is fully transparent and 1 is fully opaque.
#     color : str
#         The 6 digit color code to plot the B-spline as.
#     surface_texture : str = "" {"metallic", "glossy", ...}, optional
#         The surface texture to determine how light bounces off the surface.
#         See https://github.com/marcomusy/vedo/blob/master/examples/basic/lightings.py for options.
#     additional_plotting_elemets : list
#         Vedo plotting elements that may have been returned from previous plotting functions that should be plotted with this plot.
#     show : bool
#         A boolean on whether to show the plot or not. If the plot is not shown, the Vedo plotting element is returned.
#     '''
#     # Plot the 6 sides of the volume
#     plotting_elements = additional_plotting_elements.copy()

#     coefficients = self.coefficients.value.reshape(self.space.parametric_coefficients_shape + (self.num_physical_dimensions,))

#     for point_type in point_types:
#         if point_type == 'evaluated_points':
#             num_points_per_dimension = 50
#             linspace_dimension = np.linspace(0., 1., num_points_per_dimension)
#             linspace_meshgrid = np.meshgrid(linspace_dimension, linspace_dimension)
#             linspace_dimension1 = linspace_meshgrid[0].reshape((-1,1))
#             linspace_dimension2 = linspace_meshgrid[1].reshape((-1,1))
#             zeros_dimension = np.zeros((num_points_per_dimension**2,)).reshape((-1,1))
#             ones_dimension = np.ones((num_points_per_dimension**2,)).reshape((-1,1))

#             parametric_coordinates = []
#             parametric_coordinates.append(np.column_stack((linspace_dimension1, linspace_dimension2, zeros_dimension)))
#             parametric_coordinates.append(np.column_stack((linspace_dimension1, linspace_dimension2, ones_dimension)))
#             parametric_coordinates.append(np.column_stack((linspace_dimension1, zeros_dimension, linspace_dimension2)))
#             parametric_coordinates.append(np.column_stack((linspace_dimension1, ones_dimension, linspace_dimension2)))
#             parametric_coordinates.append(np.column_stack((zeros_dimension, linspace_dimension1, linspace_dimension2)))
#             parametric_coordinates.append(np.column_stack((ones_dimension, linspace_dimension1, linspace_dimension2)))

#             num_points_u = num_points_per_dimension
#             num_points_v = num_points_per_dimension
#             plotting_points_shape = []
#             for i in range(6):
#                 plotting_points_shape.append((num_points_u, num_points_v, self.num_physical_dimensions))

#             plotting_points = []
#             for parametric_coordinate_set in parametric_coordinates:
#                 evaluation_map = self.compute_evaluation_map(parametric_coordinates=parametric_coordinate_set, expand_map_for_physical=False)
#                 plotting_points.append(evaluation_map.dot(self.coefficients.value.reshape((-1,3))))

#             plotting_colors = []
#             if type(color) is BSpline:
#                 for parametric_coordinate_set in parametric_coordinates:
#                     plotting_colors.append(color.evaluate(parametric_coordinate_set).value)
        
#         elif point_type == 'coefficients':
#             plotting_points = []
#             plotting_points.append(coefficients[0,:,:].reshape((-1, self.num_physical_dimensions)))
#             plotting_points.append(coefficients[-1,:,:].reshape((-1, self.num_physical_dimensions)))
#             plotting_points.append(coefficients[:,0,:].reshape((-1, self.num_physical_dimensions)))
#             plotting_points.append(coefficients[:,-1,:].reshape((-1, self.num_physical_dimensions)))
#             plotting_points.append(coefficients[:,:,0].reshape((-1, self.num_physical_dimensions)))
#             plotting_points.append(coefficients[:,:,-1].reshape((-1, self.num_physical_dimensions)))

#             plotting_points_shape = []
#             plotting_points_shape.append(coefficients[0,:,:].shape)
#             plotting_points_shape.append(coefficients[-1,:,:].shape)
#             plotting_points_shape.append(coefficients[:,0,:].shape)
#             plotting_points_shape.append(coefficients[:,-1,:].shape)
#             plotting_points_shape.append(coefficients[:,:,0].shape)
#             plotting_points_shape.append(coefficients[:,:,-1].shape)


#         for i in range(6):
#             if 'point_cloud' in plot_types:
#                 plotting_elements.append(vedo.Points(plotting_points[i], r=6).opacity(opacity).color('darkred'))

#             if 'surface' in plot_types or 'wireframe' in plot_types:
#                 num_plot_u = plotting_points_shape[i][0]
#                 num_plot_v = plotting_points_shape[i][1]

#                 vertices = []
#                 faces = []
#                 plotting_points_reshaped = plotting_points[i].reshape(plotting_points_shape[i])
#                 for u_index in range(num_plot_u):
#                     for v_index in range(num_plot_v):
#                         vertex = tuple(plotting_points_reshaped[u_index, v_index, :])
#                         vertices.append(vertex)
#                         if u_index != 0 and v_index != 0:
#                             face = tuple((
#                                 (u_index-1)*num_plot_v+(v_index-1),
#                                 (u_index-1)*num_plot_v+(v_index),
#                                 (u_index)*num_plot_v+(v_index),
#                                 (u_index)*num_plot_v+(v_index-1),
#                             ))
#                             faces.append(face)

#                 mesh = vedo.Mesh([vertices, faces]).opacity(opacity).lighting(surface_texture)
#                 if type(color) is str:
#                     mesh.color(color)
#                 elif type(color) is BSpline:
#                     mesh.cmap('jet', plotting_colors[i])

#             if 'surface' in plot_types:
#                 plotting_elements.append(mesh)
#             if 'wireframe' in plot_types:
#                 mesh = vedo.Mesh([vertices, faces]).opacity(opacity)
#                 plotting_elements.append(mesh.wireframe())

    
#     if show:
#         plotter = vedo.Plotter()
#         plotter.show(plotting_elements, f'B-spline Volume: {self.name}', axes=1, viewup="y", interactive=True)
#         return plotting_elements
#     else:
#         return plotting_elements