from __future__ import annotations

from dataclasses import dataclass
import csdl_alpha as csdl
import numpy as np
import scipy.sparse as sps
import concurrent.futures



# from lsdo_function_spaces.core.function_space import FunctionSpace
import lsdo_function_spaces as lfs


def find_best_surface_chunked(chunk, functions:list[lfs.Function], options):
    # New approach - 2 stages: 
    # 1. Find lower bound of error for each surface via very fast method (eg. bounding box)
    # 2. Project point onto surfaces in order of increasing lower bound of error; 
    #    stop when next lower bound error is greater than current best error
    #
    # Each function space should have a method to compute the lower bound of error
    # eg, for b-splines, this could be the bounding box of the control points

    sorting_time = 0
    projection_time = 0
    projections_skipped = 0
    projections_performed = 0

    results = []
    for point in chunk:

        lower_bounds = {i: function._compute_distance_bounds(point, direction=options['direction']) for i, function in enumerate(functions)}
        sorted_surfaces = sorted(lower_bounds.keys(), key=lambda x: lower_bounds[x])

        # project onto the first surface
        best_surface = sorted_surfaces[0]
        function = functions[best_surface]
        best_coord = function.project(point.reshape(1,-1), direction=options['direction'], grid_search_density_parameter=options['grid_search_density_parameter'],
                                      max_newton_iterations=options['max_newton_iterations'], newton_tolerance=options['newton_tolerance'])
        projections_performed += 1
        direction = options['direction']
        if direction is None:
            best_error = np.linalg.norm(function.evaluate(best_coord, coefficients=function.coefficients.value) - point)
        else:
            function_value = function.evaluate(best_coord, coefficients=function.coefficients.value)
            displacement = (point - function_value).reshape((-1,))
            rho = 1.e-3
            best_error = ((1 + rho)*displacement.dot(displacement) - np.dot(displacement, direction)**2)**(1/2)

        for name in sorted_surfaces:
            function = functions[name]
            bound = lower_bounds[name]
            # TODO: TEMPORARY DISABLING THE BREAK BECAUSE I'M RUNNING INTO AN ERROR WHEN I HAVE A SPARSE SET OF COEFFICIENTS
            # if bound > best_error:
            #     projections_skipped += len(sorted_surfaces) - sorted_surfaces.index(name)
            #     break
            parametric_coordinate = function.project(point.reshape(1,-1), direction=options['direction'], grid_search_density_parameter=options['grid_search_density_parameter'],
                                                     max_newton_iterations=options['max_newton_iterations'], newton_tolerance=options['newton_tolerance'])
            projections_performed += 1
            if direction is None:
                error = np.linalg.norm(function.evaluate(parametric_coordinate, coefficients=function.coefficients.value) - point)
            else:
                function_value = function.evaluate(parametric_coordinate, coefficients=function.coefficients.value)
                displacement = (point - function_value).reshape((-1,))
                error = ((1 + rho)*displacement.dot(displacement) - np.dot(displacement, direction)**2)**(1/2)
            if error < best_error:
                best_surface = name
                best_coord = parametric_coordinate
                best_error = error
        results.append((best_surface, best_coord))

    return results




@dataclass
class FunctionSet:
    '''
    Function class. This class is used to represent a function in a given function space. The function space is used to evaluate the function at
    given coordinates, refit the function, and project points onto the function.

    Attributes
    ----------
    functions : list[lfs.Function]
        The functions that make up the function set.
    function_names : list[str] = None
        If they have names, the names of the functions in the function set.
    name : str = None
        The name of the function set.
    space : lfs.FunctionSetSpace = None
        The function set space that the function set is from. If None (recommended), the function set space will be inferred from the functions.
    '''
    functions: list[lfs.Function]
    function_names : list[str] = None
    name : str = None
    space : lfs.FunctionSetSpace = None

    def __post_init__(self):
        if self.function_names is None:
            self.function_names = [None for _ in range(len(self.functions))]
            for i, function in enumerate(self.functions):
                self.function_names[i] = function.name

        if self.space is None:
            self.space = lfs.FunctionSetSpace(
                num_parametric_dimensions=[function.space.num_parametric_dimensions for function in self.functions],
                spaces=[function.space for function in self.functions])
            

    def copy(self) -> lfs.FunctionSet:
        '''
        Copies the function set.

        Returns
        -------
        function_set : lfs.FunctionSet
            The copied function set.
        '''
        functions = [function.copy() for function in self.functions]
        function_set = lfs.FunctionSet(functions=functions, function_names=self.function_names, name=self.name)
        return function_set
            

    def evaluate(self, parametric_coordinates:list[tuple[int, np.ndarray]], parametric_derivative_orders:list[tuple]=None,
                 plot:bool=False) -> csdl.Variable:
        '''
        Evaluates the function.

        Parameters
        ----------
        parametric_coordinates : list[tuple[int, np.ndarray]] -- list length=num_points, tuple_length=2
            The coordinates at which to evaluate the function. The list elements correspond to the coordinate of each point.
            The tuple elements correspond to the index of the function and the parametric coordinates for that point.
            The parametric coordinates should be a numpy array of shape (num_parametric_dimensions,).
        parametric_derivative_orders : tuple = None -- shape=(num_points,num_parametric_dimensions)
            The order of the parametric derivatives to evaluate. If None, the function itself is evaluated.
        plot : bool = False
            Whether or not to plot the function with the points from the result of the evaluation.
        

        Returns
        -------
        function_values : csdl.Variable
            The function evaluated at the given coordinates.
        '''

        # Process parametric coordinates to group them by which function they belong to
        function_indices = []
        function_parametric_coordinates = []
        for parametric_coordinate in parametric_coordinates:
            function_index, coordinates = parametric_coordinate
            function_indices.append(function_index)
            function_parametric_coordinates.append(coordinates)

        # Evaluate each function at the given coordinates
        function_values_list = []
        functions_with_points = []
        for i, function in enumerate(self.functions):
            indices = np.where(np.array(function_indices) == i)[0]
            para_coords = np.array([function_parametric_coordinates[j] for j in indices]).reshape(-1, function.space.num_parametric_dimensions)
            if parametric_derivative_orders is not None:
                para_derivs = [parametric_derivative_orders[j] for j in indices]
            else:
                para_derivs = None
            if len(indices) > 0:
                function_values_list.append(function.evaluate(parametric_coordinates=para_coords,
                                                     parametric_derivative_orders=para_derivs))
                functions_with_points.append(i)

        # Arrange the function values back into the correct element of the array
        function_values = csdl.Variable(value=np.zeros((len(parametric_coordinates), function_values_list[0].shape[-1])))
        for i, function_value in enumerate(function_values_list):
            indices = list(np.where(np.array(function_indices) == functions_with_points[i])[0])
            # indices = list(np.where(np.array(function_indices) == i)[0])
            if len(indices) == 0:
                continue
            if len(indices) == function_values.shape[0]:
                function_values = function_value
            else:
                if function_value.shape[0] == 1:
                    function_value = function_value.reshape((function_value.size,))
                function_values = function_values.set(csdl.slice[indices], function_value)

        if plot:
            # Plot the function
            plotting_elements = self.plot(opacity=0.8, show=False)
            # Plot the evaluated points
            lfs.plot_points(function_values.value, color='#C69214', size=10, additional_plotting_elements=plotting_elements)

        return function_values
    

    def refit(self, new_function_spaces:list[lfs.FunctionSpace]|lfs.FunctionSpace, indices_of_functions_to_refit:list[int]=None, 
              grid_resolution:tuple=None,  parametric_coordinates:list[tuple[int,np.ndarray]]=None,
              parametric_derivative_orders:list[np.ndarray]=None, regularization_parameter:float=None) -> lfs.FunctionSet:
        '''
        Refits functions in the function set. Either a grid resolution or parametric coordinates must be provided. 
        If both are provided, the parametric coordinates will be used. If derivatives are used, the parametric derivative orders must be provided.

        NOTE: this method will NOT overwrite the coefficients or function space in this object. 
        It will return a new function object with the refitted coefficients.

        Parameters
        ----------
        new_function_spaces : list[FunctionSpace] -- list length=number of functions being refit
            The new function spaces that the functions will be picked from.
        indices_of_functions_to_refit : list[int] = None -- list length=number of functions being refit
            The indices of the functions to refit. If None, all the functions are refit.
        grid_resolution : tuple = None -- shape=(num_parametric_dimensions,)
            The resolution of the grid to refit the function.
        parametric_coordinates : np.ndarray = None -- shape=(num_points, num_parametric_dimensions)
            The coordinates at which to refit the function.
        parametric_derivative_orders : list[np.ndarray] = None --list_length=num_points, np.ndarray_shape=(num_parametric_dimensions,) 
            The orders of the parametric derivatives to refit.

        Returns
        -------
        lfs.FunctionSet
            The refitted function with the new function space and new coefficients.
        '''

        if indices_of_functions_to_refit is None:
            indices_of_functions_to_refit = np.arange(len(self.functions))

        if isinstance(new_function_spaces, lfs.FunctionSpace):
            new_function_spaces = [new_function_spaces] * len(self.functions)

        if len(new_function_spaces) != len(indices_of_functions_to_refit) and len(new_function_spaces) != 1:
            raise ValueError("The number of new function spaces must match the number of functions to refit. " +
                             f"({len(new_function_spaces)} != {len(indices_of_functions_to_refit)})")
        elif len(new_function_spaces) == 1:
            new_function_spaces = new_function_spaces * len(indices_of_functions_to_refit)

        new_functions = []
        for i, function in enumerate(self.functions):
            if i in indices_of_functions_to_refit:
                new_functions.append(function.refit(new_function_space=new_function_spaces[i], 
                                                    grid_resolution=grid_resolution,
                                                    parametric_coordinates=parametric_coordinates, 
                                                    parametric_derivative_orders=parametric_derivative_orders,
                                                    regularization_parameter=regularization_parameter))
            else:
                new_functions.append(function)

        new_function_set = lfs.FunctionSet(functions=new_functions, function_names=self.function_names)
        return new_function_set


    def project(self, points:np.ndarray, num_workers:int=8, direction:np.ndarray=None, grid_search_density_parameter:int=1, 
                max_newton_iterations:int=100, newton_tolerance:float=1e-6, plot:bool=False) -> csdl.Variable:
        '''
        Projects a set of points onto the function. The points to project must be provided. If a direction is provided, the projection will find
        the points on the function that are closest to the axis defined by the direction. If no direction is provided, the projection will find the
        points on the function that are closest to the points to project. The grid search density parameter controls the density of the grid search
        used to find the initial guess for the Newton iterations. The max newton iterations and newton tolerance control the convergence of the
        Newton iterations. If plot is True, a plot of the projection will be displayed.

        NOTE: Distance is measured by the 2-norm.

        Parameters
        ----------
        points : np.ndarray -- shape=(num_points, num_phyiscal_dimensions)
            The points to project onto the function.
        direction : np.ndarray = None -- shape=(num_parametric_dimensions,)
            The direction of the projection.
        grid_search_density_parameter : int = 1
            The density of the grid search used to find the initial guess for the Newton iterations.
        max_newton_iterations : int = 100
            The maximum number of Newton iterations.
        newton_tolerance : float = 1e-6
            The tolerance for the Newton iterations.
        plot : bool = False
            Whether or not to plot the projection.
        '''

        options = {'direction': direction, 'grid_search_density_parameter': grid_search_density_parameter,
                     'max_newton_iterations': max_newton_iterations, 'newton_tolerance': newton_tolerance}
        
        if len(points.shape) == 1:
            points = points.reshape(1, -1)
        else:
            points = points.reshape((-1, points.shape[-1]))

        # make sure there aren't more workers than points
        num_workers = min(num_workers, points.shape[0])

        # Divide the points into chunks and run in parallel
        if num_workers > 1:
            chunks = np.array_split(points, num_workers)
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                results = executor.map(find_best_surface_chunked, chunks, [self.functions]*len(chunks), [options]*len(chunks))

            output = []
            for result in results:
                output.extend(result)
        else:
            output = find_best_surface_chunked(points, self.functions, options)

        if plot:
            projection_results = self.evaluate(output).value
            plotting_elements = []
            plotting_elements.append(lfs.plot_points(points, color='#00629B', size=10, show=False))
            # plotting_elements.append(lfs.plot_points(projection_results, color='#F5F0E6', size=10, show=False))
            plotting_elements.append(lfs.plot_points(projection_results, color='#C69214', size=10, show=False))
            self.plot(opacity=0.8, additional_plotting_elements=plotting_elements, show=True)

        return output



    def _check_whether_to_load_projection(self, points:np.ndarray, direction:np.ndarray=None, grid_search_density_parameter:int=1,
                                         max_newton_iterations:int=100, newton_tolerance:float=1e-6) -> bool:
        pass


    def set_coefficients(self, coefficients:list[csdl.Variable], function_indices:list[int]=None) -> None:
        '''
        Sets the coefficients of the functions in the function set with the given indices.

        Parameters
        ----------
        coefficients : list[csdl.Variable]
            The coefficients to set the functions to.
        function_indices : list[int]
            The indices of the functions to set the coefficients of. If None, all the functions are set to the coefficients.
        '''
        if function_indices is None:
            function_indices = np.arange(len(self.functions))

        if len(coefficients) != len(function_indices):
            raise ValueError("The number of coefficients must match the number of functions to set. " +
                             f"({len(coefficients)} != {len(function_indices)})")

        for i, function_index in enumerate(function_indices):
            self.functions[function_index].coefficients = coefficients[i]


    def get_function_indices(self, function_names:list[str]) -> list[int]:
        '''
        Gets the indices of the functions in the function set with the given names.

        Parameters
        ----------
        function_names : list[str]
            The names of the functions to get the indices of.

        Returns
        -------
        function_indices : list[int]
            The indices of the functions in the function set with the given names.
        '''
        function_indices = []
        for function_name in function_names:
            function_indices.append(self.function_names.index(function_name))
        return function_indices
    

    def search_for_function_indices(self, search_strings:list[str]) -> list[int]:
        '''
        Searches for the indices of the functions in the function set with the given search string.

        Parameters
        ----------
        search_strings : str | list[str]
            The strings to search for in the function names.

        Returns
        -------
        function_indices : list[int]
            The indices of the functions in the function set with the given search string.
        '''
        if isinstance(search_strings, str):
            search_strings = [search_strings]

        function_indices = []
        for i, function_name in enumerate(self.function_names):
            for search_string in search_strings:
                if search_string in function_name:
                    function_indices.append(i)
        return function_indices
    

    def create_subset(self, function_indices:list[int]=None, function_search_names:list[str]=None, name:str=None) -> lfs.FunctionSet:
        '''
        Creates a subset of the function set with the given indices. Either the function indices or the function search names must be provided.

        Parameters
        ----------
        function_indices : list[int]
            The indices of the functions to include in the subset.
        function_search_names : list[str]
            The search strings to use to find the functions to include in the subset.
        name : str
            The name of the subset.

        Returns
        -------
        subset : lfs.FunctionSet
            The subset of the function set with the given indices.
        '''
        if function_indices is None:
            function_indices = []
        if function_search_names is not None:
            function_indices += self.search_for_function_indices(search_strings=function_search_names)
        
        # Remove duplicates
        function_indices = list(set(function_indices))

        functions = [self.functions[i] for i in function_indices]
        function_names = [self.function_names[i] for i in function_indices]
        subset = lfs.FunctionSet(functions=functions, function_names=function_names, name=name)
        return subset


    def plot(self, point_types:list=['evaluated_points'], plot_types:list=['function'],
              opacity:float=1., color:str|lfs.FunctionSet='#00629B', color_map:str='jet', surface_texture:str="",
              line_width:float=3., additional_plotting_elements:list=[], show:bool=True) -> list:
        '''
        Plots the function set.

        Parameters
        -----------
        points_type : list = ['evaluated_points']
            The type of points to be plotted. {evaluated_points, coefficients}
        plot_types : list = ['function']
            The type of plot {function, wireframe, point_cloud}
        opactity : float = 1.
            The opacity of the plot. 0 is fully transparent and 1 is fully opaque.
        color : str|lfs.FunctionSet = '#00629B'
            The 6 digit color code to plot the B-spline as. If a FunctionSet is provided, the FunctionSet will be used to color the B-spline.
        surface_texture : str = "" {"metallic", "glossy", ...}, optional
            The surface texture to determine how light bounces off the surface.
            See https://github.com/marcomusy/vedo/blob/master/examples/basic/lightings.py for options.
        color_map : str = 'jet'
            The color map to use if the color is a function.
        additional_plotting_elemets : list
            Vedo plotting elements that may have been returned from previous plotting functions that should be plotted with this plot.
        show : bool
            A boolean on whether to show the plot or not. If the plot is not shown, the Vedo plotting element is returned.

        Returns
        -------
        plotting_elements : list
            The Vedo plotting elements that were plotted.
        '''

        # Then there must be a discrete index so loop over subfunctions and plot them
        plotting_elements = additional_plotting_elements.copy()
        for i, function in enumerate(self.functions):
            plotting_elements = function.plot(point_types=point_types, plot_types=plot_types, opacity=opacity, color=color, color_map=color_map,
                                               surface_texture=surface_texture, line_width=line_width,
                                               additional_plotting_elements=plotting_elements, show=False)
        if show:
            if self.name is not None:
                lfs.show_plot(plotting_elements=plotting_elements, title=self.name)
            else:
                lfs.show_plot(plotting_elements=plotting_elements, title='Function Set Plot')
        return plotting_elements


if __name__ == "__main__":
    import csdl_alpha as csdl
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    num_coefficients1 = 10
    num_coefficients2 = 5
    degree1 = 4
    degree2 = 3
    
    # Create functions that make up set
    space_of_cubic_b_spline_surfaces_with_10_cp = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(degree1,degree1),
                                                              coefficients_shape=(num_coefficients1,num_coefficients1))
    space_of_quadratic_b_spline_surfaces_with_5_cp = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(degree2,degree2),
                                                              coefficients_shape=(num_coefficients2,num_coefficients2))

    coefficients_line = np.linspace(0., 1., num_coefficients1)
    coefficients_y, coefficients_x = np.meshgrid(coefficients_line,coefficients_line)
    coefficients1 = np.stack((coefficients_x, coefficients_y, 0.1*np.random.rand(num_coefficients1,num_coefficients1)), axis=-1)
    coefficients1 = coefficients1.reshape((num_coefficients1,num_coefficients1,3))

    b_spline1 = lfs.Function(space=space_of_cubic_b_spline_surfaces_with_10_cp, coefficients=coefficients1, name='b_spline1')

    coefficients_line = np.linspace(0., 1., num_coefficients2)
    coefficients_y, coefficients_x = np.meshgrid(coefficients_line,coefficients_line)
    coefficients_y += 1.5
    coefficients2 = np.stack((coefficients_x, coefficients_y, 0.1*np.random.rand(num_coefficients2,num_coefficients2)), axis=-1)
    coefficients2 = coefficients2.reshape((num_coefficients2,num_coefficients2,3))

    b_spline2 = lfs.Function(space=space_of_quadratic_b_spline_surfaces_with_5_cp, coefficients=coefficients2, name='b_spline2')

    # Make function set and plot
    my_b_spline_surface_set = lfs.FunctionSet(functions=[b_spline1, b_spline2], function_names=['b_spline1', 'b_spline2'])
    my_b_spline_surface_set.plot()


    # Refit the function set
    num_coefficients = 5
    space_of_linear_b_spline_surfaces_with_5_cp = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1,1),
                                                                coefficients_shape=(num_coefficients,num_coefficients))
    new_function_spaces = [space_of_linear_b_spline_surfaces_with_5_cp, space_of_linear_b_spline_surfaces_with_5_cp]
    fitting_grid_resolution = 50
    new_function_set = my_b_spline_surface_set.refit(new_function_spaces=new_function_spaces, 
                                                     grid_resolution=(fitting_grid_resolution,fitting_grid_resolution))
    new_function_set.plot()


    # Once again, refit the function set but only refit the first function
    num_coefficients = 5
    space_of_linear_b_spline_surfaces_with_5_cp = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1,1),
                                                                coefficients_shape=(num_coefficients,num_coefficients))
    new_function_spaces = [space_of_linear_b_spline_surfaces_with_5_cp]
    fitting_grid_resolution = 50
    new_function_set = my_b_spline_surface_set.refit(new_function_spaces=new_function_spaces, 
                                                     indices_of_functions_to_refit=[0],
                                                     grid_resolution=(fitting_grid_resolution,fitting_grid_resolution))
    new_function_set.plot()