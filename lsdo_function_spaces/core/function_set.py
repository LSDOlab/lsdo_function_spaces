from __future__ import annotations
from dataclasses import dataclass
import csdl_alpha as csdl
import numpy as np
import scipy.sparse as sps
import concurrent.futures
import itertools
import pickle
from pathlib import Path
import string
import random

# from lsdo_function_spaces.core.function_space import FunctionSpace
import lsdo_function_spaces as lfs

def find_best_surface_chunked(chunk, functions:dict[lfs.Function]=None, options=None):
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
    
    if functions is None:
        functions = global_functions
    if options is None:
        options = global_options

    priority_inds = options['priority_inds']
    priority_eps = options['priority_eps']

    direction = options['direction']/np.linalg.norm(options['direction']) if options['direction'] is not None else None
    extrema = options['extrema']

    for point in chunk:
        if extrema:
            n = list(functions.values())[0].space.num_parametric_dimensions
            extrema_parametric = np.array(list(itertools.product([0., 1.], repeat=n)))
            function_extrema_dict = {i: function.evaluate(extrema_parametric).value for i, function in functions.items()}
            
            best_surface = None
            best_coord = None
            best_error = None
            for i, function_extrema in function_extrema_dict.items():
                for j in range(function_extrema.shape[0]):
                    extrema_point = function_extrema[j]
                    parametric_coordinate = extrema_parametric[j]
                    if direction is None:
                        error = np.linalg.norm(extrema_point - point)
                        if best_error is None or error < best_error:
                            best_surface = i
                            best_coord = parametric_coordinate
                            best_error = error
                    else:
                        displacement = (point - extrema_point).reshape((-1,))
                        error = (np.linalg.norm(np.cross(displacement, direction)), np.linalg.norm(displacement))
                        if best_error is None:
                            best_surface = i
                            best_coord = parametric_coordinate
                            best_error = error
                        elif error[0] < best_error[0]*(1 + 1e-6) and error[1] < best_error[1]:
                            best_surface = i
                            best_coord = parametric_coordinate
                            best_error = error 
            results.append((best_surface, best_coord))
        else:
            if direction is None:
                lower_bounds = {i: function._compute_distance_bounds(point) for i, function in functions.items()}
                sorted_surfaces = sorted(lower_bounds.keys(), key=lambda x: lower_bounds[x])

            else:
                lower_bounds = {i: function._compute_distance_bounds(point, direction=direction) for i, function in functions.items()}
                distance_bounds = {i: function._compute_distance_bounds(point) for i, function in functions.items()}
                sorted_surfaces = sorted(lower_bounds.keys(), key=lambda x: (lower_bounds[x], distance_bounds[x]))

            # project onto the first surface
            best_surface = sorted_surfaces[0]
            function = functions[best_surface]
            best_coord = function.project(point.reshape(1,-1), direction=options['direction'], grid_search_density_parameter=options['grid_search_density_parameter'],
                                        max_newton_iterations=options['max_newton_iterations'], newton_tolerance=options['newton_tolerance'], do_pickles=False)
            projections_performed += 1

            if direction is None:
                best_error = np.linalg.norm(function.evaluate(best_coord, coefficients=function.coefficients.value) - point)
            else:
                function_value = function.evaluate(best_coord, coefficients=function.coefficients.value)
                displacement = (point - function_value).reshape((-1,))
                best_error = (np.linalg.norm(np.cross(displacement, direction)), np.linalg.norm(displacement)) # (directed distance, total distance)

            for name in sorted_surfaces:
                function = functions[name]
                bound = lower_bounds[name]
                # TODO: TEMPORARY DISABLING THE BREAK BECAUSE I'M RUNNING INTO AN ERROR WHEN I HAVE A SPARSE SET OF COEFFICIENTS
                if direction is None:
                    if bound > best_error:
                        projections_skipped += len(sorted_surfaces) - sorted_surfaces.index(name)
                        break
                else:
                    if bound > best_error[0]:
                        projections_skipped += len(sorted_surfaces) - sorted_surfaces.index(name)
                        break
                parametric_coordinate = function.project(point.reshape(1,-1), direction=options['direction'], grid_search_density_parameter=options['grid_search_density_parameter'],
                                                        max_newton_iterations=options['max_newton_iterations'], newton_tolerance=options['newton_tolerance'], do_pickles=False)
                projections_performed += 1
                if direction is None:
                    error = np.linalg.norm(function.evaluate(parametric_coordinate, coefficients=function.coefficients.value) - point)
                    if name in priority_inds:
                        error = error - priority_eps
                    if error < best_error:
                        best_surface = name
                        best_coord = parametric_coordinate
                        best_error = error
                else:
                    function_value = function.evaluate(parametric_coordinate, coefficients=function.coefficients.value)
                    displacement = (point - function_value).reshape((-1,))
                    error = (np.linalg.norm(np.cross(displacement, direction)), np.linalg.norm(displacement))
                    if name in priority_inds:
                        error[0] = error[0] - priority_eps
                        error[1] = error[1] - priority_eps
                    # TODO: make the 1e-6 a parameter
                    if error[0] < best_error[0]*(1 + 1e-6) and error[1] < best_error[1]:
                        best_surface = name
                        best_coord = parametric_coordinate
                        best_error = error 
            results.append((best_surface, best_coord))

    # print(f"Projections performed: {projections_performed}")
    # print(f"Projections skipped: {projections_skipped}")

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
    functions: dict[int,lfs.Function]
    function_names : dict[str] = None
    name : str = None
    space : lfs.FunctionSetSpace = None

    def __post_init__(self):
        if isinstance(self.functions, list):
            self.functions = {i:function for i, function in enumerate(self.functions)}

        if isinstance(self.function_names, list):
            self.function_names = {i:function_name for i, function_name in enumerate(self.function_names)}

        if self.function_names is None:
            self.function_names = {i:None for i in self.functions}
            for i, function in self.functions.items():
                self.function_names[i] = function.name

        if self.space is None:
            self.space = lfs.FunctionSetSpace(
                num_parametric_dimensions={i:function.space.num_parametric_dimensions for i, function in self.functions.items()},
                spaces={i:function.space for i, function in self.functions.items()})
            
    def find_surface_connections(self):
        perfect_connections = {}
        dependent_connections = {}
        search_n = 5
        ones = np.ones((search_n)).reshape((-1,1))
        zeros = np.zeros((search_n)).reshape((-1,1))
        lin = np.linspace(0, 1, search_n).reshape((-1,1))
        # find perfect connections
        function_evals = {}
        for i, function in self.functions.items():
            eval1 = function.evaluate(np.hstack((lin, zeros)), non_csdl=True)
            eval2 = function.evaluate(np.hstack((ones, lin)), non_csdl=True)
            eval3 = function.evaluate(np.hstack((lin, ones)), non_csdl=True)
            eval4 = function.evaluate(np.hstack((zeros, lin)), non_csdl=True)
            function_evals[i] = [eval1, eval2, eval3, eval4]
        
        connected_functions = set()
        for i in self.functions:
            evals = function_evals[i]
            for j in self.functions:
                if i == j:
                    continue
                if frozenset([i,j]) in connected_functions:
                    continue
                evals2 = function_evals[j]
                for k, eval1 in enumerate(evals):
                    for l, eval2 in enumerate(evals2):
                        if np.linalg.norm(eval1 - eval2) < 1e-6:
                            # points1 = [(i, eval1_i) for eval1_i in eval1]
                            # points2 = [(j, eval2_i) for eval2_i in eval2]
                            # points = np.vstack((eval1, eval2))
                            # self.project(points, plot=True)
                            perfect_connections[(i,k+1)] = (j,l+1)
                            # perfect_connections[(j,l+1)] = (i,k+1)
                            connected_functions.add(frozenset([i,j]))
                            break
                        if np.linalg.norm(eval1 - eval2[::-1]) < 1e-6:
                            perfect_connections[(i,k+1)] = (j,-l-1)
                            # perfect_connections[(j,l+1)] = (i,-k-1)
                            connected_functions.add(frozenset([i,j]))
                            break
        return perfect_connections

    def apply_surface_connections(self, geometry=None):
        connections = self.space.connections
        for face1, face2 in connections.items():
            function1 = self.functions[face1[0]]
            function2 = self.functions[face2[0]]
            coeffs1, coeffs2 = function1.space.stitch(face1[1], function1.coefficients, 
                                     function2.space, face2[1], function2.coefficients)
            function1.coefficients = coeffs1
            function2.coefficients = coeffs2




    def stack_coefficients(self) -> csdl.Variable:
        '''
        Stacks the coefficients of the functions in the function set.

        Returns
        -------
        coefficients : csdl.Variable
            The stacked coefficients of the functions in the function set.
        '''
        coefficients = []
        for i, function in self.functions.items():
            shape = function.coefficients.shape
            if len(shape) == 1:
                shape = (1, shape[0])
            if len(shape) >= 2:
                shape = (np.prod(shape[:-1]), shape[-1])
            coefficients.append([function.coefficients.reshape((shape))])
        coefficients = csdl.blockmat(coefficients)
        return coefficients

    def copy(self) -> lfs.FunctionSet:
        '''
        Copies the function set.

        Returns
        -------
        function_set : lfs.FunctionSet
            The copied function set.
        '''
        functions = {i:function.copy() for i, function in self.functions.items()}
        function_set = lfs.FunctionSet(functions=functions, function_names=self.function_names, name=self.name)
        return function_set

    def evaluate_normals(self, parametric_coordinates:list[tuple[int, np.ndarray]], plot:bool=False) -> csdl.Variable:
        '''
        Evaluates the normals of the function set at the given parametric coordinates.

        Parameters
        ----------
        parametric_coordinates : list[tuple[int, np.ndarray]] -- list length=num_points, tuple_length=2
            The coordinates at which to evaluate the function. The list elements correspond to the coordinate of each point.
            The tuple elements correspond to the index of the function and the parametric coordinates for that point.

        Returns
        -------
        normals : csdl.Variable
            The normals of the function set at the given coordinates.
        '''
        u_vectors = self.evaluate(parametric_coordinates, parametric_derivative_orders=(1,0))
        v_vectors = self.evaluate(parametric_coordinates, parametric_derivative_orders=(0,1))
        normals = csdl.cross(u_vectors, v_vectors, axis=1)
        normals = normals / (csdl.expand(csdl.norm(normals + 1e-8, axes=(1,)), (normals.shape), action='i->ij') + 1e-12)

        if plot:
            import vedo
            import lsdo_function_spaces as lfs
            scale = 1e-1
            points = self.evaluate(parametric_coordinates, non_csdl=True)
            plotting_elements = self.plot(opacity=0.8, show=False)
            varrows = vedo.Arrows(points, points+scale*normals.value, c='black')
            plotting_elements.append(varrows)
            lfs.show_plot(plotting_elements, 'normals')
        return normals

    def evaluate(self, parametric_coordinates:list[tuple[int, np.ndarray]], parametric_derivative_orders:list[tuple]=None,
                 plot:bool=False, non_csdl:bool=False) -> csdl.Variable:
        '''
        Evaluates the function.

        Parameters
        ----------
        parametric_coordinates : list[tuple[int, np.ndarray]] -- list length=num_points, tuple_length=2
            The coordinates at which to evaluate the function. The list elements correspond to the coordinate of each point.
            The tuple elements correspond to the index of the function and the parametric coordinates for that point.
            The parametric coordinates should be a numpy array of shape (num_parametric_dimensions,).
        parametric_derivative_orders : list[tuple] = None -- shape=(num_points,num_parametric_dimensions)
            The order of the parametric derivatives to evaluate. If None, the function itself is evaluated.
        plot : bool = False
            Whether or not to plot the function with the points from the result of the evaluation.
                non_csdl : bool = False
            If true, will run numpy computations instead of csdl computations, and return a numpy array.

        Returns
        -------
        function_values : csdl.Variable
            The function evaluated at the given coordinates.
        '''
        if isinstance(parametric_coordinates, tuple):
            parametric_coordinates = [parametric_coordinates]
        # if isinstance(parametric_derivative_orders, tuple):
        #     parametric_derivative_orders = [parametric_derivative_orders]

        # Process parametric coordinates to group them by which function they belong to
        function_indices = []
        function_parametric_coordinates = []
        for parametric_coordinate in parametric_coordinates:
            function_index, coordinates = parametric_coordinate
            function_indices.append(function_index)
            function_parametric_coordinates.append(coordinates)

        # Evaluate each function at the given coordinates
        basis_matrices = []
        coeff_vectors = []
        reorder_indices = []
        for i, function in self.functions.items():
            indices = np.where(np.array(function_indices) == i)[0]
            para_coords = np.array([function_parametric_coordinates[j] for j in indices]).reshape(-1, function.space.num_parametric_dimensions)
        #     if parametric_derivative_orders is not None:
        #         para_derivs = [parametric_derivative_orders[j] for j in indices]
                
        #         if len(para_derivs) >= 1:
        #             para_derivs = para_derivs[0] # TODO: Add support for a separate derivative order for each point!
        #     else:
        #         para_derivs = None
        #     if len(indices) > 0:
        #         function_values_list.append(function.evaluate(parametric_coordinates=para_coords,
        #                                              parametric_derivative_orders=para_derivs))
        #         functions_with_points.append(i)

        # # Arrange the function values back into the correct element of the array
        # if len(function_values_list) == 0:
        #     raise ValueError("No points were evaluated.")
        # if self.functions[functions_with_points[0]].num_physical_dimensions == 1:
        #     function_values = csdl.Variable(value=np.zeros((len(parametric_coordinates),)))

        # else:
        #     function_values = csdl.Variable(value=np.zeros((len(parametric_coordinates), function_values_list[0].shape[-1])))
        # for i, function_value in enumerate(function_values_list):
        #     indices = (np.array(function_indices) == functions_with_points[i]).nonzero()[0].tolist()
            # indices = list(np.where(np.array(function_indices) == i)[0])
            if len(indices) == 0:
                continue
            reorder_indices += indices.astype(int).tolist()
            para_coords = np.array([function_parametric_coordinates[j] for j in indices]).reshape(-1, function.space.num_parametric_dimensions)
            if parametric_derivative_orders is not None:
                para_derivs = parametric_derivative_orders
            else:
                para_derivs = None
            basis_matrix, coefficients = function.get_matrix_vector(parametric_coordinates=para_coords,
                                                                    parametric_derivative_orders=para_derivs,
                                                                    non_csdl=non_csdl)
            basis_matrices.append(basis_matrix)
            coeff_vectors.append(coefficients)

        if len(reorder_indices) != len(parametric_coordinates):
            raise ValueError("Some points were not evaluated.")


        basis_matrix = sps.block_diag(basis_matrices, format='csr')
        if len(coeff_vectors) == 0:
            raise ValueError("No points were evaluated.")
        elif len(coeff_vectors) == 1:
            coeff_vector = coeff_vectors[0]
        else:
            if non_csdl:
                coeff_vector = np.vstack(coeff_vectors)
            else:
                coeff_vector = csdl.vstack(coeff_vectors)

        if non_csdl:
            values = basis_matrix @ coeff_vector
        else:
            values = csdl.Variable(value=np.zeros((basis_matrix.shape[0], coeff_vector.shape[1])))
            for i in csdl.frange(coeff_vector.shape[1]):
                coefficients_column = coeff_vector[:,i].reshape((coeff_vector.shape[0],1))
                values = values.set(csdl.slice[:,i], csdl.sparse.matvec(basis_matrix, coefficients_column).reshape((basis_matrix.shape[0],)))

        indices_reorder = np.argsort(reorder_indices).tolist()
        function_values = values[indices_reorder]

        if plot:
            # Plot the function
            plotting_elements = self.plot(opacity=0.8, show=False)
            # Plot the evaluated points
            if non_csdl:
                value = function_values
            else:
                value = function_values.value
            lfs.plot_points(value, color='#C69214', size=10, additional_plotting_elements=plotting_elements)

        if not len(function_values.shape) == 1:
            if np.prod(function_values.shape) == function_values.shape[1] or np.prod(function_values.shape) == function_values.shape[0]:
                function_values = function_values.reshape((-1,))

        return function_values
    
    def integrate(self, area, grid_n=10, indices=None, quadrature_order=2) -> tuple[csdl.Variable, list[tuple[int, np.ndarray]]]:
        if indices is None:
            indices = list(self.functions)
        parametric_coordinates = []
        values = []
        # TODO: frange?
        for i in indices:
            function = self.functions[i]
            value, coords = function.integrate(area.functions[i], grid_n=grid_n, quadrature_order=quadrature_order)
            for j in range(len(coords)):
                parametric_coordinates.append((i,coords[j]))
            if len(value.shape) == 1:
                value = value.reshape((-1,1))
            values.append(value)
        values = csdl.vstack(values)
        return values, parametric_coordinates

    def refit(self, new_function_spaces:dict[lfs.FunctionSpace]|lfs.FunctionSpace, indices_of_functions_to_refit:list[int]=None, 
              grid_resolution:tuple=None,  parametric_coordinates:dict[tuple[int,np.ndarray]]=None,
              parametric_derivative_orders:list[np.ndarray]=None, regularization_parameter:float=None) -> lfs.FunctionSet:
        '''
        Refits functions in the function set. Either a grid resolution or parametric coordinates must be provided. 
        If both are provided, the parametric coordinates will be used. If derivatives are used, the parametric derivative orders must be provided.

        NOTE: this method will NOT overwrite the coefficients or function space in this object. 
        It will return a new function object with the refitted coefficients.

        Parameters
        ----------
        new_function_spaces : dict[ind, FunctionSpace] -- dictionary length=number of functions being refit
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
            indices_of_functions_to_refit = list(self.functions)

        if isinstance(new_function_spaces, lfs.FunctionSpace):
            new_function_spaces = {ind:new_function_spaces for ind in self.functions}

        if len(new_function_spaces) != len(indices_of_functions_to_refit):
            raise ValueError("The number of new function spaces must match the number of functions to refit. " +
                             f"({len(new_function_spaces)} != {len(indices_of_functions_to_refit)})")

        new_functions = {}
        for i, function in self.functions.items():
            if i in indices_of_functions_to_refit:
                new_functions[i] = function.refit(new_function_space=new_function_spaces[i], 
                                                    grid_resolution=grid_resolution,
                                                    parametric_coordinates=parametric_coordinates, 
                                                    parametric_derivative_orders=parametric_derivative_orders,
                                                    regularization_parameter=regularization_parameter)
            else:
                new_functions[i] = function

        new_function_set = lfs.FunctionSet(functions=new_functions, function_names=self.function_names)
        return new_function_set


    def project(self, points:np.ndarray, num_workers:int=None, direction:np.ndarray=None, grid_search_density_parameter:int=1, 
                max_newton_iterations:int=100, newton_tolerance:float=1e-6, plot:bool=False, extrema=False, force_reprojection=False,
                priority_inds=None, priority_eps=1e-3) -> csdl.Variable:
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
        extrema : bool = False
            Whether or not to project onto the extrema of the function.
        force_reprojection : bool = False
            Whether or not to force the projection to be recomputed.
        '''
        if num_workers is None:
            num_workers = lfs.num_workers

        if isinstance(points, csdl.Variable):
            points = points.value
        
        output = self._check_whether_to_load_projection(points, direction, 
                                                        grid_search_density_parameter, 
                                                        max_newton_iterations, 
                                                        newton_tolerance,
                                                        extrema,
                                                        priority_inds, priority_eps,
                                                        force_reprojection)
        if isinstance(output, list):
            parametric_coordinates = output
            if plot:
                projection_results = self.evaluate(parametric_coordinates).value
                plotting_elements = []
                plotting_elements.append(lfs.plot_points(points, color='#00ff00', size=10, opacity=0.6, show=False))
                # plotting_elements.append(lfs.plot_points(projection_results, color='#F5F0E6', size=10, show=False))
                plotting_elements.append(lfs.plot_points(projection_results, color='#ff0000', size=5, show=False))
                self.plot(opacity=0.3, additional_plotting_elements=plotting_elements, show=True)
            return parametric_coordinates
        else:
            name_space_dict, long_name_space = output
            
        if priority_inds is None:
            priority_inds = []

        options = {'direction': direction, 'grid_search_density_parameter': grid_search_density_parameter,
                   'max_newton_iterations': max_newton_iterations, 'newton_tolerance': newton_tolerance,
                   'extrema': extrema, 'priority_inds': priority_inds, 'priority_eps': priority_eps}
        


        if len(points.shape) == 1:
            points = points.reshape(1, -1)
        else:
            points = points.reshape(-1, points.shape[-1])

        # make sure there aren't more workers than points
        num_workers = min(num_workers, points.shape[0])

        # Divide the points into chunks and run in parallel
        if num_workers > 1:
            chunks = np.array_split(points, num_workers)

            global global_functions
            global_functions = self.functions
            global global_options
            global_options = options

            # pool = Pool(num_workers)
            # results = pool.map(find_best_surface_chunked, chunks)

            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                results = executor.map(find_best_surface_chunked, chunks)

            parametric_coordinates = []
            for result in results:
                parametric_coordinates.extend(result)
        else:
            parametric_coordinates = find_best_surface_chunked(points, self.functions, options)

        characters = string.ascii_letters + string.digits  # Alphanumeric characters
        # Generate a random string of the specified length
        random_string = ''.join(random.choice(characters) for _ in range(6))
        projections_folder = 'stored_files/projections'
        name_space_file_path = projections_folder + '/name_space_dict.pickle'
        name_space_dict[long_name_space] = random_string
        with open(name_space_file_path, 'wb+') as handle:
            pickle.dump(name_space_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(projections_folder + f'/{random_string}.pickle', 'wb+') as handle:
            pickle.dump(parametric_coordinates, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if plot:
            projection_results = self.evaluate(parametric_coordinates).value
            plotting_elements = []
            plotting_elements.append(lfs.plot_points(points, color='#00ff00', size=10, opacity=0.6, show=False))
            # plotting_elements.append(lfs.plot_points(projection_results, color='#F5F0E6', size=10, show=False))
            plotting_elements.append(lfs.plot_points(projection_results, color='#ff0000', size=5, show=False))
            self.plot(opacity=0.3, additional_plotting_elements=plotting_elements, show=True)

        return parametric_coordinates



    def _check_whether_to_load_projection(self, points:np.ndarray, direction:np.ndarray=None, grid_search_density_parameter:int=1,
                                          max_newton_iterations:int=100, newton_tolerance:float=1e-6, extrema:bool=False, 
                                          priority_inds=None, priority_eps=1e-3,
                                          force_reprojection:bool=False) -> bool:
        name_space = f'{self.name}'

        name_space = ''
        for function in self.functions.values():
            function_space = function.space

            coefficients = function.coefficients.value
            degree = function_space.degree
            coeff_shape = function_space.coefficients_shape
            knot_vectors_norm = round(np.linalg.norm(function_space.knots), 2)

            # if f'{target}_{str(degree)}_{str(coeff_shape)}_{str(knot_vectors_norm)}' in name_space:
            #     pass
            # else:
            name_space += f'_{str(coefficients)}_{str(degree)}_{str(coeff_shape)}_{str(knot_vectors_norm)}'
        
        long_name_space = name_space + f'_{str(points)}_{str(direction)}_{grid_search_density_parameter}_{max_newton_iterations}_{extrema}_{priority_inds}_{priority_eps}'

        projections_folder = 'stored_files/projections'
        name_space_file_path = projections_folder + '/name_space_dict.pickle'
        
        name_space_dict_file_path = Path(name_space_file_path)
        if name_space_dict_file_path.is_file():
            with open(name_space_file_path, 'rb') as handle:
                name_space_dict = pickle.load(handle)
        else:
            Path("stored_files/projections").mkdir(parents=True, exist_ok=True)
            name_space_dict = {}

        if long_name_space in name_space_dict.keys() and not force_reprojection:
            short_name_space = name_space_dict[long_name_space]
            saved_projections_file = projections_folder + f'/{short_name_space}.pickle'
            with open(saved_projections_file, 'rb') as handle:
                parametric_coordinates = pickle.load(handle)
                return parametric_coordinates
        else:
            Path("stored_files/projections").mkdir(parents=True, exist_ok=True)

            return name_space_dict, long_name_space


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
            function_indices = np.array(list(self.functions.keys()))

        if len(coefficients) != len(function_indices):
            raise ValueError("The number of coefficients must match the number of functions to set. " +
                             f"({len(coefficients)} != {len(function_indices)})")

        for i, function_index in enumerate(function_indices):
            coefficients_shape = self.functions[function_index].coefficients.shape
            self.functions[function_index].coefficients = coefficients[i].reshape(coefficients_shape)


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
            function_indices.append([self.function_names.keys()][[self.function_names.values()].index(function_name)])
        return function_indices
    

    def search_for_function_indices(self, search_strings:list[str], ignore_names:list[str]=[]) -> list[int]:
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
        for i, function_name in self.function_names.items():
            if any(s in function_name for s in search_strings) and not any(s in function_name for s in ignore_names):
                function_indices.append(i)
        return function_indices
    

    def create_subset(self, function_indices:list[int]=None, function_search_names:list[str]=None, ignore_names:list[str]=[], name:str=None) -> lfs.FunctionSet:
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
            function_indices += self.search_for_function_indices(search_strings=function_search_names, ignore_names=ignore_names)
        
        # Remove duplicates
        function_indices = list(set(function_indices))

        functions = {i:self.functions[i] for i in function_indices}
        function_names = {i:self.function_names[i] for i in function_indices}
        subset = lfs.FunctionSet(functions=functions, function_names=function_names, name=name)
        return subset

    def plot_but_good(self, opacity:float=1., color="777777", color_map:str='jet', surface_texture:str="", show:bool=True, grid_n=25):
        '''
        Plots the function set.

        Parameters
        -----------
        opactity : float = 1.
            The opacity of the plot. 0 is fully transparent and 1 is fully opaque.
        color : lfs.FunctionSet = None
            The FunctionSet to use to color the B-spline as.
        color_map : str = 'jet'
            The color map to use if the color is a function.
        surface_texture : str = ""
            The surface texture to determine how light bounces off the surface.
            See 

        '''
        import vedo
        from lsdo_function_spaces.utils.plotting_functions import get_surface_mesh

        vertices = []
        faces = []
        c_points = None
        for ind, function in self.functions.items():
            if isinstance(color, lfs.FunctionSet):
                function_color = color.functions[ind]
                fn_vertices, fn_faces, fn_c_points = get_surface_mesh(surface=function, color=function_color, grid_n=grid_n, offset=len(vertices))
                if c_points is None:
                    c_points = fn_c_points
                else:
                    c_points = np.hstack((c_points, fn_c_points))
            else:
                fn_vertices, fn_faces = get_surface_mesh(surface=function, grid_n=grid_n, offset=len(vertices))
            vertices.extend(fn_vertices)
            faces.extend(fn_faces)

        mesh = vedo.Mesh([vertices, faces]).opacity(opacity).lighting(surface_texture)

        if c_points is not None:
            mesh.cmap(color_map, c_points)
            mesh.add_scalarbar()
        else:
            mesh.color(color)

        if show:
            plotter = vedo.Plotter()
            plotter.show(mesh)
        return mesh


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
        import vedo

        # Then there must be a discrete index so loop over subfunctions and plot them
        plotting_elements = additional_plotting_elements.copy()
        color_min = None
        color_max = None
        for i, function in self.functions.items():
            if isinstance(color, lfs.FunctionSet):
                function_color = color.functions[i]
            else:    
                function_color = color
            out = function.plot(point_types=point_types, plot_types=plot_types, opacity=opacity, color=function_color, color_map=color_map,
                                               surface_texture=surface_texture, line_width=line_width,
                                               additional_plotting_elements=plotting_elements, show=False)
            if isinstance(out, tuple):
                plotting_elements = out[0]
                if color_min is None:
                    color_min = out[1]
                    color_max = out[2]
                else:
                    color_min = min(color_min, out[1])
                    color_max = max(color_max, out[2])
            else:
                plotting_elements = out
        if isinstance(color, lfs.FunctionSet):
            # plot some invisible points to get the scalar bar
            element = vedo.Points(np.zeros((2,3))).opacity(0)
            print('Color values', color_min, color_max)
            element.cmap(color_map, [color_min, color_max])
            plotting_elements.append(element)
            scalarbar = plotting_elements[-1].add_scalarbar()
        if show:
            if self.name is not None:
                lfs.show_plot(plotting_elements=plotting_elements, title=self.name)
            else:
                lfs.show_plot(plotting_elements=plotting_elements, title='Function Set Plot')
        return plotting_elements
    
    def create_parallel_space(self, function_space:lfs.FunctionSpace) -> lfs.FunctionSetSpace:
        '''
        Creates a parallel function set space with the given function space.

        Parameters
        ----------
        function_space : lfs.FunctionSpace
            The function space to create the parallel function set space with.

        Returns
        -------
        parallel_function_set : lfs.FunctionSetSpace
            The parallel function set space with the given function space.
        '''
        parallel_spaces = {}
        for i in self.functions.keys():
            parallel_spaces[i] = function_space
        parallel_function_set = lfs.FunctionSetSpace(num_parametric_dimensions=self.space.num_parametric_dimensions, 
                                                     spaces=parallel_spaces, connections=self.space.connections)
        return parallel_function_set

    def generate_parametric_grid(self, grid_resolution:tuple) -> list[tuple[int, np.ndarray]]:
        '''
        Generates a parametric grid for the function set.

        Parameters
        ----------
        grid_resolution : tuple
            The resolution of the grid in each parametric dimension.

        Returns
        -------
        parametric_grid : list[tuple[int, np.ndarray]]
            The grid of parametric coordinates for the FunctionSet (makes a grid of the specified resolution over each function in the set).
        '''

        return self.space.generate_parametric_grid(grid_resolution=grid_resolution)

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
