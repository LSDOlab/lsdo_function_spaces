import numpy as np
import lsdo_function_spaces as lfs

def create_b_spline_from_corners(corners:np.ndarray, degree:tuple=(3,), num_coefficients:tuple=(10,), knot_vectors:tuple=None,
                                 name:str='b_spline_hyper_volume') -> lfs.Function:
    '''
    Creates a B-Spline volume from a set of corners.

    Parameters
    ----------
    corners : np.ndarray
        The corners of the hyper-volume. The shape of the corners array should be (nu1, nu2, nu3, ..., num_physical_dimensions) 
        where u_{i} corresponds to number of corners along a parametric dimensions.
    degree : tuple
        The degree of the B-Spline in each dimension.
    num_coefficients : tuple
        The number of coefficients between each corner in each dimension.
    knot_vectors : tuple
        The knot vectors for each dimension. If None, then open uniform knot vectors are generated.
    name : str = 'b_spline_hyper_volume'
        The name of the B-Spline.

    Returns
    -------
    b_spline : lfs.Function
        The B-Spline hyper-volume.
    '''

    num_dimensions = len(corners.shape)-1
    
    if isinstance(degree, int):
        degree = (degree,)*num_dimensions
    if len(degree) != num_dimensions:
        degree = tuple(np.tile(degree, num_dimensions))
    if isinstance(num_coefficients, int):
        num_coefficients = (num_coefficients,)*num_dimensions
    if len(num_coefficients) != num_dimensions:
        num_coefficients = tuple(np.tile(num_coefficients, num_dimensions))
    if knot_vectors is not None:
        if len(knot_vectors) != num_dimensions:
            knot_vectors = tuple(np.tile(knot_vectors, num_dimensions))

        total_knot_vector = []
        for knot_vector in knot_vectors:
            total_knot_vector.append(knot_vector)
        knot_vectors = np.hstack(total_knot_vector)
    else:
        knot_vectors = None # Just let the B-spline space generate the knot vectors.

    # Build up hyper-volume based on corners given
    previous_dimension_hyper_volume = corners
    dimension_hyper_volumes = corners.copy()
    for dimension_index in np.arange(num_dimensions, 0, -1)-1:
        dimension_hyper_volumes_shape = np.array(previous_dimension_hyper_volume.shape)
        dimension_num_hyper_volumes = dimension_hyper_volumes_shape[dimension_index]-1
        dimension_hyper_volumes_shape[dimension_index] = dimension_num_hyper_volumes * (num_coefficients[dimension_index]-1) + 1
        dimension_hyper_volumes_shape = tuple(dimension_hyper_volumes_shape)
        dimension_hyper_volumes = np.zeros(dimension_hyper_volumes_shape)

        # Move dimension index to front so we can index the correct dimension
        linspace_index_front = np.moveaxis(dimension_hyper_volumes, dimension_index, 0)
        previous_index_front = np.moveaxis(previous_dimension_hyper_volume, dimension_index, 0)
        include_endpoint = False
        # Perform interpolations
        for dimension_level_index in range(previous_dimension_hyper_volume.shape[dimension_index]-1):
            if dimension_level_index == previous_dimension_hyper_volume.shape[dimension_index]-2:   # last hyper-volume/segment along dimension
                include_endpoint = True
                dimension_hyper_volume_num_sections = num_coefficients[dimension_index]
                linspace_index_front[dimension_level_index*(dimension_hyper_volume_num_sections-1):] = \
                        np.linspace(previous_index_front[dimension_level_index], previous_index_front[dimension_level_index+1],
                                dimension_hyper_volume_num_sections, endpoint=include_endpoint)
                continue

            dimension_hyper_volume_num_sections = num_coefficients[dimension_index] - 1
            linspace_index_front[dimension_level_index*dimension_hyper_volume_num_sections:
                                 (dimension_level_index+1)*dimension_hyper_volume_num_sections] = \
                    np.linspace(previous_index_front[dimension_level_index], previous_index_front[dimension_level_index+1],
                            dimension_hyper_volume_num_sections, endpoint=include_endpoint)
        # Move axis back to proper location
        dimension_hyper_volumes = np.moveaxis(linspace_index_front, 0, dimension_index)
        previous_dimension_hyper_volume = dimension_hyper_volumes.copy()

    b_spline_space = lfs.BSplineSpace(num_parametric_dimensions=num_dimensions, degree=degree, 
                                  coefficients_shape=dimension_hyper_volumes.shape[:-1], knots=knot_vectors)
    b_spline = lfs.Function(space=b_spline_space, coefficients=dimension_hyper_volumes, name=name)

    return b_spline


def create_enclosure_block(points:np.ndarray, num_coefficients:tuple[int], degree:tuple[int], knot_vectors:tuple=None, 
                                      num_parametric_dimensions:int=3, name:str='hyper_volume') -> lfs.Function:
    '''
    Creates an nd volume that tightly fits around a set of entities.

    Parameters
    ----------
    points: np.ndarray
        The points that the hyper-volume should enclose. The shape of the points array should be (num_points, num_physical_dimensions).
    num_coefficients: tuple[int]
        The number of coefficients in each dimension.
    degree: tuple[int]
        The degree of the B-Spline in each dimension.
    knot_vectors: tuple = None
        The knot vectors for each dimension. If None, then open uniform knot vectors are generated.
    num_parametric_dimensions: int = 3
        The number of parametric dimensions.
    name: str = 'hyper_volume'
        The name of the hyper-volume.

    Returns
    -------
    hyper_volume: lfs.Function
        The hyper-volume that encloses the points.
    '''
    if isinstance(num_coefficients, int):
        num_coefficients = (num_coefficients,)*num_parametric_dimensions
    if isinstance(degree, int):
        degree = (degree,)*num_parametric_dimensions

    num_physical_dimensions = points.shape[-1]

    mins = np.min(points.reshape((-1,num_physical_dimensions)), axis=0).reshape((-1,1))
    maxs = np.max(points.reshape((-1,num_physical_dimensions)), axis=0).reshape((-1,1))

    mins_and_maxs = np.hstack((mins, maxs))

    corners_shape = (2,)*num_parametric_dimensions + (num_physical_dimensions,)
    corners = np.zeros(corners_shape)
    corners_flattened = np.zeros((np.prod(corners_shape),))
    physical_dimension_index = 0
    for i in range(len(corners_flattened)):
        parametric_dimension_counter = int(i/num_physical_dimensions)
        binary_parametric_dimension_counter = bin(parametric_dimension_counter)[2:].zfill(num_physical_dimensions)
        min_or_max = int(binary_parametric_dimension_counter[physical_dimension_index])
        corners_flattened[i] = mins_and_maxs[physical_dimension_index, min_or_max]

        physical_dimension_index += 1
        if physical_dimension_index == num_physical_dimensions:
            physical_dimension_index = 0

    corners = corners_flattened.reshape(corners_shape)

    hyper_volume = create_b_spline_from_corners(name=name, corners=corners, degree=degree, num_coefficients=num_coefficients,
            knot_vectors=knot_vectors)

    return hyper_volume