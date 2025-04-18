import numpy as np

def get_projection_squared_distances(points_1, points_2, direction):
    difference = points_2 - points_1
    if direction is None:
        squared_distances = np.sum((difference)**2, axis=-1)
    else:
        direction = direction/np.linalg.norm(direction)
        distances_along_axis = np.dot(difference, direction)
        squared_distances = np.sum((difference)**2, axis=-1) - distances_along_axis**2
    return squared_distances