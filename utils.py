import numpy as np

from numpy import ndarray
from sklearn.preprocessing import OneHotEncoder


def encode_categorical(data: ndarray) -> ndarray:
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    return encoder.fit_transform(data)


def euclidean_distance(point: ndarray, all_points: ndarray, point_in_all_points: bool = True) -> ndarray:
    differences: ndarray = all_points - point
    squared_sum: ndarray = np.sum(np.square(differences), axis=1)
    if point_in_all_points:
        point_idx: int = all_points.tolist().index(point.tolist())
        squared_sum[point_idx] = np.inf  # Skip self in computations
    return np.sqrt(squared_sum)


def jaccard_coefficient(point_a: ndarray, point_b: ndarray) -> float:
    return np.sum(np.logical_and(point_a, point_b)) / np.sum(np.logical_or(point_a, point_b))



