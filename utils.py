import numpy as np

from numpy import ndarray
from sklearn.preprocessing import OneHotEncoder


def encode_categorical(data: ndarray) -> ndarray:
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    return encoder.fit_transform(data)


def euclidean_distance(point: ndarray, all_points: ndarray) -> ndarray:
    differences: ndarray = all_points - point
    point_idx: int = all_points.tolist().index(point.tolist())
    square_sum: ndarray = np.sum(np.square(differences), axis=1)
    square_sum[point_idx] = np.inf  # Skip self in computations
    return np.sqrt(square_sum)


def jaccard_coefficient(point: ndarray, all_points: ndarray) -> ndarray:
    intersection: ndarray = np.logical_and(all_points, point)
    point_idx: int = all_points.tolist().index(point.tolist())
    intersection[point_idx, :] = False  # Skip self in computations
    union: ndarray = np.logical_or(all_points, point)
    intersection: ndarray = np.sum(intersection, axis=1)
    union: ndarray = np.sum(union, axis=1)
    return intersection / union


