import numpy as np

from numpy import ndarray
from sklearn.preprocessing import OneHotEncoder


def encode_categorical(data: ndarray) -> ndarray:
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    return encoder.fit_transform(data)


def euclidean_distance(point_a: ndarray, point_b: ndarray) -> float:
    return np.sqrt(np.sum(np.square(point_a - point_b)))


def jaccard_coefficient(point_a: ndarray, point_b: ndarray) -> float:
    return np.sum(np.logical_and(point_a, point_b)) / np.sum(np.logical_or(point_a, point_b))



