"""
The :mod:`fatf.utils.distance` module includes all custom distance functions
for FAT-Forensics testing and examples.
"""

# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: BSD 3 clause

import numpy as np
from fatf.utils.validation import is_structured

def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculates the Euclidean distance between two data points.
    Args
    ----
    x : np.ndarray
        The first data point.

    y : np.ndarray
        The second data point.
    """
    if is_structured(x):
        columns = x.dtype.names
        x_minus_y = []
        for c in columns:
            x_minus_y.append(x[c] - y[c])
        distance = np.linalg.norm(np.array(x_minus_y))
    else:
        distance = np.linalg.norm(x - y)
    return distance

def euclidean_point_distance(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Calculates the element-wise Euclidean distance between a data point, y,
    and an array, X, of data points.

    Args
    ----
    y : np.ndarray
        A data point.
    X : np.ndarray
        An array of data points.
        data points.
    """
    distances = np.zeros((X.shape[0],))
    if is_structured(X):
        for i in range(0, X.shape[0]):
            distances[i] = euclidean_distance(X[i], y)
    else:
        distances = np.apply_along_axis(euclidean_distance, 1, X, y)
    return distances

def euclidean_vector_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Calculates the element-wise Euclidean distance between two arrays of data
    points.

    Args
    ----
    X : np.ndarray
        An array of data points.
    Y : np.ndarray
        An array of data points.

    Returns
    -------
    distance_matrix : np.array
        A point-wise Euclidean distance matrix between two arrays of data
        points.
    """
    distance_matrix = np.zeros((X.shape[0], Y.shape[0]))
    if is_structured(X):
        for i in range(0, X.shape[0]):
            distance_matrix[i] = euclidean_point_distance(X[i], Y)
    else:
        distance_matrix = np.apply_along_axis(euclidean_point_distance, 1, X, Y)
    return distance_matrix

def hamming_distance_string(x: str, y: str) -> float:
    if len(x) > len(y):
        x = x[len(x)-len(y):]
    elif len(y) > len(x):
        y = y[len(y)-len(x):]
    diff = 0
    for i in range(0, len(x)):
        if x[i] != y[i]:
            diff += 1
    return diff/len(x)

def hamming_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculates the Hamming distance between two data points.
    Args
    ----
    x : np.ndarray
        The first data point.

    y : np.ndarray
        The second data point.
    """
    distance = 0
    if is_structured(x):
        columns = x.dtype.names
        for c in columns:
            distance += hamming_distance_string(x[c], y[c])
    else:
        for i in range(0, x.shape[0]):
            distance += hamming_distance_string(x[i], y[i])
    return distance

def hamming_point_distance(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Calculates the element-wise Hamming distance between a data point, y,
    and an array, X, of data points.

    Args
    ----
    y : np.ndarray
        A data point.
    X : np.ndarray
        An array of data points.
        data points.
    """
    distances = np.zeros((X.shape[0],))
    if is_structured(X):
        for i in range(0, X.shape[0]):
            distances[i] = hamming_distance(X[i], y)
    else:
        distances = np.apply_along_axis(hamming_distance, 1, X, y)
    return distances


def hamming_vector_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Calculates the element-wise Hamming distance between two arrays of data
    points.

    Args
    ----
    X : np.ndarray
        An array of data points.
    Y : np.ndarray
        An array of data points.

    Returns
    -------
    distance_matrix : np.array
        A point-wise Euclidean distance matrix between two arrays of data
        points.
    """
    distance_matrix = np.zeros((X.shape[0], Y.shape[0]))
    if is_structured(X):
        for i in range(0, X.shape[0]):
            distance_matrix[i] = hamming_point_distance(X[i], Y)
    else:
        distance_matrix = np.apply_along_axis(hamming_point_distance, 1, X, Y)
    return distance_matrix
