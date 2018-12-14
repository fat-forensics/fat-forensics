"""
The :mod:`fatf.utils.distance` module includes all custom distance functions
for FAT-Forensics testing and examples.
"""

# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: BSD 3 clause

import numpy as np

def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculates the Euclidean distance between two data points.
    Args
    ----
    x : np.ndarray
        The first data point.
    y : np.ndarray
        The second data point.
    Returns
    -------
    distance : float
        Euclidean distance between the two data points.
    """
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
    Returns
    -------
    distances : np.array
        A point-wise Euclidean distance between a data point and an array of
        data points.
    """
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
    distance_matrix = np.apply_along_axis(euclidean_point_distance, 1, X, Y)
    return distance_matrix
