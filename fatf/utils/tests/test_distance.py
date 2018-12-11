import numpy as np
import pytest

import fatf.utils.distance as fud

vectorised_approx = np.vectorize(pytest.approx)

vector_1d_1 = np.array([[-2], [-1], [0], [1], [2]])
vector_1d_2 = np.array([[0], [1], [2], [-2], [-1]])
distances_1d = np.array([2, 2, 2, 3, 3])

vector_2d_1 = np.array([
        [-2, -1],
        [1, 1],
        [5,-7]
    ])
vector_2d_2 = np.array([
        [2, 2],
        [7, 9],
        [-8,-7]
    ])
distances_2d = np.array([
        [5, np.sqrt(181), np.sqrt(72)],
        [np.sqrt(2), 10, np.sqrt(145)],
        [np.sqrt(90), np.sqrt(260), 13]
    ])
distances_2d = vectorised_approx(distances_2d)

def test_euclidean_distance():
    assert ((vector_1d_1.shape[0] == vector_1d_2.shape[0]) and
            (vector_1d_2.shape[0] == distances_1d.shape[0]))
    for i in range(distances_1d.shape[0]):
        assert fud.euclidean_distance(vector_1d_1[i], vector_1d_2[i]) == \
                distances_1d[i]

    assert ((vector_2d_1.shape[0] == vector_2d_2.shape[0]) and
            (vector_2d_2.shape[0] == distances_2d.shape[0]) and
            (distances_2d.shape[0] == distances_2d.shape[1]))
    for i in range(distances_2d.shape[0]):
        assert fud.euclidean_distance(vector_2d_1[i], vector_2d_2[i]) == \
                distances_2d[i,i]

def test_euclidean_point_distance():
    assert ((vector_2d_1.shape[0] == vector_2d_2.shape[0]) and
            (vector_2d_2.shape[0] == distances_2d.shape[0]) and
            (distances_2d.shape[0] == distances_2d.shape[1]))

    for i in range(distances_2d.shape[0]):
        dist_1 = fud.euclidean_point_distance(vector_2d_1[i], vector_2d_2)
        assert np.equal(dist_1, distances_2d[i,:]).all()
        dist_2 = fud.euclidean_point_distance(vector_2d_2[i], vector_2d_1)
        assert np.equal(dist_2, distances_2d[:,i]).all()

def test_euclidean_vector_distance():
    assert ((vector_2d_1.shape[0] == vector_2d_2.shape[0]) and
            (vector_2d_2.shape[0] == distances_2d.shape[0]) and
            (distances_2d.shape[0] == distances_2d.shape[1]))

    dist_1 = fud.euclidean_vector_distance(vector_2d_1, vector_2d_2)
    assert np.equal(dist_1, distances_2d).all()
    dist_2 = fud.euclidean_vector_distance(vector_2d_2, vector_2d_1)
    assert np.equal(dist_2, distances_2d.T).all()
