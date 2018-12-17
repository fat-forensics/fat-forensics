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
vector_2d_struct_1 =  np.array([
        (-2, -1),
        (1, 1),
        (5, -7)],
    dtype=[('a', 'f'), ('b', 'i')])
vector_2d_struct_2 =  np.array([
        (2, 2),
        (7, 9),
        (-8, -7)],
    dtype=[('a', 'f'), ('b', 'i')])

distances_2d = np.array([
        [5, np.sqrt(181), np.sqrt(72)],
        [np.sqrt(2), 10, np.sqrt(145)],
        [np.sqrt(90), np.sqrt(260), 13]
    ])
distances_2d = vectorised_approx(distances_2d)

vector_str_1d_1 = np.array([['a'], ['b'], ['c'], ['de'], ['ghi']],
                           dtype=('a', '<U4'))
vector_str_1d_2 = np.array([['a'], ['g'], ['hi'], ['d'], ['ghi']],
                           dtype=('a', '<U4'))
distances_str_1d = np.array([0, 1, 1, 1, 0])

vector_str_all_1d_1 = np.array(vector_str_1d_1.tolist())
vector_str_all_1d_2 = np.array(vector_str_1d_2.tolist())

vector_str_2d_1 = np.array([
        ('x', 'Tina'),
        ('y', 'Brent'),
        ('z', 'Patricia')], 
        dtype=[('a', '<U2'), ('name', '<U9')]
    )
vector_str_2d_2 = np.array([
        ('a', 'Tina'),
        ('z', 'Ben'),
        ('x', 'Patricia')], 
        dtype=[('a', '<U2'), ('name', '<U9')]
    )

distances_str_2d = np.array([[1, 2, 0.75], [1.75, 2, 2], [1.75, 1, 1]])

vector_str_all_2d_1 = np.array(vector_str_2d_1.tolist())
vector_str_all_2d_2 = np.array(vector_str_2d_2.tolist())

def test_euclidean_distance():
    # Test unstructured arrays
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
    
    # Test structured arrays
    assert ((vector_2d_struct_1.shape[0] == vector_2d_struct_2.shape[0]) and
            (vector_2d_struct_2.shape[0] == distances_2d.shape[0]) and
            (distances_2d.shape[0] == distances_2d.shape[1]))
    for i in range(distances_2d.shape[0]):
        assert fud.euclidean_distance(vector_2d_struct_1[i], 
                                      vector_2d_struct_2[i]) == \
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

def test_hamming_distance_string():
    # Test basic hamming distance implementation
    assert(fud.hamming_distance_string('abcd', 'ca') == 0.5)
    assert(fud.hamming_distance_string('ab', 'ab') == 0)
    assert(fud.hamming_distance_string('a', 'bdeds') == 1)

def test_hamming_distance():
    # Test structured arrays
    assert ((vector_str_1d_1.shape[0] == vector_str_1d_2.shape[0]) and
            (vector_str_1d_2.shape[0] == distances_str_1d.shape[0]))
    for i in range(distances_str_1d.shape[0]):
        assert fud.hamming_distance(vector_str_1d_1[i], vector_str_1d_2[i]) \
                == distances_str_1d[i]
    
    # Test unstructured arrays
    assert ((vector_str_all_1d_1.shape[0] == vector_str_all_1d_2.shape[0]) and
            (vector_str_all_1d_2.shape[0] == distances_str_1d.shape[0]))
    for i in range(distances_str_1d.shape[0]):
        assert fud.hamming_distance(vector_str_all_1d_1[i], 
                                    vector_str_all_1d_2[i]) \
                == distances_str_1d[i]

def test_hamming_point_distance():
    # Test structured arrays
    assert ((vector_str_2d_1.shape[0] == vector_str_2d_2.shape[0]) and
            (vector_str_2d_2.shape[0] == distances_str_2d.shape[0]) and
            (distances_str_2d.shape[0] == distances_str_2d.shape[1]))

    for i in range(distances_str_2d.shape[0]):
        dist_1 = fud.hamming_point_distance(vector_str_2d_1[i], vector_str_2d_2)
        assert np.equal(dist_1, distances_str_2d[i,:]).all()
        dist_2 = fud.hamming_point_distance(vector_str_2d_2[i], vector_str_2d_1)
        assert np.equal(dist_2, distances_str_2d[:,i]).all()
    
    # Test unstructured arrays
    assert ((vector_str_all_2d_1.shape[0] == vector_str_all_2d_2.shape[0]) and
            (vector_str_all_2d_2.shape[0] == distances_str_2d.shape[0]) and
            (distances_str_2d.shape[0] == distances_str_2d.shape[1]))

    for i in range(distances_str_2d.shape[0]):
        dist_1 = fud.hamming_point_distance(vector_str_all_2d_1[i], 
                                            vector_str_all_2d_2)
        assert np.equal(dist_1, distances_str_2d[i,:]).all()
        dist_2 = fud.hamming_point_distance(vector_str_all_2d_2[i], 
                                            vector_str_all_2d_1)
        assert np.equal(dist_2, distances_str_2d[:,i]).all()

def test_hamming_vector_distance():
    # Test structured arrays
    assert ((vector_str_2d_1.shape[0] == vector_str_2d_2.shape[0]) and
            (vector_str_2d_2.shape[0] == distances_str_2d.shape[0]) and
            (distances_str_2d.shape[0] == distances_str_2d.shape[1]))

    dist_1 = fud.hamming_vector_distance(vector_str_2d_1, vector_str_2d_2)
    assert np.equal(dist_1, distances_str_2d).all()
    dist_2 = fud.hamming_vector_distance(vector_str_2d_2, vector_str_2d_1)
    assert np.equal(dist_2, distances_str_2d.T).all()

    # Test unstructured arrays
    assert ((vector_str_all_2d_1.shape[0] == vector_str_all_2d_2.shape[0]) and
            (vector_str_all_2d_2.shape[0] == distances_str_2d.shape[0]) and
            (distances_str_2d.shape[0] == distances_str_2d.shape[1]))

    dist_1 = fud.hamming_vector_distance(vector_str_all_2d_1, 
                                         vector_str_all_2d_2)
    assert np.equal(dist_1, distances_str_2d).all()
    dist_2 = fud.hamming_vector_distance(vector_str_all_2d_2, 
                                         vector_str_all_2d_1)
    assert np.equal(dist_2, distances_str_2d.T).all()
