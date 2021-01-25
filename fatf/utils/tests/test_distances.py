"""
Tests custom distance functions implemented in FAT Forensics.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

# pylint: disable=too-many-lines,invalid-name

import numpy as np

import pytest

import fatf.utils.distances as fud
from fatf.exceptions import IncorrectShapeError

VECTOR_0D = np.array(7, dtype=int)

VECTOR_1D_UNBASE_A1 = np.array([None, object(), None])

VECTOR_1D_NUMERICAL_A1 = np.array([1, 2, 3, 4, 5])
VECTOR_1D_NUMERICAL_A2 = np.array([5, 4, 3, 2, 1])
DISTANCE_1D_NUMERICAL_A = 6.325
DISTANCE_1D_A_BINARY = 4
DISTANCE_1D_A_BINARY_NORMALISED = 0.8

VECTOR_1D_NUMERICAL_B1 = np.array([4, 2])
VECTOR_1D_NUMERICAL_B2 = np.array([2, 4])
DISTANCE_1D_NUMERICAL_B = 2.828

VECTOR_1D_CATEGORICAL_A1 = np.array(['a', 'b', 'dac', 'x', 'y'])
VECTOR_1D_CATEGORICAL_A2 = np.array(['aa', 'bb', 'a', 'xyz', 'y'])
DISTANCE_1D_CATEGORICAL_A_BINARY = 4
DISTANCE_1D_CATEGORICAL_A_BINARY_NORMALISED = 0.8
DISTANCE_1D_CATEGORICAL_A_HAMMING = 7
DISTANCES_1D_CATEGORICAL_A_HAMMING = np.array([1, 1, 3, 2, 0])
DISTANCE_1D_CATEGORICAL_A_HAMMING_NORMALISED = 2.667
DISTANCES_1D_CATEGORICAL_A_HAMMING_NORMALISED = np.array(
    [0.5, 0.5, 1, 0.666, 0])

VECTORS_1D_A1 = np.array([[-2], [-1], [0], [1], [2]])
VECTORS_1D_A2 = np.array([[0], [1], [2], [-2], [-1]])
DISTANCES_1D_A = np.array([2, 2, 2, 3, 3])
DISTANCES_1D_A_BINARY = np.array([1, 1, 1, 1, 1])
DISTANCES_1D_A_BINARY_NORMALISED = DISTANCES_1D_A_BINARY

VECTORS_1D_CATEGORICAL_A1 = np.array([['a'], ['b'], ['c'], ['de'], ['ghi']])
VECTORS_1D_CATEGORICAL_A2 = np.array([['a'], ['g'], ['hi'], ['d'], ['ghi']])
DISTANCES_1D_CATEGORICAL_A = np.array([0, 1, 2, 1, 0])
DISTANCES_1D_CATEGORICAL_A_NORMALISED = np.array([0, 1, 1, 0.5, 0])
DISTANCES_1D_CATEGORICAL_A_BINARY = np.array([0, 1, 1, 1, 0])
DISTANCES_1D_CATEGORICAL_A_BINARY_NORMALISED = np.array([0, 1, 1, 1, 0])

VECTOR_2D_NUMERICAL_A1 = np.array([[-2, -1], [1, 1], [5, -7]])
VECTOR_2D_NUMERICAL_A2 = np.array([[2, 2], [7, 9], [-8, -7]])
VECTOR_2D_NUMERICAL_STRUCT_A1 = np.array([(-2, -1), (1, 1), (5, -7)],
                                         dtype=[('a', 'f'), ('b', 'i')])
VECTOR_2D_NUMERICAL_STRUCT_A2 = np.array([(2, 2), (7, 9), (-8, -7)],
                                         dtype=[('a', 'f'), ('b', 'i')])
DISTANCES_2D_NUMERICAL_A = np.array([
    [5, np.sqrt(181), np.sqrt(72)],
    [np.sqrt(2), 10, np.sqrt(145)],
    [np.sqrt(90), np.sqrt(260), 13]
])  # yapf: disable
DISTANCES_2D_NUMERICAL_A_BINARY = np.array([[2, 2, 2], [2, 2, 2], [2, 2, 1]])
DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED = np.array([
    [1, 1, 1.0],
    [1, 1, 1.0],
    [1, 1, 0.5]
])  # yapf: disable

VECTOR_2D_CATEGORICAL_A1 = np.array([['x', 'Tina'],
                                     ['y', 'Brent'],
                                     ['z', 'Patricia']])  # yapf: disable
VECTOR_2D_CATEGORICAL_A2 = np.array([['a', 'Tina'],
                                     ['z', 'Ben'],
                                     ['x', 'Patricia']])  # yapf: disable
VECTOR_2D_CATEGORICAL_STRUCT_A1 = np.array(
    [('x', 'Tina'),
     ('y', 'Brent'),
     ('z', 'Patricia')],
    dtype=[('a', '<U1'), ('name', '<U8')])  # yapf: disable
VECTOR_2D_CATEGORICAL_STRUCT_A2 = np.array(
    [('a', 'Tina'),
     ('z', 'Ben'),
     ('x', 'Patricia')],
    dtype=[('a', '<U1'), ('name', '<U8')])  # yapf: disable
DISTANCES_2D_CATEGORICAL_A = np.array([[1 + 0, 1 + 3, 0 + 8],
                                       [1 + 5, 1 + 4, 1 + 8],
                                       [1 + 8, 0 + 8, 1 + 0]])
DISTANCES_2D_CATEGORICAL_A_NORMALISED = np.array([[1 + 0, 1 + 0.75, 0 + 1],
                                                  [1 + 1, 1 + 0.80, 1 + 1],
                                                  [1 + 1, 0 + 1.00, 1 + 0]])
DISTANCES_2D_CATEGORICAL_A_BINARY = np.array([[1, 2, 1],
                                              [2, 2, 2],
                                              [2, 1, 1]])  # yapf: disable
DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED = np.array([[0.5, 1.0, 0.5],
                                                         [1.0, 1.0, 1.0],
                                                         [1.0, 0.5, 0.5]])


def test_euclidean_distance():
    """
    Tests :func:`fatf.utils.distances.euclidean_distance`.
    """
    shape_error_x = 'The x array should be 1-dimensional.'
    shape_error_y = 'The y array should be 1-dimensional.'
    value_error_x = 'The x array should be purely numerical.'
    value_error_y = 'The y array should be purely numerical.'
    shape_error_xy = 'The x and y arrays should have the same length.'

    # x or y is not 1D
    with pytest.raises(IncorrectShapeError) as exin:
        fud.euclidean_distance(VECTOR_0D, VECTOR_2D_NUMERICAL_A1)
    assert str(exin.value) == shape_error_x
    with pytest.raises(IncorrectShapeError) as exin:
        fud.euclidean_distance(VECTOR_2D_NUMERICAL_A1,
                               VECTOR_1D_CATEGORICAL_A1)
    assert str(exin.value) == shape_error_x
    with pytest.raises(IncorrectShapeError) as exin:
        fud.euclidean_distance(VECTOR_1D_CATEGORICAL_A1, VECTOR_0D)
    assert str(exin.value) == shape_error_y
    with pytest.raises(IncorrectShapeError) as exin:
        fud.euclidean_distance(VECTOR_1D_CATEGORICAL_A1,
                               VECTOR_2D_NUMERICAL_A1)
    assert str(exin.value) == shape_error_y

    # x or y is not numerical
    with pytest.raises(ValueError) as exin:
        fud.euclidean_distance(VECTOR_1D_CATEGORICAL_A1,
                               VECTOR_1D_NUMERICAL_A1)
    assert str(exin.value) == value_error_x
    with pytest.raises(ValueError) as exin:
        fud.euclidean_distance(VECTOR_1D_NUMERICAL_A1,
                               VECTOR_1D_CATEGORICAL_A1)
    assert str(exin.value) == value_error_y

    # The shape of input arrays does not match
    with pytest.raises(IncorrectShapeError) as exin:
        fud.euclidean_distance(VECTOR_1D_NUMERICAL_A1, VECTOR_1D_NUMERICAL_B1)
    assert str(exin.value) == shape_error_xy
    with pytest.raises(IncorrectShapeError) as exin:
        fud.euclidean_distance(VECTOR_1D_NUMERICAL_A1,
                               VECTOR_2D_NUMERICAL_STRUCT_A1[0])
    assert str(exin.value) == shape_error_xy

    # Test unstructured arrays
    assert ((VECTORS_1D_A1.shape[0] == VECTORS_1D_A2.shape[0])
            and (VECTORS_1D_A2.shape[0] == DISTANCES_1D_A.shape[0]))
    for i in range(DISTANCES_1D_A.shape[0]):
        dist = fud.euclidean_distance(VECTORS_1D_A1[i], VECTORS_1D_A2[i])
        assert dist == DISTANCES_1D_A[i]

    dist = fud.euclidean_distance(VECTOR_1D_NUMERICAL_A1,
                                  VECTOR_1D_NUMERICAL_A2)
    assert dist == pytest.approx(DISTANCE_1D_NUMERICAL_A, rel=1e-3)

    dist = fud.euclidean_distance(VECTOR_1D_NUMERICAL_B1,
                                  VECTOR_1D_NUMERICAL_B2)
    assert dist == pytest.approx(DISTANCE_1D_NUMERICAL_B, rel=1e-3)

    assert (
        (VECTOR_2D_NUMERICAL_A1.shape[0] == VECTOR_2D_NUMERICAL_A2.shape[0])
        and
        (VECTOR_2D_NUMERICAL_A2.shape[0] == DISTANCES_2D_NUMERICAL_A.shape[0])
        and
        (DISTANCES_2D_NUMERICAL_A.shape[0] == DISTANCES_2D_NUMERICAL_A.shape[1]
         ))
    for i in range(DISTANCES_2D_NUMERICAL_A.shape[0]):
        dist = fud.euclidean_distance(VECTOR_2D_NUMERICAL_A1[i],
                                      VECTOR_2D_NUMERICAL_A2[i])
        assert DISTANCES_2D_NUMERICAL_A[i, i] == pytest.approx(dist)

    # Test structured arrays
    assert ((VECTOR_2D_NUMERICAL_STRUCT_A1.shape[0] ==
             VECTOR_2D_NUMERICAL_STRUCT_A2.shape[0])
            and (VECTOR_2D_NUMERICAL_STRUCT_A2.shape[0] ==
                 DISTANCES_2D_NUMERICAL_A.shape[0])
            and (DISTANCES_2D_NUMERICAL_A.shape[0] ==
                 DISTANCES_2D_NUMERICAL_A.shape[1]))
    for i in range(DISTANCES_2D_NUMERICAL_A.shape[0]):
        dist = fud.euclidean_distance(VECTOR_2D_NUMERICAL_STRUCT_A1[i],
                                      VECTOR_2D_NUMERICAL_STRUCT_A2[i])
        assert DISTANCES_2D_NUMERICAL_A[i, i] == pytest.approx(dist)

    # Test unstructured-structured mixture
    assert (
        (VECTOR_2D_NUMERICAL_A1.shape[0] == VECTOR_2D_NUMERICAL_A2.shape[0])
        and (VECTOR_2D_NUMERICAL_A2.shape[0] ==
             VECTOR_2D_NUMERICAL_STRUCT_A1.shape[0])
        and (VECTOR_2D_NUMERICAL_STRUCT_A1.shape[0] ==
             VECTOR_2D_NUMERICAL_STRUCT_A2.shape[0])
        and (VECTOR_2D_NUMERICAL_STRUCT_A2.shape[0] ==
             DISTANCES_2D_NUMERICAL_A.shape[0])
        and (DISTANCES_2D_NUMERICAL_A.shape[0] ==
             DISTANCES_2D_NUMERICAL_A.shape[1]))  # yapf: disable
    for i in range(DISTANCES_2D_NUMERICAL_A.shape[0]):
        dist = fud.euclidean_distance(VECTOR_2D_NUMERICAL_STRUCT_A1[i],
                                      VECTOR_2D_NUMERICAL_A2[i])
        assert DISTANCES_2D_NUMERICAL_A[i, i] == pytest.approx(dist)
        dist = fud.euclidean_distance(VECTOR_2D_NUMERICAL_A1[i],
                                      VECTOR_2D_NUMERICAL_STRUCT_A2[i])
        assert DISTANCES_2D_NUMERICAL_A[i, i] == pytest.approx(dist)

    # Test euclidean distance for 1-element voids
    dist = fud.euclidean_distance(VECTOR_2D_NUMERICAL_STRUCT_A1[['a']][0],
                                  VECTOR_2D_NUMERICAL_STRUCT_A2[['a']][0])
    assert dist == 4


def test_euclidean_point_distance():
    """
    Tests :func:`fatf.utils.distances.euclidean_point_distance`.
    """
    shape_error_y = 'The y array should be 1-dimensional.'
    shape_error_X = 'The X array should be 2-dimensional.'
    value_error_y = 'The y array should be purely numerical.'
    value_error_X = 'The X array should be purely numerical.'
    shape_error_yX = ('The number of columns in the X array should the same '
                      'as the number of elements in the y array.')

    # X is not 2D or y is not 1D
    with pytest.raises(IncorrectShapeError) as exin:
        fud.euclidean_point_distance(VECTOR_0D, VECTOR_2D_NUMERICAL_A1)
    assert str(exin.value) == shape_error_y
    with pytest.raises(IncorrectShapeError) as exin:
        fud.euclidean_point_distance(VECTOR_2D_NUMERICAL_A1,
                                     VECTOR_1D_CATEGORICAL_A1)
    assert str(exin.value) == shape_error_y
    with pytest.raises(IncorrectShapeError) as exin:
        fud.euclidean_point_distance(VECTOR_1D_CATEGORICAL_A1, VECTOR_0D)
    assert str(exin.value) == shape_error_X
    with pytest.raises(IncorrectShapeError) as exin:
        fud.euclidean_point_distance(VECTOR_1D_CATEGORICAL_A1,
                                     VECTOR_1D_CATEGORICAL_A1)
    assert str(exin.value) == shape_error_X

    # X or y is not numerical
    with pytest.raises(ValueError) as exin:
        fud.euclidean_point_distance(VECTOR_1D_CATEGORICAL_A1,
                                     VECTOR_2D_NUMERICAL_A1)
    assert str(exin.value) == value_error_y
    with pytest.raises(ValueError) as exin:
        fud.euclidean_point_distance(VECTOR_1D_NUMERICAL_A1,
                                     VECTOR_2D_CATEGORICAL_STRUCT_A1)
    assert str(exin.value) == value_error_X

    # The shape of input arrays does not match
    with pytest.raises(IncorrectShapeError) as exin:
        fud.euclidean_point_distance(VECTOR_1D_NUMERICAL_A1,
                                     VECTOR_2D_NUMERICAL_A1)
    assert str(exin.value) == shape_error_yX
    with pytest.raises(IncorrectShapeError) as exin:
        fud.euclidean_point_distance(VECTOR_2D_NUMERICAL_STRUCT_A1[0],
                                     DISTANCES_2D_NUMERICAL_A)
    assert str(exin.value) == shape_error_yX

    # Test unstructured arrays
    assert (
        (VECTOR_2D_NUMERICAL_A1.shape[0] == DISTANCES_2D_NUMERICAL_A.shape[0])
        and
        (VECTOR_2D_NUMERICAL_A2.shape[0] == DISTANCES_2D_NUMERICAL_A.shape[1]))
    for i in range(VECTOR_2D_NUMERICAL_A1.shape[0]):
        dist = fud.euclidean_point_distance(VECTOR_2D_NUMERICAL_A1[i],
                                            VECTOR_2D_NUMERICAL_A2)
        assert np.isclose(
            DISTANCES_2D_NUMERICAL_A[i, :], dist, rtol=1e-3).all()

    assert (
        (VECTOR_2D_NUMERICAL_A2.shape[0] == DISTANCES_2D_NUMERICAL_A.shape[1])
        and
        (VECTOR_2D_NUMERICAL_A1.shape[0] == DISTANCES_2D_NUMERICAL_A.shape[0]))
    for i in range(VECTOR_2D_NUMERICAL_A2.shape[0]):
        dist = fud.euclidean_point_distance(VECTOR_2D_NUMERICAL_A2[i],
                                            VECTOR_2D_NUMERICAL_A1)
        assert np.isclose(
            DISTANCES_2D_NUMERICAL_A[:, i], dist, rtol=1e-3).all()

    # Test structured arrays
    assert ((VECTOR_2D_NUMERICAL_STRUCT_A1.shape[0] ==
             DISTANCES_2D_NUMERICAL_A.shape[0])
            and (VECTOR_2D_NUMERICAL_STRUCT_A2.shape[0] ==
                 DISTANCES_2D_NUMERICAL_A.shape[1]))
    for i in range(VECTOR_2D_NUMERICAL_STRUCT_A1.shape[0]):
        dist = fud.euclidean_point_distance(VECTOR_2D_NUMERICAL_STRUCT_A1[i],
                                            VECTOR_2D_NUMERICAL_STRUCT_A2)
        assert np.isclose(
            DISTANCES_2D_NUMERICAL_A[i, :], dist, rtol=1e-3).all()

    assert ((VECTOR_2D_NUMERICAL_STRUCT_A2.shape[0] ==
             DISTANCES_2D_NUMERICAL_A.shape[1])
            and (VECTOR_2D_NUMERICAL_STRUCT_A1.shape[0] ==
                 DISTANCES_2D_NUMERICAL_A.shape[0]))
    for i in range(VECTOR_2D_NUMERICAL_STRUCT_A2.shape[0]):
        dist = fud.euclidean_point_distance(VECTOR_2D_NUMERICAL_STRUCT_A2[i],
                                            VECTOR_2D_NUMERICAL_STRUCT_A1)
        assert np.isclose(
            DISTANCES_2D_NUMERICAL_A[:, i], dist, rtol=1e-3).all()

    # Test unstructured-structured mixture
    assert (
        (VECTOR_2D_NUMERICAL_A1.shape[0] == DISTANCES_2D_NUMERICAL_A.shape[0])
        and (VECTOR_2D_NUMERICAL_STRUCT_A2.shape[0] ==
             DISTANCES_2D_NUMERICAL_A.shape[1]))
    for i in range(VECTOR_2D_NUMERICAL_A1.shape[0]):
        dist = fud.euclidean_point_distance(VECTOR_2D_NUMERICAL_A1[i],
                                            VECTOR_2D_NUMERICAL_STRUCT_A2)
        assert np.isclose(
            DISTANCES_2D_NUMERICAL_A[i, :], dist, rtol=1e-3).all()

    assert (
        (VECTOR_2D_NUMERICAL_A2.shape[0] == DISTANCES_2D_NUMERICAL_A.shape[1])
        and (VECTOR_2D_NUMERICAL_STRUCT_A1.shape[0] ==
             DISTANCES_2D_NUMERICAL_A.shape[0]))
    for i in range(VECTOR_2D_NUMERICAL_A2.shape[0]):
        dist = fud.euclidean_point_distance(VECTOR_2D_NUMERICAL_A2[i],
                                            VECTOR_2D_NUMERICAL_STRUCT_A1)
        assert np.isclose(
            DISTANCES_2D_NUMERICAL_A[:, i], dist, rtol=1e-3).all()

    # Test euclidean point distance for 1-element voids
    dist = fud.euclidean_point_distance(
        VECTOR_2D_NUMERICAL_STRUCT_A1[['a']][0],
        VECTOR_2D_NUMERICAL_STRUCT_A2[['a']])
    assert np.array_equal(dist, [4, 9, 6])


def test_euclidean_array_distance():
    """
    Tests :func:`fatf.utils.distances.euclidean_array_distance`.
    """
    shape_error_X = 'The X array should be 2-dimensional.'
    shape_error_Y = 'The Y array should be 2-dimensional.'
    value_error_X = 'The X array should be purely numerical.'
    value_error_Y = 'The Y array should be purely numerical.'
    shape_error_XY = ('The number of columns in the X array should the same '
                      'as the number of columns in Y array.')

    # X or Y is not 2D
    with pytest.raises(IncorrectShapeError) as exin:
        fud.euclidean_array_distance(VECTOR_1D_CATEGORICAL_A1, VECTOR_0D)
    assert str(exin.value) == shape_error_X
    with pytest.raises(IncorrectShapeError) as exin:
        fud.euclidean_array_distance(VECTOR_0D, VECTOR_2D_NUMERICAL_A1)
    assert str(exin.value) == shape_error_X
    with pytest.raises(IncorrectShapeError) as exin:
        fud.euclidean_array_distance(VECTOR_2D_NUMERICAL_A1, VECTOR_0D)
    assert str(exin.value) == shape_error_Y
    with pytest.raises(IncorrectShapeError) as exin:
        fud.euclidean_array_distance(VECTOR_2D_NUMERICAL_STRUCT_A2,
                                     VECTOR_1D_CATEGORICAL_A1)
    assert str(exin.value) == shape_error_Y

    # X or Y is not numerical
    with pytest.raises(ValueError) as exin:
        fud.euclidean_array_distance(VECTOR_2D_CATEGORICAL_STRUCT_A1,
                                     VECTOR_2D_NUMERICAL_A1)
    assert str(exin.value) == value_error_X
    with pytest.raises(ValueError) as exin:
        fud.euclidean_array_distance(VECTOR_2D_NUMERICAL_A1,
                                     VECTOR_2D_CATEGORICAL_STRUCT_A2)
    assert str(exin.value) == value_error_Y

    # The shape of input arrays does not match
    with pytest.raises(IncorrectShapeError) as exin:
        fud.euclidean_array_distance(DISTANCES_2D_NUMERICAL_A,
                                     VECTOR_2D_NUMERICAL_A1)
    assert str(exin.value) == shape_error_XY
    with pytest.raises(IncorrectShapeError) as exin:
        fud.euclidean_array_distance(VECTOR_2D_NUMERICAL_STRUCT_A1,
                                     DISTANCES_2D_NUMERICAL_A)
    assert str(exin.value) == shape_error_XY

    # Test unstructured arrays
    assert (
        (VECTOR_2D_NUMERICAL_A1.shape[0] == DISTANCES_2D_NUMERICAL_A.shape[0])
        and
        (VECTOR_2D_NUMERICAL_A2.shape[0] == DISTANCES_2D_NUMERICAL_A.shape[1]))
    dist = fud.euclidean_array_distance(VECTOR_2D_NUMERICAL_A1,
                                        VECTOR_2D_NUMERICAL_A2)
    assert np.isclose(DISTANCES_2D_NUMERICAL_A, dist, rtol=1e-3).all()
    dist = fud.euclidean_array_distance(VECTOR_2D_NUMERICAL_A2,
                                        VECTOR_2D_NUMERICAL_A1)
    assert np.isclose(DISTANCES_2D_NUMERICAL_A.T, dist, rtol=1e-3).all()

    # Test structured arrays
    assert ((VECTOR_2D_NUMERICAL_STRUCT_A1.shape[0] ==
             DISTANCES_2D_NUMERICAL_A.shape[0])
            and (VECTOR_2D_NUMERICAL_STRUCT_A2.shape[0] ==
                 DISTANCES_2D_NUMERICAL_A.shape[1]))
    dist = fud.euclidean_array_distance(VECTOR_2D_NUMERICAL_STRUCT_A1,
                                        VECTOR_2D_NUMERICAL_STRUCT_A2)
    assert np.isclose(DISTANCES_2D_NUMERICAL_A, dist, rtol=1e-3).all()
    dist = fud.euclidean_array_distance(VECTOR_2D_NUMERICAL_STRUCT_A2,
                                        VECTOR_2D_NUMERICAL_STRUCT_A1)
    assert np.isclose(DISTANCES_2D_NUMERICAL_A.T, dist, rtol=1e-3).all()

    # Test unstructured-structured mixture
    assert ((VECTOR_2D_NUMERICAL_A1.shape[0] ==
             DISTANCES_2D_NUMERICAL_A.shape[0])
            and (VECTOR_2D_NUMERICAL_STRUCT_A2.shape[0] ==
                 DISTANCES_2D_NUMERICAL_A.shape[1]))  # yapf: disable
    dist = fud.euclidean_array_distance(VECTOR_2D_NUMERICAL_A1,
                                        VECTOR_2D_NUMERICAL_STRUCT_A2)
    assert np.isclose(DISTANCES_2D_NUMERICAL_A, dist, rtol=1e-3).all()

    assert ((VECTOR_2D_NUMERICAL_STRUCT_A1.shape[0] ==
             DISTANCES_2D_NUMERICAL_A.shape[0])
            and (VECTOR_2D_NUMERICAL_A2.shape[0] ==
                 DISTANCES_2D_NUMERICAL_A.shape[1]))  # yapf: disable
    dist = fud.euclidean_array_distance(VECTOR_2D_NUMERICAL_A2,
                                        VECTOR_2D_NUMERICAL_STRUCT_A1)
    assert np.isclose(DISTANCES_2D_NUMERICAL_A.T, dist, rtol=1e-3).all()


def test_hamming_distance_base():
    """
    Tests :func:`fatf.utils.distances.hamming_distance_base`.
    """
    type_error_x = 'x should be a string.'
    type_error_y = 'y should be a string.'
    value_error = ('Input strings differ in length and the equal_length '
                   'parameter forces them to be of equal length.')

    # Incorrect type
    with pytest.raises(TypeError) as exin:
        fud.hamming_distance_base(None, 'pass')
    assert str(exin.value) == type_error_x
    with pytest.raises(TypeError) as exin:
        fud.hamming_distance_base(u'pass', None)
    assert str(exin.value) == type_error_y

    # Forced equal length
    with pytest.raises(ValueError) as exin:
        fud.hamming_distance_base(u'pass', 'lengthy_string', equal_length=True)
    assert str(exin.value) == value_error

    dist = fud.hamming_distance_base(u'pass', 'pas_string', equal_length=False)
    assert dist == 7

    # Test various inputs
    assert (
        (VECTOR_1D_CATEGORICAL_A1.shape[0] ==
         VECTOR_1D_CATEGORICAL_A2.shape[0])
        and (VECTOR_1D_CATEGORICAL_A2.shape[0] ==
             DISTANCES_1D_CATEGORICAL_A_HAMMING.shape[0])
        and (DISTANCES_1D_CATEGORICAL_A_HAMMING.shape[0] ==
             DISTANCES_1D_CATEGORICAL_A_HAMMING_NORMALISED.shape[0])
    )  # yapf: disable

    # Test not normalised -- different length
    for i in range(VECTOR_1D_CATEGORICAL_A1.shape[0]):
        dist = fud.hamming_distance_base(VECTOR_1D_CATEGORICAL_A1[i],
                                         VECTOR_1D_CATEGORICAL_A2[i])
        assert DISTANCES_1D_CATEGORICAL_A_HAMMING[i] == dist
        dist = fud.hamming_distance_base(
            VECTOR_1D_CATEGORICAL_A1[i],
            VECTOR_1D_CATEGORICAL_A2[i],
            normalise=False,
            equal_length=False)
        assert DISTANCES_1D_CATEGORICAL_A_HAMMING[i] == dist

    # Test normalised -- different length
    for i in range(VECTOR_1D_CATEGORICAL_A1.shape[0]):
        dist = fud.hamming_distance_base(
            VECTOR_1D_CATEGORICAL_A1[i],
            VECTOR_1D_CATEGORICAL_A2[i],
            normalise=True)
        assert (
            DISTANCES_1D_CATEGORICAL_A_HAMMING_NORMALISED[i] == pytest.approx(
                dist, rel=1e-3))
        dist = fud.hamming_distance_base(
            VECTOR_1D_CATEGORICAL_A1[i],
            VECTOR_1D_CATEGORICAL_A2[i],
            normalise=True,
            equal_length=False)
        assert (
            DISTANCES_1D_CATEGORICAL_A_HAMMING_NORMALISED[i] == pytest.approx(
                dist, rel=1e-3))

    # Test not normalised -- equal length
    for i in range(VECTOR_1D_CATEGORICAL_A1.shape[0]):
        if (len(VECTOR_1D_CATEGORICAL_A1[i]) == len(
                VECTOR_1D_CATEGORICAL_A2[i])):
            dist = fud.hamming_distance_base(
                VECTOR_1D_CATEGORICAL_A1[i],
                VECTOR_1D_CATEGORICAL_A2[i],
                equal_length=True)
            assert DISTANCES_1D_CATEGORICAL_A_HAMMING[i] == dist
            dist = fud.hamming_distance_base(
                VECTOR_1D_CATEGORICAL_A1[i],
                VECTOR_1D_CATEGORICAL_A2[i],
                normalise=False,
                equal_length=True)
            assert DISTANCES_1D_CATEGORICAL_A_HAMMING[i] == dist
        else:
            with pytest.raises(ValueError) as exin:
                fud.hamming_distance_base(
                    VECTOR_1D_CATEGORICAL_A1[i],
                    VECTOR_1D_CATEGORICAL_A2[i],
                    equal_length=True)
            assert str(exin.value) == value_error
            with pytest.raises(ValueError) as exin:
                fud.hamming_distance_base(
                    VECTOR_1D_CATEGORICAL_A1[i],
                    VECTOR_1D_CATEGORICAL_A2[i],
                    normalise=False,
                    equal_length=True)
            assert str(exin.value) == value_error

    # Test normalised -- equal length
    for i in range(VECTOR_1D_CATEGORICAL_A1.shape[0]):
        if (len(VECTOR_1D_CATEGORICAL_A1[i]) == len(
                VECTOR_1D_CATEGORICAL_A2[i])):
            dist = fud.hamming_distance_base(
                VECTOR_1D_CATEGORICAL_A1[i],
                VECTOR_1D_CATEGORICAL_A2[i],
                normalise=True,
                equal_length=True)
            assert (DISTANCES_1D_CATEGORICAL_A_HAMMING_NORMALISED[i] ==
                    pytest.approx(dist, rel=1e-3))
        else:
            with pytest.raises(ValueError) as exin:
                fud.hamming_distance_base(
                    VECTOR_1D_CATEGORICAL_A1[i],
                    VECTOR_1D_CATEGORICAL_A2[i],
                    normalise=True,
                    equal_length=True)
            assert str(exin.value) == value_error


def test_hamming_distance():
    """
    Tests :func:`fatf.utils.distances.hamming_distance_base`.
    """
    shape_error_x = 'The x array should be 1-dimensional.'
    shape_error_y = 'The y array should be 1-dimensional.'
    value_error_x = 'The x array should be textual.'
    value_error_y = 'The y array should be textual.'
    shape_error_xy = 'The x and y arrays should have the same length.'

    # x or y is not 1D
    with pytest.raises(IncorrectShapeError) as exin:
        fud.hamming_distance(VECTOR_0D, VECTOR_2D_NUMERICAL_A1)
    assert str(exin.value) == shape_error_x
    with pytest.raises(IncorrectShapeError) as exin:
        fud.hamming_distance(VECTOR_2D_NUMERICAL_A1, VECTOR_1D_CATEGORICAL_A1)
    assert str(exin.value) == shape_error_x
    with pytest.raises(IncorrectShapeError) as exin:
        fud.hamming_distance(VECTOR_1D_NUMERICAL_A1, VECTOR_0D)
    assert str(exin.value) == shape_error_y
    with pytest.raises(IncorrectShapeError) as exin:
        fud.hamming_distance(VECTOR_1D_CATEGORICAL_A1, VECTOR_2D_NUMERICAL_A1)
    assert str(exin.value) == shape_error_y

    # x or y is not numerical
    with pytest.raises(ValueError) as exin:
        fud.hamming_distance(VECTOR_1D_NUMERICAL_A1, VECTOR_1D_CATEGORICAL_A1)
    assert str(exin.value) == value_error_x
    with pytest.raises(ValueError) as exin:
        fud.hamming_distance(VECTOR_1D_CATEGORICAL_A1, VECTOR_1D_NUMERICAL_A1)
    assert str(exin.value) == value_error_y

    # The shape of input arrays does not match
    with pytest.raises(IncorrectShapeError) as exin:
        fud.hamming_distance(VECTOR_1D_CATEGORICAL_A1,
                             VECTORS_1D_CATEGORICAL_A1[0])
    assert str(exin.value) == shape_error_xy
    with pytest.raises(IncorrectShapeError) as exin:
        fud.hamming_distance(VECTORS_1D_CATEGORICAL_A1[0],
                             VECTOR_1D_CATEGORICAL_A1)
    assert str(exin.value) == shape_error_xy

    # Test unstructured arrays
    dist = fud.hamming_distance(VECTOR_1D_CATEGORICAL_A1,
                                VECTOR_1D_CATEGORICAL_A2)
    assert dist == DISTANCE_1D_CATEGORICAL_A_HAMMING
    dist = fud.hamming_distance(
        VECTOR_1D_CATEGORICAL_A1, VECTOR_1D_CATEGORICAL_A2, normalise=True)
    assert dist == pytest.approx(
        DISTANCE_1D_CATEGORICAL_A_HAMMING_NORMALISED, rel=1e-3)

    assert ((VECTORS_1D_CATEGORICAL_A1.shape[0] ==
             VECTORS_1D_CATEGORICAL_A2.shape[0])
            and (VECTORS_1D_CATEGORICAL_A2.shape[0] ==
                 DISTANCES_1D_CATEGORICAL_A.shape[0]))
    for i in range(DISTANCES_1D_CATEGORICAL_A.shape[0]):
        dist = fud.hamming_distance(VECTORS_1D_CATEGORICAL_A1[i],
                                    VECTORS_1D_CATEGORICAL_A2[i])
        assert dist == DISTANCES_1D_CATEGORICAL_A[i]

    assert ((VECTORS_1D_CATEGORICAL_A1.shape[0] ==
             VECTORS_1D_CATEGORICAL_A2.shape[0])
            and (VECTORS_1D_CATEGORICAL_A2.shape[0] ==
                 DISTANCES_1D_CATEGORICAL_A_NORMALISED.shape[0]))
    for i in range(DISTANCES_1D_CATEGORICAL_A_NORMALISED.shape[0]):
        dist = fud.hamming_distance(
            VECTORS_1D_CATEGORICAL_A1[i],
            VECTORS_1D_CATEGORICAL_A2[i],
            normalise=True)
        assert dist == DISTANCES_1D_CATEGORICAL_A_NORMALISED[i]

    # Test structured arrays
    assert ((VECTOR_2D_CATEGORICAL_STRUCT_A1.shape[0] ==
             VECTOR_2D_CATEGORICAL_STRUCT_A2.shape[0])
            and (VECTOR_2D_CATEGORICAL_STRUCT_A2.shape[0] ==
                 DISTANCES_2D_CATEGORICAL_A.shape[0])
            and (DISTANCES_2D_CATEGORICAL_A.shape[0] ==
                 DISTANCES_2D_CATEGORICAL_A.shape[1])
            and (DISTANCES_2D_CATEGORICAL_A.shape[1] ==
                 DISTANCES_2D_CATEGORICAL_A_NORMALISED.shape[0])
            and (DISTANCES_2D_CATEGORICAL_A_NORMALISED.shape[0] ==
                 DISTANCES_2D_CATEGORICAL_A_NORMALISED.shape[1]))
    for i in range(DISTANCES_2D_CATEGORICAL_A.shape[0]):
        dist = fud.hamming_distance(VECTOR_2D_CATEGORICAL_STRUCT_A1[i],
                                    VECTOR_2D_CATEGORICAL_STRUCT_A2[i])
        assert DISTANCES_2D_CATEGORICAL_A[i, i] == dist
        dist = fud.hamming_distance(
            VECTOR_2D_CATEGORICAL_STRUCT_A1[i],
            VECTOR_2D_CATEGORICAL_STRUCT_A2[i],
            normalise=True)
        assert DISTANCES_2D_CATEGORICAL_A_NORMALISED[i, i] == dist

    # Test unstructured-structured mixture
    assert ((VECTOR_2D_CATEGORICAL_A1.shape[0] ==
             VECTOR_2D_CATEGORICAL_STRUCT_A1.shape[0])
            and (VECTOR_2D_CATEGORICAL_STRUCT_A1.shape[0] ==
                 VECTOR_2D_CATEGORICAL_STRUCT_A2.shape[0])
            and (VECTOR_2D_CATEGORICAL_STRUCT_A2.shape[0] ==
                 DISTANCES_2D_CATEGORICAL_A.shape[0])
            and (DISTANCES_2D_CATEGORICAL_A.shape[0] ==
                 DISTANCES_2D_CATEGORICAL_A.shape[1])
            and (DISTANCES_2D_CATEGORICAL_A.shape[1] ==
                 DISTANCES_2D_CATEGORICAL_A_NORMALISED.shape[0])
            and (DISTANCES_2D_CATEGORICAL_A_NORMALISED.shape[0] ==
                 DISTANCES_2D_CATEGORICAL_A_NORMALISED.shape[1]))
    for i in range(DISTANCES_2D_CATEGORICAL_A.shape[0]):
        dist = fud.hamming_distance(VECTOR_2D_CATEGORICAL_A1[i],
                                    VECTOR_2D_CATEGORICAL_STRUCT_A2[i])
        assert DISTANCES_2D_CATEGORICAL_A[i, i] == dist
        dist = fud.hamming_distance(
            VECTOR_2D_CATEGORICAL_STRUCT_A2[i],
            VECTOR_2D_CATEGORICAL_A1[i],
            normalise=True)
        assert DISTANCES_2D_CATEGORICAL_A_NORMALISED[i, i] == dist

    # Test hamming distance for 1-element voids
    dist = fud.hamming_distance(VECTOR_2D_CATEGORICAL_STRUCT_A2[['name']][1],
                                VECTOR_2D_CATEGORICAL_STRUCT_A1[['name']][0])
    assert dist == 3


def test_hamming_point_distance():
    """
    Tests :func:`fatf.utils.distances.hamming_point_distance`.
    """
    # pylint: disable=too-many-statements
    shape_error_y = 'The y array should be 1-dimensional.'
    shape_error_X = 'The X array should be 2-dimensional.'
    value_error_y = 'The y array should be textual.'
    value_error_X = 'The X array should be textual.'
    shape_error_yX = ('The number of columns in the X array should the same '
                      'as the number of elements in the y array.')

    # X is not 2D or y is not 1D
    with pytest.raises(IncorrectShapeError) as exin:
        fud.hamming_point_distance(VECTOR_0D, VECTOR_2D_NUMERICAL_A1)
    assert str(exin.value) == shape_error_y
    with pytest.raises(IncorrectShapeError) as exin:
        fud.hamming_point_distance(VECTOR_2D_NUMERICAL_A1,
                                   VECTOR_1D_CATEGORICAL_A1)
    assert str(exin.value) == shape_error_y
    with pytest.raises(IncorrectShapeError) as exin:
        fud.hamming_point_distance(VECTOR_1D_CATEGORICAL_A1, VECTOR_0D)
    assert str(exin.value) == shape_error_X
    with pytest.raises(IncorrectShapeError) as exin:
        fud.hamming_point_distance(VECTOR_1D_CATEGORICAL_A1,
                                   VECTOR_1D_CATEGORICAL_A1)
    assert str(exin.value) == shape_error_X

    # X or y is not textual
    with pytest.raises(ValueError) as exin:
        fud.hamming_point_distance(VECTOR_1D_NUMERICAL_A1,
                                   VECTOR_2D_CATEGORICAL_STRUCT_A1)
    assert str(exin.value) == value_error_y
    with pytest.raises(ValueError) as exin:
        fud.hamming_point_distance(VECTOR_1D_CATEGORICAL_A1,
                                   VECTOR_2D_NUMERICAL_A1)
    assert str(exin.value) == value_error_X

    # The shape of input arrays does not match
    with pytest.raises(IncorrectShapeError) as exin:
        fud.hamming_point_distance(VECTOR_1D_CATEGORICAL_A1,
                                   VECTOR_2D_CATEGORICAL_STRUCT_A1)
    assert str(exin.value) == shape_error_yX
    with pytest.raises(IncorrectShapeError) as exin:
        fud.hamming_point_distance(VECTOR_2D_CATEGORICAL_STRUCT_A1[0],
                                   VECTORS_1D_CATEGORICAL_A1)
    assert str(exin.value) == shape_error_yX

    # Test unstructured arrays
    assert ((VECTOR_2D_CATEGORICAL_A1.shape[0] ==
             DISTANCES_2D_CATEGORICAL_A.shape[0])
            and (DISTANCES_2D_CATEGORICAL_A.shape[0] ==
                 DISTANCES_2D_CATEGORICAL_A_NORMALISED.shape[0])
            and (VECTOR_2D_CATEGORICAL_A2.shape[0] ==
                 DISTANCES_2D_CATEGORICAL_A.shape[1])
            and (DISTANCES_2D_CATEGORICAL_A.shape[1] ==
                 DISTANCES_2D_CATEGORICAL_A_NORMALISED.shape[1]))
    for i in range(VECTOR_2D_CATEGORICAL_A1.shape[0]):
        dist = fud.hamming_point_distance(VECTOR_2D_CATEGORICAL_A1[i],
                                          VECTOR_2D_CATEGORICAL_A2)
        assert np.isclose(DISTANCES_2D_CATEGORICAL_A[i, :], dist).all()
        dist = fud.hamming_point_distance(
            VECTOR_2D_CATEGORICAL_A1[i],
            VECTOR_2D_CATEGORICAL_A2,
            normalise=True)
        assert np.isclose(DISTANCES_2D_CATEGORICAL_A_NORMALISED[i, :],
                          dist).all()

    for i in range(VECTOR_2D_CATEGORICAL_A2.shape[0]):
        dist = fud.hamming_point_distance(VECTOR_2D_CATEGORICAL_A2[i],
                                          VECTOR_2D_CATEGORICAL_A1)
        assert np.isclose(DISTANCES_2D_CATEGORICAL_A[:, i], dist).all()
        dist = fud.hamming_point_distance(
            VECTOR_2D_CATEGORICAL_A2[i],
            VECTOR_2D_CATEGORICAL_A1,
            normalise=True)
        assert np.isclose(DISTANCES_2D_CATEGORICAL_A_NORMALISED[:, i],
                          dist).all()

    # Test structured arrays
    assert ((VECTOR_2D_CATEGORICAL_STRUCT_A1.shape[0] ==
             DISTANCES_2D_CATEGORICAL_A.shape[0])
            and (DISTANCES_2D_CATEGORICAL_A.shape[0] ==
                 DISTANCES_2D_CATEGORICAL_A_NORMALISED.shape[0])
            and (VECTOR_2D_CATEGORICAL_STRUCT_A2.shape[0] ==
                 DISTANCES_2D_CATEGORICAL_A.shape[1])
            and (DISTANCES_2D_CATEGORICAL_A.shape[1] ==
                 DISTANCES_2D_CATEGORICAL_A_NORMALISED.shape[1]))
    for i in range(VECTOR_2D_CATEGORICAL_STRUCT_A1.shape[0]):
        dist = fud.hamming_point_distance(VECTOR_2D_CATEGORICAL_STRUCT_A1[i],
                                          VECTOR_2D_CATEGORICAL_STRUCT_A2)
        assert np.isclose(DISTANCES_2D_CATEGORICAL_A[i, :], dist).all()
        dist = fud.hamming_point_distance(
            VECTOR_2D_CATEGORICAL_STRUCT_A1[i],
            VECTOR_2D_CATEGORICAL_STRUCT_A2,
            normalise=True)
        assert np.isclose(DISTANCES_2D_CATEGORICAL_A_NORMALISED[i, :],
                          dist).all()

    for i in range(VECTOR_2D_CATEGORICAL_STRUCT_A2.shape[0]):
        dist = fud.hamming_point_distance(VECTOR_2D_CATEGORICAL_STRUCT_A2[i],
                                          VECTOR_2D_CATEGORICAL_STRUCT_A1)
        assert np.isclose(DISTANCES_2D_CATEGORICAL_A[:, i], dist).all()
        dist = fud.hamming_point_distance(
            VECTOR_2D_CATEGORICAL_STRUCT_A2[i],
            VECTOR_2D_CATEGORICAL_STRUCT_A1,
            normalise=True)
        assert np.isclose(DISTANCES_2D_CATEGORICAL_A_NORMALISED[:, i],
                          dist).all()

    # Test unstructured-structured mixture
    assert ((VECTOR_2D_CATEGORICAL_STRUCT_A1.shape[0] ==
             DISTANCES_2D_CATEGORICAL_A.shape[0])
            and (DISTANCES_2D_CATEGORICAL_A.shape[0] ==
                 DISTANCES_2D_CATEGORICAL_A_NORMALISED.shape[0])
            and (VECTOR_2D_CATEGORICAL_A2.shape[0] ==
                 DISTANCES_2D_CATEGORICAL_A.shape[1])
            and (DISTANCES_2D_CATEGORICAL_A.shape[1] ==
                 DISTANCES_2D_CATEGORICAL_A_NORMALISED.shape[1]))
    for i in range(VECTOR_2D_CATEGORICAL_STRUCT_A1.shape[0]):
        dist = fud.hamming_point_distance(VECTOR_2D_CATEGORICAL_STRUCT_A1[i],
                                          VECTOR_2D_CATEGORICAL_A2)
        assert np.isclose(DISTANCES_2D_CATEGORICAL_A[i, :], dist).all()
        dist = fud.hamming_point_distance(
            VECTOR_2D_CATEGORICAL_STRUCT_A1[i],
            VECTOR_2D_CATEGORICAL_A2,
            normalise=True)
        assert np.isclose(DISTANCES_2D_CATEGORICAL_A_NORMALISED[i, :],
                          dist).all()
    for i in range(VECTOR_2D_CATEGORICAL_A2.shape[0]):
        dist = fud.hamming_point_distance(VECTOR_2D_CATEGORICAL_A2[i],
                                          VECTOR_2D_CATEGORICAL_STRUCT_A1)
        assert np.isclose(DISTANCES_2D_CATEGORICAL_A[:, i], dist).all()
        dist = fud.hamming_point_distance(
            VECTOR_2D_CATEGORICAL_A2[i],
            VECTOR_2D_CATEGORICAL_STRUCT_A1,
            normalise=True)
        assert np.isclose(DISTANCES_2D_CATEGORICAL_A_NORMALISED[:, i],
                          dist).all()

    assert ((VECTOR_2D_CATEGORICAL_A1.shape[0] ==
             DISTANCES_2D_CATEGORICAL_A.shape[0])
            and (DISTANCES_2D_CATEGORICAL_A.shape[0] ==
                 DISTANCES_2D_CATEGORICAL_A_NORMALISED.shape[0])
            and (VECTOR_2D_CATEGORICAL_STRUCT_A2.shape[0] ==
                 DISTANCES_2D_CATEGORICAL_A.shape[1])
            and (DISTANCES_2D_CATEGORICAL_A.shape[1] ==
                 DISTANCES_2D_CATEGORICAL_A_NORMALISED.shape[1]))
    for i in range(VECTOR_2D_CATEGORICAL_A1.shape[0]):
        dist = fud.hamming_point_distance(VECTOR_2D_CATEGORICAL_A1[i],
                                          VECTOR_2D_CATEGORICAL_STRUCT_A2)
        assert np.isclose(DISTANCES_2D_CATEGORICAL_A[i, :], dist).all()
        dist = fud.hamming_point_distance(
            VECTOR_2D_CATEGORICAL_STRUCT_A1[i],
            VECTOR_2D_CATEGORICAL_STRUCT_A2,
            normalise=True)
        assert np.isclose(DISTANCES_2D_CATEGORICAL_A_NORMALISED[i, :],
                          dist).all()
    for i in range(VECTOR_2D_CATEGORICAL_STRUCT_A2.shape[0]):
        dist = fud.hamming_point_distance(VECTOR_2D_CATEGORICAL_STRUCT_A2[i],
                                          VECTOR_2D_CATEGORICAL_A1)
        assert np.isclose(DISTANCES_2D_CATEGORICAL_A[:, i], dist).all()
        dist = fud.hamming_point_distance(
            VECTOR_2D_CATEGORICAL_STRUCT_A2[i],
            VECTOR_2D_CATEGORICAL_A1,
            normalise=True)
        assert np.isclose(DISTANCES_2D_CATEGORICAL_A_NORMALISED[:, i],
                          dist).all()

    # Test hamming point distance for 1-element voids
    dist = fud.hamming_point_distance(
        VECTOR_2D_CATEGORICAL_STRUCT_A2[['name']][1],
        VECTOR_2D_CATEGORICAL_STRUCT_A1[['name']])
    assert np.array_equal(dist, [3, 4, 8])


def test_hamming_array_distance():
    """
    Tests :func:`fatf.utils.distances.hamming_array_distance`.
    """
    # pylint: disable=too-many-statements
    shape_error_X = 'The X array should be 2-dimensional.'
    shape_error_Y = 'The Y array should be 2-dimensional.'
    value_error_X = 'The X array should be textual.'
    value_error_Y = 'The Y array should be textual.'
    shape_error_XY = ('The number of columns in the X array should the same '
                      'as the number of columns in Y array.')

    # X or Y is not 2D
    with pytest.raises(IncorrectShapeError) as exin:
        fud.hamming_array_distance(VECTOR_1D_CATEGORICAL_A1, VECTOR_0D)
    assert str(exin.value) == shape_error_X
    with pytest.raises(IncorrectShapeError) as exin:
        fud.hamming_array_distance(VECTOR_0D, VECTOR_2D_NUMERICAL_A1)
    assert str(exin.value) == shape_error_X
    with pytest.raises(IncorrectShapeError) as exin:
        fud.hamming_array_distance(VECTOR_2D_NUMERICAL_A1, VECTOR_0D)
    assert str(exin.value) == shape_error_Y
    with pytest.raises(IncorrectShapeError) as exin:
        fud.hamming_array_distance(VECTOR_2D_NUMERICAL_STRUCT_A2,
                                   VECTOR_1D_CATEGORICAL_A1)
    assert str(exin.value) == shape_error_Y

    # X or Y is not textual
    with pytest.raises(ValueError) as exin:
        fud.hamming_array_distance(VECTOR_2D_NUMERICAL_A1,
                                   VECTOR_2D_CATEGORICAL_A1)
    assert str(exin.value) == value_error_X
    with pytest.raises(ValueError) as exin:
        fud.hamming_array_distance(VECTOR_2D_CATEGORICAL_STRUCT_A2,
                                   VECTOR_2D_NUMERICAL_A2)
    assert str(exin.value) == value_error_Y

    # The shape of input arrays does not match
    with pytest.raises(IncorrectShapeError) as exin:
        fud.hamming_array_distance(VECTOR_2D_CATEGORICAL_STRUCT_A2,
                                   VECTORS_1D_CATEGORICAL_A2)
    assert str(exin.value) == shape_error_XY
    with pytest.raises(IncorrectShapeError) as exin:
        fud.hamming_array_distance(VECTORS_1D_CATEGORICAL_A1,
                                   VECTOR_2D_CATEGORICAL_A1)
    assert str(exin.value) == shape_error_XY

    # Test unstructured arrays
    assert ((VECTOR_2D_CATEGORICAL_A1.shape[0] ==
             DISTANCES_2D_CATEGORICAL_A.shape[0])
            and (DISTANCES_2D_CATEGORICAL_A.shape[0] ==
                 DISTANCES_2D_CATEGORICAL_A_NORMALISED.shape[0])
            and (VECTOR_2D_CATEGORICAL_A2.shape[0] ==
                 DISTANCES_2D_CATEGORICAL_A.shape[1])
            and (DISTANCES_2D_CATEGORICAL_A.shape[1] ==
                 DISTANCES_2D_CATEGORICAL_A_NORMALISED.shape[1]))
    dist = fud.hamming_array_distance(VECTOR_2D_CATEGORICAL_A1,
                                      VECTOR_2D_CATEGORICAL_A2)
    assert np.isclose(DISTANCES_2D_CATEGORICAL_A, dist).all()
    dist = fud.hamming_array_distance(
        VECTOR_2D_CATEGORICAL_A1, VECTOR_2D_CATEGORICAL_A2, normalise=True)
    assert np.isclose(DISTANCES_2D_CATEGORICAL_A_NORMALISED, dist).all()
    dist = fud.hamming_array_distance(VECTOR_2D_CATEGORICAL_A2,
                                      VECTOR_2D_CATEGORICAL_A1)
    assert np.isclose(DISTANCES_2D_CATEGORICAL_A.T, dist).all()
    dist = fud.hamming_array_distance(
        VECTOR_2D_CATEGORICAL_A2, VECTOR_2D_CATEGORICAL_A1, normalise=True)
    assert np.isclose(DISTANCES_2D_CATEGORICAL_A_NORMALISED.T, dist).all()

    # Test structured arrays
    assert ((VECTOR_2D_CATEGORICAL_STRUCT_A1.shape[0] ==
             DISTANCES_2D_CATEGORICAL_A.shape[0])
            and (DISTANCES_2D_CATEGORICAL_A.shape[0] ==
                 DISTANCES_2D_CATEGORICAL_A_NORMALISED.shape[0])
            and (VECTOR_2D_CATEGORICAL_STRUCT_A2.shape[0] ==
                 DISTANCES_2D_CATEGORICAL_A.shape[1])
            and (DISTANCES_2D_CATEGORICAL_A.shape[1] ==
                 DISTANCES_2D_CATEGORICAL_A_NORMALISED.shape[1]))
    dist = fud.hamming_array_distance(VECTOR_2D_CATEGORICAL_STRUCT_A1,
                                      VECTOR_2D_CATEGORICAL_STRUCT_A2)
    assert np.isclose(DISTANCES_2D_CATEGORICAL_A, dist).all()
    dist = fud.hamming_array_distance(
        VECTOR_2D_CATEGORICAL_STRUCT_A1,
        VECTOR_2D_CATEGORICAL_STRUCT_A2,
        normalise=True)
    assert np.isclose(DISTANCES_2D_CATEGORICAL_A_NORMALISED, dist).all()
    dist = fud.hamming_array_distance(VECTOR_2D_CATEGORICAL_STRUCT_A2,
                                      VECTOR_2D_CATEGORICAL_STRUCT_A1)
    assert np.isclose(DISTANCES_2D_CATEGORICAL_A.T, dist).all()
    dist = fud.hamming_array_distance(
        VECTOR_2D_CATEGORICAL_STRUCT_A2,
        VECTOR_2D_CATEGORICAL_STRUCT_A1,
        normalise=True)
    assert np.isclose(DISTANCES_2D_CATEGORICAL_A_NORMALISED.T, dist).all()

    # Test unstructured-structured mixture
    assert ((VECTOR_2D_CATEGORICAL_A1.shape[0] ==
             DISTANCES_2D_CATEGORICAL_A.shape[0])
            and (DISTANCES_2D_CATEGORICAL_A.shape[0] ==
                 DISTANCES_2D_CATEGORICAL_A_NORMALISED.shape[0])
            and (VECTOR_2D_CATEGORICAL_STRUCT_A2.shape[0] ==
                 DISTANCES_2D_CATEGORICAL_A.shape[1])
            and (DISTANCES_2D_CATEGORICAL_A.shape[1] ==
                 DISTANCES_2D_CATEGORICAL_A_NORMALISED.shape[1]))
    dist = fud.hamming_array_distance(VECTOR_2D_CATEGORICAL_A1,
                                      VECTOR_2D_CATEGORICAL_STRUCT_A2)
    assert np.isclose(DISTANCES_2D_CATEGORICAL_A, dist).all()
    dist = fud.hamming_array_distance(
        VECTOR_2D_CATEGORICAL_A1,
        VECTOR_2D_CATEGORICAL_STRUCT_A2,
        normalise=True)
    assert np.isclose(DISTANCES_2D_CATEGORICAL_A_NORMALISED, dist).all()
    dist = fud.hamming_array_distance(VECTOR_2D_CATEGORICAL_STRUCT_A2,
                                      VECTOR_2D_CATEGORICAL_A1)
    assert np.isclose(DISTANCES_2D_CATEGORICAL_A.T, dist).all()
    dist = fud.hamming_array_distance(
        VECTOR_2D_CATEGORICAL_STRUCT_A2,
        VECTOR_2D_CATEGORICAL_A1,
        normalise=True)
    assert np.isclose(DISTANCES_2D_CATEGORICAL_A_NORMALISED.T, dist).all()

    assert ((VECTOR_2D_CATEGORICAL_STRUCT_A1.shape[0] ==
             DISTANCES_2D_CATEGORICAL_A.shape[0])
            and (DISTANCES_2D_CATEGORICAL_A.shape[0] ==
                 DISTANCES_2D_CATEGORICAL_A_NORMALISED.shape[0])
            and (VECTOR_2D_CATEGORICAL_A2.shape[0] ==
                 DISTANCES_2D_CATEGORICAL_A.shape[1])
            and (DISTANCES_2D_CATEGORICAL_A.shape[1] ==
                 DISTANCES_2D_CATEGORICAL_A_NORMALISED.shape[1]))
    dist = fud.hamming_array_distance(VECTOR_2D_CATEGORICAL_STRUCT_A1,
                                      VECTOR_2D_CATEGORICAL_A2)
    assert np.isclose(DISTANCES_2D_CATEGORICAL_A, dist).all()
    dist = fud.hamming_array_distance(
        VECTOR_2D_CATEGORICAL_STRUCT_A1,
        VECTOR_2D_CATEGORICAL_A2,
        normalise=True)
    assert np.isclose(DISTANCES_2D_CATEGORICAL_A_NORMALISED, dist).all()
    dist = fud.hamming_array_distance(VECTOR_2D_CATEGORICAL_A2,
                                      VECTOR_2D_CATEGORICAL_STRUCT_A1)
    assert np.isclose(DISTANCES_2D_CATEGORICAL_A.T, dist).all()
    dist = fud.hamming_array_distance(
        VECTOR_2D_CATEGORICAL_A2,
        VECTOR_2D_CATEGORICAL_STRUCT_A1,
        normalise=True)
    assert np.isclose(DISTANCES_2D_CATEGORICAL_A_NORMALISED.T, dist).all()


def test_binary_distance():
    """
    Tests :func:`fatf.utils.distances.binary_distance`.
    """
    # pylint: disable=too-many-statements
    shape_error_x = 'The x array should be 1-dimensional.'
    shape_error_y = 'The y array should be 1-dimensional.'
    shape_error_xy = 'The x and y arrays should have the same length.'

    # x or y is not 1D
    with pytest.raises(IncorrectShapeError) as exin:
        fud.binary_distance(VECTOR_0D, VECTOR_1D_UNBASE_A1)
    assert str(exin.value) == shape_error_x
    with pytest.raises(IncorrectShapeError) as exin:
        fud.binary_distance(VECTOR_2D_NUMERICAL_A1, VECTOR_1D_CATEGORICAL_A1)
    assert str(exin.value) == shape_error_x
    with pytest.raises(IncorrectShapeError) as exin:
        fud.binary_distance(VECTOR_1D_NUMERICAL_A1, VECTOR_0D)
    assert str(exin.value) == shape_error_y
    with pytest.raises(IncorrectShapeError) as exin:
        fud.binary_distance(VECTOR_1D_CATEGORICAL_A1, VECTOR_2D_NUMERICAL_A1)
    assert str(exin.value) == shape_error_y

    # The shape of input arrays does not match
    with pytest.raises(IncorrectShapeError) as exin:
        fud.binary_distance(VECTOR_1D_CATEGORICAL_A1,
                            VECTORS_1D_CATEGORICAL_A1[0])
    assert str(exin.value) == shape_error_xy
    with pytest.raises(IncorrectShapeError) as exin:
        fud.binary_distance(VECTORS_1D_CATEGORICAL_A1[0],
                            VECTOR_1D_CATEGORICAL_A1)
    assert str(exin.value) == shape_error_xy
    with pytest.raises(IncorrectShapeError) as exin:
        fud.binary_distance(VECTOR_1D_NUMERICAL_A1, VECTOR_1D_NUMERICAL_B1)
    assert str(exin.value) == shape_error_xy
    with pytest.raises(IncorrectShapeError) as exin:
        fud.binary_distance(VECTOR_1D_NUMERICAL_A1,
                            VECTOR_2D_NUMERICAL_STRUCT_A1[0])
    assert str(exin.value) == shape_error_xy
    with pytest.raises(IncorrectShapeError) as exin:
        fud.binary_distance(VECTOR_1D_NUMERICAL_A1,
                            VECTORS_1D_CATEGORICAL_A1[0])
    assert str(exin.value) == shape_error_xy

    # Test unstructured arrays
    dist = fud.binary_distance(VECTOR_1D_CATEGORICAL_A1,
                               VECTOR_1D_CATEGORICAL_A2)
    assert dist == DISTANCE_1D_CATEGORICAL_A_BINARY
    dist = fud.binary_distance(
        VECTOR_1D_CATEGORICAL_A1, VECTOR_1D_CATEGORICAL_A2, normalise=True)
    assert dist == DISTANCE_1D_CATEGORICAL_A_BINARY_NORMALISED

    dist = fud.binary_distance(VECTOR_1D_NUMERICAL_A1, VECTOR_1D_NUMERICAL_A2)
    assert dist == DISTANCE_1D_A_BINARY
    dist = fud.binary_distance(
        VECTOR_1D_NUMERICAL_A1, VECTOR_1D_NUMERICAL_A2, normalise=True)
    assert dist == DISTANCE_1D_A_BINARY_NORMALISED

    assert ((VECTORS_1D_CATEGORICAL_A1.shape[0] ==
             VECTORS_1D_CATEGORICAL_A2.shape[0])
            and (VECTORS_1D_CATEGORICAL_A2.shape[0] ==
                 DISTANCES_1D_CATEGORICAL_A_BINARY.shape[0])
            and (DISTANCES_1D_CATEGORICAL_A_BINARY.shape[0] ==
                 DISTANCES_1D_CATEGORICAL_A_BINARY_NORMALISED.shape[0]))
    for i in range(DISTANCES_1D_CATEGORICAL_A.shape[0]):
        dist = fud.binary_distance(VECTORS_1D_CATEGORICAL_A1[i],
                                   VECTORS_1D_CATEGORICAL_A2[i])
        assert dist == DISTANCES_1D_CATEGORICAL_A_BINARY[i]
        dist = fud.binary_distance(
            VECTORS_1D_CATEGORICAL_A1[i],
            VECTORS_1D_CATEGORICAL_A2[i],
            normalise=True)
        assert dist == DISTANCES_1D_CATEGORICAL_A_BINARY_NORMALISED[i]

    assert ((VECTORS_1D_A1.shape[0] == VECTORS_1D_A2.shape[0])
            and (VECTORS_1D_A2.shape[0] == DISTANCES_1D_A_BINARY.shape[0])
            and (DISTANCES_1D_A_BINARY.shape[0] ==
                 DISTANCES_1D_A_BINARY_NORMALISED.shape[0]))
    for i in range(DISTANCES_1D_A_BINARY.shape[0]):
        dist = fud.binary_distance(VECTORS_1D_A1[i], VECTORS_1D_A2[i])
        assert dist == DISTANCES_1D_A_BINARY[i]
        dist = fud.binary_distance(
            VECTORS_1D_A1[i], VECTORS_1D_A2[i], normalise=True)
        assert dist == DISTANCES_1D_A_BINARY_NORMALISED[i]

    assert (
        (VECTOR_2D_CATEGORICAL_A1.shape[0] ==
         VECTOR_2D_CATEGORICAL_A2.shape[0])
        and (VECTOR_2D_CATEGORICAL_A2.shape[0] ==
             DISTANCES_2D_CATEGORICAL_A_BINARY.shape[0])
        and (DISTANCES_2D_CATEGORICAL_A_BINARY.shape[0] ==
             DISTANCES_2D_CATEGORICAL_A_BINARY.shape[1])
        and (DISTANCES_2D_CATEGORICAL_A_BINARY.shape[1] ==
             DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED.shape[0])
        and (DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED.shape[0] ==
             DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED.shape[1])
    )  # yapf: disable
    for i in range(DISTANCES_2D_CATEGORICAL_A_BINARY.shape[0]):
        dist = fud.binary_distance(VECTOR_2D_CATEGORICAL_A1[i],
                                   VECTOR_2D_CATEGORICAL_A2[i])
        assert DISTANCES_2D_CATEGORICAL_A_BINARY[i, i] == dist
        dist = fud.binary_distance(
            VECTOR_2D_CATEGORICAL_A1[i],
            VECTOR_2D_CATEGORICAL_A2[i],
            normalise=True)
        assert DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED[i, i] == dist

    assert (
        (VECTOR_2D_NUMERICAL_A1.shape[0] == VECTOR_2D_NUMERICAL_A2.shape[0])
        and (VECTOR_2D_NUMERICAL_A2.shape[0] ==
             DISTANCES_2D_NUMERICAL_A_BINARY.shape[0])
        and (DISTANCES_2D_NUMERICAL_A_BINARY.shape[0] ==
             DISTANCES_2D_NUMERICAL_A_BINARY.shape[1])
        and (DISTANCES_2D_NUMERICAL_A_BINARY.shape[1] ==
             DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED.shape[0])
        and (DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED.shape[0] ==
             DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED.shape[1]))
    for i in range(DISTANCES_2D_NUMERICAL_A_BINARY.shape[0]):
        dist = fud.binary_distance(VECTOR_2D_NUMERICAL_A1[i],
                                   VECTOR_2D_NUMERICAL_A2[i])
        assert DISTANCES_2D_NUMERICAL_A_BINARY[i, i] == dist
        dist = fud.binary_distance(
            VECTOR_2D_NUMERICAL_A1[i],
            VECTOR_2D_NUMERICAL_A2[i],
            normalise=True)
        assert DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED[i, i] == dist

    # Test structured arrays
    assert ((VECTOR_2D_CATEGORICAL_STRUCT_A1.shape[0] ==
             VECTOR_2D_CATEGORICAL_STRUCT_A2.shape[0])
            and (VECTOR_2D_CATEGORICAL_STRUCT_A2.shape[0] ==
                 DISTANCES_2D_CATEGORICAL_A_BINARY.shape[0])
            and (DISTANCES_2D_CATEGORICAL_A_BINARY.shape[0] ==
                 DISTANCES_2D_CATEGORICAL_A_BINARY.shape[1])
            and (DISTANCES_2D_CATEGORICAL_A_BINARY.shape[1] ==
                 DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED.shape[0])
            and (DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED.shape[0] ==
                 DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED.shape[1]))
    for i in range(DISTANCES_2D_CATEGORICAL_A_BINARY.shape[0]):
        dist = fud.binary_distance(VECTOR_2D_CATEGORICAL_STRUCT_A1[i],
                                   VECTOR_2D_CATEGORICAL_STRUCT_A2[i])
        assert DISTANCES_2D_CATEGORICAL_A_BINARY[i, i] == dist
        dist = fud.binary_distance(
            VECTOR_2D_CATEGORICAL_STRUCT_A1[i],
            VECTOR_2D_CATEGORICAL_STRUCT_A2[i],
            normalise=True)
        assert DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED[i, i] == dist

    assert ((VECTOR_2D_NUMERICAL_STRUCT_A1.shape[0] ==
             VECTOR_2D_NUMERICAL_STRUCT_A2.shape[0])
            and (VECTOR_2D_NUMERICAL_STRUCT_A2.shape[0] ==
                 DISTANCES_2D_NUMERICAL_A_BINARY.shape[0])
            and (DISTANCES_2D_NUMERICAL_A_BINARY.shape[0] ==
                 DISTANCES_2D_NUMERICAL_A_BINARY.shape[1])
            and (DISTANCES_2D_NUMERICAL_A_BINARY.shape[1] ==
                 DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED.shape[0])
            and (DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED.shape[0] ==
                 DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED.shape[1]))
    for i in range(DISTANCES_2D_NUMERICAL_A_BINARY.shape[0]):
        dist = fud.binary_distance(VECTOR_2D_NUMERICAL_STRUCT_A1[i],
                                   VECTOR_2D_NUMERICAL_STRUCT_A2[i])
        assert DISTANCES_2D_NUMERICAL_A_BINARY[i, i] == dist
        dist = fud.binary_distance(
            VECTOR_2D_NUMERICAL_STRUCT_A1[i],
            VECTOR_2D_NUMERICAL_STRUCT_A2[i],
            normalise=True)
        assert DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED[i, i] == dist

    # Test unstructured-structured mixture
    assert ((VECTOR_2D_CATEGORICAL_A1.shape[0] ==
             VECTOR_2D_CATEGORICAL_STRUCT_A1.shape[0])
            and (VECTOR_2D_CATEGORICAL_STRUCT_A1.shape[0] ==
                 VECTOR_2D_CATEGORICAL_STRUCT_A2.shape[0])
            and (VECTOR_2D_CATEGORICAL_STRUCT_A2.shape[0] ==
                 DISTANCES_2D_CATEGORICAL_A_BINARY.shape[0])
            and (DISTANCES_2D_CATEGORICAL_A_BINARY.shape[0] ==
                 DISTANCES_2D_CATEGORICAL_A_BINARY.shape[1])
            and (DISTANCES_2D_CATEGORICAL_A_BINARY.shape[1] ==
                 DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED.shape[0])
            and (DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED.shape[0] ==
                 DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED.shape[1]))
    for i in range(DISTANCES_2D_CATEGORICAL_A_BINARY.shape[0]):
        dist = fud.binary_distance(VECTOR_2D_CATEGORICAL_A1[i],
                                   VECTOR_2D_CATEGORICAL_STRUCT_A2[i])
        assert DISTANCES_2D_CATEGORICAL_A_BINARY[i, i] == dist
        dist = fud.binary_distance(
            VECTOR_2D_CATEGORICAL_STRUCT_A2[i],
            VECTOR_2D_CATEGORICAL_A1[i],
            normalise=True)
        assert DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED[i, i] == dist

    assert ((VECTOR_2D_NUMERICAL_A1.shape[0] ==
             VECTOR_2D_NUMERICAL_STRUCT_A1.shape[0])
            and (VECTOR_2D_NUMERICAL_STRUCT_A1.shape[0] ==
                 VECTOR_2D_NUMERICAL_STRUCT_A2.shape[0])
            and (VECTOR_2D_NUMERICAL_STRUCT_A2.shape[0] ==
                 DISTANCES_2D_NUMERICAL_A_BINARY.shape[0])
            and (DISTANCES_2D_NUMERICAL_A_BINARY.shape[0] ==
                 DISTANCES_2D_NUMERICAL_A_BINARY.shape[1])
            and (DISTANCES_2D_NUMERICAL_A_BINARY.shape[1] ==
                 DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED.shape[0])
            and (DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED.shape[0] ==
                 DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED.shape[1]))
    for i in range(DISTANCES_2D_NUMERICAL_A_BINARY.shape[0]):
        dist = fud.binary_distance(VECTOR_2D_NUMERICAL_A1[i],
                                   VECTOR_2D_NUMERICAL_STRUCT_A2[i])
        assert DISTANCES_2D_NUMERICAL_A_BINARY[i, i] == dist
        dist = fud.binary_distance(
            VECTOR_2D_NUMERICAL_STRUCT_A2[i],
            VECTOR_2D_NUMERICAL_A1[i],
            normalise=True)
        assert DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED[i, i] == dist

    # Test binary distance for 1-element voids
    dist = fud.binary_distance(VECTOR_2D_NUMERICAL_STRUCT_A1[['b']][2],
                               VECTOR_2D_NUMERICAL_STRUCT_A2[['b']][0])
    assert dist == 1
    dist = fud.binary_distance(VECTOR_2D_NUMERICAL_STRUCT_A1[['b']][2],
                               VECTOR_2D_NUMERICAL_STRUCT_A2[['b']][2])
    assert dist == 0


def test_binary_point_distance():
    """
    Tests :func:`fatf.utils.distances.binary_point_distance`.
    """
    # pylint: disable=too-many-statements,too-many-branches
    shape_error_y = 'The y array should be 1-dimensional.'
    shape_error_X = 'The X array should be 2-dimensional.'
    shape_error_yX = ('The number of columns in the X array should the same '
                      'as the number of elements in the y array.')

    # X is not 2D or y is not 1D
    with pytest.raises(IncorrectShapeError) as exin:
        fud.binary_point_distance(VECTOR_0D, VECTOR_2D_NUMERICAL_A1)
    assert str(exin.value) == shape_error_y
    with pytest.raises(IncorrectShapeError) as exin:
        fud.binary_point_distance(VECTOR_2D_NUMERICAL_A1,
                                  VECTOR_1D_CATEGORICAL_A1)
    assert str(exin.value) == shape_error_y
    with pytest.raises(IncorrectShapeError) as exin:
        fud.binary_point_distance(VECTOR_1D_CATEGORICAL_A1, VECTOR_0D)
    assert str(exin.value) == shape_error_X
    with pytest.raises(IncorrectShapeError) as exin:
        fud.binary_point_distance(VECTOR_1D_CATEGORICAL_A1,
                                  VECTOR_1D_CATEGORICAL_A1)
    assert str(exin.value) == shape_error_X

    # The shape of input arrays does not match
    with pytest.raises(IncorrectShapeError) as exin:
        fud.binary_point_distance(VECTOR_1D_CATEGORICAL_A1,
                                  VECTOR_2D_CATEGORICAL_STRUCT_A1)
    assert str(exin.value) == shape_error_yX
    with pytest.raises(IncorrectShapeError) as exin:
        fud.binary_point_distance(VECTOR_2D_CATEGORICAL_STRUCT_A1[0],
                                  VECTORS_1D_CATEGORICAL_A1)
    assert str(exin.value) == shape_error_yX
    with pytest.raises(IncorrectShapeError) as exin:
        fud.binary_point_distance(VECTOR_2D_CATEGORICAL_STRUCT_A1[0],
                                  VECTORS_1D_A1)
    assert str(exin.value) == shape_error_yX

    # Test unstructured arrays
    assert ((VECTOR_2D_CATEGORICAL_A1.shape[0] ==
             DISTANCES_2D_CATEGORICAL_A_BINARY.shape[0])
            and (DISTANCES_2D_CATEGORICAL_A_BINARY.shape[0] ==
                 DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED.shape[0])
            and (VECTOR_2D_CATEGORICAL_A2.shape[0] ==
                 DISTANCES_2D_CATEGORICAL_A_BINARY.shape[1])
            and (DISTANCES_2D_CATEGORICAL_A_BINARY.shape[1] ==
                 DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED.shape[1]))
    for i in range(VECTOR_2D_CATEGORICAL_A1.shape[0]):
        dist = fud.binary_point_distance(VECTOR_2D_CATEGORICAL_A1[i],
                                         VECTOR_2D_CATEGORICAL_A2)
        assert np.isclose(DISTANCES_2D_CATEGORICAL_A_BINARY[i, :], dist).all()
        dist = fud.binary_point_distance(
            VECTOR_2D_CATEGORICAL_A1[i],
            VECTOR_2D_CATEGORICAL_A2,
            normalise=True)
        assert np.isclose(DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED[i, :],
                          dist).all()
    for i in range(VECTOR_2D_CATEGORICAL_A2.shape[0]):
        dist = fud.binary_point_distance(VECTOR_2D_CATEGORICAL_A2[i],
                                         VECTOR_2D_CATEGORICAL_A1)
        assert np.isclose(DISTANCES_2D_CATEGORICAL_A_BINARY[:, i], dist).all()
        dist = fud.binary_point_distance(
            VECTOR_2D_CATEGORICAL_A2[i],
            VECTOR_2D_CATEGORICAL_A1,
            normalise=True)
        assert np.isclose(DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED[:, i],
                          dist).all()

    assert ((VECTOR_2D_NUMERICAL_A1.shape[0] ==
             DISTANCES_2D_NUMERICAL_A_BINARY.shape[0])
            and (DISTANCES_2D_NUMERICAL_A_BINARY.shape[0] ==
                 DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED.shape[0])
            and (VECTOR_2D_NUMERICAL_A2.shape[0] ==
                 DISTANCES_2D_NUMERICAL_A_BINARY.shape[1])
            and (DISTANCES_2D_NUMERICAL_A_BINARY.shape[1] ==
                 DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED.shape[1]))
    for i in range(VECTOR_2D_NUMERICAL_A1.shape[0]):
        dist = fud.binary_point_distance(VECTOR_2D_NUMERICAL_A1[i],
                                         VECTOR_2D_NUMERICAL_A2)
        assert np.isclose(DISTANCES_2D_NUMERICAL_A_BINARY[i, :], dist).all()
        dist = fud.binary_point_distance(
            VECTOR_2D_NUMERICAL_A1[i], VECTOR_2D_NUMERICAL_A2, normalise=True)
        assert np.isclose(DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED[i, :],
                          dist).all()
    for i in range(VECTOR_2D_NUMERICAL_A2.shape[0]):
        dist = fud.binary_point_distance(VECTOR_2D_NUMERICAL_A2[i],
                                         VECTOR_2D_NUMERICAL_A1)
        assert np.isclose(DISTANCES_2D_NUMERICAL_A_BINARY[:, i], dist).all()
        dist = fud.binary_point_distance(
            VECTOR_2D_NUMERICAL_A2[i], VECTOR_2D_NUMERICAL_A1, normalise=True)
        assert np.isclose(DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED[:, i],
                          dist).all()

    # Test structured arrays
    assert ((VECTOR_2D_CATEGORICAL_STRUCT_A1.shape[0] ==
             DISTANCES_2D_CATEGORICAL_A_BINARY.shape[0])
            and (DISTANCES_2D_CATEGORICAL_A_BINARY.shape[0] ==
                 DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED.shape[0])
            and (VECTOR_2D_CATEGORICAL_STRUCT_A2.shape[0] ==
                 DISTANCES_2D_CATEGORICAL_A_BINARY.shape[1])
            and (DISTANCES_2D_CATEGORICAL_A_BINARY.shape[1] ==
                 DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED.shape[1]))
    for i in range(VECTOR_2D_CATEGORICAL_STRUCT_A1.shape[0]):
        dist = fud.binary_point_distance(VECTOR_2D_CATEGORICAL_STRUCT_A1[i],
                                         VECTOR_2D_CATEGORICAL_STRUCT_A2)
        assert np.isclose(DISTANCES_2D_CATEGORICAL_A_BINARY[i, :], dist).all()
        dist = fud.binary_point_distance(
            VECTOR_2D_CATEGORICAL_STRUCT_A1[i],
            VECTOR_2D_CATEGORICAL_STRUCT_A2,
            normalise=True)
        assert np.isclose(DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED[i, :],
                          dist).all()
    for i in range(VECTOR_2D_CATEGORICAL_STRUCT_A2.shape[0]):
        dist = fud.binary_point_distance(VECTOR_2D_CATEGORICAL_STRUCT_A2[i],
                                         VECTOR_2D_CATEGORICAL_STRUCT_A1)
        assert np.isclose(DISTANCES_2D_CATEGORICAL_A_BINARY[:, i], dist).all()
        dist = fud.binary_point_distance(
            VECTOR_2D_CATEGORICAL_STRUCT_A2[i],
            VECTOR_2D_CATEGORICAL_STRUCT_A1,
            normalise=True)
        assert np.isclose(DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED[:, i],
                          dist).all()

    assert ((VECTOR_2D_NUMERICAL_STRUCT_A1.shape[0] ==
             DISTANCES_2D_NUMERICAL_A_BINARY.shape[0])
            and (DISTANCES_2D_NUMERICAL_A_BINARY.shape[0] ==
                 DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED.shape[0])
            and (VECTOR_2D_NUMERICAL_STRUCT_A2.shape[0] ==
                 DISTANCES_2D_NUMERICAL_A_BINARY.shape[1])
            and (DISTANCES_2D_NUMERICAL_A_BINARY.shape[1] ==
                 DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED.shape[1]))
    for i in range(VECTOR_2D_NUMERICAL_STRUCT_A1.shape[0]):
        dist = fud.binary_point_distance(VECTOR_2D_NUMERICAL_STRUCT_A1[i],
                                         VECTOR_2D_NUMERICAL_STRUCT_A2)
        assert np.isclose(DISTANCES_2D_NUMERICAL_A_BINARY[i, :], dist).all()
        dist = fud.binary_point_distance(
            VECTOR_2D_NUMERICAL_STRUCT_A1[i],
            VECTOR_2D_NUMERICAL_STRUCT_A2,
            normalise=True)
        assert np.isclose(DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED[i, :],
                          dist).all()
    for i in range(VECTOR_2D_NUMERICAL_STRUCT_A2.shape[0]):
        dist = fud.binary_point_distance(VECTOR_2D_NUMERICAL_STRUCT_A2[i],
                                         VECTOR_2D_NUMERICAL_STRUCT_A1)
        assert np.isclose(DISTANCES_2D_NUMERICAL_A_BINARY[:, i], dist).all()
        dist = fud.binary_point_distance(
            VECTOR_2D_NUMERICAL_STRUCT_A2[i],
            VECTOR_2D_NUMERICAL_STRUCT_A1,
            normalise=True)
        assert np.isclose(DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED[:, i],
                          dist).all()

    # Test unstructured-structured mixture
    assert ((VECTOR_2D_CATEGORICAL_STRUCT_A1.shape[0] ==
             DISTANCES_2D_CATEGORICAL_A_BINARY.shape[0])
            and (DISTANCES_2D_CATEGORICAL_A_BINARY.shape[0] ==
                 DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED.shape[0])
            and (VECTOR_2D_CATEGORICAL_A2.shape[0] ==
                 DISTANCES_2D_CATEGORICAL_A_BINARY.shape[1])
            and (DISTANCES_2D_CATEGORICAL_A_BINARY.shape[1] ==
                 DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED.shape[1]))
    for i in range(VECTOR_2D_CATEGORICAL_STRUCT_A1.shape[0]):
        dist = fud.binary_point_distance(VECTOR_2D_CATEGORICAL_STRUCT_A1[i],
                                         VECTOR_2D_CATEGORICAL_A2)
        assert np.isclose(DISTANCES_2D_CATEGORICAL_A_BINARY[i, :], dist).all()
        dist = fud.binary_point_distance(
            VECTOR_2D_CATEGORICAL_STRUCT_A1[i],
            VECTOR_2D_CATEGORICAL_A2,
            normalise=True)
        assert np.isclose(DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED[i, :],
                          dist).all()
    for i in range(VECTOR_2D_CATEGORICAL_A2.shape[0]):
        dist = fud.binary_point_distance(VECTOR_2D_CATEGORICAL_A2[i],
                                         VECTOR_2D_CATEGORICAL_STRUCT_A1)
        assert np.isclose(DISTANCES_2D_CATEGORICAL_A_BINARY[:, i], dist).all()
        dist = fud.binary_point_distance(
            VECTOR_2D_CATEGORICAL_A2[i],
            VECTOR_2D_CATEGORICAL_STRUCT_A1,
            normalise=True)
        assert np.isclose(DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED[:, i],
                          dist).all()

    assert ((VECTOR_2D_CATEGORICAL_A1.shape[0] ==
             DISTANCES_2D_CATEGORICAL_A_BINARY.shape[0])
            and (DISTANCES_2D_CATEGORICAL_A_BINARY.shape[0] ==
                 DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED.shape[0])
            and (VECTOR_2D_CATEGORICAL_STRUCT_A2.shape[0] ==
                 DISTANCES_2D_CATEGORICAL_A_BINARY.shape[1])
            and (DISTANCES_2D_CATEGORICAL_A_BINARY.shape[1] ==
                 DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED.shape[1]))
    for i in range(VECTOR_2D_CATEGORICAL_A1.shape[0]):
        dist = fud.binary_point_distance(VECTOR_2D_CATEGORICAL_A1[i],
                                         VECTOR_2D_CATEGORICAL_STRUCT_A2)
        assert np.isclose(DISTANCES_2D_CATEGORICAL_A_BINARY[i, :], dist).all()
        dist = fud.binary_point_distance(
            VECTOR_2D_CATEGORICAL_STRUCT_A1[i],
            VECTOR_2D_CATEGORICAL_STRUCT_A2,
            normalise=True)
        assert np.isclose(DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED[i, :],
                          dist).all()
    for i in range(VECTOR_2D_CATEGORICAL_STRUCT_A2.shape[0]):
        dist = fud.binary_point_distance(VECTOR_2D_CATEGORICAL_STRUCT_A2[i],
                                         VECTOR_2D_CATEGORICAL_A1)
        assert np.isclose(DISTANCES_2D_CATEGORICAL_A_BINARY[:, i], dist).all()
        dist = fud.binary_point_distance(
            VECTOR_2D_CATEGORICAL_STRUCT_A2[i],
            VECTOR_2D_CATEGORICAL_A1,
            normalise=True)
        assert np.isclose(DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED[:, i],
                          dist).all()

    assert ((VECTOR_2D_NUMERICAL_STRUCT_A1.shape[0] ==
             DISTANCES_2D_NUMERICAL_A_BINARY.shape[0])
            and (DISTANCES_2D_NUMERICAL_A_BINARY.shape[0] ==
                 DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED.shape[0])
            and (VECTOR_2D_NUMERICAL_A2.shape[0] ==
                 DISTANCES_2D_NUMERICAL_A_BINARY.shape[1])
            and (DISTANCES_2D_NUMERICAL_A_BINARY.shape[1] ==
                 DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED.shape[1]))
    for i in range(VECTOR_2D_NUMERICAL_STRUCT_A1.shape[0]):
        dist = fud.binary_point_distance(VECTOR_2D_NUMERICAL_STRUCT_A1[i],
                                         VECTOR_2D_NUMERICAL_A2)
        assert np.isclose(DISTANCES_2D_NUMERICAL_A_BINARY[i, :], dist).all()
        dist = fud.binary_point_distance(
            VECTOR_2D_NUMERICAL_STRUCT_A1[i],
            VECTOR_2D_NUMERICAL_A2,
            normalise=True)
        assert np.isclose(DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED[i, :],
                          dist).all()
    for i in range(VECTOR_2D_NUMERICAL_A2.shape[0]):
        dist = fud.binary_point_distance(VECTOR_2D_NUMERICAL_A2[i],
                                         VECTOR_2D_NUMERICAL_STRUCT_A1)
        assert np.isclose(DISTANCES_2D_NUMERICAL_A_BINARY[:, i], dist).all()
        dist = fud.binary_point_distance(
            VECTOR_2D_NUMERICAL_A2[i],
            VECTOR_2D_NUMERICAL_STRUCT_A1,
            normalise=True)
        assert np.isclose(DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED[:, i],
                          dist).all()

    assert ((VECTOR_2D_NUMERICAL_A1.shape[0] ==
             DISTANCES_2D_NUMERICAL_A_BINARY.shape[0])
            and (DISTANCES_2D_NUMERICAL_A_BINARY.shape[0] ==
                 DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED.shape[0])
            and (VECTOR_2D_NUMERICAL_STRUCT_A2.shape[0] ==
                 DISTANCES_2D_NUMERICAL_A_BINARY.shape[1])
            and (DISTANCES_2D_NUMERICAL_A_BINARY.shape[1] ==
                 DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED.shape[1]))
    for i in range(VECTOR_2D_NUMERICAL_A1.shape[0]):
        dist = fud.binary_point_distance(VECTOR_2D_NUMERICAL_A1[i],
                                         VECTOR_2D_NUMERICAL_STRUCT_A2)
        assert np.isclose(DISTANCES_2D_NUMERICAL_A_BINARY[i, :], dist).all()
        dist = fud.binary_point_distance(
            VECTOR_2D_NUMERICAL_STRUCT_A1[i],
            VECTOR_2D_NUMERICAL_STRUCT_A2,
            normalise=True)
        assert np.isclose(DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED[i, :],
                          dist).all()
    for i in range(VECTOR_2D_NUMERICAL_STRUCT_A2.shape[0]):
        dist = fud.binary_point_distance(VECTOR_2D_NUMERICAL_STRUCT_A2[i],
                                         VECTOR_2D_NUMERICAL_A1)
        assert np.isclose(DISTANCES_2D_NUMERICAL_A_BINARY[:, i], dist).all()
        dist = fud.binary_point_distance(
            VECTOR_2D_NUMERICAL_STRUCT_A2[i],
            VECTOR_2D_NUMERICAL_A1,
            normalise=True)
        assert np.isclose(DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED[:, i],
                          dist).all()

    # Test binary point distance for 1-element voids
    dist = fud.binary_point_distance(
        VECTOR_2D_CATEGORICAL_STRUCT_A1[['name']][1],
        VECTOR_2D_CATEGORICAL_STRUCT_A1[['name']])
    assert np.array_equal(dist, [1, 0, 1])


def test_binary_array_distance():
    """
    Tests :func:`fatf.utils.distances.binary_array_distance`.
    """
    # pylint: disable=too-many-statements
    shape_error_X = 'The X array should be 2-dimensional.'
    shape_error_Y = 'The Y array should be 2-dimensional.'
    shape_error_XY = ('The number of columns in the X array should the same '
                      'as the number of columns in Y array.')

    # X or Y is not 2D
    with pytest.raises(IncorrectShapeError) as exin:
        fud.binary_array_distance(VECTOR_1D_CATEGORICAL_A1, VECTOR_0D)
    assert str(exin.value) == shape_error_X
    with pytest.raises(IncorrectShapeError) as exin:
        fud.binary_array_distance(VECTOR_0D, VECTOR_2D_NUMERICAL_A1)
    assert str(exin.value) == shape_error_X
    with pytest.raises(IncorrectShapeError) as exin:
        fud.binary_array_distance(VECTOR_2D_NUMERICAL_A1, VECTOR_0D)
    assert str(exin.value) == shape_error_Y
    with pytest.raises(IncorrectShapeError) as exin:
        fud.binary_array_distance(VECTOR_2D_NUMERICAL_STRUCT_A2,
                                  VECTOR_1D_CATEGORICAL_A1)
    assert str(exin.value) == shape_error_Y

    # The shape of input arrays does not match
    with pytest.raises(IncorrectShapeError) as exin:
        fud.binary_array_distance(VECTOR_2D_CATEGORICAL_STRUCT_A2,
                                  VECTORS_1D_CATEGORICAL_A2)
    assert str(exin.value) == shape_error_XY
    with pytest.raises(IncorrectShapeError) as exin:
        fud.binary_array_distance(VECTORS_1D_CATEGORICAL_A1,
                                  VECTOR_2D_CATEGORICAL_A1)
    assert str(exin.value) == shape_error_XY

    # Test unstructured arrays
    assert ((VECTOR_2D_CATEGORICAL_A1.shape[0] ==
             DISTANCES_2D_CATEGORICAL_A_BINARY.shape[0])
            and (DISTANCES_2D_CATEGORICAL_A_BINARY.shape[0] ==
                 DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED.shape[0])
            and (VECTOR_2D_CATEGORICAL_A2.shape[0] ==
                 DISTANCES_2D_CATEGORICAL_A_BINARY.shape[1])
            and (DISTANCES_2D_CATEGORICAL_A_BINARY.shape[1] ==
                 DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED.shape[1]))
    dist = fud.binary_array_distance(VECTOR_2D_CATEGORICAL_A1,
                                     VECTOR_2D_CATEGORICAL_A2)
    assert np.isclose(DISTANCES_2D_CATEGORICAL_A_BINARY, dist).all()
    dist = fud.binary_array_distance(
        VECTOR_2D_CATEGORICAL_A1, VECTOR_2D_CATEGORICAL_A2, normalise=True)
    assert np.isclose(DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED, dist).all()
    dist = fud.binary_array_distance(VECTOR_2D_CATEGORICAL_A2,
                                     VECTOR_2D_CATEGORICAL_A1)
    assert np.isclose(DISTANCES_2D_CATEGORICAL_A_BINARY.T, dist).all()
    dist = fud.binary_array_distance(
        VECTOR_2D_CATEGORICAL_A2, VECTOR_2D_CATEGORICAL_A1, normalise=True)
    assert np.isclose(DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED.T,
                      dist).all()

    assert ((VECTOR_2D_NUMERICAL_A1.shape[0] ==
             DISTANCES_2D_NUMERICAL_A_BINARY.shape[0])
            and (DISTANCES_2D_NUMERICAL_A_BINARY.shape[0] ==
                 DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED.shape[0])
            and (VECTOR_2D_NUMERICAL_A2.shape[0] ==
                 DISTANCES_2D_NUMERICAL_A_BINARY.shape[1])
            and (DISTANCES_2D_NUMERICAL_A_BINARY.shape[1] ==
                 DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED.shape[1]))
    dist = fud.binary_array_distance(VECTOR_2D_NUMERICAL_A1,
                                     VECTOR_2D_NUMERICAL_A2)
    assert np.isclose(DISTANCES_2D_NUMERICAL_A_BINARY, dist).all()
    dist = fud.binary_array_distance(
        VECTOR_2D_NUMERICAL_A1, VECTOR_2D_NUMERICAL_A2, normalise=True)
    assert np.isclose(DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED, dist).all()
    dist = fud.binary_array_distance(VECTOR_2D_NUMERICAL_A2,
                                     VECTOR_2D_NUMERICAL_A1)
    assert np.isclose(DISTANCES_2D_NUMERICAL_A_BINARY.T, dist).all()
    dist = fud.binary_array_distance(
        VECTOR_2D_NUMERICAL_A2, VECTOR_2D_NUMERICAL_A1, normalise=True)
    assert np.isclose(DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED.T, dist).all()

    # Test structured arrays
    assert ((VECTOR_2D_CATEGORICAL_STRUCT_A1.shape[0] ==
             DISTANCES_2D_CATEGORICAL_A_BINARY.shape[0])
            and (DISTANCES_2D_CATEGORICAL_A_BINARY.shape[0] ==
                 DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED.shape[0])
            and (VECTOR_2D_CATEGORICAL_STRUCT_A2.shape[0] ==
                 DISTANCES_2D_CATEGORICAL_A_BINARY.shape[1])
            and (DISTANCES_2D_CATEGORICAL_A_BINARY.shape[1] ==
                 DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED.shape[1]))
    dist = fud.binary_array_distance(VECTOR_2D_CATEGORICAL_STRUCT_A1,
                                     VECTOR_2D_CATEGORICAL_STRUCT_A2)
    assert np.isclose(DISTANCES_2D_CATEGORICAL_A_BINARY, dist).all()
    dist = fud.binary_array_distance(
        VECTOR_2D_CATEGORICAL_STRUCT_A1,
        VECTOR_2D_CATEGORICAL_STRUCT_A2,
        normalise=True)
    assert np.isclose(DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED, dist).all()
    dist = fud.binary_array_distance(VECTOR_2D_CATEGORICAL_STRUCT_A2,
                                     VECTOR_2D_CATEGORICAL_STRUCT_A1)
    assert np.isclose(DISTANCES_2D_CATEGORICAL_A_BINARY.T, dist).all()
    dist = fud.binary_array_distance(
        VECTOR_2D_CATEGORICAL_STRUCT_A2,
        VECTOR_2D_CATEGORICAL_STRUCT_A1,
        normalise=True)
    assert np.isclose(DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED.T,
                      dist).all()

    assert ((VECTOR_2D_NUMERICAL_STRUCT_A1.shape[0] ==
             DISTANCES_2D_NUMERICAL_A_BINARY.shape[0])
            and (DISTANCES_2D_NUMERICAL_A_BINARY.shape[0] ==
                 DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED.shape[0])
            and (VECTOR_2D_NUMERICAL_STRUCT_A2.shape[0] ==
                 DISTANCES_2D_NUMERICAL_A_BINARY.shape[1])
            and (DISTANCES_2D_NUMERICAL_A_BINARY.shape[1] ==
                 DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED.shape[1]))
    dist = fud.binary_array_distance(VECTOR_2D_NUMERICAL_STRUCT_A1,
                                     VECTOR_2D_NUMERICAL_STRUCT_A2)
    assert np.isclose(DISTANCES_2D_NUMERICAL_A_BINARY, dist).all()
    dist = fud.binary_array_distance(
        VECTOR_2D_NUMERICAL_STRUCT_A1,
        VECTOR_2D_NUMERICAL_STRUCT_A2,
        normalise=True)
    assert np.isclose(DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED, dist).all()
    dist = fud.binary_array_distance(VECTOR_2D_NUMERICAL_STRUCT_A2,
                                     VECTOR_2D_NUMERICAL_STRUCT_A1)
    assert np.isclose(DISTANCES_2D_NUMERICAL_A_BINARY.T, dist).all()
    dist = fud.binary_array_distance(
        VECTOR_2D_NUMERICAL_STRUCT_A2,
        VECTOR_2D_NUMERICAL_STRUCT_A1,
        normalise=True)
    assert np.isclose(DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED.T, dist).all()

    # Test unstructured-structured mixture
    assert ((VECTOR_2D_CATEGORICAL_A1.shape[0] ==
             DISTANCES_2D_CATEGORICAL_A_BINARY.shape[0])
            and (DISTANCES_2D_CATEGORICAL_A_BINARY.shape[0] ==
                 DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED.shape[0])
            and (VECTOR_2D_CATEGORICAL_STRUCT_A2.shape[0] ==
                 DISTANCES_2D_CATEGORICAL_A_BINARY.shape[1])
            and (DISTANCES_2D_CATEGORICAL_A_BINARY.shape[1] ==
                 DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED.shape[1]))
    dist = fud.binary_array_distance(VECTOR_2D_CATEGORICAL_A1,
                                     VECTOR_2D_CATEGORICAL_STRUCT_A2)
    assert np.isclose(DISTANCES_2D_CATEGORICAL_A_BINARY, dist).all()
    dist = fud.binary_array_distance(
        VECTOR_2D_CATEGORICAL_A1,
        VECTOR_2D_CATEGORICAL_STRUCT_A2,
        normalise=True)
    assert np.isclose(DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED, dist).all()
    dist = fud.binary_array_distance(VECTOR_2D_CATEGORICAL_STRUCT_A2,
                                     VECTOR_2D_CATEGORICAL_A1)
    assert np.isclose(DISTANCES_2D_CATEGORICAL_A_BINARY.T, dist).all()
    dist = fud.binary_array_distance(
        VECTOR_2D_CATEGORICAL_STRUCT_A2,
        VECTOR_2D_CATEGORICAL_A1,
        normalise=True)
    assert np.isclose(DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED.T,
                      dist).all()

    assert ((VECTOR_2D_CATEGORICAL_STRUCT_A1.shape[0] ==
             DISTANCES_2D_CATEGORICAL_A_BINARY.shape[0])
            and (DISTANCES_2D_CATEGORICAL_A_BINARY.shape[0] ==
                 DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED.shape[0])
            and (VECTOR_2D_CATEGORICAL_A2.shape[0] ==
                 DISTANCES_2D_CATEGORICAL_A_BINARY.shape[1])
            and (DISTANCES_2D_CATEGORICAL_A_BINARY.shape[1] ==
                 DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED.shape[1]))
    dist = fud.binary_array_distance(VECTOR_2D_CATEGORICAL_STRUCT_A1,
                                     VECTOR_2D_CATEGORICAL_A2)
    assert np.isclose(DISTANCES_2D_CATEGORICAL_A_BINARY, dist).all()
    dist = fud.binary_array_distance(
        VECTOR_2D_CATEGORICAL_STRUCT_A1,
        VECTOR_2D_CATEGORICAL_A2,
        normalise=True)
    assert np.isclose(DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED, dist).all()
    dist = fud.binary_array_distance(VECTOR_2D_CATEGORICAL_A2,
                                     VECTOR_2D_CATEGORICAL_STRUCT_A1)
    assert np.isclose(DISTANCES_2D_CATEGORICAL_A_BINARY.T, dist).all()
    dist = fud.binary_array_distance(
        VECTOR_2D_CATEGORICAL_A2,
        VECTOR_2D_CATEGORICAL_STRUCT_A1,
        normalise=True)
    assert np.isclose(DISTANCES_2D_CATEGORICAL_A_BINARY_NORMALISED.T,
                      dist).all()

    assert ((VECTOR_2D_NUMERICAL_A1.shape[0] ==
             DISTANCES_2D_NUMERICAL_A_BINARY.shape[0])
            and (DISTANCES_2D_NUMERICAL_A_BINARY.shape[0] ==
                 DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED.shape[0])
            and (VECTOR_2D_NUMERICAL_STRUCT_A2.shape[0] ==
                 DISTANCES_2D_NUMERICAL_A_BINARY.shape[1])
            and (DISTANCES_2D_NUMERICAL_A_BINARY.shape[1] ==
                 DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED.shape[1]))
    dist = fud.binary_array_distance(VECTOR_2D_NUMERICAL_A1,
                                     VECTOR_2D_NUMERICAL_STRUCT_A2)
    assert np.isclose(DISTANCES_2D_NUMERICAL_A_BINARY, dist).all()
    dist = fud.binary_array_distance(
        VECTOR_2D_NUMERICAL_A1, VECTOR_2D_NUMERICAL_STRUCT_A2, normalise=True)
    assert np.isclose(DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED, dist).all()
    dist = fud.binary_array_distance(VECTOR_2D_NUMERICAL_STRUCT_A2,
                                     VECTOR_2D_NUMERICAL_A1)
    assert np.isclose(DISTANCES_2D_NUMERICAL_A_BINARY.T, dist).all()
    dist = fud.binary_array_distance(
        VECTOR_2D_NUMERICAL_STRUCT_A2, VECTOR_2D_NUMERICAL_A1, normalise=True)
    assert np.isclose(DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED.T, dist).all()

    assert ((VECTOR_2D_NUMERICAL_STRUCT_A1.shape[0] ==
             DISTANCES_2D_NUMERICAL_A_BINARY.shape[0])
            and (DISTANCES_2D_NUMERICAL_A_BINARY.shape[0] ==
                 DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED.shape[0])
            and (VECTOR_2D_NUMERICAL_A2.shape[0] ==
                 DISTANCES_2D_NUMERICAL_A_BINARY.shape[1])
            and (DISTANCES_2D_NUMERICAL_A_BINARY.shape[1] ==
                 DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED.shape[1]))
    dist = fud.binary_array_distance(VECTOR_2D_NUMERICAL_STRUCT_A1,
                                     VECTOR_2D_NUMERICAL_A2)
    assert np.isclose(DISTANCES_2D_NUMERICAL_A_BINARY, dist).all()
    dist = fud.binary_array_distance(
        VECTOR_2D_NUMERICAL_STRUCT_A1, VECTOR_2D_NUMERICAL_A2, normalise=True)
    assert np.isclose(DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED, dist).all()
    dist = fud.binary_array_distance(VECTOR_2D_NUMERICAL_A2,
                                     VECTOR_2D_NUMERICAL_STRUCT_A1)
    assert np.isclose(DISTANCES_2D_NUMERICAL_A_BINARY.T, dist).all()
    dist = fud.binary_array_distance(
        VECTOR_2D_NUMERICAL_A2, VECTOR_2D_NUMERICAL_STRUCT_A1, normalise=True)
    assert np.isclose(DISTANCES_2D_NUMERICAL_A_BINARY_NORMALISED.T, dist).all()


def test_get_distance_matrix():
    """
    Tests :func:`fatf.utils.distances.get_distance_matrix` function.
    """
    shape_error_data = ('The data_array has to be a 2-dimensional (structured '
                        'or unstructured) numpy array.')
    type_error_data = ('The data_array has to be of a base type (strings '
                       'and/or numbers).')
    type_error_func = ('The distance function should be a Python callable '
                       '(function).')
    attribute_error_func = ('The distance function must require exactly 2 '
                            'parameters. Given function requires {} '
                            'parameters.')

    with pytest.raises(IncorrectShapeError) as exin:
        fud.get_distance_matrix(VECTOR_0D, None)
    assert str(exin.value) == shape_error_data

    with pytest.raises(TypeError) as exin:
        fud.get_distance_matrix(np.array([[None]]), None)
    assert str(exin.value) == type_error_data

    with pytest.raises(TypeError) as exin:
        fud.get_distance_matrix(VECTOR_2D_CATEGORICAL_A1, None)
    assert str(exin.value) == type_error_func

    def bad_distance(x):
        return x + 5  # pragma: nocover

    with pytest.raises(AttributeError) as exin:
        fud.get_distance_matrix(VECTOR_2D_CATEGORICAL_A1, bad_distance)
    assert str(exin.value) == attribute_error_func.format(1)

    ###########################################################################

    true_distances = fud.euclidean_array_distance(VECTOR_2D_NUMERICAL_A1,
                                                  VECTOR_2D_NUMERICAL_A1)
    # Numpy array numerical
    distances = fud.get_distance_matrix(VECTOR_2D_NUMERICAL_A1,
                                        fud.euclidean_distance)
    assert np.allclose(distances, true_distances, atol=1e-3)

    # Structured array numerical
    def struct_dist(x, y):
        return fud.euclidean_distance(
            np.array([x['a'], x['b']]), np.array([y['a'], y['b']]))

    distances = fud.get_distance_matrix(VECTOR_2D_NUMERICAL_STRUCT_A1,
                                        struct_dist)
    assert np.allclose(distances, true_distances, atol=1e-3)

    ###########################################################################

    true_distances = fud.binary_array_distance(VECTOR_2D_CATEGORICAL_A1,
                                               VECTOR_2D_CATEGORICAL_A1)
    # Numpy array string-based
    distances = fud.get_distance_matrix(VECTOR_2D_CATEGORICAL_A1,
                                        fud.binary_distance)
    assert np.allclose(distances, true_distances, atol=1e-3)

    # Structured array string-based
    def struct_dist(x, y):
        return fud.binary_distance(
            np.array([x['a'], x['name']]), np.array([y['a'], y['name']]))

    distances = fud.get_distance_matrix(VECTOR_2D_CATEGORICAL_STRUCT_A1,
                                        struct_dist)
    assert np.allclose(distances, true_distances, atol=1e-3)

    ###########################################################################

    # Mixed array
    mixed_array = np.array(
        [('a', 5, 'bb', 14),
         ('b', 3, 'aa', 12),
         ('a', 4, 'ab', 11)],
        dtype=[('a1', 'U1'), ('b1', int),
               ('a2', 'U2'), ('b2', int)])  # yapf: disable
    true_distances = np.array([[0, 1 + 1 + np.sqrt(8), 0 + 1 + np.sqrt(10)],
                               [1 + 1 + np.sqrt(8), 0, 1 + 1 + np.sqrt(2)],
                               [0 + 1 + np.sqrt(10), 1 + 1 + np.sqrt(2), 0]])

    def mix_dist(x, y):
        num = fud.euclidean_distance(
            np.array([x['b1'], x['b2']]), np.array([y['b1'], y['b2']]))
        cat = fud.binary_distance(
            np.array([x['a1'], x['a2']]), np.array([y['a1'], y['a2']]))
        return num + cat

    distances = fud.get_distance_matrix(mixed_array, mix_dist)
    assert np.allclose(distances, true_distances, atol=1e-3)


def test_get_point_distance():
    """
    Tests :func:`fatf.utils.distances.get_point_distance` function.
    """
    # test_get_distance_matrix already tests for data_array and data_point
    # validity
    shape_error_data_point = ('The data point has to be 1-dimensional numpy '
                              'array or numpy void (for structured arrays).')
    shape_error_columns = ('The data point has different number of columns '
                           '(features) than the data set.')
    type_error_base = ('The data point has to be of a base type (strings '
                       'and/or numbers).')
    type_error_dtype = ('The dtypes of the data set and the data point are '
                        'too different.')

    with pytest.raises(IncorrectShapeError) as exin:
        fud.get_point_distance(VECTOR_2D_CATEGORICAL_A1,
                               VECTOR_2D_CATEGORICAL_A1, fud.binary_distance)
    assert str(exin.value) == shape_error_data_point

    with pytest.raises(IncorrectShapeError) as exin:
        fud.get_point_distance(VECTOR_2D_NUMERICAL_A1, np.array([1, 2, 3]),
                               fud.binary_distance)
    assert str(exin.value) == shape_error_columns

    with pytest.raises(TypeError) as exin:
        fud.get_point_distance(VECTOR_2D_NUMERICAL_A1, np.array([1, None, 3]),
                               fud.binary_distance)
    assert str(exin.value) == type_error_base

    with pytest.raises(TypeError) as exin:
        fud.get_point_distance(VECTOR_2D_NUMERICAL_A1, np.array([1, '2', 3]),
                               fud.binary_distance)
    assert str(exin.value) == type_error_dtype
    with pytest.raises(TypeError) as exin:
        fud.get_point_distance(VECTOR_2D_CATEGORICAL_STRUCT_A1,
                               np.array([1, '2', 3]), fud.binary_distance)
    assert str(exin.value) == type_error_dtype
    with pytest.raises(TypeError) as exin:
        fud.get_point_distance(VECTOR_2D_CATEGORICAL_STRUCT_A1,
                               VECTOR_2D_NUMERICAL_STRUCT_A1[0],
                               fud.binary_distance)
    assert str(exin.value) == type_error_dtype

    ###########################################################################

    true_distances = fud.euclidean_array_distance(VECTOR_2D_NUMERICAL_A1,
                                                  VECTOR_2D_NUMERICAL_A2)
    # Numpy array numerical
    distance = fud.get_point_distance(VECTOR_2D_NUMERICAL_A1,
                                      VECTOR_2D_NUMERICAL_A2[0],
                                      fud.euclidean_distance)
    assert np.allclose(distance, true_distances[:, 0], atol=1e-3)

    # Structured array numerical
    def struct_dist(x, y):
        return fud.euclidean_distance(
            np.array([x['a'], x['b']]), np.array([y['a'], y['b']]))

    distance = fud.get_point_distance(VECTOR_2D_NUMERICAL_STRUCT_A1,
                                      VECTOR_2D_NUMERICAL_STRUCT_A2[0],
                                      struct_dist)
    assert np.allclose(distance, true_distances[:, 0], atol=1e-3)

    ###########################################################################

    true_distances = fud.binary_array_distance(VECTOR_2D_CATEGORICAL_A1,
                                               VECTOR_2D_CATEGORICAL_A2)
    # Numpy array string-based
    distance = fud.get_point_distance(VECTOR_2D_CATEGORICAL_A1,
                                      VECTOR_2D_CATEGORICAL_A2[0],
                                      fud.binary_distance)
    assert np.allclose(distance, true_distances[:, 0], atol=1e-3)

    # Structured array string-based
    def struct_dist(x, y):
        return fud.binary_distance(
            np.array([x['a'], x['name']]), np.array([y['a'], y['name']]))

    distance = fud.get_point_distance(VECTOR_2D_CATEGORICAL_STRUCT_A1,
                                      VECTOR_2D_CATEGORICAL_STRUCT_A2[0],
                                      struct_dist)
    assert np.allclose(distance, true_distances[:, 0], atol=1e-3)

    ###########################################################################

    # Mixed array
    mixed_array = np.array(
        [('a', 5, 'bb', 14),
         ('b', 3, 'aa', 12),
         ('a', 4, 'ab', 11)],
        dtype=[('a1', 'U1'), ('b1', int),
               ('a2', 'U2'), ('b2', int)])  # yapf: disable
    true_distances = np.array([[0, 1 + 1 + np.sqrt(8), 0 + 1 + np.sqrt(10)],
                               [1 + 1 + np.sqrt(8), 0, 1 + 1 + np.sqrt(2)],
                               [0 + 1 + np.sqrt(10), 1 + 1 + np.sqrt(2), 0]])

    def mix_dist(x, y):
        num = fud.euclidean_distance(
            np.array([x['b1'], x['b2']]), np.array([y['b1'], y['b2']]))
        cat = fud.binary_distance(
            np.array([x['a1'], x['a2']]), np.array([y['a1'], y['a2']]))
        return num + cat

    distance = fud.get_point_distance(mixed_array, mixed_array[0], mix_dist)
    assert np.allclose(distance, true_distances[0], atol=1e-3)


def test_check_distance_functionality():
    """
    Tests :func:`fatf.utils.distances.check_distance_functionality` function.
    """
    distance_type_error = ('The distance_function parameter should be a '
                           'Python callable.')
    suppress_type_error = 'The suppress_warning parameter should be a boolean.'

    error_msg = ("The '{}' distance function has incorrect number "
                 '({}) of the required parameters. It needs to have '
                 'exactly 2 required parameters. Try using optional '
                 'parameters if you require more functionality.')

    def function1():
        pass  # pragma: no cover

    def function2(x):
        pass  # pragma: no cover

    def function3(x, y):
        pass  # pragma: no cover

    def function4(x, y, z=3):
        pass  # pragma: no cover

    def function5(x=3, y=3):
        pass  # pragma: no cover

    def function6(x, y, **kwargs):
        pass  # pragma: no cover

    with pytest.raises(TypeError) as exin:
        fud.check_distance_functionality('callable')
    assert str(exin.value) == distance_type_error
    with pytest.raises(TypeError) as exin:
        fud.check_distance_functionality('callable', 'True')
    assert str(exin.value) == distance_type_error
    with pytest.raises(TypeError) as exin:
        fud.check_distance_functionality(function1, 'True')
    assert str(exin.value) == suppress_type_error

    with pytest.warns(UserWarning) as warning:
        assert fud.check_distance_functionality(function1) is False
    assert len(warning) == 1
    assert str(warning[0].message) == error_msg.format('function1', 0)
    #
    with pytest.warns(UserWarning) as warning:
        assert fud.check_distance_functionality(function1, False) is False
    assert len(warning) == 1
    assert str(warning[0].message) == error_msg.format('function1', 0)
    #
    assert fud.check_distance_functionality(function1, True) is False

    with pytest.warns(UserWarning) as warning:
        assert fud.check_distance_functionality(function5) is False
    assert len(warning) == 1
    assert str(warning[0].message) == error_msg.format('function5', 0)
    #
    with pytest.warns(UserWarning) as warning:
        assert fud.check_distance_functionality(function5, False) is False
    assert len(warning) == 1
    assert str(warning[0].message) == error_msg.format('function5', 0)
    #
    assert fud.check_distance_functionality(function5, True) is False

    with pytest.warns(UserWarning) as warning:
        assert fud.check_distance_functionality(function2) is False
    assert len(warning) == 1
    assert str(warning[0].message) == error_msg.format('function2', 1)
    #
    with pytest.warns(UserWarning) as warning:
        assert fud.check_distance_functionality(function2, False) is False
    assert len(warning) == 1
    assert str(warning[0].message) == error_msg.format('function2', 1)
    #
    assert fud.check_distance_functionality(function2, True) is False

    assert fud.check_distance_functionality(function3) is True
    assert fud.check_distance_functionality(function4, True) is True
    assert fud.check_distance_functionality(function6, False) is True
