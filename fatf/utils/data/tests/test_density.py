"""
Functions for testing data density check classes.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
#         Rafael Poyiadzi <rp13102@bristol.ac.uk>
# License: new BSD

import pytest

import numpy as np

from fatf.exceptions import IncorrectShapeError

import fatf.utils.data.density as fudd
import fatf.utils.tools as fut

_NUMPY_VERSION = [int(i) for i in np.version.version.split('.')]
_NUMPY_1_14 = fut.at_least_verion([1, 14], _NUMPY_VERSION)

# yapf: disable
NUMERICAL_NP_ARRAY = np.array(
    [[74, 52], [3, 86], [26, 56], [70, 57], [48, 57], [30, 98], [41, 73],
     [24, 1], [44, 66], [62, 96], [63, 51], [26, 88], [94, 64], [59, 19],
     [14, 88], [16, 15], [94, 48], [41, 25], [36, 57], [37, 52]])
NUMERICAL_STRUCT_ARRAY = np.array(
    [(74, 52), (3, 86), (26, 56), (70, 57), (48, 57), (30, 98), (41, 73),
     (24, 1), (44, 66), (62, 96), (63, 51), (26, 88), (94, 64), (59, 19),
     (14, 88), (16, 15), (94, 48), (41, 25), (36, 57), (37, 52)],
    dtype=[('a', int), ('b', np.int16)])
CATEGORICAL_NP_ARRAY = np.array(
    [['a', 'aa'], ['a', 'ab'], ['a', 'ba'], ['b', 'bb'], ['b', 'ba'],
     ['b', 'ab'], ['a', 'ac'], ['a', 'ca'], ['b', 'bc'], ['b', 'cb']])
CATEGORICAL_STRUCT_ARRAY = np.array(
    [('a', 'aa'), ('a', 'ab'), ('a', 'ba'), ('b', 'bb'), ('b', 'ba'),
     ('b', 'ab'), ('a', 'ac'), ('a', 'ca'), ('b', 'bc'), ('b', 'cb')],
    dtype=[('x', 'U1'), ('xx', 'U2')])
MIXED_ARRAY = np.array(
    [(74, 52, 'a', 'aa'), (3, 86, 'a', 'ab'), (26, 56, 'a', 'ba'),
     (70, 57, 'b', 'bb'), (48, 57, 'b', 'ba'), (30, 98, 'b', 'ab'),
     (41, 73, 'a', 'ac'), (24, 1, 'a', 'ca'), (44, 66, 'b', 'bc'),
     (62, 96, 'b', 'cb')],
    dtype=[('i', int), ('ii', int), ('x', 'U1'), ('xx', 'U2')])

NUMERICAL_SCORES = np.array([
    36.249, 45.618, 34.176, 33.121, 22.023, 41.437, 27.313, 60.926, 27.514,
    45.011, 31.113, 32.573, 50.040, 39.661, 38.013, 52.802, 53.141, 34.059,
    32.388, 27.295])
NUMERICAL_DISTS_A = np.array([
    [0, 35, 5, 6, 6, 47, 22, 52, 15, 45, 2, 37, 13, 34, 37, 38, 5, 28, 6, 1],
    [
        35, 0, 31, 30, 30, 13, 14, 86, 21, 11, 36, 3, 23, 68, 3, 72, 39, 62,
        30, 35
    ],
    [5, 31, 0, 2, 2, 43, 18, 56, 11, 41, 6, 32, 9, 38, 33, 42, 9, 32, 2, 5],
    [6, 30, 2, 0, 1, 42, 17, 57, 10, 40, 7, 32, 8, 39, 32, 43, 10, 33, 1, 6],
    [6, 30, 2, 1, 0, 42, 17, 57, 10, 40, 7, 32, 8, 39, 32, 43, 10, 33, 1, 6],
    [
        47, 13, 43, 42, 42, 0, 26, 98, 33, 3, 48, 11, 35, 80, 11, 84, 51, 74,
        42, 47
    ],
    [
        22, 14, 18, 17, 17, 26, 0, 73, 8, 24, 23, 16, 10, 55, 16, 59, 26, 48,
        17, 22
    ],
    [
        52, 86, 56, 57, 57, 98, 73, 0, 66, 96, 51, 88, 64, 19, 88, 15, 48, 25,
        57, 52
    ],
    [
        15, 21, 11, 10, 10, 33, 8, 66, 0, 31, 16, 23, 3, 48, 23, 52, 19, 42,
        10, 15
    ],
    [
        45, 11, 41, 40, 40, 3, 24, 96, 31, 0, 46, 9, 33, 78, 9, 82, 49, 72,
        40, 45
    ],
    [2, 36, 6, 7, 7, 48, 23, 51, 16, 46, 0, 38, 14, 33, 38, 37, 4, 27, 7, 2],
    [
        37, 3, 32, 32, 32, 11, 16, 88, 23, 9, 38, 0, 25, 70, 1, 74, 41, 64,
        32, 37
    ],
    [13, 23, 9, 8, 8, 35, 10, 64, 3, 33, 14, 25, 0, 46, 25, 50, 16, 40, 8, 13],
    [
        34, 68, 38, 39, 39, 80, 55, 19, 48, 78, 33, 70, 46, 0, 70, 5, 30, 7,
        39, 34
    ],
    [
        37, 3, 33, 32, 32, 11, 16, 88, 23, 9, 38, 1, 25, 70, 0, 74, 41, 64,
        32, 37
    ],
    [
        38, 72, 42, 43, 43, 84, 59, 15, 52, 82, 37, 74, 50, 5, 74, 0, 34, 11,
        43, 38
    ],
    [
        5, 39, 9, 10, 10, 51, 26, 48, 19, 49, 4, 41, 16, 30, 41, 34, 0, 24,
        10, 5
    ],
    [
        28, 62, 32, 33, 33, 74, 48, 25, 42, 72, 27, 64, 40, 7, 64, 11, 24, 0,
        33, 28
    ],
    [6, 30, 2, 1, 1, 42, 17, 57, 10, 40, 7, 32, 8, 39, 32, 43, 10, 33, 0, 6],
    [1, 35, 5, 6, 6, 47, 22, 52, 15, 45, 2, 37, 13, 34, 37, 38, 5, 28, 6, 0]])
NUMERICAL_SCORES_A = np.array([
    6, 23, 9, 8, 8, 35, 17, 52, 15, 33, 7, 25, 13, 34, 25, 38, 10, 28, 8, 6])
CATEGORICAL_DISTS = np.array([[0, 1, 1, 2, 2, 2, 1, 1, 2, 2],
                              [1, 0, 1, 2, 2, 1, 1, 1, 2, 2],
                              [1, 1, 0, 2, 1, 2, 1, 1, 2, 2],
                              [2, 2, 2, 0, 1, 1, 2, 2, 1, 1],
                              [2, 2, 1, 1, 0, 1, 2, 2, 1, 1],
                              [2, 1, 2, 1, 1, 0, 2, 2, 1, 1],
                              [1, 1, 1, 2, 2, 2, 0, 1, 2, 2],
                              [1, 1, 1, 2, 2, 2, 1, 0, 2, 2],
                              [2, 2, 2, 1, 1, 1, 2, 2, 0, 1],
                              [2, 2, 2, 1, 1, 1, 2, 2, 1, 0]])
CATEGORICAL_SCORES = np.array([1, 0, 0, 1, 0, 0, 1, 1, 1, 1])
MIXED_DISTS = np.array([
    [0, 79.721, 49.166, 8.403, 28.476, 65.655, 40.115, 72.421, 35.106, 47.607],
    [
        79.721, 0, 38.802, 75.007, 55.535, 30.547, 41.162, 88.556,
        47.618, 61.841],
    [
        49.166, 38.802, 0, 46.011, 23.023, 44.190, 23.672, 56.036,
        22.591, 55.814
    ],
    [8.403, 75.007, 46.011, 0, 23, 58.280, 35.121, 74.471, 28.514, 40.812],
    [28.476, 55.535, 23.023, 23, 0, 45.777, 19.464, 62.926, 10.849, 42.437],
    [
        65.655, 30.547, 44.190, 58.280, 45.777, 0, 29.313, 99.185,
        35.928, 33.062
    ],
    [40.115, 41.162, 23.672, 35.121, 19.464, 29.313, 0, 74.980, 9.616, 33.145],
    [
        72.421, 88.556, 56.036, 74.471, 62.926, 99.185, 74.980, 0,
        70.007, 104.318
    ],
    [35.106, 47.618, 22.591, 28.514, 10.849, 35.928, 9.616, 70.007, 0, 35.986],
    [
        47.607, 61.841, 55.814, 40.812, 42.4367, 33.062, 33.145, 104.318,
        35.986, 0
    ]])
MIXED_SCORES = np.array([
    0.416, 0.588, 0.342, 0.268, 0, 0.342, 0.101, 1, 0.144, 0.304])
# yapf: enable


def test_validate_input_dc():
    """
    Tests :func:`fatf.utils.data.density._validate_input_dc` function.
    """
    incorrect_shape = 'The data set should be a 2-dimensional numpy array.'
    type_error_data = ('The data set is not of a base type (numbers and/or '
                       'strings.')
    #
    index_error = ('The following indices are invalid for the input data set: '
                   '{}.')
    type_error_ind = ('The categorical_indices parameter must be a Python '
                      'list or None.')
    #
    type_error_neigh = 'The neighbours number parameter has to be an integer.'
    value_error_neigh = ('The neighbours number parameter has to be between 1 '
                         'and number of data points (rows) in the data set '
                         'array.')
    #
    attribute_error_fn = ('The distance function must require exactly 2 '
                          'parameters. Given function requires {} parameters.')
    type_error_fn = 'The distance function should be a Python (function).'
    #
    type_error_norm = 'The normalise scores parameter should be a boolean.'

    with pytest.raises(IncorrectShapeError) as exin:
        fudd._validate_input_dc(np.array([4, 2]), None, None, None, None)
    assert str(exin.value) == incorrect_shape
    with pytest.raises(TypeError) as exin:
        fudd._validate_input_dc(np.array([[None, 2]]), None, None, None, None)
    assert str(exin.value) == type_error_data

    with pytest.raises(TypeError) as exin:
        fudd._validate_input_dc(NUMERICAL_NP_ARRAY, '42', None, None, None)
    assert str(exin.value) == type_error_ind
    with pytest.raises(IndexError) as exin:
        fudd._validate_input_dc(NUMERICAL_NP_ARRAY, [42], None, None, None)
    assert str(exin.value) == index_error.format('[42]')

    with pytest.raises(TypeError) as exin:
        fudd._validate_input_dc(NUMERICAL_NP_ARRAY, None, '42', None, None)
    assert str(exin.value) == type_error_neigh
    with pytest.raises(ValueError) as exin:
        fudd._validate_input_dc(NUMERICAL_NP_ARRAY, None, 0, None, None)
    assert str(exin.value) == value_error_neigh
    with pytest.raises(ValueError) as exin:
        fudd._validate_input_dc(NUMERICAL_NP_ARRAY, None, 21, None, None)
    assert str(exin.value) == value_error_neigh

    with pytest.raises(TypeError) as exin:
        fudd._validate_input_dc(NUMERICAL_NP_ARRAY, None, 20, '42', None)
    assert str(exin.value) == type_error_fn

    def one_param(param):
        return param  # pragma: nocover

    with pytest.raises(AttributeError) as exin:
        fudd._validate_input_dc(NUMERICAL_NP_ARRAY, None, 20, one_param, None)
    assert str(exin.value) == attribute_error_fn.format('1')

    with pytest.raises(TypeError) as exin:
        fudd._validate_input_dc(NUMERICAL_NP_ARRAY, None, 20, None, None)
    assert str(exin.value) == type_error_norm

    assert fudd._validate_input_dc(NUMERICAL_NP_ARRAY, None, 20, None, False)


def cat_dist(x, y):
    """
    A dummy implementation of a binary distance metric.
    """
    return int(x['x'] != y['x']) + int(x['xx'] != y['xx'])


def mix_dist(x, y):
    """
    A dummy implementation of ``DensityCheck``'s  mixed distance.
    """
    return int(x['x'] != y['x']) + int(x['xx'] != y['xx']) + np.sqrt(
        (x['i'] - y['i'])**2 + (x['ii'] - y['ii'])**2)


class TestDensityCheck(object):
    """
    Tests the :class:`fatf.utils.data.density.DensityCheck` class.
    """
    num_np_dc = fudd.DensityCheck(NUMERICAL_NP_ARRAY, normalise_scores=False)
    num_sc_dc = fudd.DensityCheck(
        NUMERICAL_STRUCT_ARRAY, normalise_scores=False)
    num_sc_i_dc = fudd.DensityCheck(
        NUMERICAL_STRUCT_ARRAY,
        normalise_scores=False,
        categorical_indices=['a'])
    cat_np_dc = fudd.DensityCheck(
        CATEGORICAL_NP_ARRAY, normalise_scores=True, neighbours=5)
    cat_sc_dc = fudd.DensityCheck(
        CATEGORICAL_STRUCT_ARRAY, neighbours=5, distance_function=cat_dist)
    mix_sc_dc = fudd.DensityCheck(
        MIXED_ARRAY, neighbours=5, distance_function=mix_dist)

    _user_warning = ('Some of the string-based columns in the input data '
                     'set were not selected as categorical features via '
                     'the categorical_indices parameter. String-based '
                     'columns cannot be treated as numerical features, '
                     'therefore they will be also treated as categorical '
                     'features (in addition to the ones selected with the '
                     'categorical_indices parameter).')
    with pytest.warns(UserWarning) as _w:
        cat_sc_zero = fudd.DensityCheck(
            CATEGORICAL_STRUCT_ARRAY, categorical_indices=['x'])
    assert len(_w) == 1
    assert str(_w[0].message) == _user_warning

    def test_density_check_init(self):
        """
        Tests the :class:`fatf.utils.data.density.DensityCheck` class init.
        """
        assert np.array_equal(self.cat_sc_zero.data_set,
                              CATEGORICAL_STRUCT_ARRAY)
        assert self.cat_sc_zero.neighbours == 7
        assert self.cat_sc_zero.normalise_scores is True
        assert np.array_equal(self.cat_sc_zero.distance_matrix,
                              CATEGORICAL_DISTS)
        assert np.array_equal(self.cat_sc_zero.scores, np.zeros((10, )))
        assert self.cat_sc_zero.scores_min == 2
        assert self.cat_sc_zero.scores_max == 2
        #
        assert self.cat_sc_zero._samples_number == 10
        assert self.cat_sc_zero._numerical_indices == []
        assert self.cat_sc_zero._categorical_indices == ['x', 'xx']
        assert self.cat_sc_zero._is_structured is True
        if _NUMPY_1_14:
            dist_func = self.cat_sc_zero._mixed_distance_n  # pragma: nocover
        else:
            dist_func = self.cat_sc_zero._mixed_distance_o  # pragma: nocover
        assert self.cat_sc_zero._distance_function == dist_func

        #######################################################################

        assert np.array_equal(self.num_sc_i_dc.data_set,
                              NUMERICAL_STRUCT_ARRAY)
        assert self.num_sc_i_dc.neighbours == 7
        assert self.num_sc_i_dc.normalise_scores is False
        assert np.array_equal(self.num_sc_i_dc.distance_matrix,
                              NUMERICAL_DISTS_A)
        assert np.array_equal(self.num_sc_i_dc.scores, NUMERICAL_SCORES_A)
        assert self.num_sc_i_dc.scores_min == 6
        assert self.num_sc_i_dc.scores_max == 52
        #
        assert self.num_sc_i_dc._samples_number == 20
        assert self.num_sc_i_dc._numerical_indices == ['b']
        assert self.num_sc_i_dc._categorical_indices == ['a']
        assert self.num_sc_i_dc._is_structured is True
        if _NUMPY_1_14:
            dist_func = self.num_sc_i_dc._mixed_distance_n  # pragma: nocover
        else:
            dist_func = self.num_sc_i_dc._mixed_distance_o  # pragma: nocover
        assert self.num_sc_i_dc._distance_function == dist_func

        #######################################################################

        assert np.array_equal(self.cat_np_dc.data_set, CATEGORICAL_NP_ARRAY)
        assert self.cat_np_dc.neighbours == 5
        assert self.cat_np_dc.normalise_scores is True
        assert np.array_equal(self.cat_np_dc.distance_matrix,
                              CATEGORICAL_DISTS)
        assert np.array_equal(self.cat_np_dc.scores, CATEGORICAL_SCORES)
        assert self.cat_np_dc.scores_min == 1
        assert self.cat_np_dc.scores_max == 2
        #
        assert self.cat_np_dc._samples_number == 10
        assert self.cat_np_dc._numerical_indices == []
        assert self.cat_np_dc._categorical_indices == [0, 1]
        assert self.cat_np_dc._is_structured is False
        assert (self.cat_np_dc._distance_function
                == self.cat_np_dc._mixed_distance_n)  # yapf: disable

        #######################################################################

        assert np.array_equal(self.mix_sc_dc.data_set, MIXED_ARRAY)
        assert self.mix_sc_dc.neighbours == 5
        assert self.mix_sc_dc.normalise_scores is True
        assert np.allclose(
            self.mix_sc_dc.distance_matrix, MIXED_DISTS, atol=1e-3)
        assert np.allclose(self.mix_sc_dc.scores, MIXED_SCORES, atol=1e-3)
        assert pytest.approx(self.mix_sc_dc.scores_min, abs=1e-3) == 28.476
        assert pytest.approx(self.mix_sc_dc.scores_max, abs=1e-3) == 74.471
        #
        assert self.mix_sc_dc._samples_number == 10
        assert self.mix_sc_dc._numerical_indices == ['i', 'ii']
        assert self.mix_sc_dc._categorical_indices == ['x', 'xx']
        assert self.mix_sc_dc._is_structured is True
        assert self.mix_sc_dc._distance_function == mix_dist

    def test_mixed_distance_o(self):
        """
        Tests :func:`~fatf.utils.data.density.DensityCheck._mixed_distance_o`.

        This implementation only handles structured arrays.
        """
        num_dist = np.sqrt((3 - 30)**2 + (86 - 98)**2)

        dist = self.num_sc_dc._mixed_distance_o(NUMERICAL_STRUCT_ARRAY[1],
                                                NUMERICAL_STRUCT_ARRAY[5])
        assert pytest.approx(dist, abs=1e-3) == num_dist

        dist = self.cat_sc_dc._mixed_distance_o(CATEGORICAL_STRUCT_ARRAY[1],
                                                CATEGORICAL_STRUCT_ARRAY[5])
        assert dist == 1

        dist = self.mix_sc_dc._mixed_distance_o(MIXED_ARRAY[0], MIXED_ARRAY[0])
        assert dist == 0
        dist = self.mix_sc_dc._mixed_distance_o(MIXED_ARRAY[1], MIXED_ARRAY[5])
        assert pytest.approx(dist, abs=1e-3) == (0 + 1 + num_dist)

    def test_mixed_distance_n(self):
        """
        Tests :func:`~fatf.utils.data.density.DensityCheck._mixed_distance_n`.
        """
        num_dist = np.sqrt((3 - 30)**2 + (86 - 98)**2)

        dist = self.num_np_dc._mixed_distance_n(NUMERICAL_NP_ARRAY[0],
                                                NUMERICAL_NP_ARRAY[0])
        assert dist == 0

        dist = self.num_np_dc._mixed_distance_n(NUMERICAL_NP_ARRAY[1],
                                                NUMERICAL_NP_ARRAY[5])
        assert pytest.approx(dist, abs=1e-3) == num_dist

        dist = self.cat_np_dc._mixed_distance_n(CATEGORICAL_NP_ARRAY[1],
                                                CATEGORICAL_NP_ARRAY[5])
        assert dist == 1

    def test_compute_scores(self):
        """
        Tests :func:`~fatf.utils.data.density.DensityCheck._compute_scores`.
        """
        assert np.allclose(self.num_np_dc.scores, NUMERICAL_SCORES, atol=1e-3)
        scores = self.num_np_dc._compute_scores()
        assert np.allclose(scores, NUMERICAL_SCORES, atol=1e-3)
        #
        assert np.allclose(self.num_sc_dc.scores, NUMERICAL_SCORES, atol=1e-3)
        scores = self.num_sc_dc._compute_scores()
        assert np.allclose(scores, NUMERICAL_SCORES, atol=1e-3)

        assert np.allclose(
            self.num_sc_i_dc.scores, NUMERICAL_SCORES_A, atol=1e-3)
        scores = self.num_sc_i_dc._compute_scores()
        assert np.allclose(scores, NUMERICAL_SCORES_A, atol=1e-3)

        assert np.allclose(
            self.cat_np_dc.scores, CATEGORICAL_SCORES, atol=1e-3)
        scores = self.cat_np_dc._compute_scores()
        scores -= scores.min()
        scores /= scores.max()
        assert np.allclose(scores, CATEGORICAL_SCORES, atol=1e-3)
        #
        assert np.allclose(
            self.cat_sc_dc.scores, CATEGORICAL_SCORES, atol=1e-3)
        scores = self.cat_sc_dc._compute_scores()
        scores -= scores.min()
        scores /= scores.max()
        assert np.allclose(scores, CATEGORICAL_SCORES, atol=1e-3)

        assert np.allclose(self.mix_sc_dc.scores, MIXED_SCORES, atol=1e-3)
        scores = self.mix_sc_dc._compute_scores()
        scores -= scores.min()
        scores /= scores.max()
        assert np.allclose(scores, MIXED_SCORES, atol=1e-3)

    def test_filter_data_set(self):
        """
        Tests :func:`~fatf.utils.data.density.DensityCheck.filter_data_set`.
        """
        type_error = 'The alpha parameter has to be a number.'
        value_error_norm = ('The alpha parameter has to be between 0 and 1 '
                            'for normalised scores.')
        value_error_unnorm = ('The alpha parameter has to be equal to or '
                              'larger than 0.')
        user_warning = ('Chosen alpha parameter is too large and none of the '
                        'data points were selected.')

        with pytest.raises(TypeError) as exin:
            self.num_np_dc.filter_data_set('42')
        assert str(exin.value) == type_error
        with pytest.raises(ValueError) as exin:
            self.mix_sc_dc.filter_data_set(1.0000042)
        assert str(exin.value) == value_error_norm
        with pytest.raises(ValueError) as exin:
            self.mix_sc_dc.filter_data_set(-0.0000042)
        assert str(exin.value) == value_error_norm
        with pytest.raises(ValueError) as exin:
            self.num_np_dc.filter_data_set(-0.0000042)
        assert str(exin.value) == value_error_unnorm

        with pytest.warns(UserWarning) as w:
            filtered = self.num_np_dc.filter_data_set(42424242)
        assert len(w) == 1
        assert str(w[0].message) == user_warning
        #
        assert filtered.size == 0

        filtered = self.num_np_dc.filter_data_set()
        assert np.array_equal(filtered, NUMERICAL_NP_ARRAY)
        filtered = self.num_sc_dc.filter_data_set(55)
        assert np.array_equal(filtered, NUMERICAL_STRUCT_ARRAY[[7]])

        filtered = self.num_sc_i_dc.filter_data_set(42)
        assert np.array_equal(filtered, NUMERICAL_STRUCT_ARRAY[[7]])

        filtered = self.cat_np_dc.filter_data_set()
        assert np.array_equal(filtered,
                              CATEGORICAL_NP_ARRAY[[0, 3, 6, 7, 8, 9]])
        filtered = self.cat_sc_dc.filter_data_set()
        assert np.array_equal(filtered,
                              CATEGORICAL_STRUCT_ARRAY[[0, 3, 6, 7, 8, 9]])

        filtered = self.mix_sc_dc.filter_data_set()
        assert np.array_equal(filtered, MIXED_ARRAY[[7]])

    def test_validate_data_point(self):
        """
        Tests data point validation.

        This function tests
        :func:`~fatf.utils.data.density.DensityCheck._validate_data_point`
        method.
        """
        shape_error_point = ('The data point has to be 1-dimensional numpy '
                             'array or numpy void (for structured arrays).')
        shape_error_columns = ('The data point has different number of '
                               'columns (features) than the data set used to '
                               'initialise this class.')
        type_error_base = ('The data point has to be of a base type (strings '
                           'and/or numbers).')
        type_error_dtype = ('The dtypes of the data set used to initialise '
                            'this class and the provided data point are too '
                            'different.')
        type_error_clip = 'The clip parameter has to be a boolean.'

        with pytest.raises(IncorrectShapeError) as exin:
            self.num_np_dc._validate_data_point(NUMERICAL_NP_ARRAY, None)
        assert str(exin.value) == shape_error_point

        with pytest.raises(TypeError) as exin:
            self.num_np_dc._validate_data_point(np.array([4, None, 2]), None)
        assert str(exin.value) == type_error_base

        with pytest.raises(TypeError) as exin:
            self.num_np_dc._validate_data_point(np.array(['4', '2']), None)
        assert str(exin.value) == type_error_dtype
        array = np.array([(4, 2)], dtype=[('a', int), ('B', np.int16)])[0]
        with pytest.raises(TypeError) as exin:
            self.num_sc_dc._validate_data_point(array, None)
        assert str(exin.value) == type_error_dtype

        with pytest.raises(IncorrectShapeError) as exin:
            self.num_np_dc._validate_data_point(np.array([1, 2, 3]), None)
        assert str(exin.value) == shape_error_columns

        with pytest.raises(TypeError) as exin:
            self.num_np_dc._validate_data_point(np.array([4, 2]), None)
        assert str(exin.value) == type_error_clip

        assert self.num_np_dc._validate_data_point(np.array([4, 2]), True)

    def test_score_data_point(self):
        """
        Tests :func:`~fatf.utils.data.density.DensityCheck.score_data_point`.
        """
        user_warning = ('The minimum and maximum scores are the same, '
                        'therefore the score normalisation is ill-defined.')
        dc = fudd.DensityCheck(np.array([[0, 1], [1, 0]]), neighbours=1)
        #
        with pytest.warns(UserWarning) as w:
            score_n = dc.score_data_point(np.array([4, 2]))
            score_un = dc.score_data_point(np.array([4, 2]), False)
        assert len(w) == 2
        assert str(w[0].message) == user_warning
        assert str(w[1].message) == user_warning
        assert score_n == 1
        assert score_un == 1
        #
        with pytest.warns(UserWarning) as w:
            score_n = dc.score_data_point(np.array([0.5, 0.5]))
            score_un = dc.score_data_point(np.array([0.5, 0.5]), False)
        assert len(w) == 2
        assert str(w[0].message) == user_warning
        assert str(w[1].message) == user_warning
        assert score_n == 0
        assert score_un == 0
        #
        with pytest.warns(UserWarning) as w:
            score_n = dc.score_data_point(np.array([0, -1]))
            score_un = dc.score_data_point(np.array([0, -1]), False)
        assert len(w) == 2
        assert str(w[0].message) == user_warning
        assert str(w[1].message) == user_warning
        assert score_n == 0
        assert score_un == 0

        array = np.array([(42, 42, 'a', 'aa')], dtype=MIXED_ARRAY.dtype)[0]
        score = self.mix_sc_dc.score_data_point(array)
        true_score = 33.526
        true_score -= self.mix_sc_dc.scores_min
        true_score /= self.mix_sc_dc.scores_max - self.mix_sc_dc.scores_min
        assert pytest.approx(score, abs=1e-3) == true_score
        #
        array = np.array([(0, 0, '0', '0')], dtype=MIXED_ARRAY.dtype)[0]
        score = self.mix_sc_dc.score_data_point(array, False)
        true_score = 1.244
        assert pytest.approx(score, abs=1e-3) == true_score
        score = self.mix_sc_dc.score_data_point(array)
        assert score == 1

        true_score = 24.083
        score = self.num_np_dc.score_data_point(np.array([42, 42]))
        assert pytest.approx(score, abs=1e-3) == true_score
        score = self.num_np_dc.score_data_point(np.array([42, 42]), True)
        assert pytest.approx(score, abs=1e-3) == true_score
        score = self.num_np_dc.score_data_point(np.array([42, 42]), False)
        assert pytest.approx(score, abs=1e-3) == true_score
