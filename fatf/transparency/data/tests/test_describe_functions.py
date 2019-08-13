"""
Tests describing arrays.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
#         Rafael Poyiadzi <rp13102@bristol.ac.uk>
# License: new BSD

import pytest

import numpy as np

import fatf.transparency.data.describe_functions as ftddf
import fatf.utils.tools as fut

from fatf.exceptions import IncorrectShapeError

_NUMPY_VERSION = [int(i) for i in np.version.version.split('.')]
_NUMPY_1_17 = fut.at_least_verion([1, 17], _NUMPY_VERSION)
_NUMPY_1_16 = fut.at_least_verion([1, 16], _NUMPY_VERSION)
_NUMPY_1_14_4 = fut.at_least_verion([1, 14, 4], _NUMPY_VERSION)
_NUMPY_1_11 = fut.at_least_verion([1, 11], _NUMPY_VERSION)
_NUMPY_1_10 = fut.at_least_verion([1, 10], _NUMPY_VERSION)


def test_describe_numerical_array():
    """
    Tests :func:`fatf.transparency.data.describe.describe_numerical_array`.
    """
    rwpf = 'percentile' if _NUMPY_1_11 else 'median'
    runtime_warning_percentile = 'Invalid value encountered in {}'.format(rwpf)
    # numpy<1.16 throws a reduce warning for nan arrays when computing min/max
    runtime_warning_minmax = 'invalid value encountered in reduce'
    #
    incorrect_shape_error = 'The input array should be 1-dimensional.'
    value_error_non_numerical = 'The input array should be purely numerical.'
    value_error_empty = 'The input array cannot be empty.'

    # Wrong shape
    array = np.array([[5, 33], [22, 17]])
    with pytest.raises(IncorrectShapeError) as exin:
        ftddf.describe_numerical_array(array)
    assert str(exin.value) == incorrect_shape_error

    # Wrong type
    array = np.array(['string', 33, 22, 17])
    with pytest.raises(ValueError) as exin:
        ftddf.describe_numerical_array(array)
    assert str(exin.value) == value_error_non_numerical

    # Empty array
    array = np.array([], dtype=np.int32)
    with pytest.raises(ValueError) as exin:
        ftddf.describe_numerical_array(array)
    assert str(exin.value) == value_error_empty

    # Array with nans -- structured row; ignore nans + default parameter
    array = np.array([(33, 22, np.nan, 11, np.nan, 4)],
                     dtype=[('a', int), ('b', int), ('c', np.float),
                            ('d', np.int32), ('e', np.float), ('f', int)])
    description = {'count': 6, 'mean': 17.5, 'std': 11.011, 'max': 33,
                   'min': 4, '25%': 9.25, '50%': 16.5, '75%': 24.75,
                   'nan_count': 2}  # yapf: disable
    array_description = ftddf.describe_numerical_array(array[0])
    assert set(ftddf.NUMERICAL_KEYS) == set(description.keys())
    assert set(ftddf.NUMERICAL_KEYS) == set(array_description.keys())
    for i in ftddf.NUMERICAL_KEYS:
        assert pytest.approx(array_description[i], abs=1e-3) == description[i]
    # ...
    array_description = ftddf.describe_numerical_array(
        array[0], skip_nans=True)
    assert set(ftddf.NUMERICAL_KEYS) == set(description.keys())
    assert set(ftddf.NUMERICAL_KEYS) == set(array_description.keys())
    for i in ftddf.NUMERICAL_KEYS:
        assert pytest.approx(array_description[i], abs=1e-3) == description[i]

    # Array with nans -- classic array; do not ignore nans
    array = np.array([33, 22, np.nan, 11, np.nan, 4])
    if _NUMPY_1_10:  # pragma: no cover
        description = {'count': 6, 'mean': np.nan, 'std': np.nan,
                       'max': np.nan, 'min': np.nan, '25%': np.nan,
                       '50%': np.nan, '75%': np.nan, 'nan_count': 2
                       }  # yapf: disable
        if _NUMPY_1_17:
            array_description = ftddf.describe_numerical_array(
                array, skip_nans=False)
        else:
            with pytest.warns(RuntimeWarning) as w:
                array_description = ftddf.describe_numerical_array(
                    array, skip_nans=False)
            if _NUMPY_1_16:
                assert len(w) == 3
                for i in range(len(w)):
                    assert str(
                        w[i].message).startswith(runtime_warning_percentile)
            elif _NUMPY_1_14_4:
                assert len(w) == 5
                assert str(w[0].message).startswith(runtime_warning_minmax)
                for i in range(1, len(w) - 1):
                    assert str(
                        w[i].message).startswith(runtime_warning_percentile)
                assert str(w[-1].message).startswith(runtime_warning_minmax)
            else:
                assert len(w) == 3
                for i in range(len(w)):
                    assert str(
                        w[i].message).startswith(runtime_warning_percentile)
    else:  # pragma: no cover
        description = {'count': 6, 'mean': np.nan, 'std': np.nan,
                       'max': np.nan, 'min': np.nan, '25%': 13.75,
                       '50%': 27.5, '75%': np.nan, 'nan_count': 2
                       }  # yapf: disable
        array_description = ftddf.describe_numerical_array(
            array, skip_nans=False)
    assert set(ftddf.NUMERICAL_KEYS) == set(description.keys())
    assert set(ftddf.NUMERICAL_KEYS) == set(array_description.keys())
    for i in ftddf.NUMERICAL_KEYS:
        true = description[i]
        computed = array_description[i]
        if np.isnan(true) and np.isnan(computed):
            assert True
        else:
            assert pytest.approx(computed, abs=1e-3) == true

    # Array without nans -- classic array; ignore nans
    array = np.array([33, 22, 11, 4])
    description = {'count': 4, 'mean': 17.5, 'std': 11.011, 'max': 33,
                   'min': 4, '25%': 9.25, '50%': 16.5, '75%': 24.75,
                   'nan_count': 0}  # yapf: disable
    array_description = ftddf.describe_numerical_array(array, skip_nans=True)
    assert set(ftddf.NUMERICAL_KEYS) == set(description.keys())
    assert set(ftddf.NUMERICAL_KEYS) == set(array_description.keys())
    for i in ftddf.NUMERICAL_KEYS:
        assert pytest.approx(array_description[i], abs=1e-3) == description[i]

    # Array without nans -- classic array; do not ignore nans
    array_description = ftddf.describe_numerical_array(array, skip_nans=False)
    assert set(ftddf.NUMERICAL_KEYS) == set(description.keys())
    assert set(ftddf.NUMERICAL_KEYS) == set(array_description.keys())
    for i in ftddf.NUMERICAL_KEYS:
        assert pytest.approx(array_description[i], abs=1e-3) == description[i]


def test_describe_categorical_array():
    """
    Tests :func:`fatf.transparency.data.describe.describe_categorical_array`.
    """
    incorrect_shape_error = 'The input array should be 1-dimensional.'
    value_error_empty = 'The input array cannot be empty.'
    user_warning_non_numerical = ('The input array is not purely categorical. '
                                  'Converting the input array into a textual '
                                  'type to facilitate a categorical '
                                  'description.')
    # Wrong shape
    array = np.array([[5, 33], [22, 17]])
    with pytest.raises(IncorrectShapeError) as exin:
        ftddf.describe_categorical_array(array)
    assert str(exin.value) == incorrect_shape_error

    # Empty array
    array = np.array([], dtype=np.int32)
    with pytest.raises(ValueError) as exin:
        ftddf.describe_categorical_array(array)
    assert str(exin.value) == value_error_empty

    # Wrong type -- will be treated as textual
    array = np.array([5, 33, 5])
    description = {'count': 3, 'unique': np.array(['33', '5']),
                   'unique_counts': np.array([1, 2]), 'top': '5',
                   'freq': 2, 'is_top_unique': True}  # yapf: disable
    with pytest.warns(UserWarning) as w:
        array_description = ftddf.describe_categorical_array(array)
    assert len(w) == 1
    assert str(w[0].message) == user_warning_non_numerical
    #
    assert set(ftddf.CATEGORICAL_KEYS) == set(description.keys())
    assert set(ftddf.CATEGORICAL_KEYS) == set(array_description.keys())
    for i in ftddf.CATEGORICAL_KEYS:
        if isinstance(description[i], (str, bool, int)):
            assert array_description[i] == description[i]
        elif isinstance(description[i], np.ndarray):
            assert np.array_equal(array_description[i], description[i])
        else:  # pragma: no cover
            assert False, 'Unrecognised type!'

    # Structured row
    array = np.array([('aa', 'bb', 'abc', 'd', 'bb')],
                     dtype=[('a', 'U2'), ('b', 'U2'), ('c', 'U3'), ('d', 'U4'),
                            ('e', 'U2')])
    description = {'count': 5, 'unique': np.array(['aa', 'abc', 'bb', 'd']),
                   'unique_counts': np.array([1, 1, 2, 1]), 'top': 'bb',
                   'freq': 2, 'is_top_unique': True}  # yapf: disable
    array_description = ftddf.describe_categorical_array(array[0])
    assert set(ftddf.CATEGORICAL_KEYS) == set(description.keys())
    assert set(ftddf.CATEGORICAL_KEYS) == set(array_description.keys())
    for i in ftddf.CATEGORICAL_KEYS:
        if isinstance(description[i], (str, bool, int)):
            assert array_description[i] == description[i]
        elif isinstance(description[i], np.ndarray):
            assert np.array_equal(array_description[i], description[i])
        else:  # pragma: no cover
            assert False, 'Unrecognised type!'

    # Classic array -- more than one top
    array = np.array(['d', 'bb', 'abc', 'd', 'bb'])
    description = {'count': 5, 'unique': np.array(['abc', 'bb', 'd']),
                   'unique_counts': np.array([1, 2, 2]), 'top': 'bb',
                   'freq': 2, 'is_top_unique': False}  # yapf: disable
    array_description = ftddf.describe_categorical_array(array)
    assert set(ftddf.CATEGORICAL_KEYS) == set(description.keys())
    assert set(ftddf.CATEGORICAL_KEYS) == set(array_description.keys())
    for i in ftddf.CATEGORICAL_KEYS:
        if isinstance(description[i], (str, bool, int)):
            assert array_description[i] == description[i]
        elif isinstance(description[i], np.ndarray):
            assert np.array_equal(array_description[i], description[i])
        else:  # pragma: no cover
            assert False, 'Unrecognised type!'


def test_describe_array():
    """
    Tests :func:`fatf.transparency.data.describe.describe_array`.
    """
    incorrect_shape_error = 'The input array should be 1- or 2-dimensional.'
    value_error_non_base = ('The input array should be of a base type (a '
                            'mixture of numerical and textual types).')
    user_warning = ('The input array is 1-dimensional. Ignoring include and '
                    'exclude parameters.')
    value_error_0_columns = 'The input array cannot have 0 columns.'
    value_error_include_index = ('The following include index is not a valid '
                                 'index: {}.')
    value_error_include_indices = ('The following include indices are not '
                                   'valid indices: ')
    type_error_include = ('The include parameter can either be a string, an '
                          'integer or a list of these two types.')
    value_error_exclude_index = ('The following exclude index is not a valid '
                                 'index: {}.')
    value_error_exclude_indices = ('The following exclude indices are not '
                                   'valid indices: ')
    type_error_exclude = ('The exclude parameter can either be a string, an '
                          'integer or a list of these two types.')
    runtime_error = 'None of the columns were selected to be described.'
    rwpf = 'percentile' if _NUMPY_1_11 else 'median'
    runtime_warning_percentile = 'Invalid value encountered in {}'.format(rwpf)
    # numpy<1.16 throws a reduce warning for nan arrays when computing min/max
    runtime_warning_minmax = 'invalid value encountered in reduce'

    # Wrong shape
    array = np.ones((2, 2, 2), dtype=np.int32)
    with pytest.raises(IncorrectShapeError) as exin:
        ftddf.describe_array(array)
    assert str(exin.value) == incorrect_shape_error

    # Wrong type
    array = np.array([[2, None, 2], [7, 4, 7]])
    with pytest.raises(ValueError) as exin:
        ftddf.describe_array(array)
    assert str(exin.value) == value_error_non_base

    # 1D categorical array -- no include & exclude
    array = np.array([2, '44', 2, 44])
    description = {'count': 4, 'unique': np.array(['2', '44']),
                   'unique_counts': np.array([2, 2]), 'top': '2', 'freq': 2,
                   'is_top_unique': False}  # yapf: disable
    array_description = ftddf.describe_array(array)
    assert set(ftddf.CATEGORICAL_KEYS) == set(description.keys())
    assert set(ftddf.CATEGORICAL_KEYS) == set(array_description.keys())
    for i in ftddf.CATEGORICAL_KEYS:
        if isinstance(description[i], (str, bool, int)):
            assert array_description[i] == description[i]
        elif isinstance(description[i], np.ndarray):
            assert np.array_equal(array_description[i], description[i])
        else:  # pragma: no cover
            assert False, 'Unrecognised type!'
    # Skip nans
    array_description = ftddf.describe_array(array, skip_nans=False)
    assert set(ftddf.CATEGORICAL_KEYS) == set(description.keys())
    assert set(ftddf.CATEGORICAL_KEYS) == set(array_description.keys())
    for i in ftddf.CATEGORICAL_KEYS:
        if isinstance(description[i], (str, bool, int)):
            assert array_description[i] == description[i]
        elif isinstance(description[i], np.ndarray):
            assert np.array_equal(array_description[i], description[i])
        else:  # pragma: no cover
            assert False, 'Unrecognised type!'

    # 1D categorical array -- include & not exclude
    array = np.array([2, '44', 2, 44])
    description = {'count': 4, 'unique': np.array(['2', '44']),
                   'unique_counts': np.array([2, 2]), 'top': '2', 'freq': 2,
                   'is_top_unique': False}  # yapf: disable
    with pytest.warns(UserWarning) as w:
        array_description = ftddf.describe_array(array, exclude='unimportant')
    assert len(w) == 1
    assert str(w[0].message) == user_warning
    #
    assert set(ftddf.CATEGORICAL_KEYS) == set(description.keys())
    assert set(ftddf.CATEGORICAL_KEYS) == set(array_description.keys())
    for i in ftddf.CATEGORICAL_KEYS:
        if isinstance(description[i], (str, bool, int)):
            assert array_description[i] == description[i]
        elif isinstance(description[i], np.ndarray):
            assert np.array_equal(array_description[i], description[i])
        else:  # pragma: no cover
            assert False, 'Unrecognised type!'

    # 1D structured numerical array -- include & exclude -- ignore nans
    array = np.array([(33, 22, np.nan, 11, np.nan, 4)],
                     dtype=[('a', int), ('b', int), ('c', np.float),
                            ('d', np.int32), ('e', np.float), ('f', int)])
    description = {'count': 6, 'mean': 17.5, 'std': 11.011, 'max': 33,
                   'min': 4, '25%': 9.25, '50%': 16.5, '75%': 24.75,
                   'nan_count': 2}  # yapf: disable
    with pytest.warns(UserWarning) as w:
        array_description = ftddf.describe_array(
            array[0], exclude='unimportant', include='nope')
    assert len(w) == 1
    assert str(w[0].message) == user_warning
    #
    assert set(ftddf.NUMERICAL_KEYS) == set(description.keys())
    assert set(ftddf.NUMERICAL_KEYS) == set(array_description.keys())
    for i in ftddf.NUMERICAL_KEYS:
        assert pytest.approx(array_description[i], abs=1e-3) == description[i]

    # 1D numerical array -- include & not exclude -- do not filter nans
    if _NUMPY_1_10:  # pragma: no cover
        description = {'count': 6, 'mean': np.nan, 'std': np.nan,
                       'max': np.nan, 'min': np.nan, '25%': np.nan,
                       '50%': np.nan, '75%': np.nan, 'nan_count': 2
                       }  # yapf: disable
        with pytest.warns(None) as w:
            array_description = ftddf.describe_array(
                array[0], include='unimportant', skip_nans=False)
        assert issubclass(w[0].category, UserWarning)
        assert str(w[0].message) == user_warning
        if _NUMPY_1_17:
            pass
        elif _NUMPY_1_16:
            assert len(w) == 4
            for i in range(1, len(w)):
                assert issubclass(w[i].category, RuntimeWarning)
                assert str(w[i].message).startswith(runtime_warning_percentile)
        elif _NUMPY_1_14_4:
            assert len(w) == 6
            assert issubclass(w[1].category, RuntimeWarning)
            assert str(w[1].message).startswith(runtime_warning_minmax)
            for i in range(2, len(w) - 1):
                assert issubclass(w[i].category, RuntimeWarning)
                assert str(w[i].message).startswith(runtime_warning_percentile)
            assert issubclass(w[-1].category, RuntimeWarning)
            assert str(w[-1].message).startswith(runtime_warning_minmax)
        else:
            assert len(w) == 4
            for i in range(1, len(w)):
                assert issubclass(w[i].category, RuntimeWarning)
                assert str(w[i].message).startswith(runtime_warning_percentile)
    else:  # pragma: no cover
        description = {'count': 6, 'mean': np.nan, 'std': np.nan,
                       'max': np.nan, 'min': np.nan, '25%': 13.75,
                       '50%': 27.5, '75%': np.nan, 'nan_count': 2
                       }  # yapf: disable
        with pytest.warns(UserWarning) as w:
            array_description = ftddf.describe_array(
                array[0], include='unimportant', skip_nans=False)
        assert len(w) == 1
        assert str(w[0].message) == user_warning
    #
    assert set(ftddf.NUMERICAL_KEYS) == set(description.keys())
    assert set(ftddf.NUMERICAL_KEYS) == set(array_description.keys())
    for i in ftddf.NUMERICAL_KEYS:
        true = description[i]
        computed = array_description[i]
        if np.isnan(true) and np.isnan(computed):
            assert True
        else:
            assert true == computed

    # A 2D array with 0 columns
    array = np.ndarray((10, 0), dtype=np.int32)
    with pytest.raises(ValueError) as exin:
        ftddf.describe_array(array)
    assert str(exin.value) == value_error_0_columns

    # Testing arrays
    numerical_indices_struct = ['a', 'b', 'd']
    array_num = np.array([[33, 0.5, 11], [17, 2.22, 22], [22, 3.33, -5],
                          [0, np.nan, 0]])
    array_cat = np.array([['one', 'four'], ['six', 'xyz'], ['one', 'xyz'],
                          ['s', 'four']])
    array_struct = np.array([(33, 0.5, 'one', 11, 'four'),
                             (17, 2.22, 'six', 22, 'xyz'),
                             (22, 3.33, 'one', -5, 'xyz'),
                             (0, np.nan, 's', 0, 'four')],
                            dtype=[('a', int), ('b', float), ('c', 'U3'),
                                   ('d', np.int32), ('e', 'U4')])
    # yapf: disable
    num_c0 = {'count': 4, 'mean': 18, 'std': 11.895, 'max': 33, 'min': 0,
              '25%': 12.75, '50%': 19.5, '75%': 24.75, 'nan_count': 0}
    num_c1 = {'count': 4, 'mean': 2.017, 'std': 1.164, 'max': 3.33, 'min': 0.5,
              '25%': 1.36, '50%': 2.22, '75%': 2.775, 'nan_count': 1}
    num_c1_nan = {'count': 4, 'mean': np.nan, 'std': np.nan, 'max': np.nan,
                  'min': np.nan, '25%': np.nan, '50%': np.nan, '75%': np.nan,
                  'nan_count': 1}
    num_c1_nann = {'count': 4, 'mean': np.nan, 'std': np.nan, 'max': np.nan,
                   'min': np.nan, '25%': 1.79, '50%': 2.775, '75%': np.nan,
                   'nan_count': 1}
    num_c2 = {'count': 4, 'mean': 7, 'std': 10.416, 'max': 22, 'min': -5,
              '25%': -1.25, '50%': 5.5, '75%': 13.75, 'nan_count': 0}
    description_num = {'a': num_c0, 'b': num_c1, 'd': num_c2, 0: num_c0,
                       1: num_c1, 3: num_c2, 2: num_c2}
    description_num_nan = {'a': num_c0, 'b': num_c1_nan, 'd': num_c2,
                           0: num_c0, 1: num_c1_nan, 3: num_c2, 2: num_c2}
    description_num_nann = {'a': num_c0, 'b': num_c1_nann, 'd': num_c2,
                            0: num_c0, 1: num_c1_nann, 3: num_c2, 2: num_c2}
    cat_c0 = {'count': 4, 'unique': np.array(['one', 's', 'six']),
              'unique_counts': np.array([2, 1, 1]), 'top': 'one', 'freq': 2,
              'is_top_unique': True}  # yapf: disable
    cat_c1 = {'count': 4, 'unique': np.array(['four', 'xyz']),
              'unique_counts': np.array([2, 2]), 'top': 'four', 'freq': 2,
              'is_top_unique': False}  # yapf: disable
    description_cat = {'c': cat_c0, 'e': cat_c1, 0: cat_c0, 1: cat_c1}
    # yapf: enable

    # 2D structured mixture -- ignore nans -- no include/ exclude
    description = ftddf.describe_array(array_struct)
    assert set(description) == set(array_struct.dtype.names)
    for col_id, column_description in description.items():
        if col_id in numerical_indices_struct:
            assert set(ftddf.NUMERICAL_KEYS) == set(column_description.keys())
            assert set(ftddf.NUMERICAL_KEYS) == set(
                description_num[col_id].keys())
            for i in ftddf.NUMERICAL_KEYS:
                col_d = column_description[i]
                gt_d = description_num[col_id][i]
                if np.isnan(col_d) and np.isnan(gt_d):  # pragma: no cover
                    assert True
                else:
                    assert pytest.approx(col_d, abs=1e-3) == gt_d
        else:
            assert set(ftddf.CATEGORICAL_KEYS) == set(
                column_description.keys())
            assert set(ftddf.CATEGORICAL_KEYS) == set(
                description_cat[col_id].keys())
            for i in ftddf.CATEGORICAL_KEYS:
                col_d = column_description[i]
                gt_d = description_cat[col_id][i]
                if isinstance(col_d, (str, bool, int, np.int64, np.bool_)):
                    assert col_d == gt_d
                elif isinstance(col_d, np.ndarray):
                    assert np.array_equal(col_d, gt_d)
                else:  # pragma: no cover
                    assert False, 'Unrecognised type!'

    # 2D structured mixture -- do not ignore nans -- no include/ exclude
    if _NUMPY_1_10:  # pragma: no cover
        description_num_test = description_num_nan
        if _NUMPY_1_17:
            description = ftddf.describe_array(array_struct, skip_nans=False)
        else:
            with pytest.warns(RuntimeWarning) as w:
                description = ftddf.describe_array(
                    array_struct, skip_nans=False)
            if _NUMPY_1_16:
                assert len(w) == 3
                for i in range(len(w)):
                    assert str(
                        w[i].message).startswith(runtime_warning_percentile)
            elif _NUMPY_1_14_4:
                assert len(w) == 5
                assert str(w[0].message).startswith(runtime_warning_minmax)
                for i in range(1, len(w) - 1):
                    assert str(
                        w[i].message).startswith(runtime_warning_percentile)
                assert str(w[-1].message).startswith(runtime_warning_minmax)
            else:
                assert len(w) == 3
                for i in range(len(w)):
                    assert str(
                        w[i].message).startswith(runtime_warning_percentile)
    else:  # pragma: no cover
        description_num_test = description_num_nann
        description = ftddf.describe_array(array_struct, skip_nans=False)
    #
    assert set(description) == set(array_struct.dtype.names)
    for col_id, column_description in description.items():
        if col_id in numerical_indices_struct:
            assert set(ftddf.NUMERICAL_KEYS) == set(column_description.keys())
            assert set(ftddf.NUMERICAL_KEYS) == set(
                description_num_test[col_id].keys())
            for i in ftddf.NUMERICAL_KEYS:
                col_d = column_description[i]
                gt_d = description_num_test[col_id][i]
                if np.isnan(col_d) and np.isnan(gt_d):
                    assert True
                else:
                    assert pytest.approx(col_d, abs=1e-3) == gt_d
        else:
            assert set(ftddf.CATEGORICAL_KEYS) == set(
                column_description.keys())
            assert set(ftddf.CATEGORICAL_KEYS) == set(
                description_cat[col_id].keys())
            for i in ftddf.CATEGORICAL_KEYS:
                col_d = column_description[i]
                gt_d = description_cat[col_id][i]
                if isinstance(col_d, (str, bool, int, np.int64, np.bool_)):
                    assert col_d == gt_d
                elif isinstance(col_d, np.ndarray):
                    assert np.array_equal(col_d, gt_d)
                else:  # pragma: no cover
                    assert False, 'Unrecognised type!'

    # 2D structured mixture -- include categorical/ exclude string
    description = ftddf.describe_array(
        array_struct, include='categorical', exclude='c')
    assert set(description) == set(['e'])
    for col_id, column_description in description.items():
        assert set(ftddf.CATEGORICAL_KEYS) == set(column_description.keys())
        assert set(ftddf.CATEGORICAL_KEYS) == set(
            description_cat[col_id].keys())
        for i in ftddf.CATEGORICAL_KEYS:
            col_d = column_description[i]
            gt_d = description_cat[col_id][i]
            if isinstance(col_d, (str, bool, int, np.int64, np.bool_)):
                assert col_d == gt_d
            elif isinstance(col_d, np.ndarray):
                assert np.array_equal(col_d, gt_d)
            else:  # pragma: no cover
                assert False, 'Unrecognised type!'

    # 2D structured mixture -- include numerical/ exclude list
    description = ftddf.describe_array(
        array_struct, include='numerical', exclude=['a', 'b', 'c', 'e'])
    assert set(description) == set('d')
    for col_id, column_description in description.items():
        assert set(ftddf.NUMERICAL_KEYS) == set(column_description.keys())
        assert set(ftddf.NUMERICAL_KEYS) == set(
            description_num_nan[col_id].keys())
        for i in ftddf.NUMERICAL_KEYS:
            col_d = column_description[i]
            gt_d = description_num_nan[col_id][i]
            if np.isnan(col_d) and np.isnan(gt_d):  # pragma: no cover
                assert True
            else:
                assert pytest.approx(col_d, abs=1e-3) == gt_d

    # 2D structured mixture -- include='numerical', exclude='numerical'
    with pytest.raises(RuntimeError) as exin:
        ftddf.describe_array(
            array_struct, include='numerical', exclude='numerical')
    assert str(exin.value) == runtime_error

    # Invalid indices -- include
    with pytest.raises(IndexError) as exin:
        ftddf.describe_array(array_struct, include='f', exclude='x')
    assert str(exin.value) == value_error_include_index.format('f')

    # Invalid indices -- include
    with pytest.raises(TypeError) as exin:
        ftddf.describe_array(array_struct, include=7.5, exclude='x')
    assert str(exin.value) == type_error_include

    # Invalid indices -- include
    with pytest.raises(IndexError) as exin:
        ftddf.describe_array(
            array_struct, include=['a', 7.5, 'b', 'y'], exclude='x')
    exin_message = str(exin.value)
    assert (exin_message.startswith(value_error_include_indices)
            and '{' in exin_message and '}' in exin_message
            and '7.5' in exin_message and 'y' in exin_message)

    # Invalid indices -- exclude
    with pytest.raises(IndexError) as exin:
        ftddf.describe_array(array_struct, include='b', exclude='x')
    assert str(exin.value) == value_error_exclude_index.format('x')

    # Invalid indices -- include
    with pytest.raises(IndexError) as exin:
        ftddf.describe_array(
            array_struct, include='a', exclude=['a', 7.5, 'b', 'z'])
    exin_message = str(exin.value)
    assert (exin_message.startswith(value_error_exclude_indices)
            and '{' in exin_message and '}' in exin_message
            and '7.5' in exin_message and 'z' in exin_message)

    # Invalid indices -- exclude
    with pytest.raises(TypeError) as exin:
        ftddf.describe_array(array_struct, include='b', exclude=4.2)
    assert str(exin.value) == type_error_exclude

    # Include list
    description = ftddf.describe_array(
        array_struct, include=['a', 'd'], exclude='categorical')
    assert set(description) == set(['a', 'd'])
    for col_id, column_description in description.items():
        assert set(ftddf.NUMERICAL_KEYS) == set(column_description.keys())
        assert set(ftddf.NUMERICAL_KEYS) == set(
            description_num_nan[col_id].keys())
        for i in ftddf.NUMERICAL_KEYS:
            col_d = column_description[i]
            gt_d = description_num_nan[col_id][i]
            if np.isnan(col_d) and np.isnan(gt_d):  # pragma: no cover
                assert True
            else:
                assert pytest.approx(col_d, abs=1e-3) == gt_d

    # include string
    description = ftddf.describe_array(array_struct, include='a')
    assert set(description) == set(['a'])
    for col_id, column_description in description.items():
        assert set(ftddf.NUMERICAL_KEYS) == set(column_description.keys())
        assert set(ftddf.NUMERICAL_KEYS) == set(
            description_num_nan[col_id].keys())
        for i in ftddf.NUMERICAL_KEYS:
            col_d = column_description[i]
            gt_d = description_num_nan[col_id][i]
            if np.isnan(col_d) and np.isnan(gt_d):  # pragma: no cover
                assert True
            else:
                assert pytest.approx(col_d, abs=1e-3) == gt_d

    # Include categorical
    description = ftddf.describe_array(array_struct, include='categorical')
    assert set(description) == set(['c', 'e'])
    for col_id, column_description in description.items():
        assert set(ftddf.CATEGORICAL_KEYS) == set(column_description.keys())
        assert set(ftddf.CATEGORICAL_KEYS) == set(
            description_cat[col_id].keys())
        for i in ftddf.CATEGORICAL_KEYS:
            col_d = column_description[i]
            gt_d = description_cat[col_id][i]
            if isinstance(col_d, (str, bool, int, np.int64, np.bool_)):
                assert col_d == gt_d
            elif isinstance(col_d, np.ndarray):
                assert np.array_equal(col_d, gt_d)
            else:  # pragma: no cover
                assert False, 'Unrecognised type!'

    # Include numerical
    description = ftddf.describe_array(array_struct, include='numerical')
    assert set(description) == set(['a', 'b', 'd'])
    for col_id, column_description in description.items():
        assert set(ftddf.NUMERICAL_KEYS) == set(column_description.keys())
        assert set(ftddf.NUMERICAL_KEYS) == set(description_num[col_id].keys())
        for i in ftddf.NUMERICAL_KEYS:
            col_d = column_description[i]
            gt_d = description_num[col_id][i]
            if np.isnan(col_d) and np.isnan(gt_d):  # pragma: no cover
                assert True
            else:
                assert pytest.approx(col_d, abs=1e-3) == gt_d

    # 2D structured mixture -- include=None, exclude='categorical'
    description = ftddf.describe_array(array_struct, exclude='categorical')
    assert set(description) == set(['a', 'b', 'd'])
    for col_id, column_description in description.items():
        assert set(ftddf.NUMERICAL_KEYS) == set(column_description.keys())
        assert set(ftddf.NUMERICAL_KEYS) == set(description_num[col_id].keys())
        for i in ftddf.NUMERICAL_KEYS:
            col_d = column_description[i]
            gt_d = description_num[col_id][i]
            if np.isnan(col_d) and np.isnan(gt_d):  # pragma: no cover
                assert True
            else:
                assert pytest.approx(col_d, abs=1e-3) == gt_d

    # 2D structured mixture -- include=None, exclude='numerical'
    description = ftddf.describe_array(array_struct, exclude='numerical')
    assert set(description) == set(['c', 'e'])
    for col_id, column_description in description.items():
        assert set(ftddf.CATEGORICAL_KEYS) == set(column_description.keys())
        assert set(ftddf.CATEGORICAL_KEYS) == set(
            description_cat[col_id].keys())
        for i in ftddf.CATEGORICAL_KEYS:
            col_d = column_description[i]
            gt_d = description_cat[col_id][i]
            if isinstance(col_d, (str, bool, int, np.int64, np.bool_)):
                assert col_d == gt_d
            elif isinstance(col_d, np.ndarray):
                assert np.array_equal(col_d, gt_d)
            else:  # pragma: no cover
                assert False, 'Unrecognised type!'

    # 2D classic numerical -- only exclude numerical
    with pytest.raises(RuntimeError) as exin:
        ftddf.describe_array(array_num, exclude='numerical')
    assert str(exin.value) == runtime_error

    # 2D classic numerical -- only exclude=1
    description = ftddf.describe_array(array_num, exclude=1)
    assert set(description) == set([0, 2])
    for col_id, column_description in description.items():
        assert set(ftddf.NUMERICAL_KEYS) == set(column_description.keys())
        assert set(ftddf.NUMERICAL_KEYS) == set(description_num[col_id].keys())
        for i in ftddf.NUMERICAL_KEYS:
            col_d = column_description[i]
            gt_d = description_num[col_id][i]
            if np.isnan(col_d) and np.isnan(gt_d):  # pragma: no cover
                assert True
            else:
                assert pytest.approx(col_d, abs=1e-3) == gt_d

    # 2D classic numerical -- only include=[0,2]
    description = ftddf.describe_array(array_num, include=[0, 2])
    assert set(description) == set([0, 2])
    for col_id, column_description in description.items():
        assert set(ftddf.NUMERICAL_KEYS) == set(column_description.keys())
        assert set(ftddf.NUMERICAL_KEYS) == set(description_num[col_id].keys())
        for i in ftddf.NUMERICAL_KEYS:
            col_d = column_description[i]
            gt_d = description_num[col_id][i]
            if np.isnan(col_d) and np.isnan(gt_d):  # pragma: no cover
                assert True
            else:
                assert pytest.approx(col_d, abs=1e-3) == gt_d

    # 2D classic categorical -- only exclude categorical
    with pytest.raises(RuntimeError) as exin:
        ftddf.describe_array(array_cat, exclude='categorical')
    assert str(exin.value) == runtime_error

    # 2D classic categorical -- only exclude=0
    description = ftddf.describe_array(array_cat, exclude=0)
    assert set(description) == set([1])
    for col_id, column_description in description.items():
        assert set(ftddf.CATEGORICAL_KEYS) == set(column_description.keys())
        assert set(ftddf.CATEGORICAL_KEYS) == set(
            description_cat[col_id].keys())
        for i in ftddf.CATEGORICAL_KEYS:
            col_d = column_description[i]
            gt_d = description_cat[col_id][i]
            if isinstance(col_d, (str, bool, int, np.int64, np.bool_)):
                assert col_d == gt_d
            elif isinstance(col_d, np.ndarray):
                assert np.array_equal(col_d, gt_d)
            else:  # pragma: no cover
                assert False, 'Unrecognised type!'

    # 2D classic categorical -- only include=1, exclude=numerical
    description = ftddf.describe_array(
        array_cat, include=1, exclude='numerical')
    assert set(description) == set([1])
    for col_id, column_description in description.items():
        assert set(ftddf.CATEGORICAL_KEYS) == set(column_description.keys())
        assert set(ftddf.CATEGORICAL_KEYS) == set(
            description_cat[col_id].keys())
        for i in ftddf.CATEGORICAL_KEYS:
            col_d = column_description[i]
            gt_d = description_cat[col_id][i]
            if isinstance(col_d, (str, bool, int, np.int64, np.bool_)):
                assert col_d == gt_d
            elif isinstance(col_d, np.ndarray):
                assert np.array_equal(col_d, gt_d)
            else:  # pragma: no cover
                assert False, 'Unrecognised type!'


def test_filter_include_indices():
    """
    Tests :func:`fatf.transparency.data.describe._filter_include_indices`.
    """
    index_error_a = 'The following include index is not a valid index: {}.'
    index_error_b = 'The following include indices are not valid indices: '
    type_error = ('The include parameter can either be a string, an integer '
                  'or a list of these two types.')
    n_set = set(['n1', 'n2', 'n3'])
    c_set = set(['c1', 'c2', 'c3'])
    nc_set = n_set.union(c_set)
    n_set_num = set([1, 2, 4])
    c_set_num = set([3, 5])
    nc_set_num = n_set_num.union(c_set_num)

    # None index
    include = None
    c_ind, n_ind = ftddf._filter_include_indices(c_set, n_set, include, nc_set)
    assert c_ind == c_set and n_ind == n_set

    # 'numerical' index
    include = 'numerical'
    c_ind, n_ind = ftddf._filter_include_indices(c_set, n_set, include, nc_set)
    assert c_ind == set() and n_ind == n_set

    # 'categorical' index
    include = 'categorical'
    c_ind, n_ind = ftddf._filter_include_indices(c_set, n_set, include, nc_set)
    assert c_ind == c_set and n_ind == set()

    # Numbered index
    # ...int
    include = 5
    with pytest.raises(IndexError) as exin:
        ftddf._filter_include_indices(c_set, n_set, include, nc_set)
    assert str(exin.value) == index_error_a.format(include)
    #
    c_ind, n_ind = ftddf._filter_include_indices(c_set_num, n_set_num, include,
                                                 nc_set_num)
    assert c_ind == set([5]) and n_ind == set()
    # ...float (and object testing)
    include = 5.
    with pytest.raises(TypeError) as exin:
        ftddf._filter_include_indices(c_set, n_set, include, nc_set)
    assert str(exin.value) == type_error.format(include)

    # String index
    include = 'c2'
    c_ind, n_ind = ftddf._filter_include_indices(c_set, n_set, include, nc_set)
    assert c_ind == set(['c2']) and n_ind == set()

    # List index
    # ...with float
    include = ['c1', 'c5', 77, 7.7, 'n3']
    with pytest.raises(IndexError) as exin:
        ftddf._filter_include_indices(c_set, n_set, include, nc_set)
    exin_msg = str(exin.value)
    assert (exin_msg.startswith(index_error_b) and '77' in exin_msg
            and '7.7' in exin_msg and "'c5'" in exin_msg)
    # ..normal
    include = ['c1', 'c3', 'n3']
    c_ind, n_ind = ftddf._filter_include_indices(
        set(['c1', 'c2']), n_set, include, nc_set)
    assert c_ind == set(['c1']) and n_ind == set(['n3'])


def test_filter_exclude_indices():
    """
    Tests :func:`fatf.transparency.data.describe._filter_exclude_indices`.
    """
    index_error_a = 'The following exclude index is not a valid index: {}.'
    index_error_b = 'The following exclude indices are not valid indices: '
    type_error = ('The exclude parameter can either be a string, an integer '
                  'or a list of these two types.')
    n_set = set(['n1', 'n2', 'n3'])
    c_set = set(['c1', 'c2', 'c3'])
    nc_set = n_set.union(c_set)
    n_set_num = set([1, 2, 4])
    c_set_num = set([3, 5])
    nc_set_num = n_set_num.union(c_set_num)

    # None index
    exclude = None
    c_ind, n_ind = ftddf._filter_exclude_indices(c_set, n_set, exclude, nc_set)
    assert c_ind == c_set and n_ind == n_set

    # 'numerical' index
    exclude = 'numerical'
    c_ind, n_ind = ftddf._filter_exclude_indices(c_set, n_set, exclude, nc_set)
    assert c_ind == c_set and n_ind == set()

    # 'categorical' index
    exclude = 'categorical'
    c_ind, n_ind = ftddf._filter_exclude_indices(c_set, n_set, exclude, nc_set)
    assert c_ind == set() and n_ind == n_set

    # Numbered index
    # ...int
    exclude = 5
    with pytest.raises(IndexError) as exin:
        ftddf._filter_exclude_indices(c_set, n_set, exclude, nc_set)
    assert str(exin.value) == index_error_a.format(exclude)
    #
    c_ind, n_ind = ftddf._filter_exclude_indices(c_set_num, n_set_num, exclude,
                                                 nc_set_num)
    assert c_ind == set([3]) and n_ind == set([1, 2, 4])
    # ...float (and object testing)
    exclude = 5.
    with pytest.raises(TypeError) as exin:
        ftddf._filter_exclude_indices(c_set, n_set, exclude, nc_set)
    assert str(exin.value) == type_error.format(exclude)

    # String index
    exclude = 'c2'
    c_ind, n_ind = ftddf._filter_exclude_indices(c_set, n_set, exclude, nc_set)
    assert c_ind == set(['c1', 'c3']) and n_ind == set(['n1', 'n2', 'n3'])

    # List index
    # ...with float
    exclude = ['c1', 'c5', 77, 7.7, 'n3']
    with pytest.raises(IndexError) as exin:
        ftddf._filter_exclude_indices(c_set, n_set, exclude, nc_set)
    exin_msg = str(exin.value)
    assert (exin_msg.startswith(index_error_b) and '77' in exin_msg
            and '7.7' in exin_msg and "'c5'" in exin_msg)
    # ..normal
    exclude = ['c1', 'c3', 'n3']
    c_ind, n_ind = ftddf._filter_exclude_indices(
        set(['c1', 'c2']), n_set, exclude, nc_set)
    assert c_ind == set(['c2']) and n_ind == set(['n1', 'n2'])
