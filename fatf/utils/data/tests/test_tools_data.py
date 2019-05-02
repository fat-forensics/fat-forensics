"""
Tests data tools and utilities.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
#         Matt Clifford <mc15445@bristol.ac.uk>
# License: new BSD

import pytest

import numpy as np

import fatf.utils.data.tools as fudt

from fatf.exceptions import IncorrectShapeError


def test_group_by_column_errors():
    """
    Tests :func:`fatf.utils.data.tools.group_by_column` for errors.
    """
    incorrect_shape_error_data = 'The input array should be 2-dimensional.'
    value_error_data = ('The input array should be of a base type (a mixture '
                        'of numerical and textual types).')
    #
    index_error_index = ('*{}* is not a valid column index for the input '
                         'dataset.')
    type_error_index = 'The column index can either be a string or an integer.'
    #
    value_error_bins = 'The numerical_bins_number needs to be at least 2.'
    type_error_bins = ('The numerical_bins_number parameter has to be an '
                       'integer.')
    #
    value_error_grouping_num_empty = ('A numerical grouping list has to '
                                      'contain at least one element.')
    type_error_grouping_num_inner = ('For a numerical column all of the '
                                     'grouping items must be numbers. *{}* is '
                                     'not a number.')
    value_error_grouping_num_monotonicity = ('The numbers in the groupings '
                                             'list have to be monotonically '
                                             'increasing.')
    type_error_grouping_num_general = ('Since a numerical column was chosen '
                                       'the grouping must be a list of bin '
                                       'boundaries or None.')
    #
    type_error_grouping_cat_general = ('Since a categorical column was chosen '
                                       'the grouping must be a list of tuples '
                                       'representing categorical values '
                                       'grouping or None for the default '
                                       'grouping.')
    type_error_grouping_cat_tuple = ('For a categorical column all of the '
                                     'grouping items must be tuples. *{}* '
                                     'is not a tuple.')
    value_error_grouping_cat_empty = ('A categorical grouping list has to '
                                      'contain at least one element.')
    value_error_grouping_cat_extra = ('*{}* value is not present in the '
                                      'selected column.')
    value_error_grouping_cat_unique = ('Some values are duplicated across '
                                       'tuples.')
    #
    user_warning_val = ('The following values in the selected column were '
                        'not accounted for in the grouping tuples:\n{}.')
    user_warning_ind = ('The following row indices could not be accounted for:'
                        '\n{}.\n For a numerical column there may have been '
                        'some numpy.nan therein. For a categorical column '
                        'some of the column values were probably not '
                        'specified in the grouping, in which case there '
                        'should be a separate user warning.')

    num_array = np.array([[1, 2], [3, 4]])
    cat_array = np.array([['a', 'b'], [3, 4]])

    with pytest.raises(IncorrectShapeError) as exin:
        fudt.group_by_column(np.ones((2, 2, 2)), 1)
    assert str(exin.value) == incorrect_shape_error_data
    with pytest.raises(ValueError) as exin:
        fudt.group_by_column(np.array([[1, 2], [3, None]]), None)
    assert str(exin.value) == value_error_data

    with pytest.raises(IndexError) as exin:
        fudt.group_by_column(num_array, 3)
    assert str(exin.value) == index_error_index.format(3)
    with pytest.raises(TypeError) as exin:
        fudt.group_by_column(num_array, None)
    assert str(exin.value) == type_error_index

    with pytest.raises(ValueError) as exin:
        fudt.group_by_column(num_array, 1, numerical_bins_number=1)
    assert str(exin.value) == value_error_bins
    with pytest.raises(TypeError) as exin:
        fudt.group_by_column(num_array, 1, numerical_bins_number='1')
    assert str(exin.value) == type_error_bins

    with pytest.raises(TypeError) as exin:
        fudt.group_by_column(num_array, 1, groupings='a')
    assert str(exin.value) == type_error_grouping_num_general
    with pytest.raises(ValueError) as exin:
        fudt.group_by_column(num_array, 1, groupings=[])
    assert str(exin.value) == value_error_grouping_num_empty
    with pytest.raises(TypeError) as exin:
        fudt.group_by_column(num_array, 1, groupings=[5, 7.3, 8, 'a'])
    assert str(exin.value) == type_error_grouping_num_inner.format('a')
    with pytest.raises(ValueError) as exin:
        fudt.group_by_column(num_array, 1, groupings=[5, 7.3, 8, 7.9, 11])
    assert str(exin.value) == value_error_grouping_num_monotonicity

    with pytest.raises(TypeError) as exin:
        fudt.group_by_column(cat_array, 1, groupings='a')
    assert str(exin.value) == type_error_grouping_cat_general
    with pytest.raises(TypeError) as exin:
        fudt.group_by_column(cat_array, 0, groupings=[('3', ), ['a'], ('a', )])
    assert str(exin.value) == type_error_grouping_cat_tuple.format("['a']")
    with pytest.raises(ValueError) as exin:
        fudt.group_by_column(cat_array, 1, groupings=[])
    assert str(exin.value) == value_error_grouping_cat_empty
    with pytest.raises(ValueError) as exin:
        fudt.group_by_column(cat_array, 0, groupings=[('3', 'a'), ('1', )])
    assert str(exin.value) == value_error_grouping_cat_extra.format('1')
    with pytest.raises(ValueError) as exin:
        fudt.group_by_column(cat_array, 0, groupings=[('3', 'a'), ('a', )])
    assert str(exin.value) == value_error_grouping_cat_unique

    with pytest.warns(UserWarning) as warning:
        grp, grpn = fudt.group_by_column(cat_array, 0, groupings=[('3', )])
    assert len(warning) == 2
    assert user_warning_val.format("{'a'}") == str(warning[0].message)
    assert user_warning_ind.format('{0}') == str(warning[1].message)
    assert grp == [[1]]
    assert grpn == ["('3',)"]
    #
    nan_array = np.array([[0, np.inf], [0, 7], [0, -np.inf], [0, np.nan]])
    with pytest.warns(UserWarning) as warning:
        grp, grpn = fudt.group_by_column(nan_array, 1, groupings=[1])
    assert len(warning) == 1
    assert user_warning_ind.format('{3}') == str(warning[0].message)
    assert grp == [[2], [0, 1]]
    assert grpn == ['x <= 1', '1 < x']


def test_group_by_column():
    """
    Tests :func:`fatf.utils.data.tools.group_by_column`.
    """
    # Structured array
    # Classic array
    pass


def test_apply_to_column_grouping_errors():
    """
    Tests :func:`fatf.utils.data.tools.apply_to_column_grouping` for errors.
    """
    shape_error_labels = 'The labels array should be 1-dimensional.'
    shape_error_gt = 'The predictions array should be 1-dimensional.'
    shape_error_dim = ('The labels and predictions arrays should be of the '
                       'same length.')
    #
    type_error_rg = 'The row_grouping parameter has to be a list.'
    type_error_rg_inner = ('All of the elements of the row_grouping list have '
                           'to be lists.')
    type_error_rg_in_inner = ('All of the elements of the inner lists in the '
                              'row_grouping have to be integers.')
    value_error_rg_empty = ('The row_grouping parameter cannot be an empty '
                            'list.')
    value_error_rg_inner_empty = ('All of the elements of the row_grouping '
                                  'list must be non-empty lists.')
    value_error_rg_dup = ('Some of the values in the row_grouping are '
                          'duplicated.')
    #
    type_error_fnc = 'The fnc parameter is not callable (a function).'
    attribute_error_fnc = ('Provided function (fnc) does not require 2 input '
                           'parameters. The first required parameter should '
                           'be ground truth labels and the second one '
                           'predictions.')

    labels = np.array(['a', 'b', 'b', 'b', 'a', 'b', 'b', 'a'])
    ground_truth = np.array(['b', 'b', 'a', 'b', 'b', 'b', 'a', 'b'])
    groupings = [[0, 1, 2, 6, 7], [3, 4, 5]]

    two_d_ones = np.ones((2, 2))

    def fnc(x, y, z):
        return x + y + z

    with pytest.raises(IncorrectShapeError) as exin:
        fudt.apply_to_column_grouping(two_d_ones, two_d_ones, [], fnc)
    assert str(exin.value) == shape_error_labels
    with pytest.raises(IncorrectShapeError) as exin:
        fudt.apply_to_column_grouping(labels, two_d_ones, [], fnc)
    assert str(exin.value) == shape_error_gt
    with pytest.raises(IncorrectShapeError) as exin:
        fudt.apply_to_column_grouping(labels, ground_truth[:-1], [], fnc)
    assert str(exin.value) == shape_error_dim

    with pytest.raises(TypeError) as exin:
        fudt.apply_to_column_grouping(labels, ground_truth, None, fnc)
    assert str(exin.value) == type_error_rg
    with pytest.raises(ValueError) as exin:
        fudt.apply_to_column_grouping(labels, ground_truth, [], fnc)
    assert str(exin.value) == value_error_rg_empty
    with pytest.raises(TypeError) as exin:
        fudt.apply_to_column_grouping(labels, ground_truth, [[1], 'a'], fnc)
    assert str(exin.value) == type_error_rg_inner
    with pytest.raises(ValueError) as exin:
        fudt.apply_to_column_grouping(labels, ground_truth, [[1], []], fnc)
    assert str(exin.value) == value_error_rg_inner_empty
    with pytest.raises(TypeError) as exin:
        fudt.apply_to_column_grouping(labels, ground_truth, [[1], ['b']], fnc)
    assert str(exin.value) == type_error_rg_in_inner
    with pytest.raises(ValueError) as exin:
        fudt.apply_to_column_grouping(labels, ground_truth, [[1], [2, 1]], fnc)
    assert str(exin.value) == value_error_rg_dup

    with pytest.raises(TypeError) as exin:
        fudt.apply_to_column_grouping(labels, ground_truth, [[1], [2]], None)
    assert str(exin.value) == type_error_fnc
    with pytest.raises(AttributeError) as exin:
        fudt.apply_to_column_grouping(labels, ground_truth, [[1], [2]], fnc)
    assert str(exin.value) == attribute_error_fnc


def test_apply_to_column_grouping():
    """
    Tests :func:`fatf.utils.data.tools.apply_to_column_grouping`.
    """
    labels = np.array(['a', 'b', 'b', 'b', 'a', 'b', 'b', 'a'])
    ground_truth = np.array(['b', 'b', 'a', 'b', 'b', 'b', 'a', 'b'])
    groupings = [[0, 1, 2, 6, 7], [3, 4, 5]]

    def fnc(x, y):
        return (x != y).sum() / x.shape[0]

    vls = fudt.apply_to_column_grouping(labels, ground_truth, groupings, fnc)
    assert vls == [4 / 5, 1 / 3]
