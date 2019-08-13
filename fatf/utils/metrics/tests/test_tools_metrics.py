"""
Tests implementations of model metric tools.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import pytest

import numpy as np

from fatf.exceptions import IncorrectShapeError

import fatf.utils.metrics.tools as fumt

USER_WARNING = ('Some of the given labels are not present in either of the '
                'input arrays: {}.')
MISSING_LABEL_WARNING = ('Some of the given labels are not present in either '
                         'of the input arrays: {2}.')

DATASET = np.array([['0', '3', '0'], ['0', '5', '0'], ['0', '7', '0'],
                    ['0', '5', '0'], ['0', '7', '0'], ['0', '3', '0'],
                    ['0', '5', '0'], ['0', '3', '0'], ['0', '7', '0'],
                    ['0', '5', '0'], ['0', '7', '0'], ['0', '7', '0'],
                    ['0', '5', '0'], ['0', '7', '0'], ['0', '7', '0']])
_INDICES_PER_BIN = [[0, 5, 7], [1, 6, 9, 3, 12], [2, 4, 8, 10, 11, 13, 14]]
GROUND_TRUTH = np.zeros((15, ), dtype=int)
GROUND_TRUTH[_INDICES_PER_BIN[0]] = [0, 1, 0]
GROUND_TRUTH[_INDICES_PER_BIN[1]] = [0, 1, 2, 1, 0]
GROUND_TRUTH[_INDICES_PER_BIN[2]] = [0, 1, 2, 0, 1, 2, 0]
PREDICTIONS = np.zeros((15, ), dtype=int)
PREDICTIONS[_INDICES_PER_BIN[0]] = [0, 0, 0]
PREDICTIONS[_INDICES_PER_BIN[1]] = [0, 2, 2, 2, 0]
PREDICTIONS[_INDICES_PER_BIN[2]] = [0, 1, 2, 2, 1, 0, 0]


def test_validate_confusion_matrix():
    """
    Tests :func:`fatf.utils.metrics.tools.validate_confusion_matrix`.
    """
    incorrect_shape_error_2d = ('The confusion matrix has to be a '
                                '2-dimensional numpy array.')
    incorrect_shape_error_square = ('The confusion matrix has to be a square '
                                    '(equal width and height) numpy array.')
    incorrect_shape_error_2 = 'The confusion matrix needs to be at least 2x2.'
    value_error = 'The confusion matrix cannot be a structured numpy array.'
    type_error_cm = 'The confusion matrix has to be of integer kind.'
    type_error_index = 'The label index has to be an integer.'
    index_error = ('The label index {} is not a valid index for the confusion '
                   'matrix of shape {}x{}.')

    three_d_array = np.array([[[0], [2]], [[3]]])
    two_d_array_rect = np.array([[0, 2], [3, 5], [7, 9]])
    two_d_array_one = np.array([[0]])
    struct_array = np.array([(1, 2), (3, 4)], dtype=[('a', 'i'), ('b', 'i')])
    non_int_array = np.array([[2.0, 2], [7, 8]])

    two_d_array = np.array([[1, 2], [3, 4]])

    with pytest.raises(IncorrectShapeError) as exi:
        fumt.validate_confusion_matrix(three_d_array)
    assert str(exi.value) == incorrect_shape_error_2d

    with pytest.raises(IncorrectShapeError) as exi:
        fumt.validate_confusion_matrix(two_d_array_rect)
    assert str(exi.value) == incorrect_shape_error_square

    with pytest.raises(IncorrectShapeError) as exi:
        fumt.validate_confusion_matrix(two_d_array_one)
    assert str(exi.value) == incorrect_shape_error_2

    with pytest.raises(ValueError) as exi:
        fumt.validate_confusion_matrix(struct_array)
    assert str(exi.value) == value_error

    with pytest.raises(TypeError) as exi:
        fumt.validate_confusion_matrix(non_int_array)
    assert str(exi.value) == type_error_cm

    with pytest.raises(TypeError) as exi:
        fumt.validate_confusion_matrix(two_d_array, 'a')
    assert str(exi.value) == type_error_index

    with pytest.raises(IndexError) as exi:
        fumt.validate_confusion_matrix(two_d_array, -1)
    assert str(exi.value) == index_error.format(-1, 2, 2)

    with pytest.raises(IndexError) as exi:
        fumt.validate_confusion_matrix(two_d_array, 2)
    assert str(exi.value) == index_error.format(2, 2, 2)


def test_validate_confusion_matrix_size():
    """
    Tests ``validate_confusion_matrix_size`` function.

    Tests :func:`fatf.utils.metrics.tools.validate_confusion_matrix_size`
    function.
    """
    incorrect_shape_error = ('The confusion matrix is of shape {}x{} but '
                             '{}x{} is the requirement.')

    two_d = np.array([[0, 1], [2, 3]])
    three_d = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])

    with pytest.raises(IncorrectShapeError) as exi:
        fumt.validate_confusion_matrix_size(two_d, 3)
    assert str(exi.value) == incorrect_shape_error.format(2, 2, 3, 3)

    with pytest.raises(IncorrectShapeError) as exi:
        fumt.validate_confusion_matrix_size(three_d, 2)
    assert str(exi.value) == incorrect_shape_error.format(3, 3, 2, 2)


def test_get_confusion_matrix_errors():
    """
    Tests :func:`fatf.utils.metrics.tools.get_confusion_matrix` errors.
    """
    incorrect_shape_error_gt = ('The ground truth vector has to be '
                                '1-dimensional numpy array.')
    incorrect_shape_error_pred = ('The predictions vector has to be '
                                  '1-dimensional numpy array.')
    incorrect_shape_error_gtp = ('Both the ground truth and the predictions '
                                 'vectors have to have the same length.')
    value_error_labels_empty = 'The labels list cannot be empty.'
    value_error_labels_duplicates = 'The labels list contains duplicates.'
    value_error_labels_missing = ('The following labels are present in the '
                                  'input arrays but were not given in the '
                                  'labels parameter: {}.')
    type_error_labels = 'The labels parameter has to either a list or None.'

    two_d_array = np.array([[1, 2], [3, 4]])
    one_d_array_4 = np.array([1, 2, 3, 4])
    one_d_array_5 = np.array([1, 2, 3, 4, 5])

    cma_true = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0],
                         [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]])

    with pytest.raises(IncorrectShapeError) as exi:
        fumt.get_confusion_matrix(two_d_array, two_d_array)
    assert str(exi.value) == incorrect_shape_error_gt
    #
    with pytest.raises(IncorrectShapeError) as exi:
        fumt.get_confusion_matrix(one_d_array_4, two_d_array)
    assert str(exi.value) == incorrect_shape_error_pred
    #
    with pytest.raises(IncorrectShapeError) as exi:
        fumt.get_confusion_matrix(one_d_array_4, one_d_array_5)
    assert str(exi.value) == incorrect_shape_error_gtp

    with pytest.raises(TypeError) as exi:
        fumt.get_confusion_matrix(one_d_array_4, one_d_array_4, 'a')
    assert str(exi.value) == type_error_labels
    #
    with pytest.raises(ValueError) as exi:
        fumt.get_confusion_matrix(one_d_array_4, one_d_array_4, [])
    assert str(exi.value) == value_error_labels_empty
    #
    with pytest.raises(ValueError) as exi:
        fumt.get_confusion_matrix(one_d_array_4, one_d_array_4, [2, 3, 2])
    assert str(exi.value) == value_error_labels_duplicates
    #
    with pytest.raises(ValueError) as exi:
        fumt.get_confusion_matrix(one_d_array_4, one_d_array_4, [2, 4, 3])
    assert str(exi.value) == value_error_labels_missing.format('{1}')

    with pytest.warns(UserWarning) as w:
        cma = fumt.get_confusion_matrix(one_d_array_4, one_d_array_4,
                                        [1, 2, 3, 4, 5])
    assert len(w) == 1
    assert str(w[0].message) == USER_WARNING.format('{5}')
    assert np.array_equal(cma, cma_true)


def test_get_confusion_matrix():
    """
    Tests :func:`fatf.utils.metrics.tools.get_confusion_matrix` function.
    """
    # [[1, 1, 1],
    #  [1, 2, 1],
    #  [1, 1, 1]]
    ground_truth = np.array(['a', 'b', 'b', 'b', 'a', 'a', 'b', 'c', 'c', 'c'])
    predictions = np.array(['b', 'a', 'b', 'c', 'a', 'c', 'b', 'a', 'c', 'b'])
    # [[3, 11],
    #  [7, 5 ]]
    ground_truth_bin = np.array([
        'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b', 'a', 'a', 'a', 'a', 'a', 'a',
        'a', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b'
    ])
    predictions_bin = np.array([
        'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b',
        'b', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'
    ])

    cmx = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]])
    cmx_bin = np.array([[3, 11], [7, 5]])
    cmx_bb = np.array([[1, 1, 0, 1], [1, 2, 0, 1], [0, 0, 0, 0], [1, 1, 0, 1]])

    # Default labeling
    cma = fumt.get_confusion_matrix(ground_truth, predictions)
    assert np.array_equal(cmx, cma)
    cma = fumt.get_confusion_matrix(ground_truth_bin, predictions_bin)
    assert np.array_equal(cmx_bin, cma)

    # Custom non-existing labeling
    with pytest.warns(UserWarning) as w:
        cma = fumt.get_confusion_matrix(ground_truth, predictions,
                                        ['a', 'b', 'bb', 'c'])
    assert len(w) == 1
    assert str(w[0].message) == USER_WARNING.format("{'bb'}")
    assert np.array_equal(cmx_bb, cma)


def test_confusion_matrix_per_subgroup():
    """
    Tests calculating confusion matrix per sub-population.

    Tests
    :func:`fatf.utils.metrics.tools.confusion_matrix_per_subgroup`
    function.
    """

    mx1 = np.array([[2, 1, 0], [0, 0, 0], [0, 0, 0]])
    mx2 = np.array([[2, 0, 0], [0, 0, 0], [0, 2, 1]])
    mx3 = np.array([[2, 0, 1], [0, 2, 0], [1, 0, 1]])

    with pytest.warns(UserWarning) as w:
        pcmxs, bin_names = fumt.confusion_matrix_per_subgroup(
            DATASET, GROUND_TRUTH, PREDICTIONS, 1)
    assert len(w) == 1
    assert str(w[0].message) == MISSING_LABEL_WARNING

    assert len(pcmxs) == 3
    assert np.array_equal(pcmxs[0], mx1)
    assert np.array_equal(pcmxs[1], mx2)
    assert np.array_equal(pcmxs[2], mx3)
    assert bin_names == ["('3',)", "('5',)", "('7',)"]


def test_confusion_matrix_per_subgroup_indexed():
    """
    Tests calculating confusion matrix per index-based sub-population.

    Tests
    :func:`fatf.utils.metrics.tools.confusion_matrix_per_subgroup_indexed`
    function.
    """
    incorrect_shape_error_gt = ('The ground_truth parameter should be a '
                                '1-dimensional numpy array.')
    incorrect_shape_error_p = ('The predictions parameter should be a '
                               '1-dimensional numpy array.')

    flat = np.array([1, 2])
    square = np.array([[1, 2], [3, 4]])
    with pytest.raises(IncorrectShapeError) as exin:
        fumt.confusion_matrix_per_subgroup_indexed([[0]], square, square)
    assert str(exin.value) == incorrect_shape_error_gt

    with pytest.raises(IncorrectShapeError) as exin:
        fumt.confusion_matrix_per_subgroup_indexed([[0]], flat, square)
    assert str(exin.value) == incorrect_shape_error_p

    mx1 = np.array([[2, 1, 0], [0, 0, 0], [0, 0, 0]])
    mx2 = np.array([[2, 0, 0], [0, 0, 0], [0, 2, 1]])
    mx3 = np.array([[2, 0, 1], [0, 2, 0], [1, 0, 1]])

    with pytest.warns(UserWarning) as w:
        pcmxs_1 = fumt.confusion_matrix_per_subgroup_indexed(
            _INDICES_PER_BIN, GROUND_TRUTH, PREDICTIONS, labels=[0, 1, 2])
        pcmxs_2 = fumt.confusion_matrix_per_subgroup_indexed(
            _INDICES_PER_BIN, GROUND_TRUTH, PREDICTIONS)
    assert len(w) == 2
    wmsg = ('Some of the given labels are not present in either of the input '
            'arrays: {2}.')
    assert str(w[0].message) == wmsg
    assert str(w[1].message) == wmsg
    assert len(pcmxs_1) == 3
    assert len(pcmxs_2) == 3
    assert np.array_equal(pcmxs_1[0], mx1)
    assert np.array_equal(pcmxs_2[0], mx1)
    assert np.array_equal(pcmxs_1[1], mx2)
    assert np.array_equal(pcmxs_2[1], mx2)
    assert np.array_equal(pcmxs_1[2], mx3)
    assert np.array_equal(pcmxs_2[2], mx3)
