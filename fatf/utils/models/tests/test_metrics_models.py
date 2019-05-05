"""
Holds custom distance functions used for FAT-Forensics examples and testing.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import numpy as np
import pytest

import fatf.utils.models.metrics as fumm

from fatf.exceptions import IncorrectShapeError

USER_WARNING = ('Some of the given labels are not present in either of the '
                'input arrays: {}.')
GROUND_TRUTH = np.array(['a', 'b', 'b', 'b', 'a', 'a', 'b', 'c', 'c', 'c'])
PREDICTIONS = np.array(['b', 'a', 'b', 'c', 'a', 'c', 'b', 'a', 'c', 'b'])
# [[3, 11],
#  [7, 5 ]]
GROUND_TRUTH_BIN = np.array([
    'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b', 'a', 'a', 'a', 'a', 'a', 'a', 'a',
    'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b'
])
# [[1, 1, 1],
#  [1, 2, 1],
#  [1, 1, 1]]
PREDICTIONS_BIN = np.array([
    'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b',
    'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'
])


def test_validate_confusion_matrix():
    """
    Tests :func:`fatf.utils.models.metrics.validate_confusion_matrix`.
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
        fumm.validate_confusion_matrix(three_d_array)
    assert str(exi.value) == incorrect_shape_error_2d

    with pytest.raises(IncorrectShapeError) as exi:
        fumm.validate_confusion_matrix(two_d_array_rect)
    assert str(exi.value) == incorrect_shape_error_square

    with pytest.raises(IncorrectShapeError) as exi:
        fumm.validate_confusion_matrix(two_d_array_one)
    assert str(exi.value) == incorrect_shape_error_2

    with pytest.raises(ValueError) as exi:
        fumm.validate_confusion_matrix(struct_array)
    assert str(exi.value) == value_error

    with pytest.raises(TypeError) as exi:
        fumm.validate_confusion_matrix(non_int_array)
    assert str(exi.value) == type_error_cm

    with pytest.raises(TypeError) as exi:
        fumm.validate_confusion_matrix(two_d_array, 'a')
    assert str(exi.value) == type_error_index

    with pytest.raises(IndexError) as exi:
        fumm.validate_confusion_matrix(two_d_array, -1)
    assert str(exi.value) == index_error.format(-1, 2, 2)

    with pytest.raises(IndexError) as exi:
        fumm.validate_confusion_matrix(two_d_array, 2)
    assert str(exi.value) == index_error.format(2, 2, 2)


def test_validate_confusion_matrix_size():
    """
    Tests :func:`fatf.utils.models.metrics._validate_confusion_matrix_size`.
    """
    incorrect_shape_error = ('The confusion matrix is of shape {}x{} but '
                             '{}x{} is the requirement.')

    two_d = np.array([[0, 1], [2, 3]])
    three_d = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])

    with pytest.raises(IncorrectShapeError) as exi:
        fumm._validate_confusion_matrix_size(two_d, 3)
    assert str(exi.value) == incorrect_shape_error.format(2, 2, 3, 3)

    with pytest.raises(IncorrectShapeError) as exi:
        fumm._validate_confusion_matrix_size(three_d, 2)
    assert str(exi.value) == incorrect_shape_error.format(3, 3, 2, 2)


def test_get_confusion_matrix_errors():
    """
    Tests :func:`fatf.utils.models.metrics.get_confusion_matrix` for errors.
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
        fumm.get_confusion_matrix(two_d_array, two_d_array)
    assert str(exi.value) == incorrect_shape_error_gt
    #
    with pytest.raises(IncorrectShapeError) as exi:
        fumm.get_confusion_matrix(one_d_array_4, two_d_array)
    assert str(exi.value) == incorrect_shape_error_pred
    #
    with pytest.raises(IncorrectShapeError) as exi:
        fumm.get_confusion_matrix(one_d_array_4, one_d_array_5)
    assert str(exi.value) == incorrect_shape_error_gtp

    with pytest.raises(TypeError) as exi:
        fumm.get_confusion_matrix(one_d_array_4, one_d_array_4, 'a')
    assert str(exi.value) == type_error_labels
    #
    with pytest.raises(ValueError) as exi:
        fumm.get_confusion_matrix(one_d_array_4, one_d_array_4, [])
    assert str(exi.value) == value_error_labels_empty
    #
    with pytest.raises(ValueError) as exi:
        fumm.get_confusion_matrix(one_d_array_4, one_d_array_4, [2, 3, 2])
    assert str(exi.value) == value_error_labels_duplicates
    #
    with pytest.raises(ValueError) as exi:
        fumm.get_confusion_matrix(one_d_array_4, one_d_array_4, [2, 4, 3])
    assert str(exi.value) == value_error_labels_missing.format('{1}')

    with pytest.warns(UserWarning) as w:
        cma = fumm.get_confusion_matrix(one_d_array_4, one_d_array_4,
                                        [1, 2, 3, 4, 5])
    assert len(w) == 1
    assert str(w[0].message) == USER_WARNING.format('{5}')
    assert np.array_equal(cma, cma_true)


def test_get_confusion_matrix():
    """
    Tests the :func:`fatf.utils.models.metrics.get_confusion_matrix` function.
    """
    cmx = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]])
    cmx_bb = np.array([[1, 1, 0, 1], [1, 2, 0, 1], [0, 0, 0, 0], [1, 1, 0, 1]])

    # Default labeling
    cma = fumm.get_confusion_matrix(GROUND_TRUTH, PREDICTIONS)
    assert np.array_equal(cmx, cma)

    # Custom non-existing labeling
    with pytest.warns(UserWarning) as w:
        cma = fumm.get_confusion_matrix(GROUND_TRUTH, PREDICTIONS,
                                        ['a', 'b', 'bb', 'c'])
    assert len(w) == 1
    assert str(w[0].message) == USER_WARNING.format("{'bb'}")
    assert np.array_equal(cmx_bb, cma)


def test_multiclass_true_positive_rate():
    """
    Tests :func:`fatf.utils.models.metrics.multiclass_true_positive_rate`.
    """
    cma = fumm.get_confusion_matrix(GROUND_TRUTH, PREDICTIONS)

    mtpr_0 = fumm.multiclass_true_positive_rate(cma, 0)
    assert mtpr_0 == pytest.approx(0.333, abs=1e-3)
    mtpr_1 = fumm.multiclass_true_positive_rate(cma, 1)
    assert mtpr_1 == 0.5
    mtpr_2 = fumm.multiclass_true_positive_rate(cma, 2)
    assert mtpr_2 == pytest.approx(0.333, abs=1e-3)


def test_multiclass_true_negative_rate():
    """
    Tests :func:`fatf.utils.models.metrics.multiclass_true_negative_rate`.
    """
    cma = fumm.get_confusion_matrix(GROUND_TRUTH, PREDICTIONS)

    type_error = 'The strict parameter has to be a boolean.'
    with pytest.raises(TypeError) as exi:
        fumm.multiclass_true_negative_rate(cma, 0, 'one')
    assert str(exi.value) == type_error

    metric = pytest.approx(5 / 7)
    mtpr_0_n = fumm.multiclass_true_negative_rate(cma, 0)
    assert mtpr_0_n == metric
    mtpr_0_n = fumm.multiclass_true_negative_rate(cma, 0, False)
    assert mtpr_0_n == metric
    #
    metric = pytest.approx(4 / 6)
    mtpr_1_n = fumm.multiclass_true_negative_rate(cma, 1)
    assert mtpr_1_n == metric
    mtpr_1_n = fumm.multiclass_true_negative_rate(cma, 1, False)
    assert mtpr_1_n == metric
    #
    metric = pytest.approx(5 / 7)
    mtpr_2_n = fumm.multiclass_true_negative_rate(cma, 2)
    assert mtpr_2_n == metric
    mtpr_2_n = fumm.multiclass_true_negative_rate(cma, 2, False)
    assert mtpr_2_n == metric

    metric = pytest.approx(3 / 7)
    mtpr_0_p = fumm.multiclass_true_negative_rate(cma, 0, True)
    assert mtpr_0_p == metric
    metric = pytest.approx(2 / 6)
    mtpr_1_p = fumm.multiclass_true_negative_rate(cma, 1, True)
    assert mtpr_1_p == metric
    metric = pytest.approx(3 / 7)
    mtpr_2_p = fumm.multiclass_true_negative_rate(cma, 2, True)
    assert mtpr_2_p == metric


def test_multiclass_false_positive_rate():
    """
    Tests :func:`fatf.utils.models.metrics.multiclass_false_positive_rate`.
    """
    cma = fumm.get_confusion_matrix(GROUND_TRUTH, PREDICTIONS)

    mtpr_0 = fumm.multiclass_false_positive_rate(cma, 0)
    assert mtpr_0 == pytest.approx(2 / 7, abs=1e-3)
    mtpr_1 = fumm.multiclass_false_positive_rate(cma, 1)
    assert mtpr_1 == pytest.approx(2 / 6, abs=1e-3)
    mtpr_2 = fumm.multiclass_false_positive_rate(cma, 2)
    assert mtpr_2 == pytest.approx(2 / 7, abs=1e-3)


def test_multiclass_false_negative_rate():
    """
    Tests :func:`fatf.utils.models.metrics.multiclass_false_negative_rate`.
    """
    cma = fumm.get_confusion_matrix(GROUND_TRUTH, PREDICTIONS)

    mtpr_0 = fumm.multiclass_false_negative_rate(cma, 0)
    assert mtpr_0 == pytest.approx(2 / 3, abs=1e-3)
    mtpr_1 = fumm.multiclass_false_negative_rate(cma, 1)
    assert mtpr_1 == pytest.approx(2 / 4, abs=1e-3)
    mtpr_2 = fumm.multiclass_false_negative_rate(cma, 2)
    assert mtpr_2 == pytest.approx(2 / 3, abs=1e-3)


def test_true_positive_rate():
    """
    Tests :func:`fatf.utils.models.metrics.true_positive_rate`.
    """
    cma = fumm.get_confusion_matrix(GROUND_TRUTH_BIN, PREDICTIONS_BIN)
    mtpr = fumm.true_positive_rate(cma)
    assert mtpr == pytest.approx(3 / 10, abs=1e-3)


def test_true_negative_rate():
    """
    Tests :func:`fatf.utils.models.metrics.true_negative_rate`.
    """
    cma = fumm.get_confusion_matrix(GROUND_TRUTH_BIN, PREDICTIONS_BIN)
    mtpr = fumm.true_negative_rate(cma)
    assert mtpr == pytest.approx(5 / 16, abs=1e-3)


def test_false_negative_rate():
    """
    Tests :func:`fatf.utils.models.metrics.false_negative_rate`.
    """
    cma = fumm.get_confusion_matrix(GROUND_TRUTH_BIN, PREDICTIONS_BIN)
    mtpr = fumm.false_negative_rate(cma)
    assert mtpr == pytest.approx(7 / 10, abs=1e-3)


def test_false_positive_rate():
    """
    Tests :func:`fatf.utils.models.metrics.false_positive_rate`.
    """
    cma = fumm.get_confusion_matrix(GROUND_TRUTH_BIN, PREDICTIONS_BIN)
    mtpr = fumm.false_positive_rate(cma)
    assert mtpr == pytest.approx(11 / 16, abs=1e-3)


def test_multiclass_positive_predictive_value():
    """
    :func:`fatf.utils.models.metrics.multiclass_positive_predictive_value`.
    """
    cma = fumm.get_confusion_matrix(GROUND_TRUTH, PREDICTIONS)

    mtpr_0 = fumm.multiclass_positive_predictive_value(cma, 0)
    assert mtpr_0 == pytest.approx(1 / 3, abs=1e-3)
    mtpr_1 = fumm.multiclass_positive_predictive_value(cma, 1)
    assert mtpr_1 == pytest.approx(2 / 4, abs=1e-3)
    mtpr_2 = fumm.multiclass_positive_predictive_value(cma, 2)
    assert mtpr_2 == pytest.approx(1 / 3, abs=1e-3)


def test_multiclass_negative_predictive_value():
    """
    :func:`fatf.utils.models.metrics.multiclass_negative_predictive_value`.
    """
    cma = fumm.get_confusion_matrix(GROUND_TRUTH, PREDICTIONS)

    type_error = 'The strict parameter has to be a boolean.'
    with pytest.raises(TypeError) as exi:
        fumm.multiclass_negative_predictive_value(cma, 0, 'one')
    assert str(exi.value) == type_error

    metric = pytest.approx(5 / 7)
    mtpr_0_n = fumm.multiclass_negative_predictive_value(cma, 0)
    assert mtpr_0_n == metric
    mtpr_0_n = fumm.multiclass_negative_predictive_value(cma, 0, False)
    assert mtpr_0_n == metric
    #
    metric = pytest.approx(4 / 6)
    mtpr_1_n = fumm.multiclass_negative_predictive_value(cma, 1)
    assert mtpr_1_n == metric
    mtpr_1_n = fumm.multiclass_negative_predictive_value(cma, 1, False)
    assert mtpr_1_n == metric
    #
    metric = pytest.approx(5 / 7)
    mtpr_2_n = fumm.multiclass_negative_predictive_value(cma, 2)
    assert mtpr_2_n == metric
    mtpr_2_n = fumm.multiclass_negative_predictive_value(cma, 2, False)
    assert mtpr_2_n == metric

    metric = pytest.approx(3 / 7)
    mtpr_0_p = fumm.multiclass_negative_predictive_value(cma, 0, True)
    assert mtpr_0_p == metric
    metric = pytest.approx(2 / 6)
    mtpr_1_p = fumm.multiclass_negative_predictive_value(cma, 1, True)
    assert mtpr_1_p == metric
    metric = pytest.approx(3 / 7)
    mtpr_2_p = fumm.multiclass_negative_predictive_value(cma, 2, True)
    assert mtpr_2_p == metric


def test_positive_predictive_value():
    """
    :func:`fatf.utils.models.metrics.positive_predictive_value`.
    """
    cma = fumm.get_confusion_matrix(GROUND_TRUTH_BIN, PREDICTIONS_BIN)
    mtpr = fumm.positive_predictive_value(cma)
    assert mtpr == pytest.approx(3 / 14, abs=1e-3)


def test_negative_predictive_value():
    """
    :func:`fatf.utils.models.metrics.negative_predictive_value`.
    """
    cma = fumm.get_confusion_matrix(GROUND_TRUTH_BIN, PREDICTIONS_BIN)
    mtpr = fumm.negative_predictive_value(cma)
    assert mtpr == pytest.approx(5 / 12, abs=1e-3)


def test_accuracy():
    """
    Tests :func:`fatf.utils.models.metrics.accuracy` function.
    """
    cma = fumm.get_confusion_matrix(GROUND_TRUTH, PREDICTIONS)
    acc = fumm.accuracy(cma)
    assert acc == pytest.approx(4 / 10, abs=1e-3)

    cma = fumm.get_confusion_matrix(GROUND_TRUTH_BIN, PREDICTIONS_BIN)
    acc = fumm.accuracy(cma)
    assert acc == pytest.approx(8 / 26, abs=1e-3)


def test_multiclass_treatment():
    """
    Tests :func:`fatf.utils.models.metrics.multiclass_treatment` function.
    """
    cma = fumm.get_confusion_matrix(GROUND_TRUTH, PREDICTIONS)

    mtpr_0 = fumm.multiclass_treatment(cma, 0)
    assert mtpr_0 == pytest.approx(2 / 6, abs=1e-3)
    mtpr_1 = fumm.multiclass_treatment(cma, 1)
    assert mtpr_1 == pytest.approx(2 / 6, abs=1e-3)
    mtpr_2 = fumm.multiclass_treatment(cma, 2)
    assert mtpr_2 == pytest.approx(2 / 6, abs=1e-3)


def test_treatment():
    """
    Tests :func:`fatf.utils.models.metrics.treatment` function.
    """
    cma = fumm.get_confusion_matrix(GROUND_TRUTH_BIN, PREDICTIONS_BIN)
    mtpr = fumm.treatment(cma)
    assert mtpr == pytest.approx(11 / 18, abs=1e-3)
