"""
Holds model performance metrics.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import warnings

from numbers import Number
from typing import List, Optional, Union

import numpy as np

from fatf.exceptions import IncorrectShapeError

import fatf.utils.array.validation as fuav

__all__ = ['get_confusion_matrix',
           'multiclass_true_positive_rate',
           'multiclass_true_negative_rate',
           'multiclass_false_positive_rate',
           'multiclass_false_negative_rate',
           'true_positive_rate',
           'true_negative_rate',
           'false_positive_rate',
           'false_negative_rate',
           'multiclass_positive_predictive_value',
           'multiclass_negative_predictive_value',
           'positive_predictive_value',
           'negative_predictive_value',
           'accuracy',
           'multiclass_treatment',
           'treatment']  # yapf: disable


def validate_confusion_matrix(confusion_matrix: np.ndarray,
                              label_index: Optional[int] = None) -> bool:
    """
    Validates a confusion matrix.

    This function checks whether the ``confusion_matrix`` is 2-dimensional,
    square, unstructured and of integer kind.

    If the ``label_index`` parameter is given, it is checked to be a valid
    index for the given confusion matrix.

    Parameters
    ----------
    confusion_matrix : numpy.ndarray
        A confusion matrix to be validated.
    label_index : integer, optional (default=None)
        An index which validity will be checked for the confusion matrix (if
        not ``None``).

    Raises
    ------
    IncorrectShapeError
        The confusion matrix is not a 2-dimensional numpy array, it is not
        square (equal width and height) or its dimension is not at least 2x2.
    IndexError
        The ``label_index`` (if given) is not valid for the confusion matrix.
    TypeError
        The confusion matrix is not of an integer kind (e.g. ``int``,
        ``numpy.int32``, ``numpy.int64``). The ``label_index`` is not an
        integer.
    ValueError
        The confusion matrix is a structured numpy array.

    Returns
    -------
    is_valid : boolean
        ``True`` if the confusion matrix is valid, ``False`` otherwise.
    """
    is_valid = False

    if not fuav.is_2d_array(confusion_matrix):
        raise IncorrectShapeError('The confusion matrix has to be a '
                                  '2-dimensional numpy array.')
    if fuav.is_structured_array(confusion_matrix):
        raise ValueError('The confusion matrix cannot be a structured numpy '
                         'array.')
    if confusion_matrix.shape[0] != confusion_matrix.shape[1]:
        raise IncorrectShapeError('The confusion matrix has to be a square '
                                  '(equal width and height) numpy array.')
    if confusion_matrix.shape[1] < 2:
        raise IncorrectShapeError('The confusion matrix needs to be at least '
                                  '2x2.')
    if confusion_matrix.dtype.kind != 'i':
        raise TypeError('The confusion matrix has to be of integer kind.')

    if label_index is not None:
        if not isinstance(label_index, int):
            raise TypeError('The label index has to be an integer.')
        if label_index < 0 or label_index >= confusion_matrix.shape[0]:
            msg = ('The label index {} is not a valid index for the confusion '
                   'matrix of shape {}x{}.')
            msg = msg.format(label_index, confusion_matrix.shape[0],
                             confusion_matrix.shape[1])
            raise IndexError(msg)

    is_valid = True
    return is_valid


def _validate_confusion_matrix_size(confusion_matrix: np.ndarray,
                                    size: int) -> bool:
    """
    Validates the exact shape of the confusion matrix.

    Parameters
    ----------
    confusion_matrix : numpy.ndarray
        A confusion matrix to be validated.
    size : integer, optional (default=None)
        The expected width and height of the confusion matrix.

    Raises
    ------
    IncorrectShapeError
        If the confusion matrix is not of a given size (height and width).

    Returns
    -------
    is_valid : boolean
        ``True`` if the confusion matrix size is valid, ``False`` otherwise.
    """
    is_valid = False

    assert isinstance(size, int), 'Size has to be an integer.'
    assert size > 0, 'Size has to be a positive integer.'

    if confusion_matrix.shape[1] != size:
        msg = ('The confusion matrix is of shape {}x{} but {}x{} is the '
               'requirement.')
        msg = msg.format(confusion_matrix.shape[0], confusion_matrix.shape[1],
                         size, size)
        raise IncorrectShapeError(msg)

    is_valid = True
    return is_valid


def get_confusion_matrix(
        ground_truth: np.ndarray,
        predictions: np.ndarray,
        labels: Optional[List[Union[str, Number]]] = None) -> np.ndarray:
    """
    Computes a confusion matrix based on predictions and ground truth vectors.

    The confusion matrix (a.k.a. contingency table) has predictions in rows
    and ground truth in columns. If the value order is not provide via the
    ``labels`` parameter, the ordering is based on the alphanumeric sorting
    of the unique values in both of the input arrays.

    Parameters
    ----------
    ground_truth : numpy.ndarray
        An array holding the *true* target values.
    predictions : numpy.ndarray
        An array holding *predictions* of the target values.
    labels : List[string, number], optional (default=None)
        If a certain ordering of the labels in the confusion matrix is desired,
        it can be specified via this parameter. By default alphanumeric sorting
        is used.

    Warns
    -----
    UserWarning
        Some of the labels provided by the user are not present in either of
        the input arrays.

    Raises
    ------
    IncorrectShapeError
        The ``ground_truth`` and/or ``labels`` vectors are not 1-dimensional.
        The length of these two arrays does not agree.
    TypeError
        The ``labels`` parameter is not a list.
    ValueError
        The ``labels`` list empty, it contains duplicate entries or some of the
        labels present in either of the input array are not accounted for by
        the ``labels`` list.

    Returns
    -------
    confusion_matrix : numpy.ndarray
        A confusion matrix.
    """
    if not fuav.is_1d_array(ground_truth):
        raise IncorrectShapeError('The ground truth vector has to be '
                                  '1-dimensional numpy array.')
    if not fuav.is_1d_array(predictions):
        raise IncorrectShapeError('The predictions vector has to be '
                                  '1-dimensional numpy array.')
    if ground_truth.shape[0] != predictions.shape[0]:
        raise IncorrectShapeError('Both the ground truth and the predictions '
                                  'vectors have to have the same length.')

    all_values = np.concatenate([ground_truth, predictions])
    if labels is None:
        ordering = np.sort(np.unique(all_values)).tolist()
    elif isinstance(labels, list):
        if not labels:
            raise ValueError('The labels list cannot be empty.')
        labels_set = set(labels)
        if len(labels_set) != len(labels):
            raise ValueError('The labels list contains duplicates.')

        extra_labels = labels_set.difference(all_values)
        if extra_labels:
            warnings.warn(
                'Some of the given labels are not present in either of the '
                'input arrays: {}.'.format(extra_labels), UserWarning)

        unaccounted_labels = set(all_values).difference(labels_set)
        if unaccounted_labels:
            raise ValueError('The following labels are present in the input '
                             'arrays but were not given in the labels '
                             'parameter: {}.'.format(unaccounted_labels))

        ordering = labels
    else:
        raise TypeError('The labels parameter has to either a list or None.')

    confusion_matrix_list = []
    for pred in ordering:
        pdt = predictions == pred
        row = [np.logical_and(pdt, ground_truth == i).sum() for i in ordering]
        confusion_matrix_list.append(row)

    confusion_matrix = np.array(confusion_matrix_list)
    return confusion_matrix


def multiclass_true_positive_rate(confusion_matrix: np.ndarray,
                                  label_index: int) -> Number:
    """
    Calculates the "true positive rate" for a multi-class confusion matrix.

    See the documentation of
    :func:`fatf.utils.models.metrics.validate_confusion_matrix` for all the
    possible errors and exceptions.

    Parameters
    ----------
    confusion_matrix : numpy.ndarray
        A confusion matrix based on which the metric will be computed.
    label_index : integer
        The index of a label that should be treated as "positive".

    Returns
    -------
    metric : number
        The "true positive rate".
    """
    assert validate_confusion_matrix(confusion_matrix, label_index), \
        'The input parameters are invalid.'

    true_positive = confusion_matrix[label_index, label_index]
    condition_positive = confusion_matrix[:, label_index].sum()

    metric = true_positive / condition_positive if condition_positive else 0
    return metric


def multiclass_true_negative_rate(confusion_matrix: np.ndarray,
                                  label_index: int,
                                  strict: bool = False):
    """
    Calculates the "true negative rate" for a multi-class confusion matrix.

    There are two possible ways of calculating it:

    strict
        The true negatives are all non-positive ground truth predicted
        correctly.
    relaxed
        The true negatives are defined as all non-positive ground truth
        predicted as any non-positive.

    See the documentation of
    :func:`fatf.utils.models.metrics.validate_confusion_matrix` for all the
    possible errors and exceptions.

    Parameters
    ----------
    confusion_matrix : numpy.ndarray
        A confusion matrix based on which the metric will be computed.
    label_index : integer
        The index of a label that should be treated as "positive". All the
        other labels will be treated as "negative".
    strict : boolean, optional (default=False)
        If ``True``, the "true negatives" are calculated "strictly", otherwise
        a generalised approach to "true negatives" is used.

    Raises
    ------
    TypeError
        The ``strict`` parameter is not a boolean.

    Returns
    -------
    metric : number
        The "true negative rate".
    """
    assert validate_confusion_matrix(confusion_matrix, label_index), \
        'The input parameters are invalid.'
    if not isinstance(strict, bool):
        raise TypeError('The strict parameter has to be a boolean.')

    if strict:
        true_negative = (np.diagonal(confusion_matrix).sum() -
                         confusion_matrix[label_index, label_index])
    else:
        true_negative = (
            confusion_matrix.sum() - confusion_matrix[:, label_index].sum() -
            confusion_matrix[label_index, :].sum() +
            confusion_matrix[label_index, label_index])

    condition_negative = (
        confusion_matrix.sum() - confusion_matrix[:, label_index].sum())

    metric = true_negative / condition_negative if condition_negative else 0
    return metric


def multiclass_false_positive_rate(confusion_matrix: np.ndarray,
                                   label_index: int) -> Number:
    """
    Calculates the "false positive rate" for a multi-class confusion matrix.

    See the documentation of
    :func:`fatf.utils.models.metrics.validate_confusion_matrix` for all the
    possible errors and exceptions.

    Parameters
    ----------
    confusion_matrix : numpy.ndarray
        A confusion matrix based on which the metric will be computed.
    label_index : integer
        The index of a label that should be treated as "positive". All the
        other labels will be treated as "negative".

    Returns
    -------
    metric : number
        The "false positive rate".
    """
    assert validate_confusion_matrix(confusion_matrix, label_index), \
        'The input parameters are invalid.'

    false_positive = (confusion_matrix[label_index, :].sum() -
                      confusion_matrix[label_index, label_index])
    condition_negative = (
        confusion_matrix.sum() - confusion_matrix[:, label_index].sum())

    metric = false_positive / condition_negative if condition_negative else 0
    return metric


def multiclass_false_negative_rate(confusion_matrix: np.ndarray,
                                   label_index: int) -> Number:
    """
    Calculates the "false negative rate" for a multi-class confusion matrix.

    See the documentation of
    :func:`fatf.utils.models.metrics.validate_confusion_matrix` for all the
    possible errors and exceptions.

    Parameters
    ----------
    confusion_matrix : numpy.ndarray
        A confusion matrix based on which the metric will be computed.
    label_index : integer
        The index of a label that should be treated as "positive". All the
        other labels will be treated as "negative".

    Returns
    -------
    metric : number
        The "false negative rate".
    """
    assert validate_confusion_matrix(confusion_matrix, label_index), \
        'The input parameters are invalid.'

    condition_positive = confusion_matrix[:, label_index].sum()
    false_negative = (
        condition_positive - confusion_matrix[label_index, label_index])

    metric = false_negative / condition_positive if condition_positive else 0
    return metric


def true_positive_rate(confusion_matrix: np.ndarray) -> Number:
    """
    Calculates the true positive rate for a binary confusion matrix.

    See the documentation of
    :func:`fatf.utils.models.metrics.validate_confusion_matrix` and
    :func:`fatf.utils.models.metrics._validate_confusion_matrix_size` for all
    the possible errors and exceptions.

    Parameters
    ----------
    confusion_matrix : numpy.ndarray
        A confusion matrix based on which the metric will be computed.

    Returns
    -------
    metric : number
        The true positive rate.
    """
    metric = multiclass_true_positive_rate(confusion_matrix, 0)
    assert _validate_confusion_matrix_size(confusion_matrix, 2), \
        'The confusion matrix has to be 2x2.'

    return metric


def true_negative_rate(confusion_matrix: np.ndarray) -> Number:
    """
    Calculates the true negative rate for a binary confusion matrix.

    See the documentation of
    :func:`fatf.utils.models.metrics.validate_confusion_matrix` and
    :func:`fatf.utils.models.metrics._validate_confusion_matrix_size` for all
    the possible errors and exceptions.

    Parameters
    ----------
    confusion_matrix : numpy.ndarray
        A confusion matrix based on which the metric will be computed.

    Returns
    -------
    metric : number
        The true negative rate.
    """
    metric = multiclass_true_negative_rate(confusion_matrix, 0)
    assert _validate_confusion_matrix_size(confusion_matrix, 2), \
        'The confusion matrix has to be 2x2.'

    return metric


def false_positive_rate(confusion_matrix: np.ndarray) -> Number:
    """
    Calculates the false positive rate for a binary confusion matrix.

    See the documentation of
    :func:`fatf.utils.models.metrics.validate_confusion_matrix` and
    :func:`fatf.utils.models.metrics._validate_confusion_matrix_size` for all
    the possible errors and exceptions.

    Parameters
    ----------
    confusion_matrix : numpy.ndarray
        A confusion matrix based on which the metric will be computed.

    Returns
    -------
    metric : number
        The false positive rate.
    """
    metric = multiclass_false_positive_rate(confusion_matrix, 0)
    assert _validate_confusion_matrix_size(confusion_matrix, 2), \
        'The confusion matrix has to be 2x2.'

    return metric


def false_negative_rate(confusion_matrix: np.ndarray) -> Number:
    """
    Calculates the false negative rate for a binary confusion matrix.

    See the documentation of
    :func:`fatf.utils.models.metrics.validate_confusion_matrix` and
    :func:`fatf.utils.models.metrics._validate_confusion_matrix_size` for all
    the possible errors and exceptions.

    Parameters
    ----------
    confusion_matrix : numpy.ndarray
        A confusion matrix based on which the metric will be computed.

    Returns
    -------
    metric : number
        The false negative rate.
    """
    metric = multiclass_false_negative_rate(confusion_matrix, 0)
    assert _validate_confusion_matrix_size(confusion_matrix, 2), \
        'The confusion matrix has to be 2x2.'

    return metric


def multiclass_positive_predictive_value(confusion_matrix: np.ndarray,
                                         label_index: int) -> Number:
    """
    Gets the "positive predictive value" for a multi-class confusion matrix.

    The positive predictive value is also known as *precision*.

    See the documentation of
    :func:`fatf.utils.models.metrics.validate_confusion_matrix` for all the
    possible errors and exceptions.

    Parameters
    ----------
    confusion_matrix : numpy.ndarray
        A confusion matrix based on which the metric will be computed.
    label_index : integer
        The index of a label that should be treated as "positive". All the
        other labels will be treated as "negative".

    Returns
    -------
    metric : number
        The "positive predictive value".
    """
    assert validate_confusion_matrix(confusion_matrix, label_index), \
        'The input parameters are invalid.'

    true_positive = confusion_matrix[label_index, label_index]
    predicted_condition_positive = confusion_matrix[label_index, :].sum()

    metric = (true_positive / predicted_condition_positive
              if predicted_condition_positive else 0)
    return metric


def multiclass_negative_predictive_value(confusion_matrix: np.ndarray,
                                         label_index: int,
                                         strict: bool = False) -> Number:
    """
    Gets the "negative predictive value" for a multi-class confusion matrix.

    There are two possible ways of calculating it:

    strict
        The true negatives are all non-positive ground truth predicted
        correctly.
    relaxed
        The true negatives are defined as all non-positive ground truth
        predicted as any non-positive.

    See the documentation of
    :func:`fatf.utils.models.metrics.validate_confusion_matrix` for all the
    possible errors and exceptions.

    Parameters
    ----------
    confusion_matrix : numpy.ndarray
        A confusion matrix based on which the metric will be computed.
    label_index : integer
        The index of a label that should be treated as "positive". All the
        other labels will be treated as "negative".
    strict : boolean, optional (default=False)
        If ``True``, the "true negatives" are calculated "strictly", otherwise
        a generalised approach to "true negatives" is used.

    Raises
    ------
    TypeError
        The ``strict`` parameter is not a boolean.

    Returns
    -------
    metric : number
        The "negative predictive value".
    """
    assert validate_confusion_matrix(confusion_matrix, label_index), \
        'The input parameters are invalid.'

    if not isinstance(strict, bool):
        raise TypeError('The strict parameter has to be a boolean.')

    if strict:
        true_negative = (np.diagonal(confusion_matrix).sum() -
                         confusion_matrix[label_index, label_index])
    else:
        true_negative = (
            confusion_matrix.sum() - confusion_matrix[:, label_index].sum() -
            confusion_matrix[label_index, :].sum() +
            confusion_matrix[label_index, label_index])

    predicted_condition_negative = (
        confusion_matrix.sum() - confusion_matrix[label_index, :].sum())

    metric = (true_negative / predicted_condition_negative
              if predicted_condition_negative else 0)
    return metric


def positive_predictive_value(confusion_matrix: np.ndarray) -> Number:
    """
    Calculates the positive predictive value for a binary confusion matrix.

    See the documentation of
    :func:`fatf.utils.models.metrics.validate_confusion_matrix` and
    :func:`fatf.utils.models.metrics._validate_confusion_matrix_size` for all
    the possible errors and exceptions.

    Parameters
    ----------
    confusion_matrix : numpy.ndarray
        A confusion matrix based on which the metric will be computed.

    Returns
    -------
    metric : number
        The positive predictive value.
    """
    metric = multiclass_positive_predictive_value(confusion_matrix, 0)
    assert _validate_confusion_matrix_size(confusion_matrix, 2), \
        'The confusion matrix has to be 2x2.'

    return metric


def negative_predictive_value(confusion_matrix: np.ndarray) -> Number:
    """
    Calculates the negative predictive value for a binary confusion matrix.

    See the documentation of
    :func:`fatf.utils.models.metrics.validate_confusion_matrix` and
    :func:`fatf.utils.models.metrics._validate_confusion_matrix_size` for all
    the possible errors and exceptions.

    Parameters
    ----------
    confusion_matrix : numpy.ndarray
        A confusion matrix based on which the metric will be computed.

    Returns
    -------
    metric : number
        The negative predictive value.
    """
    metric = multiclass_negative_predictive_value(confusion_matrix, 0)
    assert _validate_confusion_matrix_size(confusion_matrix, 2), \
        'The confusion matrix has to be 2x2.'

    return metric


def accuracy(confusion_matrix: np.ndarray) -> Number:
    """
    Computes the accuracy for an arbitrary confusion matrix.

    See the documentation of
    :func:`fatf.utils.models.metrics.validate_confusion_matrix` for all the
    possible errors and exceptions.

    Parameters
    ----------
    confusion_matrix : numpy.ndarray
        A confusion matrix based on which the metric will be computed.

    Returns
    -------
    metric : number
        The accuracy.
    """
    assert validate_confusion_matrix(confusion_matrix), \
        'The input parameters are invalid.'

    tp_tn = np.diagonal(confusion_matrix).sum()
    total = confusion_matrix.sum()
    metric = tp_tn / total if total else 0
    return metric


def multiclass_treatment(confusion_matrix: np.ndarray,
                         label_index: int) -> Number:
    """
    Computes the "treatment" metric for a multi-class confusion matrix.

    A "treatment" is the proportion of all the predictions of a selected class
    that are incorrect to all incorrectly predicted instances.

    See the documentation of
    :func:`fatf.utils.models.metrics.validate_confusion_matrix` for all the
    possible errors and exceptions.

    Parameters
    ----------
    confusion_matrix : numpy.ndarray
        A confusion matrix based on which the metric will be computed.
    label_index : integer
        The index of a label that should be treated as "positive". All the
        other labels will be treated as "negative".

    Returns
    -------
    metric : number
        The "treatment" measurement.
    """
    assert validate_confusion_matrix(confusion_matrix, label_index), \
        'The input parameters are invalid.'

    row_incorrect = (confusion_matrix[label_index, :].sum() -
                     confusion_matrix[label_index, label_index])
    all_incorrect = (
        confusion_matrix.sum() - np.diagonal(confusion_matrix).sum())

    metric = row_incorrect / all_incorrect if all_incorrect else 0
    return metric


def treatment(confusion_matrix: np.ndarray) -> Number:
    """
    Computes the "treatment" metric for a binary confusion matrix.

    A "treatment" is the proportion of all the false positives to the sum of
    false positives and false negatives.

    See the documentation of
    :func:`fatf.utils.models.metrics.validate_confusion_matrix` and
    :func:`fatf.utils.models.metrics._validate_confusion_matrix_size` for all
    the possible errors and exceptions.

    Parameters
    ----------
    confusion_matrix : numpy.ndarray
        A confusion matrix based on which the metric will be computed.

    Returns
    -------
    metric : number
        The "treatment" measurement.
    """
    metric = multiclass_treatment(confusion_matrix, 0)
    assert _validate_confusion_matrix_size(confusion_matrix, 2), \
        'The confusion matrix has to be 2x2.'

    return metric
