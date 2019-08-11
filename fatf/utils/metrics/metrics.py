"""
The :mod:`fatf.utils.metrics.metrics` module holds common performance metrics.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import numpy as np

import fatf.utils.metrics.tools as fumt

__all__ = ['multiclass_true_positive_rate',
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


def multiclass_true_positive_rate(confusion_matrix: np.ndarray,
                                  label_index: int) -> float:
    """
    Calculates the "true positive rate" for a multi-class confusion matrix.

    See the documentation of
    :func:`fatf.utils.metrics.tools.validate_confusion_matrix` for all the
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
    assert fumt.validate_confusion_matrix(confusion_matrix, label_index), \
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
    :func:`fatf.utils.metrics.tools.validate_confusion_matrix` for all the
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
    assert fumt.validate_confusion_matrix(confusion_matrix, label_index), \
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
                                   label_index: int) -> float:
    """
    Calculates the "false positive rate" for a multi-class confusion matrix.

    See the documentation of
    :func:`fatf.utils.metrics.tools.validate_confusion_matrix` for all the
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
    assert fumt.validate_confusion_matrix(confusion_matrix, label_index), \
        'The input parameters are invalid.'

    false_positive = (confusion_matrix[label_index, :].sum() -
                      confusion_matrix[label_index, label_index])
    condition_negative = (
        confusion_matrix.sum() - confusion_matrix[:, label_index].sum())

    metric = false_positive / condition_negative if condition_negative else 0
    return metric


def multiclass_false_negative_rate(confusion_matrix: np.ndarray,
                                   label_index: int) -> float:
    """
    Calculates the "false negative rate" for a multi-class confusion matrix.

    See the documentation of
    :func:`fatf.utils.metrics.tools.validate_confusion_matrix` for all the
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
    assert fumt.validate_confusion_matrix(confusion_matrix, label_index), \
        'The input parameters are invalid.'

    condition_positive = confusion_matrix[:, label_index].sum()
    false_negative = (
        condition_positive - confusion_matrix[label_index, label_index])

    metric = false_negative / condition_positive if condition_positive else 0
    return metric


def true_positive_rate(confusion_matrix: np.ndarray) -> float:
    """
    Calculates the true positive rate for a binary confusion matrix.

    See the documentation of
    :func:`fatf.utils.metrics.tools.validate_confusion_matrix` and
    :func:`fatf.utils.metrics.tools.validate_confusion_matrix_size` for all the
    possible errors and exceptions.

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
    assert fumt.validate_confusion_matrix_size(confusion_matrix, 2), \
        'The confusion matrix has to be 2x2.'

    return metric


def true_negative_rate(confusion_matrix: np.ndarray) -> float:
    """
    Calculates the true negative rate for a binary confusion matrix.

    See the documentation of
    :func:`fatf.utils.metrics.tools.validate_confusion_matrix` and
    :func:`fatf.utils.metrics.tools.validate_confusion_matrix_size` for all the
    possible errors and exceptions.

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
    assert fumt.validate_confusion_matrix_size(confusion_matrix, 2), \
        'The confusion matrix has to be 2x2.'

    return metric


def false_positive_rate(confusion_matrix: np.ndarray) -> float:
    """
    Calculates the false positive rate for a binary confusion matrix.

    See the documentation of
    :func:`fatf.utils.metrics.tools.validate_confusion_matrix` and
    :func:`fatf.utils.metrics.tools.validate_confusion_matrix_size` for all the
    possible errors and exceptions.

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
    assert fumt.validate_confusion_matrix_size(confusion_matrix, 2), \
        'The confusion matrix has to be 2x2.'

    return metric


def false_negative_rate(confusion_matrix: np.ndarray) -> float:
    """
    Calculates the false negative rate for a binary confusion matrix.

    See the documentation of
    :func:`fatf.utils.metrics.tools.validate_confusion_matrix` and
    :func:`fatf.utils.metrics.tools.validate_confusion_matrix_size` for all the
    possible errors and exceptions.

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
    assert fumt.validate_confusion_matrix_size(confusion_matrix, 2), \
        'The confusion matrix has to be 2x2.'

    return metric


def multiclass_positive_predictive_value(confusion_matrix: np.ndarray,
                                         label_index: int) -> float:
    """
    Gets the "positive predictive value" for a multi-class confusion matrix.

    The positive predictive value is also known as *precision*.

    See the documentation of
    :func:`fatf.utils.metrics.tools.validate_confusion_matrix` for all the
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
    assert fumt.validate_confusion_matrix(confusion_matrix, label_index), \
        'The input parameters are invalid.'

    true_positive = confusion_matrix[label_index, label_index]
    predicted_condition_positive = confusion_matrix[label_index, :].sum()

    metric = (true_positive / predicted_condition_positive
              if predicted_condition_positive else 0)
    return metric


def multiclass_negative_predictive_value(confusion_matrix: np.ndarray,
                                         label_index: int,
                                         strict: bool = False) -> float:
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
    :func:`fatf.utils.metrics.tools.validate_confusion_matrix` for all the
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
    assert fumt.validate_confusion_matrix(confusion_matrix, label_index), \
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


def positive_predictive_value(confusion_matrix: np.ndarray) -> float:
    """
    Calculates the positive predictive value for a binary confusion matrix.

    See the documentation of
    :func:`fatf.utils.metrics.tools.validate_confusion_matrix` and
    :func:`fatf.utils.metrics.tools.validate_confusion_matrix_size` for all the
    possible errors and exceptions.

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
    assert fumt.validate_confusion_matrix_size(confusion_matrix, 2), \
        'The confusion matrix has to be 2x2.'

    return metric


def negative_predictive_value(confusion_matrix: np.ndarray) -> float:
    """
    Calculates the negative predictive value for a binary confusion matrix.

    See the documentation of
    :func:`fatf.utils.metrics.tools.validate_confusion_matrix` and
    :func:`fatf.utils.metrics.tools.validate_confusion_matrix_size` for all the
    possible errors and exceptions.

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
    assert fumt.validate_confusion_matrix_size(confusion_matrix, 2), \
        'The confusion matrix has to be 2x2.'

    return metric


def accuracy(confusion_matrix: np.ndarray) -> float:
    """
    Computes the accuracy for an arbitrary confusion matrix.

    See the documentation of
    :func:`fatf.utils.metrics.tools.validate_confusion_matrix` for all the
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
    assert fumt.validate_confusion_matrix(confusion_matrix), \
        'The input parameters are invalid.'

    tp_tn = np.diagonal(confusion_matrix).sum()
    total = confusion_matrix.sum()
    metric = tp_tn / total if total else 0
    return metric


def multiclass_treatment(confusion_matrix: np.ndarray,
                         label_index: int) -> float:
    """
    Computes the "treatment" metric for a multi-class confusion matrix.

    A "treatment" is the proportion of all the predictions of a selected class
    that are incorrect to all incorrectly predicted instances.

    See the documentation of
    :func:`fatf.utils.metrics.tools.validate_confusion_matrix` for all the
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
    assert fumt.validate_confusion_matrix(confusion_matrix, label_index), \
        'The input parameters are invalid.'

    row_incorrect = (confusion_matrix[label_index, :].sum() -
                     confusion_matrix[label_index, label_index])
    all_incorrect = (
        confusion_matrix.sum() - np.diagonal(confusion_matrix).sum())

    metric = row_incorrect / all_incorrect if all_incorrect else 0
    return metric


def treatment(confusion_matrix: np.ndarray) -> float:
    """
    Computes the "treatment" metric for a binary confusion matrix.

    A "treatment" is the proportion of all the false positives to the sum of
    false positives and false negatives.

    See the documentation of
    :func:`fatf.utils.metrics.tools.validate_confusion_matrix` and
    :func:`fatf.utils.metrics.tools.validate_confusion_matrix_size` for all the
    possible errors and exceptions.

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
    assert fumt.validate_confusion_matrix_size(confusion_matrix, 2), \
        'The confusion matrix has to be 2x2.'

    return metric
