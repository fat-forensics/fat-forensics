"""
The :mod:`fatf.utils.metrics.tools` module holds tools for performance metrics.

These tools -- mainly confusion matrix computation and manipulation -- are
useful for constructing new performance metrics.
Examples of how they are used can be found in :mod:`fatf.utils.metrics.metrics`
module.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import warnings

from typing import List, Optional, Tuple, Union

import numpy as np

from fatf.exceptions import IncorrectShapeError

import fatf.utils.array.validation as fuav
import fatf.utils.data.tools as fudt

__all__ = ['get_confusion_matrix',
           'confusion_matrix_per_subgroup',
           'confusion_matrix_per_subgroup_indexed',
           'validate_confusion_matrix_size',
           'validate_confusion_matrix']  # yapf: disable

Index = Union[int, str]  # A column index type


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


def validate_confusion_matrix_size(confusion_matrix: np.ndarray,
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
        labels: Optional[List[Union[str, float]]] = None) -> np.ndarray:
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


def confusion_matrix_per_subgroup(
        dataset: np.ndarray,
        #
        ground_truth: np.ndarray,
        predictions: np.ndarray,
        #
        column_index: Index,
        groupings: Optional[List[Union[float, Tuple[str]]]] = None,
        numerical_bins_number: int = 5,
        treat_as_categorical: Optional[bool] = None,
        #
        labels: Optional[List[Union[str, float]]] = None
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Computes confusion matrices for every defined sub-population.

    This is useful for computing a variety of performance metrics for each
    sub-population.

    For warnings raised by this method please see the documentation of
    :func:`fatf.utils.data.tools.validate_indices_per_bin` function.

    Parameters
    ----------
    dataset, column_index, groupings, numerical_bins_number, \
and treat_as_categorical
        These parameters are described in the documentation of
        :func:`fatf.utils.data.tools.group_by_column` function and are used to
        define a grouping (i.e. sub-populations). If you have your own
        index-based grouping and would like to get sub-population-based
        confusion matrices, please consider using
        :func:`fatf.utils.metrics.tools.confusion_matrix_per_subgroup_indexed`
        function.
    ground_truth, predictions, and labels
        These parameters are described in the documentation of
        :func:`fatf.utils.metrics.tools.get_confusion_matrix` function and are
        used to calculate confusion matrices.

    Returns
    -------
    population_confusion_matrix : List[numpy.ndarray]
        A list of confusion matrices for each sub-population.
    bin_names : List[strings]
        The name of every sub-population (binning results) defined by the
        feature ranges for a numerical feature and feature value sets for a
        categorical feature.
    """
    # pylint: disable=too-many-arguments
    indices_per_bin, bin_names = fudt.group_by_column(
        dataset, column_index, groupings, numerical_bins_number,
        treat_as_categorical)

    assert fudt.validate_indices_per_bin(indices_per_bin), \
        'Binned indices list is invalid.'

    population_confusion_matrix = confusion_matrix_per_subgroup_indexed(
        indices_per_bin, ground_truth, predictions, labels)
    return population_confusion_matrix, bin_names


def confusion_matrix_per_subgroup_indexed(
        indices_per_bin: List[np.ndarray],
        ground_truth: np.ndarray,
        predictions: np.ndarray,
        labels: Optional[List[Union[str, float]]] = None) -> List[np.ndarray]:
    """
    Computes confusion matrices for every defined sub-population.

    This is useful for computing a variety of performance metrics based on
    predefined instance index binning for each sub-population.

    This is an alternative to
    :func:`fatf.utils.metrics.tools.confusion_matrix_per_subgroup` function,
    which can be used when one already has the desired instance binning.

    For warnings and errors raised by this method please see the documentation
    of :func:`fatf.utils.data.tools.validate_indices_per_bin` function.

    Parameters
    ----------
    indices_per_bin : List[List[integer]]
        A list of lists with the latter one holding row indices of a particular
        group (sub-population).
    ground_truth, predictions, and labels
        These parameters are described in the documentation of
        :func:`fatf.utils.metrics.tools.get_confusion_matrix` function and are
        used to calculate confusion matrices.

    Returns
    -------
    population_confusion_matrix : List[numpy.ndarray]
        A list of confusion matrices for each sub-population.
    """
    assert fudt.validate_indices_per_bin(indices_per_bin), \
        'Binned indices list is invalid.'

    if labels is None:
        if not fuav.is_1d_array(ground_truth):
            raise IncorrectShapeError('The ground_truth parameter should be a '
                                      '1-dimensional numpy array.')
        if not fuav.is_1d_array(predictions):
            raise IncorrectShapeError('The predictions parameter should be a '
                                      '1-dimensional numpy array.')
        labels = np.sort(
            np.unique(np.concatenate([ground_truth, predictions]))).tolist()

    population_confusion_matrix = []
    for bin_indices in indices_per_bin:
        confusion_matrix = get_confusion_matrix(
            ground_truth[bin_indices], predictions[bin_indices], labels)
        population_confusion_matrix.append(confusion_matrix)
    return population_confusion_matrix
