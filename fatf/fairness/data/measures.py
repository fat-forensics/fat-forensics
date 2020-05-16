"""
The :mod:`fatf.fairness.data.measures` module holds data fairness measures.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
#         Rafael Poyiadzi <rp13102@bristol.ac.uk>
# License: new BSD

from typing import List, Union

import numpy as np
import numpy.lib.recfunctions as recfn

from fatf.exceptions import IncorrectShapeError

import fatf.utils.array.tools as fuat
import fatf.utils.array.validation as fuav
import fatf.utils.data.tools as fudt

__all__ = ['systemic_bias', 'systemic_bias_check']

Index = Union[int, str]  # A column index type


def systemic_bias(dataset: np.ndarray, ground_truth: np.ndarray,
                  protected_features: List[Index]) -> np.ndarray:
    """
    Checks for systemic bias in a dataset.

    This function checks whether there exist data points that share the same
    unprotected features but differ in protected features. For all of these
    instances their label (ground truth) will be checked and if it is
    different, a particular data points pair will be indicated to be biased.
    This dependency is represented as a boolean, square numpy array that shows
    whether systemic bias exists (``True``) for any pair of data points.

    Parameters
    ----------
    dataset : numpy.ndarray
        A dataset to be evaluated for systemic bias.
    ground_truth : numpy.ndarray
        The labels corresponding to the dataset.
    protected_features : List[column index]
        A list of column indices in the dataset that hold protected attributes.

    Raises
    ------
    IncorrectShapeError
        The dataset is not a 2-dimensional numpy array, the ground truth is not
        a 1-dimensional numpy array or the number of rows in the dataset is not
        equal to the number of elements in the ground truth array.
    IndexError
        Some of the column indices given in the ``protected_features`` list are
        not valid for the input dataset.
    TypeError
        The ``protected_features`` parameter is not a list.
    ValueError
        There are duplicate values in the protected feature indices list.

    Returns
    -------
    systemic_bias_matrix : numpy.ndarray
        A square, diagonally symmetrical and boolean numpy array that indicates
        which pair of data point share the same unprotected features but differ
        in protected features and the ground truth annotation.
    """
    # pylint: disable=too-many-branches
    if not fuav.is_2d_array(dataset):
        raise IncorrectShapeError('The dataset should be a 2-dimensional '
                                  'numpy array.')
    if not fuav.is_1d_array(ground_truth):
        raise IncorrectShapeError('The ground truth should be a 1-dimensional '
                                  'numpy array.')
    if ground_truth.shape[0] != dataset.shape[0]:
        raise IncorrectShapeError('The number of rows in the dataset and the '
                                  'ground truth should be equal.')
    if isinstance(protected_features, list):
        pfa = np.asarray(protected_features)
        if not fuat.are_indices_valid(dataset, pfa):
            iid = np.sort(fuat.get_invalid_indices(dataset, pfa)).tolist()
            raise IndexError('The following protected feature indices are not '
                             'valid for the dataset array: {}.'.format(iid))
        if len(set(protected_features)) != len(protected_features):
            raise ValueError('Some of the protected indices are duplicated.')
    else:
        raise TypeError('The protected_features parameter should be a list.')

    is_structured = fuav.is_structured_array(dataset)

    if is_structured:
        unprotected_features_array = recfn.drop_fields(dataset,
                                                       protected_features)
        # Needed for numpy<1.18
        if unprotected_features_array is None:
            unprotected_features_array = np.ones(  # pragma: nocover
                (dataset.shape[0], ),
                dtype=[('ones', int)])
    else:
        unprotected_features_array = np.delete(
            dataset, protected_features, axis=1)
        if not unprotected_features_array.size:
            unprotected_features_array = np.ones((dataset.shape[0], 1))

    assert unprotected_features_array.shape[0] == dataset.shape[0], \
        'Must share rows number.'

    systemic_bias_columns = []
    for i in range(unprotected_features_array.shape[0]):
        if is_structured:
            equal_unprotected = (
                unprotected_features_array == unprotected_features_array[i])
        else:
            equal_unprotected = np.apply_along_axis(
                np.array_equal, 1, unprotected_features_array,
                unprotected_features_array[i, :])

        equal_unprotected_indices = np.where(equal_unprotected)

        # Check whether the ground truth is different for these rows
        equal_unprotected[equal_unprotected_indices] = (
            ground_truth[i] != ground_truth[equal_unprotected_indices])
        systemic_bias_columns.append(equal_unprotected)

    systemic_bias_matrix = np.stack(systemic_bias_columns, axis=1)
    assert np.array_equal(systemic_bias_matrix, systemic_bias_matrix.T), \
        'The matrix has to be diagonally symmetric.'
    assert not np.diagonal(systemic_bias_matrix).any(), \
        'Same elements cannot be systemically biased.'
    return systemic_bias_matrix


def systemic_bias_check(systemic_bias_matrix: np.ndarray) -> bool:
    """
    Indicates whether a dataset has a systemic bias.

    Parameters
    ----------
    systemic_bias_matrix : numpy.ndarray
        A square (equal number of rows and columns) boolean numpy array that
        indicates which pair of data points share the same unprotected features
        but differ in protected features and ground truth annotation. (The
        number of rows/columns should be equal to the number of data points in
        the original data set.)

    Raises
    ------
    IncorrectShapeError
        The systemic bias matrix is not 2-dimensional or square.
    TypeError
        The systemic bias matrix is not of boolean type.
    ValueError
        The systemic bias matrix is a structured numpy array or is not
        diagonally symmetric.

    Returns
    -------
    systemic_bias_present : boolean
        ``True`` if systemic bias is present, ``False`` otherwise.
    """
    assert fudt.validate_binary_matrix(systemic_bias_matrix,
                                       'systemic bias'), 'Invalid matrix.'
    systemic_bias_present = systemic_bias_matrix.any()
    return systemic_bias_present
