"""
Implementing different methods for choosing features.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
# License: new BSD

from typing import List, Union

import numpy as np

from sklearn.linear_model import lars_path

import fatf.utils.array.validation as fuav
import fatf.utils.array.tools as fuat
from fatf.exceptions import IncorrectShapeError

__all__ = ['lasso_path']

Index = Union[int, str]


def _is_input_valid(dataset: np.ndarray,
                    target: np.ndarray,
                    weights: np.ndarray,
                    num_features: int) -> bool:
    """
    Validates the input parameters of lasso path function..

    For the input parameter description, warnings and exceptions please see
    the documentation of the :func`fatf.transparency.models.blimey.lasso_path`
    function.

    Returns
    -------
    is_input_ok : boolean
        ``True`` if input is valid, ``False`` otherwise.
    """
    is_input_ok = False

    if not fuav.is_2d_array(dataset):
        raise IncorrectShapeError('The input dataset must be a 2-dimensional '
                                  'array.')

    if not fuav.is_numerical_array(dataset):
        raise TypeError('The input dataset must only contain numerical '
                        'dtypes')

    if not fuav.is_1d_array(target):
        raise IncorrectShapeError('The input target array must a '
                                  '1-dimensional array.')

    if target.shape[0] != dataset.shape[0]:
        raise IncorrectShapeError('The number of labels in target must be the '
                                  'same as the number of samples in dataset.')

    if not fuav.is_1d_array(weights):
        raise IncorrectShapeError('The input weights array must a '
                                  '1-dimensional array.')

    if weights.shape[0] != dataset.shape[0]:
        raise IncorrectShapeError('The number distances in weights must be '
                                  'the same as the number of samples in '
                                  'dataset.')

    if not isinstance(num_features, int):
        raise TypeError('num_features must be an integer.')

    if num_features < 1:
        raise ValueError('num_features must be an integer greater than zero.')

    is_input_ok = True

    return is_input_ok


def lasso_path(dataset: np.ndarray,
               target: np.ndarray,
               weights: np.ndarray,
               num_features: int) -> List[Index]:
    """
    Computes the features to be used in training a local model using LASSO.

    Parameters
    ----------
    dataset : numpy.ndarray
        A 2-dimensional numpy array with the dataset.
    target : numpy.ndarray
        The class/regression assignment of each row in dataset.
    weights : numpy.ndarray
        An array of weights to assign to values in dataset when calculating
        lasso path.
    num_features : integer
        Number of features to return.

    Raises
    ------
    IncorrectShapeError
        `dataset
    Returns
    -------
    features : List[Index]
        List of indices of the features selecte by lasso path.
    """
    assert _is_input_valid(dataset, target, weights,
                           num_features), 'Input is invalid.'

    if fuav.is_structured_array(dataset):
        indices = np.array(dataset.dtype.names)
        dataset = fuat.as_unstructured(dataset)
    else:
        indices = np.arange(0, dataset.shape[1], 1)

    weighted_data = (dataset - np.average(dataset, axis=0, weights=weights))
    weighted_data = weighted_data * np.sqrt(weights[:, np.newaxis])
    weighted_target = target - np.average(target, weights=weights)
    weighted_target = weighted_target * np.sqrt(weights)

    alphas, _, coefs = lars_path(
        weighted_data, weighted_target, method='lasso', verbose=False)
    for i in range(coefs.shape[1] - 1, 0, -1):
        nonzero = coefs[:, i].nonzero()[0]
        if len(nonzero) <= num_features:
            break

    features = indices[nonzero]

    return features
