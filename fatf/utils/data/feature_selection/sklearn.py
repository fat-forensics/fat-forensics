"""
.. versionadded:: 0.0.2

The :mod:`fatf.utils.data.feature_selection.sklearn` module implements
scikit-learn-based feature selection approaches.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
#         Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

from typing import List, Union, Optional

import logging
import warnings

try:
    import sklearn.linear_model
except ImportError:
    _warning_msg = (  # pylint: disable=invalid-name
        'scikit-learn (sklearn) Python module is not installed on your '
        'system. You must install it in order to use '
        'fatf.utils.data.feature_selection.sklearn functionality. '
        'One possibility is to install scikit-learn alongside this package '
        'via machine learning dependencies with: pip install fatf[ml].')
    warnings.warn(_warning_msg, ImportWarning)

import numpy as np

import fatf.utils.array.tools as fuat
import fatf.utils.array.validation as fuav

from fatf.exceptions import IncorrectShapeError

__all__ = ['lasso_path']

Index = Union[int, str]

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def _validate_input_lasso_path(dataset: np.ndarray, target: np.ndarray,
                               weights: Union[np.ndarray, None],
                               features_number: Union[int, None],
                               features_percentage: int) -> bool:
    """
    Validates the input parameters of the ``lasso_path`` function.

    For the input parameter description, warnings and exceptions please see
    the documentation of the
    :func:`fatf.utils.data.feature_selection.sklearn.lasso_path` function.

    Returns
    -------
    input_is_valid : boolean
        ``True`` if the input is valid, ``False`` otherwise.
    """
    # pylint: disable=too-many-branches
    input_is_valid = False

    if not fuav.is_2d_array(dataset):
        raise IncorrectShapeError('The input data set must be a 2-dimensional '
                                  'array.')
    if not fuav.is_numerical_array(dataset):
        raise TypeError('The input data set must be purely numerical. (The '
                        'lasso path feature selection is based on '
                        'sklearn.linear_model.lars_path function.)')

    if not fuav.is_1d_array(target):
        raise IncorrectShapeError('The target array must be a 1-dimensional '
                                  'array.')
    if not fuav.is_numerical_array(target):
        raise TypeError('The target array must be numerical since this '
                        'feature selection method is based on Lasso '
                        'regression.')
    if target.shape[0] != dataset.shape[0]:
        raise IncorrectShapeError('The number of labels in the target array '
                                  'must agree with the number of samples in '
                                  'the data set.')

    if weights is not None:
        if not fuav.is_1d_array(weights):
            raise IncorrectShapeError('The weights array must 1-dimensional.')
        if not fuav.is_numerical_array(weights):
            raise TypeError('The weights array must be purely numerical.')
        if weights.shape[0] != dataset.shape[0]:
            raise IncorrectShapeError('The number of weights in the weights '
                                      'array must be the same as the number '
                                      'of samples in the input data set.')

    if features_number is not None:
        if not isinstance(features_number, int):
            raise TypeError('The features_number parameter must be an '
                            'integer.')
        if features_number < 1:
            raise ValueError('The features_number parameter must be a '
                             'positive integer.')

    if not isinstance(features_percentage, int):
        raise TypeError('The feature_percentage parameter must be an integer.')
    if features_percentage < 0 or features_percentage > 100:
        raise ValueError('The feature_percentage parameter must be between 0 '
                         'and 100 (inclusive).')

    input_is_valid = True
    return input_is_valid


def lasso_path(dataset: np.ndarray,
               target: np.ndarray,
               weights: Optional[np.ndarray] = None,
               features_number: Optional[int] = None,
               features_percentage: int = 100) -> List[Index]:
    """
    Selects the specified number of features based on Lasso path coefficients.

    .. versionadded:: 0.0.2

    It may be the case that the specified number of features cannot be selected
    as a lasso path does not give enough non-zero coefficients, in which case
    the biggest number of features (smaller than the specified number) will be
    returned. In case all of the features are assigned 0 weight or all of the
    paths have a non-zero number of coefficients larger than the specified
    number, all of the features are selected. If the exact number of features
    specified by the user cannot be selected an appropriate message will be
    logged. Also, if the value of ``feature_percentage`` results in selecting
    0 features, 1 feature will be selected and a warning will be logged.

    The ``weights`` provided as the input parameter are incorporated into the
    feature selection process by centering the ``dataset`` around their
    weighted average (if no weights are provided, the average is simply not
    weighted) and scaling by the square root of the ``weights``.
    The ``target`` array is treated in the same way.

    This feature selection method is based on the default feature selection
    mechanism implemented by LIME_ (Local Interpretable Model-agnostic
    Explanations. The original implementation can be found in the
    ``lime.lime_base.LimeBase.feature_selection`` method in the
    `official LIME package`_.

    .. _LIME: https://github.com/marcotcr/lime
    .. _`official LIME package`: https://github.com/marcotcr/lime/blob/master/
       lime/lime_base.py#L116

    Parameters
    ----------
    dataset : numpy.ndarray
        A 2-dimensional numpy array with holding a data set.
    target : numpy.ndarray
        The class/probabilities/regression values of each row in the input data
        set.
    weights : numpy.ndarray, optional (default=None)
        An array of (importance) weights for each data point in the input data
        set. If ``None``, all of the data points are the same important when
        computing the Lasso path.
    features_number : integer, optional (default=None)
        The number of (top) features to be selected. If ``None``, the top x% of
        the features are selected where x is given by the
        ``features_percentage`` parameter. It may be the case that exactly the
        exact number of features cannot be extracted in which case a warning
        will be logged and the next biggest subset of features will be
        selected.
    features_percentage : integer, optional (default=100)
        The percentage of (top) features to be selected. By default all of the
        features are returned if ``features_number`` is ``None``.

    Warns
    -----
    UserWarning
        The specified ``features_number`` is larger than the number of features
        in the ``dataset`` array; all of the features are selected.

    Raises
    ------
    IncorrectShapeError
        The ``dataset`` array is not 2-dimensional. The ``target`` array is not
        1-dimensional. The number of labels in the ``target`` array is
        different than the number of samples in the ``dataset`` array. The
        ``weights`` array is not 1-dimensional. The number of weights in the
        ``weights`` array does not agree with the number of samples in the
        ``dataset`` array.
    TypeError
        The one of the ``dataset``, ``target`` or ``weights`` array is not
        purely numerical. The ``features_number`` parameter is not an integer.
        The ``features_percentage`` parameter is not an integer.
    ValueError
        The ``features_number`` parameter is not a positive integer. The
        ``features_percentage`` parameter is outside of the allowed range
        0--100 (inclusive).

    Returns
    -------
    feature_indices : List[Index]
        List of indices indicating features selected by the Lasso path.
    """
    # pylint: disable=too-many-branches,too-many-locals
    assert _validate_input_lasso_path(
        dataset, target, weights, features_number, features_percentage), \
        'Input is invalid.'

    if fuav.is_structured_array(dataset):
        indices = np.array(dataset.dtype.names)
        dataset_array = fuat.as_unstructured(dataset)
    else:
        indices = np.array(range(0, dataset.shape[1]))
        dataset_array = dataset

    indices_number = indices.shape[0]
    if features_number is None:
        feature_proportion = int((features_percentage / 100) * indices_number)
        if feature_proportion:
            features_number = feature_proportion
        else:
            logger.warning(
                'Since the number of features to be extracted was not given '
                '%d%% of features will be used. This percentage translates to '
                '0 features, therefore the number of features to be used is '
                'overwritten to 1. To prevent this from happening, you should '
                'either explicitly set the number of features via the '
                'features_number parameter or increase the value of the '
                'features_percentage parameter.', features_percentage)
            features_number = feature_proportion + 1

    if features_number == indices_number:
        feature_indices = indices
    elif features_number > indices_number:
        feature_indices = indices
        warnings.warn(
            'The selected number of features is larger than the total number '
            'of features in the dataset array. All of the features are being '
            'selected.', UserWarning)
    else:
        if weights is not None:
            weights_scaled = np.sqrt(weights)
        else:
            weights_scaled = np.ones_like(target)

        dataset_avg = np.average(dataset_array, axis=0, weights=weights)
        weighted_data = (
            (dataset_array - dataset_avg) * weights_scaled[:, np.newaxis])

        target_avg = np.average(target, weights=weights)
        weighted_target = (target - target_avg) * weights_scaled

        fitted_lars_path = sklearn.linear_model.lars_path(
            weighted_data, weighted_target, method='lasso', verbose=False)
        coefs = fitted_lars_path[2]

        # numpy.count_nonzero returns a scalar (despite specifying the axis)
        # in early versions of numpy, hence the workaround of:
        # np.count_nonzero(coefs, axis=0).
        nonzero_count = (coefs != 0).sum(axis=0)

        matching_paths_user = (nonzero_count <= features_number)
        matching_paths_nonzero = (nonzero_count > 0)
        matching_paths = np.where(
            np.logical_and(matching_paths_user, matching_paths_nonzero))[0]

        if matching_paths.size:
            biggest_path = matching_paths[-1]
            nonzero_indices = coefs[:, biggest_path].nonzero()[0]
            feature_indices = indices[nonzero_indices]
            if nonzero_indices.shape[0] != features_number:
                logger.warning(
                    'The lasso path feature selection could not pick %d '
                    'features. Only %d were selected.', features_number,
                    nonzero_indices.shape[0])
        else:
            feature_indices = indices
            logger.warning('The lasso path feature selection could not pick '
                           'any feature subset. All of the features were '
                           'selected.')
    return feature_indices
