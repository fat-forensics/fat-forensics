"""
The :mod:`fatf.transparency.models.submodular_pick` module implements the
submodular pick algorithm proposed by [RIBEIRO2016WHY]_.

.. versionadded:: 0.1.1

.. [RIBEIRO2016WHY] Ribeiro, M.T., Singh, S. and Guestrin, C., 2016,
   August. Why should I trust you?: Explaining the predictions of any
   classifier. In Proceedings of the 22nd ACM SIGKDD international
   conference on knowledge discovery and data mining (pp. 1135-1144). ACM.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
#         Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

from numbers import Number
from typing import Callable, Dict, List, Set, Tuple, Union

import logging
import warnings

import numpy as np

from fatf.exceptions import IncorrectShapeError

import fatf.utils.array.validation as fuav
import fatf.utils.validation as fuv

__all__ = ['submodular_pick']

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

Index = Union[int, str]


def _validate_input(dataset: np.ndarray, explain_instance: Callable,
                    sample_size: int, explanations_number: int) -> bool:
    """
    Validates input for submodular pick.

    For the input parameters description, warnings and exceptions please see
    the documentation of the :func:`fatf.transparency.models.submodular_pick`
    function.

    Returns
    -------
    is_valid : boolean
        ``True`` if the input is valid, ``False`` otherwise.
    """
    is_valid = False

    if not fuav.is_2d_array(dataset):
        raise IncorrectShapeError(
            'The input data set must be a 2-dimensional array.')
    if not fuav.is_base_array(dataset):
        raise ValueError('The input data set must only contain base types '
                         '(strings and numbers).')

    if not isinstance(sample_size, int):
        raise TypeError('sample_size must be an integer.')
    if sample_size < 0:
        raise ValueError('sample_size must be a non-negative integer.')

    if not isinstance(explanations_number, int):
        raise TypeError('explanations_number must be an integer.')
    if explanations_number is not None and explanations_number < 0:
        raise ValueError('explanations_number must be a non-negative integer.')

    if (sample_size and explanations_number
            and sample_size < explanations_number):
        raise ValueError('The number of explanations cannot be larger than '
                         'the number of samples.')

    if callable(explain_instance):
        params_n = fuv.get_required_parameters_number(explain_instance)
        if params_n != 1:
            raise RuntimeError('The explain_instance callable must accept '
                               'exactly one required parameter.')
    else:
        raise TypeError('The explain_instance should be a Python callable '
                        '(function or method).')

    is_valid = True
    return is_valid


def submodular_pick(dataset: np.ndarray,
                    explain_instance: Callable,
                    sample_size: int = 0,
                    explanations_number: int = 5
                    ) -> Tuple[List[Dict[Union[int, str], Number]], List[int]]:
    """
    Applies submodular pick to explanations of a given subset of data.

    .. versionadded:: 0.1.1

    Chooses the most informative data point explanations using the submodular
    pick algorithm introduced by [RIBEIRO2016WHY]_.
    Submodular pick applies a greedy optimisation to maximise the coverage
    function for explanations of a subset of points taken from the input
    data set.
    The explanation function (``explain_instance``) must have exactly one
    required parameter and return a dictionary mapping feature names or indices
    to their respective importance.

    References
    ----------
    .. [RIBEIRO2016WHY] Ribeiro, M.T., Singh, S. and Guestrin, C., 2016,
       August. Why should I trust you?: Explaining the predictions of any
       classifier. In Proceedings of the 22nd ACM SIGKDD international
       conference on knowledge discovery and data mining (pp. 1135-1144). ACM.

    Parameters
    ----------
    dataset : numpy.ndarray
        A data set from which to select individual instances to be explained.
    explain_instance : callable
        A reference to a function or method that can generate an explanation
        from an array representing an individual instance.
        This callable must accept exactly one required parameter and return
        an explanation around the selected data point -- a dictionary mapping
        feature names or indices to their importance.
    sample_size : integer, optional (default=0)
        The number of (randomly selected) data points for which to generate
        explanations. If ``0``, explanations for all the data points in the
        ``dataset`` will be generated.
    explanations_number : integer, optional (default=5)
        The number of explanations to return. If ``0``, an ordered list of
        all explanations generated for the selected data subset are returned.

    Warns
    -----
    UserWarning
        ``sample_size`` is larger than the number of instances (rows) available
        in the ``dataset``, in which case the entire data set is used.
        The number of the requested explanations is larger than the number of
        instances selected to generate explanations -- explanations for all
        the data points in the sample will be generated.

    Raises
    ------
    IncorrectShapeError
        The input data set is not a 2-dimensional numpy array.
    TypeError
        ``sample_size`` or ``explanations_number`` is not an integer.
        ``explain_instance`` is not Python callable (function or method).
    ValueError
        The input data set must only contain base types (strings and numbers).
        ``sample_size`` or ``explanations_number`` is a negative integer.
        The number of requested explanations is larger than the number of
        samples in the data set.
    RuntimeError
        The ``explain_instance`` callable does not require exactly one
        parameter.

    Returns
    -------
    sp_explanations : List[Dictionary[Union[integer, string], Number]]
        List of explanations chosen by the submodular pick algorithm.
    sp_indices : List[integer]
        List of indices for rows in the ``dataset`` chosen (and explained)
        by the submodular pick algorithm.
    """
    # pylint: disable=too-many-locals
    assert _validate_input(dataset, explain_instance, sample_size,
                           explanations_number), 'Invalid input.'

    if sample_size == 0:
        sample_size = dataset.shape[0]
        sample_indices = np.arange(sample_size, dtype=int)
    elif sample_size > dataset.shape[0]:
        warnings.warn(
            'sample_size is larger than the number of samples in the data '
            'set. The whole dataset will be used.', UserWarning)
        sample_size = dataset.shape[0]
        sample_indices = np.arange(sample_size, dtype=int)
    else:
        sample_indices = np.random.choice(
            dataset.shape[0], sample_size, replace=False)

    if explanations_number == 0:
        explanations_number = sample_size
    if explanations_number > sample_size:
        warnings.warn(
            'The number of explanations cannot be larger than the number of '
            'instances (rows) in the data set.', UserWarning)
        explanations_number = sample_size

    # Explain selected rows
    explanations = []
    for row in dataset[sample_indices]:
        explanations.append(explain_instance(row))
    explanations_no = len(explanations)

    # Get unique feature names and fix index mapping
    feature_names = set()  # type: Set[str]
    for exp in explanations:
        feature_names = feature_names.union(exp.keys())
    feature_names_no = len(feature_names)
    feature_map = {id_: i for i, id_ in enumerate(sorted(feature_names))}

    # Compute explanation matrix -- explanations X importances -- i.e., weights
    weights = np.zeros((explanations_no, feature_names_no))
    for i, exp in enumerate(explanations):
        keys = sorted(exp.keys())
        idx = [feature_map[key] for key in keys]
        val = [exp[key] for key in keys]
        weights[i, idx] = val

    importance = np.abs(weights).sum(axis=0)**.5

    indices = set(range(explanations_no))
    sp_indices = []  # type: List[int]
    for _ in range(explanations_number):
        coverage = []
        for i in indices:
            indices_ = sp_indices + [i]
            global_importance = np.abs(weights)[indices_].sum(axis=0)
            global_importance_indicator = global_importance > 0
            coverage_ = np.dot(global_importance_indicator, importance)
            coverage.append((i, coverage_))
        coverage = sorted(coverage, key=lambda i: i[1], reverse=True)
        best_idx = coverage[0][0]
        sp_indices.append(best_idx)
        indices -= {best_idx}

    sp_explanations = [explanations[i] for i in sp_indices]
    sp_indices = sample_indices[sp_indices].tolist()

    return sp_explanations, sp_indices
