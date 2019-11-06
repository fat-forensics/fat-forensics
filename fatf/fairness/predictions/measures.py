"""
The :mod:`fatf.fairness.predictions.measures` module measures predictions
fairness.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
#         Rafael Poyiadzi <rp13102@bristol.ac.uk>
# License: new BSD

from numbers import Number
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from fatf.exceptions import IncorrectShapeError

import fatf.transparency.predictions.counterfactuals as ftpc
import fatf.utils.array.validation as fuav

__all__ = ['counterfactual_fairness', 'counterfactual_fairness_check']

FeatureRange = Union[Tuple[float, float], List[Union[float, str]]]
Index = Union[int, str]  # Possible types of column indices


def counterfactual_fairness(
        instance: Union[np.ndarray, np.void],
        protected_feature_indices: List[Index],
        #
        counterfactual_class: Optional[Union[int, str]] = None,
        #
        model: Optional[object] = None,
        predictive_function: Optional[Callable] = None,
        dataset: Optional[np.ndarray] = None,
        categorical_indices: Optional[List[Index]] = None,
        numerical_indices: Optional[List[Index]] = None,
        max_counterfactual_length: int = 2,
        feature_ranges: Optional[Dict[Index, FeatureRange]] = None,
        distance_functions: Optional[Dict[Index, Callable]] = None,
        step_sizes: Optional[Dict[Index, float]] = None,
        default_numerical_step_size: float = 1.0,
        #
        normalise_distance: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Checks counterfactual fairness of a prediction given a model.

    This is an example of **disparate treatment** fairness approach, i.e.
    individual fairness. It checks whether there are two "similar" individuals
    (who only differ in the protected attributes) who are treated differently,
    i.e. get a different prediction.

    The counterfactual fairness function is based on the  object.
    It is based on the :obj:`fatf.transparency.predictions.counterfactuals.\
CounterfactualExplainer` object. For all the errors, warnings and
    exceptions please see the documentation of :obj:`fatf.transparency.\
predictions.counterfactuals.CounterfactualExplainer` object and its methods.

    Parameters
    ----------
    instance, counterfactual_class, and normalise_distance
        For the description of these parameters please see the documentation of
        :func:`fatf.transparency.predictions.counterfactuals.\
CounterfactualExplainer.explain_instance` method.
    protected_feature_indices, model, predictive_function, dataset, \
categorical_indices, numerical_indices, max_counterfactual_length, \
feature_ranges, distance_functions, step_sizes, and default_numerical_step_size
        For the desctiption of these parameters please see the documentation of
        :obj:`fatf.transparency.predictions.counterfactuals.\
CounterfactualExplainer` object. The only difference is that the
        `counterfactual_feature_indices` parameter is renamed to
        `protected_feature_indices` and is required by this function.

    Returns
    -------
    counterfactuals : numpy.ndarray
        A 2-dimensional numpy array with counterfactually unfair data points.
    distances : numpy.ndarray
        A 1-dimensional numpy array with distances from the input ``instance``
        to every counterfactual data point.
    predictions : numpy.ndarray
        A 1-dimensional numpy array with predictions for every counterfactual
        data point.
    """
    # pylint: disable=too-many-arguments,too-many-locals
    cfex = ftpc.CounterfactualExplainer(
        model=model,
        predictive_function=predictive_function,
        dataset=dataset,
        categorical_indices=categorical_indices,
        numerical_indices=numerical_indices,
        counterfactual_feature_indices=protected_feature_indices,
        max_counterfactual_length=max_counterfactual_length,
        feature_ranges=feature_ranges,
        distance_functions=distance_functions,
        step_sizes=step_sizes,
        default_numerical_step_size=default_numerical_step_size)

    counterfactuals, distances, predictions = cfex.explain_instance(
        instance, counterfactual_class, normalise_distance)

    return counterfactuals, distances, predictions


def counterfactual_fairness_check(
        unfair_counterfactuals: Optional[np.ndarray] = None,
        distances: Optional[np.ndarray] = None,
        threshold: Optional[float] = None) -> bool:
    """
    Checks for counterfactual fairness using a counterfactual fairness arrays.

    There are two different approaches to evaluate counterfactual fairness. The
    first one is to take the ``distances`` to the counterfactual examples and
    see whether there are any that exceed a certain ``threshold`` in which case
    a given instance is considered to be treated unfairly. Alternatively by
    using the ``unfair_counterfactuals`` array this function checks whether
    there are any unfair counterfactual instances. In case all the input
    parameters are given **the distance-based approach takes the precedence**.

    Parameters
    ----------
    unfair_counterfactuals : numpy.ndarray, optional (default=None)
        A 2-dimensional numpy array with counterfactual examples that expose
        unfairness of a prediction.
    distances : numpy.ndarray, optional (default=None)
        A 1-dimensional numpy array with .
    threshold : number, optional (default=None)
        A numerical threshold above which a counterfactual instance is too far,
        therefore it is considered to be an exemplar of individual unfairness.

    Raises
    ------
    IncorrectShapeError
        The ``unfair_counterfactuals`` parameter is not a 2-dimensional array.
        The ``distances`` parameter is not a 1-dimensional array.
    RuntimeError
        Either of the required input parameters were not given:
        ``unfair_counterfactuals`` or ``distances`` and ``threshold``.
    TypeError
        The ``threshold`` parameter is not a number.
    ValueError
        The ``distances`` array is not purely numerical.

    Returns
    -------
    counterfactually_unfair : boolean
        ``True`` if there are any counterfactually unfair instances, ``False``
        otherwise.
    """
    if distances is not None and threshold is not None:
        if not fuav.is_1d_array(distances):
            raise IncorrectShapeError('The distances parameter has to be a '
                                      '1-dimensional array.')
        if not fuav.is_numerical_array(distances):
            raise ValueError('The distances array has to be purely numerical.')
        if not isinstance(threshold, Number):
            raise TypeError('The threshold parameter has to be a number.')

        counterfactually_unfair = (distances > threshold).any()
    elif unfair_counterfactuals is not None:
        if not fuav.is_2d_array(unfair_counterfactuals):
            raise IncorrectShapeError('The unfair counterfactuals parameter '
                                      'has to be a 2-dimensional numpy array.')
        counterfactually_unfair = bool(unfair_counterfactuals.size)
    else:
        raise RuntimeError('Either of the two is required to run this '
                           'function: unfair_counterfactuals parameter or '
                           'both distances and threshold parameters.')

    return counterfactually_unfair
