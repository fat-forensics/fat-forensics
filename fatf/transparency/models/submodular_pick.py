"""
Implements submodular pick algorithm.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
# License: new BSD

from typing import Optional, Union, List, Dict
import warnings

import numpy as np

import fatf.utils.array.validation as fuav
import fatf.utils.validation as fuv

from fatf.exceptions import IncompatibleExplainerError, IncorrectShapeError

Index = Union[int, str]

__all__ = ['submodular_pick']


def _input_is_valid(dataset: np.ndarray,
                    explainer: object,
                    sample_size: Optional[int],
                    num_explanations: Optional[int]) -> bool:
    """
    Validates input for submodular pick.

    For the input parameter description, warnings and exceptions please see the
    documentation of the :func`fatf.transparency.models.submodular_pick`
    function.

    Returns
    -------
    is_input_ok : boolean
        ``True`` if the input is valid, ``False`` otherwise.
    """
    is_input_ok = True

    if not fuav.is_2d_array(dataset):
        raise IncorrectShapeError('The input dataset must be a 2-dimensional '
                                  'array.')

    if not fuav.is_base_array(dataset):
        raise ValueError('The input dataset must only contain base types '
                         '(textual and numerical).')

    if sample_size is not None and not isinstance(sample_size, int):
        raise TypeError('sample_size must be an integer or None.')

    if sample_size is not None and sample_size <= 0:
        raise ValueError('sample_size must be a positive integer or None.')

    if num_explanations is not None and not isinstance(num_explanations, int):
        raise TypeError('num_explanations must be an integer or None.')

    if num_explanations is not None and num_explanations < 0:
        raise ValueError('num_explanations must be a positive integer or '
                         'None.')

    if ((sample_size is not None and num_explanations is not None) and
            (sample_size < num_explanations)):
        raise ValueError('sample_size must be larger or equal to '
                         'num_explanations.')

    if not fuv.check_explainer_functionality(explainer, True):
        raise IncompatibleExplainerError(
            'explainer object must be method \'explain_instance\' which has '
            'exactly one required parameter. Other named parameters can be '
            'passed to the submodular pick method.')

    return is_input_ok


def submodular_pick(dataset: np.ndarray,
                    explainer: object,
                    sample_size: Optional[int] = None,
                    num_explanations: Optional[int] = 5,
                    **kwargs) -> List[Dict[Union[int, str], np.float64]]:
    """
    Performs submodular pick with given explainer.

    Chooses the most informative explanations based on the submodular pick
    algorithm described in the original LIME paper [1]_.
    Performs a greedy optimisation to maximise the coverage function for
    explanations of various points in the dataset.
    The explainer object should contain a method called ``explain_instance``
    that has exactly one required parameter and returns a dictionary mapping
    feature names or indices to a float value indicating that features
    importance. In order to pass additioanl arguments to
    ``explainer.explain_instnace`` function please use ``**kwargs``.

    Parameters
    ----------
    dataset : numpy.ndarray
        A dataset based on which submodule pick will be computed.
    explainer : object
        An object used to get explainations for a model around instances. This
        object must contain a method ``explain_instance`` that returns an
        explanation around a provided row in the dataset. An explanation
        should be of type of `Dict[Union[int, str], np.float64]]`,
        mapping feature names or indices to the corresponding feature
        importance.
    sample_size : integer, optional (default=None)
        Number of samples in the subset of instances used to generate
        explainations for. If `None`, then explanations will be generated for
        the whole dataset.
    num_explanations : integer, optional (default=5)
        Number of explanations to return. If `None`, then an ordered list of
        all explanations sampled data are returned.
    **kwargs : explainer.explain_instance
        `explain_instance` optional parameters (e.g. number of sampels to
        generation around instance during explanation)

    Warns
    -----
    UserWarning
        ``sample_size`` is larger than the number of samples in the dataset,
        and as such the entire dataset will be used.

    Raises
    ------
    IncorrectShapeError
        The input dataset is not a 2-dimensional numpy array.
    TypeError
        ``sample_size`` is not None` or integer, ``num_explanations`` is
        not `None or integer.
    ValueError
        The input dataset must only contain base types (textual and numerical
        values). ``sample_size`` is not positive integer, ``num_explanations``
        is not positive integer.
    IncompatibleExplainerError
        Explainer object either has no method called ``explain_instance`` or
        ``explain_instance`` method does not have exactly one required
        parameter.

    Returns
    -------
    submodular_pick_explanations : List[Dictionary[Union[integer, string],
            numpy.float64]]
        List of explanations chosen by submodular pick algorithm.
    submodular_pick_indices : List[integer]
        List of indices of rows in dataset which were chosen by submodular
        pick.

    References
    ----------
    .. [1] Ribeiro, Marco Tulio, Sameer Singh, and Carlos Guestrin. "Why
       should i trust you?: Explaining the predictions of any classifier."
       Proceedings of the 22nd ACM SIGKDD international conference on
       knowledge discovery and data mining. ACM, 2016.
    """
    assert _input_is_valid(dataset, explainer, sample_size,
                           num_explanations), 'Input must be valid.'

    sampled = False
    if sample_size is None:
        rows = dataset
    elif sample_size > dataset.shape[0]:
        msg = ('sample_size is larger than the number of sampels in the '
               'dataset. The whole dataset will be used.')
        warnings.warn(msg, UserWarning)
        rows = dataset
    else:
        random_indices = np.random.choice(dataset.shape[0], sample_size)
        rows = dataset[random_indices]
        sampled = True

    if num_explanations is None:
        num_explanations = rows.shape[0]

    explanations = []
    for row in rows:
        explanations.append(explainer.explain_instance(row, **kwargs))

    feature_set = set().union(*(d.keys() for d in explanations))
    feature_dict = dict(zip(list(feature_set), list(range(len(feature_set)))))

    weights = np.zeros((len(explanations), len(feature_set)))
    for i, exp in enumerate(explanations):
        ind = np.array([feature_dict[key] for key in exp.keys()])
        val = np.array([exp[key] for key in exp.keys()])
        weights[i, ind] = val
    importance = np.sum(np.abs(weights), axis=0)**.5

    remaining_indices = set(range(len(explanations)))
    submodular_pick_indices = []
    for _ in range(num_explanations):
        coverages = []
        for i in remaining_indices:
            weights_abs = np.abs(weights)[submodular_pick_indices + [i]]
            coverage = np.dot((np.sum(weights_abs, axis=0) > 0), importance)
            coverages.append((i, coverage))
        best_ind = coverages[np.argmax([k[1] for k in coverages])][0]
        submodular_pick_indices.append(best_ind)
        remaining_indices -= {best_ind}

    submodular_pick_explanations = [explanations[i] for i in
                                    submodular_pick_indices]

    if sampled:
        submodular_pick_indices = \
            random_indices[submodular_pick_indices].tolist()

    return submodular_pick_explanations, submodular_pick_indices
