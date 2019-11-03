"""
.. versionadded:: 0.0.2

The :mod:`fatf.utils.transparency.explainers` module holds utilities for
building custom explainer objects.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

from typing import Union

import abc
import warnings

import numpy as np

import fatf.utils.validation as fuv

__all__ = ['check_instance_explainer_functionality', 'Explainer']


class Explainer(abc.ABC):
    """
    A base class for any explainer object implemented in the package.

    .. versionadded:: 0.0.2
    """

    def feature_importance(self) -> np.ndarray:
        """
        Computes feature importance.
        """
        raise NotImplementedError('Feature importance not implemented.')

    def explain_model(self) -> np.ndarray:
        """
        Generates a model explanation.
        """
        raise NotImplementedError('Model explanation (global) not '
                                  'implemented.')

    def explain_instance(self) -> np.ndarray:
        """
        Generates an explanation of a single data point (instance).

        This can be an explanation of a data point from a data set or of a
        prediction provided by a predictive model.
        """
        raise NotImplementedError('Data point explanation (local) not '
                                  'implemented.')


def check_instance_explainer_functionality(
        explainer_object: Union[object, type],
        suppress_warning: bool = False) -> bool:
    """
    Checks whether an explainer object can explain a data point (instance).

    .. versionadded:: 0.0.2

    The explainer object to be checked can either be an uninitialised object
    reference or an initialised object instance.

    This function examines the ``explainer_object`` and ensures that it has
    an ``explain_instance`` method with exactly 1 required parameter.

    Parameters
    ----------
    explainer_object : object
        A Python object (either an object reference or an instance of an
        initialised object) that represents an explanation generator.
    suppress_warning : boolean, optional (default=False)
        A boolean parameter that indicates whether the function should suppress
        the warning message detailing the lacking functionality of the
        explainer object. Defaults to False.

    Warns
    -----
    UserWarning
        Warns about the required functionality that the explainer object lacks
        if it has not been disabled with the ``suppress_warning`` parameter.
        Warns if the explainer object does not inherit from the
        :class:`fatf.utils.transparency.explainers.Explainer` abstract class.

    Raises
    ------
    TypeError
        The ``suppress_warning`` parameter is not a boolean.

    Returns
    -------
    is_functional : boolean
        A boolean variable that indicates whether the explainer object has all
        the desired functionality.
    """
    if not isinstance(suppress_warning, bool):
        raise TypeError('The suppress_warning parameter should be a boolean.')

    methods = {'explain_instance': 1}

    is_functional, message = fuv.check_object_functionality(
        explainer_object, methods, object_reference_name='explainer')

    if not is_functional and not suppress_warning:
        warnings.warn(message, category=UserWarning)

    if isinstance(explainer_object, type):
        inherits_correctly = issubclass(explainer_object, Explainer)
    else:
        inherits_correctly = isinstance(explainer_object, Explainer)
    if not inherits_correctly:
        warnings.warn(
            'Every explainer object should inherit from '
            'fatf.utils.transparency.explainers.Explainer abstract class.',
            category=UserWarning)

    return is_functional
