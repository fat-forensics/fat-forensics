"""
The :mod:`fatf.utils.models.validation` module validates models functionality.

This module holds functions responsible for validating models functionality
across the FAT Forensics package.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
#         Alex Hepburn <ah13558@bristol.ac.uk>
# License: new BSD

from typing import Union

import warnings

import fatf.utils.validation as fuv

__all__ = ['check_model_functionality']


def check_model_functionality(model_object: Union[object, type],
                              require_probabilities: bool = False,
                              suppress_warning: bool = False) -> bool:
    """
    Checks whether a model object has all the required functionality.

    Examines a ``model_object`` and ensures that it has all the required
    methods with the correct number of parameters (excluding ``self``):
    ``__init__`` (at least 0), ``fit`` (at least 1 for unsupervised and at
    least 2 for supervised models), ``predict`` (at least 1) and,
    if required (``require_probabilities=True``), ``predict_proba`` (at
    least 1, in which case ``fit`` requires at least 2).

    Parameters
    ----------
    model_object : Union[object, type]
        A Python object (either instantiated or just an object reference) that
        represents a predictive model.

        .. versionchanged:: 0.0.2
           Added the possibility of checking functionality of non-initialised
           objects.
    require_probabilities : boolean, optional (default=False)
        A boolean parameter that indicates whether the model object should
        contain a ``predict_proba`` method. Defaults to False.
    suppress_warning : boolean, optional (default=False)
        A boolean parameter that indicates whether the function should suppress
        its warning message. Defaults to False.

    Raises
    ------
    TypeError
        The ``require_probabilities`` or ``suppress_warning`` parameter is not
        a boolean.

    Warns
    -----
    UserWarning
        Warns about the required functionality that the model object lacks.

    Returns
    -------
    is_functional : boolean
        A boolean variable that indicates whether the model object has all the
        desired functionality.
    """
    if not isinstance(require_probabilities, bool):
        raise TypeError('The require_probabilities parameter must be boolean.')
    if not isinstance(suppress_warning, bool):
        raise TypeError('The suppress_warning parameter must be boolean.')

    methods_unsupervised = {'fit': 1, 'predict': 1}
    methods_supervised = {'fit': 2, 'predict': 1}
    if require_probabilities:
        methods_supervised['predict_proba'] = 1

    is_functional_us, message_us = fuv.check_object_functionality(
        model_object, methods_unsupervised, object_reference_name='model')
    is_functional_s, message_s = fuv.check_object_functionality(
        model_object, methods_supervised, object_reference_name='model')
    is_functional = is_functional_us or is_functional_s

    if not is_functional and not suppress_warning:
        message = ('Model object characteristics are neither consistent with '
                   'supervised nor unsupervised models.\n\n'
                   '--> Unsupervised models <--\n{}'
                   '\n\n--> Supervised models <--\n{}').format(
                       message_us, message_s)
        warnings.warn(message, category=UserWarning)

    return is_functional
