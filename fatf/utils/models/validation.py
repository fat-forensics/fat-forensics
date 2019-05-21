"""
Holds functions responsible for models validation across FAT-Forensics.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
#         Alex Hepburn <ah13558@bristol.ac.uk>
# License: new BSD

from typing import Dict, Tuple
import inspect
import warnings

import fatf.utils.validation as fuv

__all__ = ['check_model_functionality']


def check_model_functionality(model_object: object,
                              require_probabilities: bool = False,
                              suppress_warning: bool = False,
                              is_instance: bool = True) -> bool:
    """
    Checks whether a model object has all the required functionality.

    Examines a ``model_object`` and ensures that it has all the required
    methods with the correct number of parameters (excluding ``self``):
    ``__init__`` (at least 0), ``fit`` (at least 2), ``predict`` (at least 1)
    and, if required (``require_probabilities=True``), ``predict_proba`` (at
    least 1).

    Parameters
    ----------
    model_object : object
        A Python object that represents a predictive model.
    require_probabilities : boolean, optional (default=False)
        A boolean parameter that indicates whether the model object should
        contain a ``predict_proba`` method. Defaults to False.
    suppress_warning : boolean, optional (default=False)
        A boolean parameter that indicates whether the function should suppress
        its warning message. Defaults to False.
    is_instance : boolean, optional (default=True)
        A boolean parameter that indices whether the model is an instanatiated
        object or just an object reference.

    Warns
    -----
    UserWarning
        Warns about the required functionality that the model object lacks.

    Returns
    -------
    is_functional : boolean
        A Boolean variable that indicates whether the model object has all the
        desired functionality.
    """
    if is_instance:
        methods = {'fit': 2, 'predict': 1}
    else:
        methods = {'fit': 3, 'predict': 2}
    if require_probabilities:
        methods['predict_proba'] = 1 if is_instance else 2

    is_functional, message = fuv._check_object_functionality(
        model_object, 'model', methods, is_instance=is_instance)

    if not is_functional and not suppress_warning:
        warnings.warn(message, category=UserWarning)

    return is_functional
