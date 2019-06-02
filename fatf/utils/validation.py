"""
Holds functions responsible for models validation across FAT-Forensics.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
#         Alex Hepburn <ah13558@bristol.ac.uk>
# License: new BSD

from typing import Dict, Tuple
import inspect
import warnings

__all__ = ['check_explainer_functionality']


def _check_object_functionality(given_object: object,
                                object_name: str,
                                methods: Dict[str, int],
                                is_instance: bool = True) -> Tuple[bool, str]:
    """
    Checks whether an object has specified methods with nubmer of parameters.

    Parameters
    ----------
    given_object : object
        A Python object to check methods.
    object_name : str
        Name of type of object that `given_object` is to use in warning
        messages
    methods : Dictionary[string, integer]
        Dictionary where key is method name required in `given_object` and
        value is number of required parameters in specified method.
    """
    is_functional = True

    message_strings = []
    for method in methods:
        if not hasattr(given_object, method):
            is_functional = False
            message_strings.append(
                'The {} class is missing \'{}\' method.'.format(
                    object_name, method))
        else:
            method_object = getattr(given_object, method)
            required_param_n = 0
            params = inspect.signature(method_object).parameters
            for param in params:
                if params[param].default is params[param].empty:
                    required_param_n += 1
            if required_param_n != methods[method]:
                is_functional = False
                if not is_instance:
                    required_param_n -= 1 # Remove ``self``` if not instance.
                    methods[method] -= 1
                message_strings.append(
                    ('The \'{}\' method of the class has incorrect number '
                     '({}) of the required parameters. It needs to have '
                     'exactly {} required parameters. Try using optional '
                     'parameters if you require more functionality.').format(
                         method, required_param_n, methods[method]))

    message = '\n'.join(message_strings)

    return is_functional, message


def check_explainer_functionality(explainer_object: object,
                                  suppress_warning: bool = False) -> bool:
    """
    Checks whether a explainer object has all the required functionality.

    Examines a ``explainer_object`` and ensures that it has all the required
    methods with the correct number of parameters (excluding ``self``):
    ``__init__`` (at least 0), ``explain_instance`` (at least 2).

    Parameters
    ----------
    explainer_object : object
        A Python object that represents a object that generates explanations.
    suppress_warning : boolean, optional (default=False)
        A boolean parameter that indicates whether the function should suppress
        its warning message. Defaults to False.

    Warns
    -----
    UserWarning
        Warns about the required functionality that the explainer object lacks.

    Returns
    -------
    is_functional : boolean
        A Boolean variable that indicates whether the explainer object has all
        the desired functionality.
    """
    is_functional = True

    methods = {'explain_instance': 1}

    is_functional, message = _check_object_functionality(
        explainer_object, 'explainer', methods)

    if not is_functional and not suppress_warning:
        warnings.warn(message, category=UserWarning)

    return is_functional
