"""
.. versionadded:: 0.0.2

The :mod:`fatf.utils.validation` module validates functions and objects.

This module holds functions responsible for validating generic functions and
objects implemented across the FAT Forensics package.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
#         Alex Hepburn <ah13558@bristol.ac.uk>
# License: new BSD

from typing import Callable, Dict, Optional, Tuple, Union

import inspect

__all__ = ['get_required_parameters_number', 'check_object_functionality']


def get_required_parameters_number(callable_object: Callable) -> int:
    """
    Checks if a callable object has the correct number of required parameters.

    .. versionadded:: 0.0.2

    A callable object can be a function or a method.

    Parameters
    ----------
    callable_object : callable
        A callable object to be tested.

    Raises
    ------
    TypeError
        The ``callable_object`` is not a Python callable, i.e., a function or
        a method.

    Returns
    -------
    required_param_n : integer
        The number of required parameters that ``callable_object`` takes.
    """
    if not callable(callable_object):
        raise TypeError('The callable_object should be Python callable, e.g., '
                        'a function or a method.')

    required_param_n = 0
    params = inspect.signature(callable_object).parameters
    for param in params:
        if param != 'kwargs':
            if params[param].default is params[param].empty:
                required_param_n += 1

    return required_param_n


def check_object_functionality(
        an_object: Union[object, type],
        methods: Dict[str, int],
        object_reference_name: Optional[str] = None) -> Tuple[bool, str]:
    """
    Checks if an object has specified methods with given number of parameters.

    .. versionadded:: 0.0.2

    The object to be checked can either be an uninitialised object reference or
    an initialised object instance.

    Parameters
    ----------
    an_object : Union[object, type]
        A Python object (either uninitialised object reference or an
        initialised object) to be checked.
    methods : Dictionary[string, integer]
        A dictionary where keys are method names required to be in `an_object`
        and values are the number of required parameters in for these methods.
    object_reference_name : string, optional (default='')
        A reference name of ``an_object`` used to provide more information
        in the generated (warning) ``message``. If ``None``, this information
        will not be included in the ``message``.

    Raises
    ------
    TypeError
        The ``methods`` parameter is not a dictionary, one of its keys is not a
        string or one of its values is not an integer. The
        ``object_reference_name`` parameter is neither a string nor ``None``.
    ValueError
        The ``methods`` dictionary is empty or one of its values is a negative
        integer.

    Returns
    -------
    is_functional : boolean
        A boolean variable that indicates whether the object has all
        the desired functionality.
    message : string
        A message detailing the lacking functionality of ``an_object``.
    """
    # pylint: disable=too-many-branches,too-many-locals
    if isinstance(methods, dict):
        if methods:
            # The splitting is necessary for consistent error raising with
            # Python 3.5 (and, possibly, below).
            key_str = sorted([i for i in methods.keys() if isinstance(i, str)])
            key_otr = [i for i in methods.keys() if i not in key_str]

            for key in key_str + key_otr:
                value = methods[key]
                if not isinstance(key, str):
                    raise TypeError('All of the keys in the methods '
                                    "dictionary must be strings. The '{}' key "
                                    'in not a string.'.format(key))
                if not isinstance(value, int):
                    raise TypeError('All of the values in the methods '
                                    "dictionary must be integers. The '{}' "
                                    "value for the '{}' key in not a "
                                    'string.'.format(value, key))
                if value < 0:
                    raise ValueError('All of the values in the methods '
                                     'dictionary must be non-negative '
                                     "integers. The '{}' value for '{}' key "
                                     'does not comply.'.format(value, key))
        else:
            raise ValueError('The methods dictionary cannot be empty.')
    else:
        raise TypeError('The methods parameter must be a dictionary.')

    if object_reference_name is not None:
        if not isinstance(object_reference_name, str):
            raise TypeError('The object_reference_name parameter must be a '
                            'string or None.')

    is_instantiated = not isinstance(an_object, type)

    if is_instantiated:
        object_name = an_object.__class__.__name__
        param_correction = 0
    else:
        object_name = an_object.__name__  # type: ignore
        # `self` is an extra parameter if the object is not instantiated
        param_correction = 1

    if object_reference_name is None:
        object_reference = '*{}* class'.format(object_name)
    else:
        object_reference = '*{}* ({}) class'.format(object_name,
                                                    object_reference_name)

    is_functional = True
    message_strings = []

    for method in methods:
        if not hasattr(an_object, method):
            is_functional = False
            message_strings.append("The {} is missing '{}' method.".format(
                object_reference, method))
        else:
            method_object = getattr(an_object, method)
            required_param_n = get_required_parameters_number(method_object)
            required_param_n -= param_correction

            if not required_param_n == methods[method]:
                is_functional = False
                message_strings.append(
                    ("The '{}' method of the {} has incorrect number "
                     '({}) of the required parameters. It needs to have '
                     'exactly {} required parameter(s). Try using optional '
                     'parameters if you require more functionality.').format(
                         method, object_reference, required_param_n,
                         methods[method]))

    message = '\n'.join(message_strings)

    return is_functional, message
