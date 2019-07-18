"""
Holds functions responsible for models validation across FAT-Forensics.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
#         Alex Hepburn <ah13558@bristol.ac.uk>
# License: new BSD

from typing import Dict, Tuple, Callable, Any
import inspect
import warnings

__all__ = ['check_explainer_functionality', 'check_kernel_functionality']


def _check_function_functionality(function: Callable[..., Any],
                                  required_param: int) -> bool:
    """
    Checks whether a method has the correct number of required parameters.

    Parameters
    ----------
    function : Callable[..., Any]
        Function to test.
    required_param : integer
        Number of required parameters that ``method`` must have.

    Returns
    -------
    is_functional : boolean
        A Boolean variable that indicates whether the object has all
        the desired functionality.
    required_param_n : integer
        Number of parameters that ``function`` requires.
    message : string
        Message detailing the functionality that the function is lacking.
    """
    is_functional = True

    required_param_n = 0
    params = inspect.signature(function).parameters
    for param in params:
        if param == 'kwargs':
            if params[param].default is params[param].empty:
                required_param_n += 1
    if required_param_n != required_param:
        is_functional = False

    return is_functional, required_param_n


def _check_object_functionality(given_object: object,
                                object_name: str,
                                methods: Dict[str, int],
                                is_instance: bool = True) -> Tuple[bool, str]:
    """
    Checks whether an object has specified methods with number of parameters.

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

    Returns
    -------
    is_functional : boolean
        A Boolean variable that indicates whether the object has all
        the desired functionality.
    message : string
        Message detailing the functionality that the object is lacking.
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
            functional, required_params, = _check_function_functionality(
                method_object, methods[method])
            if not functional:
                is_functional = False
                if not is_instance:
                    required_params -= 1  # Remove ``self``` if not instance.
                    methods[method] -= 1
                message_strings.append(
                    ('The \'{}\' method of the class has incorrect number '
                     '({}) of the required parameters. It needs to have '
                     'exactly {} required parameters. Try using optional '
                     'parameters if you require more functionality.').format(
                         method, required_params, methods[method]))

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


def check_kernel_functionality(kernel_function: Callable[..., Any],
                               suppress_warning: bool = False) -> bool:
    """
    Checks whether a kernel function has all the required functionality.

    Examines a ``kernel_function`` and ensures that it has all the required
    methods with the correct number of required parameters of 1.

    Parameters
    ----------
    kernel_function : object
        A function that represents a kernel.
    suppress_warning : boolean, optional (default=False)
        A boolean parameter that indicates whether the function should suppress
        its warning message. Defaults to False.

    Warns
    -----
    UserWarning
        Warns about the required functionality that the kernel function lacks.

    Returns
    -------
    is_functional : boolean
        A Boolean variable that indicates whether the kernel function has all
        the desired functionality.
    """
    is_functional = True

    is_functional, required_param = _check_function_functionality(
        kernel_function, 1)

    if not is_functional and not suppress_warning:
        message = ('The \'{}\' kernel function has incorrect number '
                   '({}) of the required parameters. It needs to have '
                   'exactly 1 required parameters. Try using optional '
                   'parameters if you require more functionality.').format(
                       kernel_function.__name__, required_param)
        warnings.warn(message, category=UserWarning)

    return is_functional


def check_distance_functionality(distance_function: Callable[..., Any],
                                 suppress_warning: bool = False) -> bool:
    """
    Checks whether a distance function has all the required functionality.

    Examines a ``distance_function`` and ensures that it has all the required
    methods with the correct number of required parameters of 2.

    Parameters
    ----------
    distance_function : object
        A function that represents a distance.
    suppress_warning : boolean, optional (default=False)
        A boolean parameter that indicates whether the function should suppress
        its warning message. Defaults to False.

    Warns
    -----
    UserWarning
        Warns about the required functionality that the distance function
        lacks.

    Returns
    -------
    is_functional : boolean
        A Boolean variable that indicates whether the distance function has all
        the desired functionality.
    """
    is_functional = True

    is_functional, required_param = _check_function_functionality(
        distance_function, 2)

    if not is_functional and not suppress_warning:
        message = ('The \'{}\' distance function has incorrect number '
                   '({}) of the required parameters. It needs to have '
                   'exactly 2 required parameters. Try using optional '
                   'parameters if you require more functionality.').format(
                       distance_function.__name__, required_param)
        warnings.warn(message, category=UserWarning)

    return is_functional
