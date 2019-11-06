"""
.. versionadded:: 0.0.2

The :mod:`fatf.utils.kernels` module holds distance transformation kernels.

The kernel functions implemented by this module are mainly used to transform a
distance into a similarity measure. One of their applications is to weight
training data samples when training a predictive model based on their
similarity (closeness) to a selected data point.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
#         Kacper Sokol <K.Sokol@bristol.ac.uk>
# License: new BSD

from numbers import Number
from typing import Callable

import warnings

import numpy as np

import fatf.utils.array.validation as fuav
import fatf.utils.validation as fuv

from fatf.exceptions import IncorrectShapeError

__all__ = ['exponential_kernel', 'check_kernel_functionality']


def _input_is_valid(distances: np.ndarray) -> bool:
    """
    Validates input parameters of a kernel function.

    Parameters
    ----------
    distances : numpy.ndarray
        A 1-dimensional numpy array of distances.

    Raises
    ------
    IncorrectShapeError
        The ``distances`` array is not a 1-dimensional numpy array.
    TypeError
        The ``distances`` array is a structured numpy array or it is not a
        purely numerical array.

    Returns
    -------
    is_input_ok : boolean
        ``True`` if the input is valid, ``False`` otherwise.
    """
    is_input_ok = False

    if fuav.is_structured_array(distances):
        raise TypeError('The distances array cannot be a structured array.')

    if not fuav.is_1d_array(distances):
        raise IncorrectShapeError('The distances array must be a '
                                  '1-dimensional array.')

    if not fuav.is_numerical_array(distances):
        raise TypeError('The distances array must be of numerical type.')

    is_input_ok = True
    return is_input_ok


def exponential_kernel(distances: np.ndarray, width: float = 1) -> np.ndarray:
    """
    Applies an exponential kernel to an array of distances.

    .. versionadded:: 0.0.2

    The exponential kernel is computed as:

    .. math::

       \\mathcal{K}(\\mathbf{d})
       =
       \\sqrt{exp\\left(-\\frac{\\mathbf{d}^2}{w^2}\\right)}

    where :math:`\\mathbf{d}` is the array with distances and :math:`w` is the
    kernel width.

    Parameters
    ----------
    distances : numpy.ndarray
        A 1-dimensional, numerical numpy array with distances.
    width : number, optional (default=1)
        Width of the exponential kernel, which has to be a positive number.
        If a value is not provided, the default value of 1 is used.

    Raises
    ------
    IncorrectShapeError
        The ``distances`` array is not a 1-dimensional numpy array.
    TypeError
        The ``distances`` array is a structured numpy array or it is not a
        purely numerical array. The width of the kernel is not a number.
    ValueError
        The width of the kernel is not a **positive** (greater than 0) number.

    Returns
    -------
    kernelised_distances : numpy.ndarray
        A 1-dimensional numpy array containing distances transformed with an
        exponential kernel of width ``width``.
    """
    assert _input_is_valid(distances), 'Input is invalid.'

    if not isinstance(width, Number):
        raise TypeError('The kernel width must be a number.')
    if width <= 0:
        raise ValueError('The kernel width must be a positive (greater than '
                         '0) number.')

    kernelised_distances = np.sqrt(np.exp(-(distances**2) / width**2))

    return kernelised_distances


def check_kernel_functionality(kernel_function: Callable[..., np.ndarray],
                               suppress_warning: bool = False) -> bool:
    """
    Checks whether a kernel function has exactly one required parameter.

    .. versionadded:: 0.0.2

    Parameters
    ----------
    kernel_function : Callable[[numpy.ndarray, ...], numpy.ndarray]
        A Python callable, e.g., a function or a method, which represents a
        kernel function.
    suppress_warning : boolean, optional (default=False)
        A boolean parameter that indicates whether the function should suppress
        its warning message. Defaults to False.

    Warns
    -----
    UserWarning
        Warns about the details of the required functionality that the kernel
        function lacks.

    Raises
    ------
    TypeError
        The ``kernel_function`` parameter is not a Python callable or the
        ``suppress_warning`` parameter is not a boolean.

    Returns
    -------
    is_functional : boolean
        A boolean variable that indicates whether the kernel function is valid.
    """
    if not callable(kernel_function):
        raise TypeError('The kernel_function parameter should be a Python '
                        'callable.')
    if not isinstance(suppress_warning, bool):
        raise TypeError('The suppress_warning parameter should be a boolean.')

    required_param_n = fuv.get_required_parameters_number(kernel_function)
    is_functional = required_param_n == 1

    if not is_functional and not suppress_warning:
        message = ("The '{}' kernel function has incorrect number ({}) of the "
                   'required parameters. It needs to have exactly 1 required '
                   'parameter. Try using optional parameters if you require '
                   'more functionality.').format(kernel_function.__name__,
                                                 required_param_n)
        warnings.warn(message, category=UserWarning)

    return is_functional
