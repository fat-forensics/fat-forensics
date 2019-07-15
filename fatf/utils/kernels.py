"""
Holds functions that use a kernel to transform distances.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
# License: new BSD

import numpy as np

from fatf.exceptions import IncorrectShapeError
import fatf.utils.array.validation as fuav

__all__ = ['exponential_kernel']


def _input_is_valid(distances: np.ndarray):
    """
    Validates input parameters of for kernel functions.

    Parameters
    ----------
    distances : numpy.ndarray
        A numpy array which is a vector of distances.

    Returns
    -------
    is_input_ok : boolean
        ``True`` if the input is valid, ``False`` otherwise.
    """
    is_input_ok = False

    if fuav.is_structured_array(distances):
        raise TypeError('distances cannot be a structured array.')

    if not fuav.is_1d_array(distances):
        raise IncorrectShapeError('distances must be a 1-dimensional array.')

    if not fuav.is_numerical_array(distances):
        raise TypeError('distances must be of numerical type.')

    is_input_ok = True
    return is_input_ok


def exponential_kernel(distances: np.ndarray, width:float = None):
    """
    Applies exponential kernel to distances. For additional expceptions raised
    please see :func:`fatf.utils.kernels.__input_is_valid` function.

    Parameters
    ----------
    distances : numpy.ndarray
        A numpy array which is a vector of distances.
    width : numpy.ndarray, Optional (default=None)
        Width of expoential kernel. If None then will default to 1.

    Raises
    ------
    TypeError:
        ``width`` is not a float. ``width`` is not a positive float greater
        than 0.

    Returns
    -------
    kernalised_distance: np.ndarray
        An array containing kernalised_distances using the exponential kernel
        with a width of ``width``.
    """
    assert _input_is_valid(distances), 'Input is not valid.'

    if not isinstance(width, float):
        raise TypeError('width must be a float.')

    if width <= 0:
        raise ValueError('width must be a positive float greater than 0.')

    kernalised_distance = np.sqrt(np.exp(-(distances ** 2) / width ** 2))

    return kernalised_distance
