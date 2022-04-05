"""
The :mod:`fatf.utils.models.processing` module implements model processing
functions.

.. versionadded:: 0.1.1
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import logging
from typing import Callable

import numpy as np

from fatf.exceptions import IncorrectShapeError

import fatf.utils.array.validation as fuav
import fatf.utils.validation as fuv

__all__ = ['batch_data']

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def batch_data(data: np.ndarray,
               batch_size: int = 50,
               transformation_fn: Callable = None) -> np.ndarray:
    """
    Slices ``data`` into batches and returns then sequentially.

    .. versionadded:: 0.1.1

    Since some data may be too large to fit into memory as whole,
    this function slices them into batches and yields them sequentially.
    If desired, each batch can be processed by ``transformation_fn``
    prior to returning it.

    Parameters
    ----------
    data : numpy.ndarray
        A two dimensional numpy array (either classic or structured) to be
        sliced into batches.
    batch_size : integer, optional (default=50)
        The size (number of rows) of each batch.
    transformation_fn : callable, optional (default=None)
        A callable object to apply to each batch before returning it.
        It must have exactly one required parameter.

    Raises
    ------
    IncorrectShapeError
        The ``data`` array is not 2-dimensional.
    RuntimeError
        The transformation function does not have exactly one required
        parameter.
    TypeError
        The ``batch_size`` is not an integer or the ``transformation_fn`` is
        not a callable object.
    ValueError
        The ``batch_size`` is smaller than 1.

    Yields
    ------
    slice : numpy.ndarray
        A slice of data.
    """
    if not fuav.is_2d_array(data):
        raise IncorrectShapeError('The data array must be 2-dimensional.')
    if fuav.is_structured_array(data):
        slice_fn = lambda d, a, b: d[a:b]  # noqa: E731
    else:
        slice_fn = lambda d, a, b: d[a:b, :]  # noqa: E731

    if not isinstance(batch_size, int):
        raise TypeError('The batch size must be an integer.')
    if batch_size < 1:
        raise ValueError('The batch size must be larger than 0.')

    if transformation_fn is None:
        transformation_fn = lambda slice: slice  # noqa: E731
    else:
        if not callable(transformation_fn):
            raise TypeError(
                'The transformation function must be a callable object.')
        required_params = fuv.get_required_parameters_number(transformation_fn)
        if required_params != 1:
            raise RuntimeError(
                'The transformation function must have only one required '
                'parameter; now it has {}.'.format(required_params))

    n_rows = data.shape[0]

    def _batch_data():
        for i_start in np.arange(0, n_rows, batch_size):
            i_end = np.min([i_start + batch_size, n_rows])
            data_slice_ = slice_fn(data, i_start, i_end)
            data_slice = transformation_fn(data_slice_)
            yield data_slice

    return _batch_data()
