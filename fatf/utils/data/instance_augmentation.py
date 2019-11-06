"""
.. versionadded:: 0.0.2

The :mod:`fatf.utils.data.instance_augmentation` module implements various
augmentation function for 1-dimensional numpy array-like objects.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
#         Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

from typing import Tuple  # pylint: disable=unused-import
from typing import Union

import numpy as np

import fatf.utils.array.validation as fuav

from fatf.exceptions import IncorrectShapeError

__all__ = ['binary_sampler']


def _validate_input(data_row: Union[np.ndarray, np.void],
                    samples_number: int) -> bool:
    """
    Validates input parameters of an instance sampler function.

    This function validates ``data_row`` and ``samples_number`` input
    parameters. For the description of input parameters and errors please see
    the documentation of the
    :func:`fatf.utils.data.instance_augmentation.binary_sampler` function.

    Returns
    -------
    is_valid : boolean
        ``True`` if input parameters are valid, ``False`` otherwise.
    """
    is_valid = False

    if not fuav.is_1d_like(data_row):
        raise IncorrectShapeError('The data_row must either be a '
                                  '1-dimensional numpy array or a numpy '
                                  'void object for structured rows.')

    if isinstance(samples_number, int):
        if samples_number < 1:
            raise ValueError('The samples_number parameter must be a positive '
                             'integer.')
    else:
        raise TypeError('The samples_number parameter must be an integer.')

    is_valid = True
    return is_valid


def binary_sampler(data_row: Union[np.ndarray, np.void],
                   samples_number: int = 50) -> np.ndarray:
    """
    Samples non-zero elements of the binary ``data_row`` array uniformly.

    .. versionadded:: 0.0.2

    Uniformly samples all of the features that have non-zero (i.e., 1) value in
    the input ``data_row`` from a {0, 1} set. For example, for a
    ``[0, 1, 1, 0]`` ``data_row``, only the two middle features will be
    sampled. Therefore, all of the possible data rows in the output sample have
    to be a subset of: ``[0, 1, 1, 0]``, ``[0, 1, 0, 0]``, ``[0, 0, 1, 0]`` and
    ``[0, 0, 0, 0]`` arrays.

    Raises
    ------
    IncorrectShapeError
        The ``data_row`` is not a 1-dimensional numpy array-like object.
    TypeError
        The ``data_row`` is not a binary array. The ``samples_number`` is not
        an integer.
    ValueError
        The ``samples_number`` is not a positive integer.

    Returns
    -------
    binary_samples : numpy.ndarray
        Binary data sampled based on the input ``data_row``.
    """
    assert _validate_input(data_row, samples_number), 'Input is invalid.'

    # Test if the data_row is binary
    unique_elements = set(np.unique(data_row.tolist()))
    if not unique_elements.issubset([0, 1, 0., 1., False, True]):
        raise ValueError('The data_row is not binary.')

    is_structured = fuav.is_structured_array(np.asarray(data_row))

    if is_structured:
        column_indices = data_row.dtype.names
        output_shape = (samples_number, )  # type: Tuple[int, ...]
    else:
        column_indices = range(data_row.shape[0])
        output_shape = (samples_number, data_row.shape[0])

    binary_samples = np.zeros(output_shape, dtype=data_row.dtype)

    for column_index in column_indices:
        column_mask = np.random.choice([0, 1], size=(samples_number, ))
        column_values = data_row[column_index] * column_mask
        if is_structured:
            binary_samples[column_index] = column_values
        else:
            binary_samples[:, column_index] = column_values

    return binary_samples
