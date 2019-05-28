"""
Sampler functions that only rely on instance
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
#         Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import numpy as np

import fatf.utils.array.validation as fuav
from fatf.exceptions import IncorrectShapeError

__all__ = ['binary_sampler']


def _validate_input(data_row: np.ndarray,
                    samples_number: int) -> bool:
    """
    Validates input parameters of the instance sampler methods.

    This function checks the validity of ``data_row`` and
    ``samples_number`` parameters.

    Raises
    ------
    IncorrectShapeError
        The ``data_row`` is not a 1-dimensional numpy array-like object.
    ValueError
        The ``samples_number`` parameter is not a positive integer.

    Returns
    -------
    is_valid : boolean
        ``True`` if input parameters are valid, ``False`` otherwise.
        """
    is_valid = False

    if data_row is not None:
        if not fuav.is_1d_like(data_row):
            raise IncorrectShapeError('The data_row must either be a '
                                        '1-dimensional numpy array or numpy '
                                        'void object for structured rows.')

    if isinstance(samples_number, int):
        if samples_number < 1:
            raise ValueError('The samples_number parameter must be a '
                             'positive integer.')
    else:
        raise TypeError('The samples_number parameter must be an integer.')

    is_valid = True
    return is_valid


def binary_sampler(data_row: np.ndarray,
                   samples_number: int = 50) -> np.ndarray:
    """
    Uniformally samples non-zero values in binary array.

    Uniformally samples a number of binary masks and applies them to
    ``data_row``, setting a selection of non-zero elements in ``data_row`` to
    zero. This method for sampling is described in the original LIME paper.

    For the additional documentation of errors please see the  description of
    the
    :func:`~fatf.utils.data.instance_sampler._validate_input` method.

    Raises
    ------
    TypeError
        ``data_row`` is not a binary row.

    Returns
    -------
    samples : numpy.ndarray
        Sampled data.
    """
    assert _validate_input(data_row, samples_number), 'Input is invalid.'

    is_structured = isinstance(data_row, np.void)

    if is_structured:
        indices = data_row.dtype.names
        shape = (samples_number, )  # type: Tuple[int, ...]
    else:
        indices = np.arange(0, data_row.shape[0], 1)
        shape = (samples_number, data_row.shape[0])

    # Test if data_row contains only 0, 1 as ints or floats
    if not set(np.unique(list(data_row))) <= {0, 1, 0., 1.}:
        raise ValueError('data_row is not a binary row.')

    samples = np.zeros(shape, dtype=data_row.dtype)

    for index in indices:
        column_mask = np.random.choice([0, 1], size=(samples_number, ))
        if is_structured:
            samples[index] = column_mask * data_row[index]
        else:
            samples[:, index] = column_mask * data_row[index]

    return samples
