"""
.. versionadded:: 0.0.2

The :mod:`fatf.utils.data.transformation` module holds data transformation
functions.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
#         Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

from typing import Union

import numpy as np

import fatf.utils.array.validation as fuav

from fatf.exceptions import IncorrectShapeError

__all__ = ['dataset_row_masking']


def _validate_input_drm(dataset: np.ndarray,
                        data_row: Union[np.ndarray, np.void]) -> bool:
    """
    Validates :func:`fatf.utils.data.transformation.dataset_row_masking` input.

    This function checks if ``dataset`` is a 2-dimensional array and if
    ``data_row`` is a 1-dimensional array of the same length as the number of
    columns in the ``dataset``. It also checks if they have valid and
    compatible dtypes.

    For the description of input parameters, and warnings and exceptions raised
    by this function please see the  documentation of the
    :func:`fatf.utils.data.transformation.dataset_row_masking` function.

    Returns
    -------
    is_valid : boolean
        ``True`` if input is valid, ``False`` otherwise.
    """
    is_valid = False

    if not fuav.is_2d_array(dataset):
        raise IncorrectShapeError('The input dataset must be a 2-dimensional '
                                  'numpy array.')
    if not fuav.is_base_array(dataset):
        raise TypeError('The input dataset must be of a base type -- text, '
                        'numbers or mixture of the two.')

    if not fuav.is_1d_like(data_row):
        raise IncorrectShapeError('The data row must either be a '
                                  '1-dimensional numpy array or a numpy void '
                                  'object for structured rows.')

    # For structured arrays the dtype check also checks the number of columns
    are_similar = fuav.are_similar_dtype_arrays(
        dataset, np.array([data_row]), strict_comparison=False)
    if not are_similar:
        raise TypeError('The dtype of the data row is too different from the '
                        'dtype of the dataset provided.')

    # Since the types agree both, the row and the data set, have to be
    # structured or plane
    if not fuav.is_structured_array(dataset):
        if dataset.shape[1] != data_row.shape[0]:
            raise IncorrectShapeError('The data row must contain the same '
                                      'number of elements as the number of '
                                      'columns in the provided dataset.')

    is_valid = True
    return is_valid


def dataset_row_masking(dataset: np.ndarray,
                        data_row: Union[np.ndarray, np.void]) -> np.ndarray:
    """
    Creates a binary representation of the ``dataset`` by masking its rows.

    .. versionadded:: 0.0.2

    The rows of the ``dataset`` array are compared against specified
    ``data_row`` to determine which features values are the same and which are
    different. The same values are represented as ``1`` in the binary output
    and different ones are indicated by ``0``.

    For a ``['a', 'b']`` ``data_row`` and
    ``[['x', 'b'], ['a', 'b'], ['a', 'x']]`` ``dataset`` the binary
    representation would be ``[[0, 1], [1, 1], [1, 0]]``.

    Parameters
    ----------
    dataset : numpy.ndarray
        A 2-dimensional numpy array used to generate the binary representation.
    data_row : Union[numpy.ndarray, numpy.void]
        A 1-dimensional numpy array for unstructured arrays or numpy void for
        structured rows containing feature values that will be compared against
        the ``dataset`` rows.

    Raises
    ------
    IncorrectShapeError
        The ``dataset`` is not a 2-dimensional array or ``data_row`` is not a
        1-dimensional array. The length of the ``data_row`` is different to the
        number of columns in the ``dataset``.
    TypeError
        The ``dataset`` is not of a base type or the ``data_row``\\ 's dtype is
        too different from the ``dataset``\\ 's dtype.

    Returns
    -------
    binary_representation : numpy.ndarray
        A binary (0's and 1's in an array of ``numpy.int8`` type)
        representation of the ``dataset`` (with the same shape as ``dataset``)
        achieved by "masking" it with the ``data_row``.
    """
    assert _validate_input_drm(dataset, data_row), 'Input is not valid.'

    if fuav.is_structured_array(dataset):
        dtypes = [(name, np.int8) for name in dataset.dtype.names]
        binary_representation = np.zeros_like(dataset, dtype=dtypes)
        for index in dataset.dtype.names:
            # E1337 is unsupported-assignment-operation
            binary_representation[index] = (  # pylint: disable=E1137
                dataset[index] == data_row[index])
    else:
        binary_representation = (dataset == data_row).astype(np.int8)

    return binary_representation
