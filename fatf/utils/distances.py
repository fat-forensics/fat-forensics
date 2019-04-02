"""
Holds custom distance functions used for FAT-Forensics examples and testing.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import logging

from typing import Union

import numpy as np

import fatf.utils.array.tools as fuat
import fatf.utils.array.validation as fuav

from fatf.exceptions import IncorrectShapeError

__all__ = ['euclidean_distance',
           'euclidean_point_distance',
           'euclidean_array_distance',
           'hamming_distance_base',
           'hamming_distance',
           'hamming_point_distance',
           'hamming_array_distance',
           'binary_distance',
           'binary_point_distance',
           'binary_array_distance']  # yapf: disable

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def euclidean_distance(x: Union[np.ndarray, np.void],
                       y: Union[np.ndarray, np.void]) -> float:
    """
    Calculates the Euclidean distance between two 1-dimensional numpy "arrays".

    Each of the input arrays can be either a 1D numpy array or a row of a
    structured numpy array, i.e. numpy's void.

    Parameters
    ----------
    x : Union[numpy.ndarray, numpy.void]
        The first numpy array (has to be 1-dimensional and purely numerical).
    y : Union[numpy.ndarray, numpy.void]
        The second numpy array (has to be 1-dimensional and purely numerical).

    Raises
    ------
    IncorrectShapeError
        Either of the input arrays is not 1-dimensional or they are not of the
        same length.
    ValueError
        Either of the input arrays is not purely numerical.

    Returns
    -------
    distance : float
        Euclidean distance between the two numpy arrays.
    """
    # pylint: disable=invalid-name
    if not fuav.is_1d_like(x):
        raise IncorrectShapeError('The x array should be 1-dimensional.')
    if not fuav.is_1d_like(y):
        raise IncorrectShapeError('The y array should be 1-dimensional.')

    # Transform the arrays to unstructured
    x_array = fuat.as_unstructured(x)
    y_array = fuat.as_unstructured(y)

    if not fuav.is_numerical_array(x_array):
        raise ValueError('The x array should be purely numerical.')
    if not fuav.is_numerical_array(y_array):
        raise ValueError('The y array should be purely numerical.')

    if x_array.shape[0] != y_array.shape[0]:
        raise IncorrectShapeError(('The x and y arrays should have the same '
                                   'length.'))

    distance = np.linalg.norm(x_array - y_array)
    return distance


def euclidean_point_distance(y: Union[np.ndarray, np.void],
                             X: np.ndarray) -> np.ndarray:
    """
    Calculates the Euclidean distance between ``y`` and every row of ``X``.

    ``y`` has to be a 1-dimensional numerical numpy array or a row of a
    structured numpy array (i.e. numpy's void) and ``X`` has to be a
    2-dimensional numerical numpy array. The length of ``y`` has to be the same
    as the width of ``X``.

    Parameters
    ----------
    y : Union[numpy.ndarray, numpy.void]
        A numpy array (has to be 1-dimensional and purely numerical) used to
        calculate distances from.
    X : numpy.ndarray
        A numpy array (has to be 2-dimensional and purely numerical) to which
        rows distances are calculated.

    Raises
    ------
    IncorrectShapeError
        Either ``y`` is not 1-dimensional or ``X`` is not 2-dimensional or the
        length of ``y`` is not equal to the number of columns in ``X``.
    ValueError
        Either of the input arrays is not purely numerical.

    Returns
    -------
    distances : numpy.ndarray
        An array of Euclidean distances between ``y`` and every row of ``X``.
    """
    # pylint: disable=invalid-name
    if not fuav.is_1d_like(y):
        raise IncorrectShapeError('The y array should be 1-dimensional.')
    if not fuav.is_2d_array(X):
        raise IncorrectShapeError('The X array should be 2-dimensional.')

    # Transform the arrays to unstructured
    y_array = fuat.as_unstructured(y)
    X_array = fuat.as_unstructured(X)  # pylint: disable=invalid-name

    if not fuav.is_numerical_array(y_array):
        raise ValueError('The y array should be purely numerical.')
    if not fuav.is_numerical_array(X_array):
        raise ValueError('The X array should be purely numerical.')

    # Compare shapes
    if y_array.shape[0] != X_array.shape[1]:
        raise IncorrectShapeError('The number of columns in the X array '
                                  'should the same as the number of elements '
                                  'in the y array.')

    distances = np.apply_along_axis(euclidean_distance, 1, X_array, y_array)
    return distances


def euclidean_array_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Calculates the Euclidean distance matrix between rows in ``X`` and ``Y``.

    Both ``X`` and ``Y`` have to be 2-dimensional numerical numpy arrays of the
    same width.

    Parameters
    ----------
    X : numpy.ndarray
        A numpy array -- has to be 2-dimensional and purely numerical.
    Y : numpy.ndarray
        A numpy array -- has to be 2-dimensional and purely numerical.

    Raises
    ------
    IncorrectShapeError
        Either ``X`` or ``Y`` is not 2-dimensional or ``X`` and ``Y`` do not
        have the same number of columns.
    ValueError
        Either of the input arrays is not purely numerical.

    Returns
    -------
    distance_matrix : numpy.ndarray
        An matrix of Euclidean distances between rows in ``X` and ``Y``.
    """
    # pylint: disable=invalid-name
    if not fuav.is_2d_array(X):
        raise IncorrectShapeError('The X array should be 2-dimensional.')
    if not fuav.is_2d_array(Y):
        raise IncorrectShapeError('The Y array should be 2-dimensional.')

    if not fuav.is_numerical_array(X):
        raise ValueError('The X array should be purely numerical.')
    if not fuav.is_numerical_array(Y):
        raise ValueError('The Y array should be purely numerical.')

    # Transform the arrays to unstructured
    Y_array = fuat.as_unstructured(Y)  # pylint: disable=invalid-name
    X_array = fuat.as_unstructured(X)  # pylint: disable=invalid-name

    # Compare shapes
    if Y_array.shape[1] != X_array.shape[1]:
        raise IncorrectShapeError('The number of columns in the X array '
                                  'should the same as the number of columns '
                                  'in Y array.')

    distance_matrix = np.apply_along_axis(euclidean_point_distance, 1, X_array,
                                          Y_array)

    return distance_matrix


def hamming_distance_base(x: str,
                          y: str,
                          normalise: bool = False,
                          equal_length: bool = False) -> Union[int, float]:
    """
    Calculates the Hamming distance between two strings ``x`` and ``y``.

    If the strings are of a different length they are compared up to the
    shorter one's length and the distance between them is increased by their
    difference in length.

    Parameters
    ----------
    x : string
        The first string to be compared.
    y : string
        The second string to be compared.
    normalise : boolean, optional (default=False)
        Normalises the distance to be bounded between 0 and 1.
    equal_length : boolean, optional (default=False)
        Forces the arrays to be of equal length -- raises exception if they are
        not.

    Raises
    ------
    TypeError
        Either ``x`` or ``y`` is not a string.
    ValueError
        ``x`` and ``y`` are of different length when ``equal_length`` parameter
        is set to ``True``.

    Returns
    -------
    distance : Union[integer, float]
        The Hamming distances between ``x` and ``y``.
    """
    # pylint: disable=invalid-name
    if not isinstance(x, str):
        raise TypeError('x should be a string.')
    if not isinstance(y, str):
        raise TypeError('y should be a string.')

    x_len = len(x)
    y_len = len(y)

    distance = abs(x_len - y_len)
    if distance and equal_length:
        raise ValueError('Input strings differ in length and the equal_length '
                         'parameter forces them to be of equal length.')
    elif distance:
        min_index = min(x_len, y_len)
        for i in range(min_index):
            distance += 0 if x[i] == y[i] else 1
    else:
        assert x_len == y_len, 'The strings should be of equal length.'
        for i in range(x_len):
            distance += 0 if x[i] == y[i] else 1

    if normalise:
        logger.debug('Hamming distance is being normalised.')
        distance /= max(x_len, y_len)  # type: ignore

    return distance


def hamming_distance(x: Union[np.ndarray, np.void],
                     y: Union[np.ndarray, np.void],
                     **kwargs: bool) -> Union[int, float]:
    """
    Computes the Hamming distance between 1-dimensional non-numerical arrays.

    Each of the input arrays can be either a 1D numpy array or a row of a
    structured numpy array, i.e. numpy's void.

    Parameters
    ----------
    x : Union[numpy.ndarray, numpy.void]
        The first numpy array (has to be 1-dimensional and non-numerical).
    y : Union[numpy.ndarray, numpy.void]
        The second numpy array (has to be 1-dimensional and non-numerical).
    **kwargs : boolean
        Keyword arguments that are passed to the
        :func:`fatf.utils.distances.hamming_distance_base` function responsible
        for calculating the Hamming distance.

    Raises
    ------
    IncorrectShapeError
        Either of the input arrays is not 1-dimensional or they are of a
        different length.
    ValueError
        Either of the input arrays is not purely textual.

    Returns
    -------
    distance : Union[integer, float]
        Hamming distance between the two numpy arrays.
    """
    # pylint: disable=invalid-name
    if not fuav.is_1d_like(x):
        raise IncorrectShapeError('The x array should be 1-dimensional.')
    if not fuav.is_1d_like(y):
        raise IncorrectShapeError('The y array should be 1-dimensional.')

    # Transform the arrays to unstructured
    x_array = fuat.as_unstructured(x)
    y_array = fuat.as_unstructured(y)

    if not fuav.is_textual_array(x_array):
        raise ValueError('The x array should be textual.')
    if not fuav.is_textual_array(y_array):
        raise ValueError('The y array should be textual.')

    if x_array.shape[0] != y_array.shape[0]:
        raise IncorrectShapeError('The x and y arrays should have the same '
                                  'length.')

    def kw_hamming_distance(vec):
        return hamming_distance_base(vec[0], vec[1], **kwargs)

    distance = np.apply_along_axis(kw_hamming_distance, 0,
                                   np.vstack((x_array, y_array)))
    distance = distance.sum()
    return distance


def hamming_point_distance(y: Union[np.ndarray, np.void], X: np.ndarray,
                           **kwargs: bool) -> np.ndarray:
    """
    Calculates the Hamming distance between ``y`` and every row of ``X``.

    ``y`` has to be a 1-dimensional numerical numpy array or a row of a
    structured numpy array (i.e. numpy's void) and ``X`` has to be a
    2-dimensional numerical numpy array. The length of ``y`` has to be the same
    as the width of ``X``.

    Parameters
    ----------
    y : Union[numpy.ndarray, numpy.void]
        A numpy array (has to be 1-dimensional and non-numerical) used to
        calculate the distances from.
    X : numpy.ndarray
        A numpy array (has to be 2-dimensional and non-numerical) to which
        rows the distances are calculated.
    **kwargs : boolean
        Keyword arguments that are passed to the
        :func:`fatf.utils.distances.hamming_distance_base` function responsible
        for calculating the Hamming distance.

    Raises
    ------
    IncorrectShapeError
        Either ``y`` is not 1-dimensional or ``X`` is not 2-dimensional or the
        length of ``y`` is not equal to the number of columns in ``X``.
    ValueError
        Either of the input arrays is not purely textual.

    Returns
    -------
    distances : numpy.ndarray
        An array of Hamming distances between ``y`` and every row of ``X``.
    """
    # pylint: disable=invalid-name
    if not fuav.is_1d_like(y):
        raise IncorrectShapeError('The y array should be 1-dimensional.')
    if not fuav.is_2d_array(X):
        raise IncorrectShapeError('The X array should be 2-dimensional.')

    # Transform the arrays to unstructured
    y_array = fuat.as_unstructured(y)
    X_array = fuat.as_unstructured(X)  # pylint: disable=invalid-name

    if not fuav.is_textual_array(y_array):
        raise ValueError('The y array should be textual.')
    if not fuav.is_textual_array(X_array):
        raise ValueError('The X array should be textual.')

    # Compare shapes
    if y_array.shape[0] != X_array.shape[1]:
        raise IncorrectShapeError('The number of columns in the X array '
                                  'should the same as the number of elements '
                                  'in the y array.')

    distances = np.apply_along_axis(hamming_distance, 1, X_array, y_array,
                                    **kwargs)
    return distances


def hamming_array_distance(X: np.ndarray, Y: np.ndarray,
                           **kwargs: bool) -> np.ndarray:
    """
    Calculates the Hamming distance matrix between rows in ``X`` and ``Y``.

    Both ``X`` and ``Y`` have to be 2-dimensional numerical numpy arrays of the
    same width.

    Parameters
    ----------
    X : numpy.ndarray
        A numpy array -- has to be 2-dimensional and non-numerical.
    Y : numpy.ndarray
        A numpy array -- has to be 2-dimensional and non-numerical.
    **kwargs : boolean
        Keyword arguments that are passed to the
        :func:`fatf.utils.distances.hamming_distance_base` function responsible
        for calculating the Hamming distance.

    Raises
    ------
    IncorrectShapeError
        Either ``X`` or ``Y`` is not 2-dimensional or ``X`` and ``Y`` do not
        have the same number of columns.
    ValueError
        Either of the input arrays is not purely textual.

    Returns
    -------
    distance_matrix : numpy.ndarray
        An matrix of Hamming distances between rows in ``X` and ``Y``.
    """
    # pylint: disable=invalid-name
    if not fuav.is_2d_array(X):
        raise IncorrectShapeError('The X array should be 2-dimensional.')
    if not fuav.is_2d_array(Y):
        raise IncorrectShapeError('The Y array should be 2-dimensional.')

    if not fuav.is_textual_array(X):
        raise ValueError('The X array should be textual.')
    if not fuav.is_textual_array(Y):
        raise ValueError('The Y array should be textual.')

    # Transform the arrays to unstructured
    X_array = fuat.as_unstructured(X)  # pylint: disable=invalid-name
    Y_array = fuat.as_unstructured(Y)  # pylint: disable=invalid-name

    # Compare shapes
    if X_array.shape[1] != Y_array.shape[1]:
        raise IncorrectShapeError('The number of columns in the X array '
                                  'should the same as the number of columns '
                                  'in Y array.')

    distance_matrix = np.apply_along_axis(hamming_point_distance, 1, X_array,
                                          Y_array, **kwargs)
    return distance_matrix


def binary_distance(x: Union[np.ndarray, np.void],
                    y: Union[np.ndarray, np.void],
                    normalise: bool = False) -> Union[int, float]:
    """
    Computes the binary distance between two 1-dimensional arrays.

    The distance is incremented by one for every position in the two input
    arrays where the value does not match. Each of the input arrays can be
    either a 1D numpy array or a row of a structured numpy array, i.e. numpy's
    void.

    Either of the input arrays is not of a base dtype. (See
    :func:`fatf.utils.array.validation.is_base_array` function description for
    the explanation of a base dtype.)

    Parameters
    ----------
    x : Union[numpy.ndarray, numpy.void]
        The first numpy array (has to be 1-dimensional).
    y : Union[numpy.ndarray, numpy.void]
        The second numpy array (has to be 1-dimensional).
    normalise : boolean, optional (default=False)
        Whether to normalise the binary distance using the input array length.

    Raises
    ------
    IncorrectShapeError
        Either of the input arrays is not 1-dimensional or they are of a
        different length.

    Returns
    -------
    distance : Union[integer, float]
        Binary distance between the two numpy arrays.
    """
    # pylint: disable=invalid-name
    if not fuav.is_1d_like(x):
        raise IncorrectShapeError('The x array should be 1-dimensional.')
    if not fuav.is_1d_like(y):
        raise IncorrectShapeError('The y array should be 1-dimensional.')

    # Transform the arrays to unstructured
    x_array = fuat.as_unstructured(x)
    y_array = fuat.as_unstructured(y)

    if x_array.shape[0] != y_array.shape[0]:
        raise IncorrectShapeError('The x and y arrays should have the same '
                                  'length.')

    distance = (x_array != y_array).sum()
    if normalise:
        logger.debug('Binary distance is being normalised.')
        distance /= x_array.shape[0]
    return distance


def binary_point_distance(y: Union[np.ndarray, np.void], X: np.ndarray,
                          **kwargs: bool) -> np.ndarray:
    """
    Calculates the binary distance between ``y`` and every row of ``X``.

    ``y`` has to be a 1-dimensional numpy array or a row of a structured numpy
    array (i.e. numpy's void) and ``X`` has to be a 2-dimensional numpy array.
    The length of ``y`` has to be the same as the width of ``X``.

    Either of the input arrays is not of a base dtype. (See
    :func:`fatf.utils.array.validation.is_base_array` function description for
    the explanation of a base dtype.)

    Parameters
    ----------
    y : Union[numpy.ndarray, numpy.void]
        A numpy array (has to be 1-dimensional) used to calculate the distances
        from.
    X : numpy.ndarray
        A numpy array (has to be 2-dimensional) to which rows the distances are
        calculated.
    **kwargs : boolean
        Keyword arguments that are passed to the
        :func:`fatf.utils.distances.binary_distance` function responsible for
        calculating the binary distance.

    Raises
    ------
    IncorrectShapeError
        Either ``y`` is not 1-dimensional or ``X`` is not 2-dimensional or the
        length of ``y`` is not equal to the number of columns in ``X``.

    Returns
    -------
    distances : numpy.ndarray
        An array of binary distances between ``y`` and every row of ``X``.
    """
    # pylint: disable=invalid-name
    if not fuav.is_1d_like(y):
        raise IncorrectShapeError('The y array should be 1-dimensional.')
    if not fuav.is_2d_array(X):
        raise IncorrectShapeError('The X array should be 2-dimensional.')

    # Transform the arrays to unstructured
    y_array = fuat.as_unstructured(y)
    X_array = fuat.as_unstructured(X)  # pylint: disable=invalid-name

    # Compare shapes
    if y_array.shape[0] != X_array.shape[1]:
        raise IncorrectShapeError('The number of columns in the X array '
                                  'should the same as the number of elements '
                                  'in the y array.')

    distances = np.apply_along_axis(binary_distance, 1, X_array, y_array,
                                    **kwargs)
    return distances


def binary_array_distance(X: np.ndarray, Y: np.ndarray,
                          **kwargs: bool) -> np.ndarray:
    """
    Calculates the binary distance matrix between rows in ``X`` and ``Y``.

    Both ``X`` and ``Y`` have to be 2-dimensional numpy arrays of the same
    width.

    Either of the input arrays is not of a base dtype. (See
    :func:`fatf.utils.array.validation.is_base_array` function description for
    the explanation of a base dtype.)

    Parameters
    ----------
    X : numpy.ndarray
        A numpy array -- has to be 2-dimensional.
    Y : numpy.ndarray
        A numpy array -- has to be 2-dimensional.
    **kwargs : boolean
        Keyword arguments that are passed to the
        :func:`fatf.utils.distances.binary_distance` function responsible for
        calculating the binary distance.

    Raises
    ------
    IncorrectShapeError
        Either ``X`` or ``Y`` is not 2-dimensional or ``X`` and ``Y`` do not
        have the same number of columns.

    Returns
    -------
    distance_matrix : numpy.ndarray
        An matrix of binary distances between rows in ``X` and ``Y``.
    """
    # pylint: disable=invalid-name
    if not fuav.is_2d_array(X):
        raise IncorrectShapeError('The X array should be 2-dimensional.')
    if not fuav.is_2d_array(Y):
        raise IncorrectShapeError('The Y array should be 2-dimensional.')

    # Transform the arrays to unstructured
    X_array = fuat.as_unstructured(X)
    Y_array = fuat.as_unstructured(Y)

    # Compare shapes
    if X_array.shape[1] != Y_array.shape[1]:
        raise IncorrectShapeError('The number of columns in the X array '
                                  'should the same as the number of columns '
                                  'in Y array.')

    distance_matrix = np.apply_along_axis(binary_point_distance, 1, X_array,
                                          Y_array, **kwargs)
    return distance_matrix
