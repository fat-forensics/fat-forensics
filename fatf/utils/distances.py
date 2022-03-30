"""
The :mod:`fatf.utils.distances` module holds a variety of distance metrics.

The distance metrics and tools implemented in this module are mainly used for
the :class:`fatf.utils.models.models.KNN` model implementation, to measure
distance (and similarity) of data points for various functions in this package
as well as for documentation examples and testing.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import inspect
import logging
import warnings

from typing import Callable, Union

import numpy as np

import fatf.utils.array.tools as fuat
import fatf.utils.array.validation as fuav
import fatf.utils.validation as fuv

from fatf.exceptions import IncorrectShapeError

__all__ = ['get_distance_matrix',
           'get_point_distance',
           'euclidean_distance',
           'euclidean_point_distance',
           'euclidean_array_distance',
           'hamming_distance_base',
           'hamming_distance',
           'hamming_point_distance',
           'hamming_array_distance',
           'binary_distance',
           'binary_point_distance',
           'binary_array_distance',
           'check_distance_functionality']  # yapf: disable

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def _validate_get_distance(
        data_array: np.ndarray,
        distance_function: Callable[[np.ndarray, np.ndarray], float]) -> bool:
    """
    Validates ``data_array`` and ``distance_function`` parameters.

    Parameters
    ----------
    data_array : numpy.ndarray
        A 2-dimensional numpy array.
    distance_function : Callable[[numpy.ndarray, numpy.ndarray], number]
        A Python function that takes as an input two 1-dimensional numpy arrays
        of equal length and outputs a number representing a distance between
        them.

    Raises
    ------
    AttributeError
        The distance function does not require exactly two parameters.
    IncorrectShapeError
        The data array is not a 2-dimensional numpy array.
    TypeError
        The data array is not of a base type (numbers and/or strings). The
        distance function is not a Python callable (function).

    Returns
    -------
    is_valid : boolean
        ``True`` if the parameters are valid, ``False`` otherwise.
    """
    is_valid = False

    if not fuav.is_2d_array(data_array):
        raise IncorrectShapeError('The data_array has to be a 2-dimensional '
                                  '(structured or unstructured) numpy array.')
    if not fuav.is_base_array(data_array):
        raise TypeError('The data_array has to be of a base type (strings '
                        'and/or numbers).')

    if callable(distance_function):
        required_param_n = 0
        params = inspect.signature(distance_function).parameters
        for param in params:
            if params[param].default is params[param].empty:
                required_param_n += 1
        if required_param_n != 2:
            raise AttributeError('The distance function must require exactly '
                                 '2 parameters. Given function requires {} '
                                 'parameters.'.format(required_param_n))
    else:
        raise TypeError('The distance function should be a Python callable '
                        '(function).')

    is_valid = True
    return is_valid


def get_distance_matrix(
        data_array: np.ndarray,
        distance_function: Callable[[np.ndarray, np.ndarray], float]
) -> np.ndarray:
    """
    Computes a distance matrix (2-D) between all rows of the ``data_array``.

    Parameters
    ----------
    data_array : numpy.ndarray
        A 2-dimensional numpy array for which row-to-row distances will be
        computed.
    distance_function : Callable[[numpy.ndarray, numpy.ndarray], number]
        A Python function that takes as an input two 1-dimensional numpy arrays
        of equal length and outputs a number representing a distance between
        them. **The distance function is assumed to return the same distance
        regardless of the order in which parameters are given.**

    Raises
    ------
    AttributeError
        The distance function does not require exactly two parameters.
    IncorrectShapeError
        The data array is not a 2-dimensional numpy array.
    TypeError
        The data array is not of a base type (numbers and/or strings). The
        distance function is not a Python callable (function).

    Returns
    -------
    distances : numpy.ndarray
        A square numerical numpy array with distances between all pairs of data
        points (rows) in the ``data_array``.
    """
    assert _validate_get_distance(data_array,
                                  distance_function), 'Invalid input.'

    if fuav.is_structured_array(data_array):
        distances = np.zeros((data_array.shape[0], data_array.shape[0]),
                             dtype=np.float64)
        for row_i in range(data_array.shape[0]):
            for row_j in range(row_i, data_array.shape[0]):
                dist = distance_function(data_array[row_i], data_array[row_j])
                distances[row_i, row_j] = dist
                distances[row_j, row_i] = dist
    else:

        def ddf(one_d, two_d):
            return np.apply_along_axis(distance_function, 1, two_d, one_d)

        distances = np.apply_along_axis(ddf, 1, data_array, data_array)

    return distances


def get_point_distance(
        data_array: np.ndarray, data_point: Union[np.ndarray, np.void],
        distance_function: Callable[[np.ndarray, np.ndarray], float]
) -> np.ndarray:
    """
    Computes the distance between a data point and an array of data.

    This function computes the distances between the ``data_point`` and all
    rows of the ``data_array``.

    Parameters
    ----------
    data_array : numpy.ndarray
        A 2-dimensional numpy array to which rows distances will be computed.
    data_point : Union[numpy.ndarray, numpy.void]
        A 1-dimensional numpy array or numpy void (for structured data points)
        for which distances to every row of the ``data_array`` will be
        computed.
    distance_function : Callable[[numpy.ndarray, numpy.ndarray], number]
        A Python function that takes as an input two 1-dimensional numpy arrays
        of equal length and outputs a number representing a distance between
        them. **The distance function is assumed to return the same distance
        regardless of the order in which parameters are given.**

    Raises
    ------
    AttributeError
        The distance function does not require exactly two parameters.
    IncorrectShapeError
        The data array is not a 2-dimensional numpy array. The data point is
        not 1-dimensional. The number of columns in the data array is different
        to the number of elements in the data point.
    TypeError
        The data array or the data point is not of a base type (numbers and/or
        strings). The data point and the data array have incomparable dtypes.
        The distance function is not a Python callable (function).

    Returns
    -------
    distances : numpy.ndarray
        A 1-dimensional numerical numpy array with distances between
        ``data_point`` and every row of the ``data_array``.
    """
    assert _validate_get_distance(data_array,
                                  distance_function), 'Invalid input.'

    is_structured = fuav.is_structured_array(data_array)

    if not fuav.is_1d_like(data_point):
        raise IncorrectShapeError('The data point has to be 1-dimensional '
                                  'numpy array or numpy void (for structured '
                                  'arrays).')
    data_point_array = np.asarray([data_point])
    if not fuav.is_base_array(data_point_array):
        raise TypeError('The data point has to be of a base type (strings '
                        'and/or numbers).')
    if not fuav.are_similar_dtype_arrays(data_array, data_point_array):
        raise TypeError('The dtypes of the data set and the data point are '
                        'too different.')
    # Testing only for unstructured as the dtype comparison picks up on a
    # different number of columns in a structured array
    if not is_structured:
        if data_array.shape[1] != data_point_array.shape[1]:
            raise IncorrectShapeError('The data point has different number of '
                                      'columns (features) than the data set.')

    if is_structured:
        distances = np.zeros((data_array.shape[0], ), dtype=np.float64)
        for row_i in range(data_array.shape[0]):
            distances[row_i] = distance_function(data_array[row_i], data_point)
    else:
        distances = np.apply_along_axis(distance_function, 1, data_array,
                                        data_point)

    return distances


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
    x_array = fuat.as_unstructured(x).reshape(-1)
    y_array = fuat.as_unstructured(y).reshape(-1)

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
    y_array = fuat.as_unstructured(y).reshape(-1)
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
        An matrix of Euclidean distances between rows in ``X`` and ``Y``.
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
    distance : Number
        The Hamming distances between ``x`` and ``y``.
    """
    # pylint: disable=invalid-name
    if not isinstance(x, str):
        raise TypeError('x should be a string.')
    if not isinstance(y, str):
        raise TypeError('y should be a string.')

    x_len = len(x)
    y_len = len(y)

    distance = abs(x_len - y_len)  # type: float
    if distance and equal_length:
        raise ValueError('Input strings differ in length and the equal_length '
                         'parameter forces them to be of equal length.')
    if distance:
        min_index = min(x_len, y_len)
        for i in range(min_index):
            distance += 0 if x[i] == y[i] else 1
    else:
        assert x_len == y_len, 'The strings should be of equal length.'
        for i in range(x_len):
            distance += 0 if x[i] == y[i] else 1

    if normalise:
        logger.debug('Hamming distance is being normalised.')
        distance /= max(x_len, y_len)

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
    x_array = fuat.as_unstructured(x).reshape(-1)
    y_array = fuat.as_unstructured(y).reshape(-1)

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
    y_array = fuat.as_unstructured(y).reshape(-1)
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
        An matrix of Hamming distances between rows in ``X`` and ``Y``.
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
    x_array = fuat.as_unstructured(x).reshape(-1)
    y_array = fuat.as_unstructured(y).reshape(-1)

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
    y_array = fuat.as_unstructured(y).reshape(-1)
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
        An matrix of binary distances between rows in ``X`` and ``Y``.
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


def check_distance_functionality(distance_function: Callable[..., np.ndarray],
                                 suppress_warning: bool = False) -> bool:
    """
    Checks whether a distance function takes exactly 2 required parameters.

    .. versionadded:: 0.0.2

    The distance function to be checked should calculate a distance matrix
    (2-dimensional numpy array) between all of the rows of the two
    2-dimensional numpy arrays passed as input to the ``distance_function``.

    Parameters
    ----------
    distance_function : Callable[[numpy.ndarray, numpy.ndarray, ...], \
numpy.ndarray]
        A function that calculates a distance matrix between all pairs of rows
        of the two input arrays.
    suppress_warning : boolean, optional (default=False)
        A boolean parameter that indicates whether the function should suppress
        its warning message. Defaults to False.

    Warns
    -----
    UserWarning
        Warns about the details of the required functionality that the distance
        function lacks.

    Raises
    ------
    TypeError
        The ``distance_function`` parameter is not a Python callable or the
        ``suppress_warning`` parameter is not a boolean.

    Returns
    -------
    is_functional : boolean
        A boolean variable that indicates whether the distance function is
        valid.
    """
    if not callable(distance_function):
        raise TypeError('The distance_function parameter should be a Python '
                        'callable.')
    if not isinstance(suppress_warning, bool):
        raise TypeError('The suppress_warning parameter should be a boolean.')

    required_param_n = fuv.get_required_parameters_number(distance_function)
    is_functional = required_param_n == 2

    if not is_functional and not suppress_warning:
        message = ("The '{}' distance function has incorrect number "
                   '({}) of the required parameters. It needs to have '
                   'exactly 2 required parameters. Try using optional '
                   'parameters if you require more functionality.').format(
                       distance_function.__name__, required_param_n)
        warnings.warn(message, category=UserWarning)

    return is_functional
