"""
The :mod:`fatf.utils.data.density` module implements data density estimators.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
#         Rafael Poyiadzi <rp13102@bristol.ac.uk>
# License: new BSD

import inspect
import warnings

from numbers import Number
from typing import Callable, List, Optional, Union

import numpy as np

from fatf.exceptions import IncorrectShapeError

import fatf.utils.array.tools as fuat
import fatf.utils.array.validation as fuav
import fatf.utils.distances as fud
import fatf.utils.tools as fut

__all__ = ['DensityCheck']

_NUMPY_VERSION = [int(i) for i in np.version.version.split('.')]
_NUMPY_1_14 = fut.at_least_verion([1, 14], _NUMPY_VERSION)

DataRow = Union[np.ndarray, np.void]
DistanceFunction = Callable[[DataRow, DataRow], float]
Index = Union[int, str]


def _validate_input_dc(
        data_set: np.ndarray, categorical_indices: Union[None, List[Index]],
        neighbours: int, distance_function: Union[None, DistanceFunction],
        normalise_scores: bool) -> bool:
    """
    Validates ``DensityCheck`` class initialiser's input parameters.

    Parameters
    ----------
    data_set : numpy.ndarray
        A 2-dimensional numpy array (either classic or structured) of a base
        type.
    categorical_indices : Union[None, List[column index]],
        Either ``None`` or a list of column indices to be treated as
        categorical.
    neighbours : integer
        The number of closest neighbours to be considered.
    distance_function : Union[None, Callable[[data row, data row], number]]
        Either ``None`` or a Python function that calculates a distance between
        two data points. This function takes as an input two 1-dimensional
        numpy arrays (for classic numpy arrays) or numpy voids (fro structured
        numpy arrays) of equal length and outputs a number representing a
        distance between them. **The distance function is assumed to return the
        same distance regardless of the order in which the input parameters are
        given.**
    normalise_scores : boolean
        A boolean parameter indicating whether to normalise the scores
        (``True``) or not (``False``).

    Raises
    ------
    AttributeError
        The distance function does not require exactly 2 non-optional
        parameters.
    IncorrectShapeError
        The ``data_set`` array is not 2-dimensional.
    IndexError
        Some of the provided categorical column indices are invalid for the
        ``data_set`` array.
    TypeError
        The ``data_set`` array is not of a base type (strings and/or numbers).
        The ``neighbours`` parameter is not an integer. The
        ``distance_function`` is neither ``None`` nor Python callable (a
        function). The ``normalise_scores`` parameter is not a boolean. The
        ``categorical_indices`` parameter is not a Python list.
    ValueError
        The ``neighbours`` parameter is smaller than 1 or larger than the
        number of instances (rows) in the ``data_set`` array.

    Returns
    -------
    is_valid : boolean
        ``True`` if the input is valid, ``False`` otherwise.
    """
    # pylint: disable=too-many-branches
    is_valid = False

    if not fuav.is_2d_array(data_set):
        raise IncorrectShapeError('The data set should be a 2-dimensional '
                                  'numpy array.')
    if not fuav.is_base_array(data_set):
        raise TypeError('The data set is not of a base type (numbers and/or '
                        'strings.')

    if categorical_indices is not None:
        if isinstance(categorical_indices, list):
            invalid_indices = fuat.get_invalid_indices(
                data_set, np.asarray(categorical_indices)).tolist()
            if invalid_indices:
                raise IndexError('The following indices are invalid for the '
                                 'input data set: {}.'.format(invalid_indices))
        else:
            raise TypeError('The categorical_indices parameter must be a '
                            'Python list or None.')

    if isinstance(neighbours, int):
        if neighbours < 1 or neighbours > data_set.shape[0]:
            raise ValueError('The neighbours number parameter has to be '
                             'between 1 and number of data points (rows) in '
                             'the data set array.')
    else:
        raise TypeError('The neighbours number parameter has to be an '
                        'integer.')

    if distance_function is not None:
        if callable(distance_function):
            required_param_n = 0
            params = inspect.signature(distance_function).parameters
            for param in params:
                if params[param].default is params[param].empty:
                    required_param_n += 1
            if required_param_n != 2:
                raise AttributeError('The distance function must require '
                                     'exactly 2 parameters. Given function '
                                     'requires {} '
                                     'parameters.'.format(required_param_n))
        else:
            raise TypeError('The distance function should be a Python '
                            '(function).')

    if not isinstance(normalise_scores, bool):
        raise TypeError('The normalise scores parameter should be a boolean.')

    is_valid = True
    return is_valid


class DensityCheck(object):
    """
    Checks and scores density in the ``data_set`` and for new data points.

    A density score for a particular data point is calculated by looking at the
    distance of the n-th neighbour defined by the ``neighbours`` parameter.
    If this distance is relatively large (in comparison to all the other
    data point-to-data point distances in the ``data_set``) it means that this
    particular point lies in a low density region. The scores can be normalised
    to [0, 1] range by setting the ``normalise_scores`` parameter to ``True``
    (the default value). Since the minimum and the maximum value of the scores
    in the data set are used when (normalised) scoring a new data point the
    score may go out of the [0, 1] range. To avoid this from happening please
    look into the ``clip`` parameter in the ``score_data_point`` method.

    Parameters
    ----------
    data_set : numpy.ndarray
        A 2-dimensional numpy array (either classic or structured) of a base
        type (strings and/or numbers).
    categorical_indices : List[column indices], optional (default=None)
        A list of column indices that should be treat as categorical features.
        If ``None`` the categorical column indices will be inferred by checking
        the type of the ``data_set`` for a classic numpy array and the type of
        every column for a structured numpy array.
    neighbours : integer, optional (default=7)
        The number of closest neighbours to be considered when calculating the
        density score.
    distance_function : Callable[[data row, data row], number], \
optional (default=None)
        If ``None`` the sum of Euclidean distance for numerical features and
        binary distance (0 when the values are the same and 1 otherwise) for
        categorical features will be used as a distance function.
        Alternatively, the user may provide a Python function that will be used
        to calculate a distance between two data points. This function takes as
        an input two 1-dimensional numpy arrays (for classic numpy arrays) or
        numpy voids (fro structured numpy arrays) of equal length and outputs a
        number representing a distance between them. **The distance function is
        assumed to return the same distance regardless of the order in which
        the input parameters are given.**
    normalise_scores : boolean, optional (default=True)
        A boolean parameter indicating whether to normalise the density scores
        (``True``) or not (``False``). The scores are normalised by subtracting
        the minimum value and dividing by the new (after subtracting the
        minimum) maximum value.

    Warns
    -----
    UserWarning
        If some of the string-based columns in the input data array were not
        indicated to be categorical features by the user (via the
        ``categorical_indices`` parameter) the user is warned that they will be
        added to the list of categorical features.

    Raises
    ------
    AttributeError
        The distance function does not require exactly 2 non-optional
        parameters.
    IncorrectShapeError
        The ``data_set`` array is not 2-dimensional.
    IndexError
        Some of the provided categorical column indices are invalid for the
        ``data_set`` array.
    TypeError
        The ``data_set`` array is not of a base type (strings and/or numbers).
        The ``neighbours`` parameter is not an integer. The
        ``distance_function`` is neither ``None`` nor Python callable (a
        function). The ``normalise_scores`` parameter is not a boolean. The
        ``categorical_indices`` parameter is not a Python list.
    ValueError
        The ``neighbours`` parameter is smaller than 1 or larger than the
        number of instances (rows) in the ``data_set`` array.

    Attributes
    ----------
    data_set : numpy.ndarray
        A data set used to compute the density scores.
    neighbours : integer
        The number of neighbours used to calculate the density scores.
    normalise_scores : boolean
        Indicates whether the scores should be normalised to a [0, 1] range.
    distance_matrix : numpy.ndarray
        An 2-dimensional, square and diagonally symmetric array with distances
        between every pair of rows in the ``data_set``.
    scores : numpy.ndarray
        A 1-dimensional array with a density score for every row in the
        ``data_set``.
    scores_min : number
        The minimum density score (extracted before the normalisation if one is
        performed).
    scores_max : number
        The maximum density score (extracted before the normalisation if one is
        performed).
    _samples_number : integer
        The number of data points (rows) in the ``data_set``.
    _numerical_indices : List[column indices]
        An array holding indices of numerical columns in the ``data_set``
        array.
    _categorical_indices : List[column indices]
        An array holding indices of categorical columns in the ``data_set``
        array.
    _is_structured : boolean
        Indicates whether the input ``data_set`` is a structured array
        (``True``) or a classic numpy array (``False``).
    _distance_function : Callable[[data row, data row], number]
        A Python function used to calculate distances between data points.
    """

    # pylint: disable=useless-object-inheritance,too-many-instance-attributes

    def __init__(self,
                 data_set: np.ndarray,
                 categorical_indices: Optional[List[Index]] = None,
                 neighbours: int = 7,
                 distance_function: Optional[DistanceFunction] = None,
                 normalise_scores: bool = True) -> None:
        """
        Initialises the ``DensityCheck`` class.
        """
        # pylint: disable=too-many-arguments
        assert _validate_input_dc(data_set, categorical_indices, neighbours,
                                  distance_function,
                                  normalise_scores), 'Invalid input.'

        self.data_set = data_set
        self._is_structured = fuav.is_structured_array(self.data_set)
        #
        self.neighbours = neighbours
        if distance_function is None:
            if not _NUMPY_1_14 and self._is_structured:
                distance_function = self._mixed_distance_o  # pragma: nocover
            else:
                distance_function = self._mixed_distance_n
        self._distance_function = distance_function  # type: ignore
        #
        self.normalise_scores = normalise_scores

        # Sort out column indices
        feature_indices = fuat.indices_by_type(self.data_set)
        num_indices = set(feature_indices[0])
        cat_indices = set(feature_indices[1])
        all_indices = num_indices.union(cat_indices)
        if categorical_indices is None:
            _categorical_indices = cat_indices
            _numerical_indices = num_indices
        else:
            if cat_indices.difference(categorical_indices):
                msg = ('Some of the string-based columns in the input data '
                       'set were not selected as categorical features via the '
                       'categorical_indices parameter. String-based columns '
                       'cannot be treated as numerical features, therefore '
                       'they will be also treated as categorical features '
                       '(in addition to the ones selected with the '
                       'categorical_indices parameter).')
                warnings.warn(msg, UserWarning)
                _categorical_indices = cat_indices.union(categorical_indices)
            else:
                _categorical_indices = categorical_indices  # type: ignore
            _numerical_indices = all_indices.difference(_categorical_indices)
        self._categorical_indices = sorted(list(_categorical_indices))
        self._numerical_indices = sorted(list(_numerical_indices))

        self._samples_number = self.data_set.shape[0]

        self.distance_matrix = fud.get_distance_matrix(self.data_set,
                                                       self._distance_function)
        assert self._samples_number == self.distance_matrix.shape[0]
        assert self.distance_matrix.shape[0] == self.distance_matrix.shape[1]

        self.scores = self._compute_scores()
        assert self._samples_number == self.scores.shape[0]
        self.scores_min = self.scores.min()
        self.scores_max = self.scores.max()
        if self.normalise_scores:
            if self.scores_min == self.scores_max:
                assert (self.scores == self.scores_min).all(), \
                    'All distances/scores are equal.'
                self.scores[:] = 0
            else:
                self.scores -= self.scores_min
                self.scores /= self.scores_max - self.scores_min

    def _mixed_distance_n(self, array_x: DataRow, array_y: DataRow) -> float:
        """
        Calculates a distance between two data points.

        This distance function is a mixture of Euclidean and binary (0 when the
        values are the same and 1 otherwise) distances. It is calculated by
        summing up the Euclidean distance between the numerical features and
        the binary distance between categorical features.

        .. note::
           This implementation is designed for numpy version 1.14 or greater.
           Due to structured rows (numpy void) indexing this implementation
           will result in ``IndexError`` if used with older numpy version.

        Parameters
        ----------
        array_x : Union[numpy.ndarray, numpy.void]
            1-dimensional data array of the same length as ``array_y``.
        array_y : Union[numpy.ndarray, numpy.void]
            1-dimensional data array of the same length as ``array_x``.

        Returns
        -------
        distance : number
            A distance between ``array_x`` and ``array_y``.
        """
        if self._numerical_indices:
            num_dist = fud.euclidean_distance(array_x[self._numerical_indices],
                                              array_y[self._numerical_indices])
        else:
            num_dist = 0

        if self._categorical_indices:
            cat_dist = fud.binary_distance(array_x[self._categorical_indices],
                                           array_y[self._categorical_indices])
        else:
            cat_dist = 0

        distance = num_dist + cat_dist
        return distance

    def _mixed_distance_o(self, array_x: DataRow, array_y: DataRow) -> float:
        """
        Calculates a distance between two data points.

        This distance function is a mixture of Euclidean and binary (0 when the
        values are the same and 1 otherwise) distances. It is calculated by
        summing up the Euclidean distance between the numerical features and
        the binary distance between categorical features.

        .. note::
           This implementation is designed for any numpy version and structured
           numpy arrays. It avoids extracting multiple column indices from
           structured rows (numpy void).

        Parameters
        ----------
        array_x : Union[numpy.ndarray, numpy.void]
            1-dimensional data array of the same length as ``array_y``.
        array_y : Union[numpy.ndarray, numpy.void]
            1-dimensional data array of the same length as ``array_x``.

        Returns
        -------
        distance : number
            A distance between ``array_x`` and ``array_y``.
        """
        assert self._is_structured, 'The dataset for this implementation.'
        as_array_x = np.asarray([array_x])
        as_array_y = np.asarray([array_y])
        if self._numerical_indices:
            num_dist = fud.euclidean_distance(
                as_array_x[self._numerical_indices][0],
                as_array_y[self._numerical_indices][0])
        else:
            num_dist = 0

        if self._categorical_indices:
            cat_dist = fud.binary_distance(
                as_array_x[self._categorical_indices][0],
                as_array_y[self._categorical_indices][0])
        else:
            cat_dist = 0

        distance = num_dist + cat_dist
        return distance

    def _compute_scores(self) -> np.ndarray:
        """
        Computes density scores for all data points (rows) in the ``data_set``.

        Returns
        -------
        scores : numpy.ndarray
            A 1-dimensional numpy array with a density score for every data
            point (row) in the ``data_set``.
        """
        scores = np.zeros(self._samples_number)
        # Find the distance of the furthest neighbour
        for i, row in enumerate(self.distance_matrix):
            # We do not add 1 to the neighbours number because the closest
            # neighbour will be the data point itself (+1) but the indexing
            # starts from 0 (-1)
            furthest_neighbour = np.argsort(row)[self.neighbours]
            scores[i] = row[furthest_neighbour]

        return scores

    def filter_data_set(self, alpha: float = 0.8) -> np.ndarray:
        """
        Returns the data points that are in alpha-dense areas.

        A data points in an alpha-dense region have a density score larger or
        equal to ``alpha``. For normalised scores ``alpha`` should be between 0
        and 1, whereas for unnormalised scores it must be equal to or larger
        than 0.

        Parameters
        ----------
        alpha : number, optional (default=0.8)
            The score above which instances should be kept.

        Warns
        -----
        UserWarning
            Chosen ``alpha`` parameter is too large and none of the data points
            were selected.

        Raises
        ------
        TypeError
            The ``alpha`` parameter is not a number.
        ValueError
            The alpha parameter is not between 0 and 1 for the normalised
            scores or is not larger or equal to 0 for unnormalised scores.

        Returns
        -------
        filtered_data_set : numpy.ndarray
            Data points with density score larger than ``alpha`` (extracted
            from the ``data_set``).
        """
        if isinstance(alpha, Number):
            if self.normalise_scores:
                if alpha < 0 or alpha > 1:
                    raise ValueError('The alpha parameter has to be between '
                                     '0 and 1 for normalised scores.')
            else:
                if alpha < 0:
                    raise ValueError('The alpha parameter has to be equal to '
                                     'or larger than 0.')
        else:
            raise TypeError('The alpha parameter has to be a number.')

        # Filter scores above a certain threshold
        filtered_indices = np.where(self.scores >= alpha)[0]
        data_to_keep = self.data_set[filtered_indices]

        if not data_to_keep.size:
            warnings.warn(
                'Chosen alpha parameter is too large and none of the data '
                'points were selected.', UserWarning)

        return data_to_keep

    def _validate_data_point(self, data_point: DataRow, clip: bool) -> bool:
        """
        Validates input parameters of the ``score_data_point`` method.

        Parameters
        ----------
        data_point : Union[numpy.array, numpy.void]
            A data row. For numpy arrays this will be a numpy ndarray. For
            structured numpy arrays this will be numpy void.

        Raises
        ------
        IncorrectShapeError
            The data point is not 1-dimensional numpy array (either numpy
            ndarray for classic numpy arrays or numpy void for structured numpy
            arrays). The data point does not have the same number of columns
            (features) as the data set used to initialise this class.
        TypeError
            The data point is not of a base type (strings and/or numbers). The
            dtype of the data point is too different from the dtype of the
            data set used to initialise this class. The ``clip`` parameter is
            not a boolean.

        Returns
        -------
        is_valid : boolean
            ``True`` if the input parameters are valid, ``False`` otherwise.
        """
        is_valid = False

        if not fuav.is_1d_like(data_point):
            raise IncorrectShapeError('The data point has to be 1-dimensional '
                                      'numpy array or numpy void (for '
                                      'structured arrays).')
        data_point_array = np.asarray([data_point])
        if not fuav.is_base_array(data_point_array):
            raise TypeError('The data point has to be of a base type (strings '
                            'and/or numbers).')
        if not fuav.are_similar_dtype_arrays(self.data_set, data_point_array):
            raise TypeError('The dtypes of the data set used to initialise '
                            'this class and the provided data point are too '
                            'different.')
        # Testing only for unstructured as the dtype comparison picks up on a
        # different number of columns in a structured array
        if not self._is_structured:
            if self.data_set.shape[1] != data_point_array.shape[1]:
                raise IncorrectShapeError('The data point has different '
                                          'number of columns (features) than '
                                          'the data set used to initialise '
                                          'this class.')

        if not isinstance(clip, bool):
            raise TypeError('The clip parameter has to be a boolean.')

        is_valid = True
        return is_valid

    def score_data_point(self, data_point: DataRow,
                         clip: bool = True) -> float:
        """
        Calculates a density score for the ``data_point``.

        Parameters
        ----------
        data_point : Union[numpy.array, numpy.void]
            A data row. For numpy arrays this will be a numpy ndarray. For
            structured numpy arrays this will be numpy void.
        clip : boolean, optional (default=True)
            If ``True`` and the scores are normalised (this class was
            initialised with the ``normalise_scores`` parameter set to
            ``True``, which is the default option) the score of the provided
            data point will be clipped to fit the [0, 1] range. If the scores
            are not normalised this parameter is ignored.

        Warns
        -----
        UserWarning
            The minimum and maximum score values for this class are the same,
            therefore the score normalisation cannot be performed. In this case
            the score will be 0 if it is below the min/max, 1 if it is above
            the min/max and otherwise it stays the same.

        Raises
        ------
        IncorrectShapeError
            The data point is not 1-dimensional numpy array (either numpy
            ndarray for classic numpy arrays or numpy void for structured numpy
            arrays). The data point does not have the same number of columns
            (features) as the data set used to initialise this class.
        TypeError
            The data point is not of a base type (strings and/or numbers). The
            dtype of the data point is too different from the dtype of the
            data set used to initialise this class. The ``clip`` parameter is
            not a boolean.

        Returns
        -------
        score : number
            A density score for the ``data_point``.
        """
        assert self._validate_data_point(data_point,
                                         clip), 'Invalid data point.'

        distance_array = fud.get_point_distance(self.data_set, data_point,
                                                self._distance_function)

        # Find the distance of the furthest neighbour: we subtract 1 from the
        # neighbours number because the indexing starts from 0
        furthest_neighbour = np.argsort(distance_array)[self.neighbours - 1]
        score = distance_array[furthest_neighbour]

        if self.normalise_scores:
            if self.scores_min == self.scores_max:
                warnings.warn(
                    'The minimum and maximum scores are the same, therefore '
                    'the score normalisation is ill-defined.', UserWarning)
                if score <= self.scores_min:
                    score = 0
                else:
                    score = 1
            else:
                score -= self.scores_min
                score /= self.scores_max - self.scores_min
                if clip:
                    score = max(min(1, score), 0)

        return score
