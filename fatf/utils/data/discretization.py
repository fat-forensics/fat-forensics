"""
Data discretiser classes.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
# License: new BSD

import abc
import warnings

from typing import List, Optional, Union

import numpy as np

from fatf.exceptions import IncorrectShapeError

import fatf.utils.array.tools as fuat
import fatf.utils.array.validation as fuav

__all__ = ['QuartileDiscretizer']

Index = Union[int, str]


def _validate_input(dataset: np.ndarray,
                    categorical_indices: Optional[List[Index]] = None,
                    feature_names: Optional[List[str]] = None) -> bool:
    """
    Validates the input parameters of an arbitrary discretize class.

    Parameters
    ----------
    dataset : numpy.ndarray
        A 2-dimensional numpy array with a dataset to be discretized.
    categorical_indices : List[column indices], optional (default=None)
        A list of column indices that should be treat as categorical features.

    Raises
    ------
    IncorrectShapeError
        The input ``dataset`` is not a 2-dimensional numpy array.
    IndexError
        Some of the column indices given in the ``categorical_indices``
        parameter are not valid for the input ``dataset``.
    TypeError
        The ``categorical_indices`` parameter is neither a list nor ``None``.
        The ``dataset`` is not of base (numerical and/or string) type.

    Returns
    -------
    is_valid : boolean
        ``True`` if input is valid, ``False`` otherwise.
    """
    is_valid = False

    if not fuav.is_2d_array(dataset):
        raise IncorrectShapeError('The input dataset must be a '
                                  '2-dimensional numpy array.')
    if not fuav.is_base_array(dataset):
        raise TypeError('The input dataset must be of a base type.')

    if categorical_indices is not None:
        if isinstance(categorical_indices, list):
            invalid_indices = fuat.get_invalid_indices(
                dataset, np.asarray(categorical_indices))
            if invalid_indices.size:
                raise IndexError('The following indices are invalid for the '
                                 'input dataset: {}.'.format(invalid_indices))
        else:
            raise TypeError('The categorical_indices parameter must be a '
                            'Python list or None.')

    if fuav.is_structured_array(dataset):
        features_number = len(dataset.dtype.names)
    else:
        features_number = dataset.shape[1]

    if feature_names is not None:
        if not isinstance(feature_names, list):
            raise TypeError('The feature_names parameter must be a Python '
                            'list or None.')
        else:
            for name in feature_names:
                if not isinstance(name, str):
                    raise TypeError('The feature_names must be strings.')
            if len(feature_names) != features_number:
                raise ValueError('The length of feature_names must be equal '
                                 'to the number of features in the dataset.')

    is_valid = True
    return is_valid


class Discretization(abc.ABC):
    """
    Abstract class for all discretisers

     An abstract class that all discretizer classes should inherit from. It
    contains abstract ``__init__`` and ``discretize`` methods and an input
    validator -- ``_validate_discretize_input`` -- for the ``discretize``
    method. The validation of the input parameter to the initialisation
    method is done via the :func:`fatf.utils.data.discretize._validate_input`
    function.

    .. warning::
       Attribute ``feature_value_names`` must be overwritten . This attribute
       is of type Dictionary[Index, Dictionary[Any, string]] where the
       outer dictionary is mapping a index of a feature that has been
       discretized to a dictionary. The second dictionary maps values in that
       discretized feature vector to the string description of what those
       values mean, e.g. if we discretized a feature vector into quartiles
       the inner dictionary would be {0: 'feature < q1', 1: 'q1 < feature <
       q2', 2: 'q2 < feature < q3', 3: 'feature > q3'}.

    Parameters
    ----------
    dataset : numpy.ndarray
        A 2-dimensional numpy array with a dataset that the data to be
        disretized belongs to.
    categorical_indices : List[column indices], optional (default=None)
        A list of column indices that should be treat as categorical features.
        If ``None`` is given this will be inferred from the data array:
        string-based columns will be treated as categorical features and
        numerical columns will be treated as numerical features.

    Warns
    -----
    UserWarning
        If some of the string-based columns in the input data array were not
        indicated to be categorical features by the user (via the
        ``categorical_indices`` parameter) the user is warned that they will be
        added to the list of categorical features.

    Raises
    ------
    IncorrectShapeError
        The input ``dataset`` is not a 2-dimensional numpy array.
    IndexError
        Some of the column indices given in the ``categorical_indices``
        parameter are not valid for the input ``dataset``.
    TypeError
        The ``categorical_indices`` parameter is neither a list nor ``None``.
        The ``dataset`` is not of base (numerical and/or string) type.

    Attributes
    ----------
    dataset : numpy.ndarray
        A 2-dimensional numpy array with a dataset to be discretized.
    data_points_number : integer
        The number of data points in the ``dataset``.
    is_structured : boolean
        ``True`` if the ``dataset`` is a structured numpy array, ``False``
        otherwise.
    categorical_indices : List[column indices]
        A list of column indices that should be treat as categorical features.
    numerical_indices : List[column indices]
        A list of column indices that should be treat as numerical features.
    features_number : integer
        The number of features (columns) in the input ``dataset``.
    feature_names : Dict[Index, str]
        A dictionary of feature names. If None then feature names are inferred
        by indices.
    feature_value_names : Dictionary[Index, List[string]]
        A dictionary mapping indices to list of values specifying the
        bounds used to discretize the data.
    """
    def __init__(self,
                 dataset: np.ndarray,
                 categorical_indices: Optional[List[Index]] = None,
                 feature_names: Optional[List[str]] = None):
        """
        Constructs an ``Discretization`` abstract class.
        """
        assert _validate_input(
            dataset,
            categorical_indices=categorical_indices,
            feature_names=feature_names), 'Invalid Input'

        self.dataset = dataset
        self.data_points_number = dataset.shape[0]
        self.is_structured = fuav.is_structured_array(dataset)

        # Sort out column indices
        indices = fuat.indices_by_type(dataset)
        num_indices = set(indices[0])
        cat_indices = set(indices[1])
        all_indices = num_indices.union(cat_indices)

        if categorical_indices is None:
            categorical_indices = cat_indices
            numerical_indices = num_indices
        else:
            if cat_indices.difference(categorical_indices):
                msg = ('Some of the string-based columns in the input dataset '
                       'were not selected as categorical features via the '
                       'categorical_indices parameter. String-based columns '
                       'cannot be treated as numerical features, therefore '
                       'they will be also treated as categorical features '
                       '(in addition to the ones selected with the '
                       'categorical_indices parameter).')
                warnings.warn(msg, UserWarning)
                categorical_indices = cat_indices.union(categorical_indices)
            numerical_indices = all_indices.difference(categorical_indices)

        if self.is_structured:
            indices = self.dataset.dtype.names
        else:
            indices = list(range(self.dataset.shape[1]))
        if feature_names is None:
            feature_names = [str(x) for x in indices]

        self.categorical_indices = sorted(list(categorical_indices))
        self.numerical_indices = sorted(list(numerical_indices))
        self.features_number = len(all_indices)
        self.feature_names = dict(zip(indices, feature_names))
        self.feature_value_names = {}

    @abc.abstractmethod
    def discretize(self, data: np.ndarray) -> np.ndarray:
        """
        Discretizes non-categorical features in ``data``.

        This is an abstract method that must be implemented for each child
        object. This method should return an numpy.ndarray of the same shape
        as ``data`` parameter with non-categorical features been discretized.

        Parameters
        ----------
        data : Union[numpy.ndarray, numpy.void]
            A data point or an array of data points to be discretized.

        Raises
        ------
        NotImplementedError
            This is an abstract method and has not been implemented.

        Returns
        -------
        discretized_data : numpy.ndarray
            Data that has been discretized.
        """
        assert self._validate_discretize_input(  # pragma: nocover
            data), 'Invalid discretize method input.'

        raise NotImplementedError(  # pragma: nocover
            'discretize method needs to be overwritten.')

    def _validate_discretize_input(self, data: np.ndarray) -> bool:
        """
        Validates input parameters of the ``discretize`` method.

        This function checks the validity of ``data`` which can be either a
        1-D array or 2-D array to be discretized.

        Raises
        ------
        IncorrectShapeError
            The number of features (columns) in the ``data`` is different
            to the number of features in the data array used to initialise this
            object.
        TypeError
            The dtype of the ``data`` is different than the dtype of the data
            array used to initialise this object.

        Returns
        -------
        is_valid : boolean
            ``True`` if input parameter are valid, ``False`` otherwise.
        """
        is_valid = False

        are_similar = fuav.are_similar_dtype_arrays(
            self.dataset, np.array(data), strict_comparison=False)
        if not are_similar:
            raise TypeError('The dtype of the data is different to '
                            'the dtype of the data array used to '
                            'initialise this class.')
        if not (fuav.is_1d_like(data) or fuav.is_2d_array(data)):
            raise IncorrectShapeError(
                'data must be a 1-dimensional array, 2-dimensional array or '
                'void object for structured rows.')
        if not self.is_structured:
            features_number = data.shape[0] if fuav.is_1d_like(data) \
                else data.shape[1]
            if features_number != self.dataset.shape[1]:
                raise IncorrectShapeError(
                    'The data must contain the same number of features as '
                    'the dataset used to initialise this class.')

        is_valid = True
        return is_valid


class QuartileDiscretizer(Discretization):
    """
    Discretization that discretizes data into quartiles.

    This class discretizes numerical data by mapping the values in dataset to
    which quartile the value is in for a given feature.

    For additional parameters, attributes, warnings and exceptions raised by
    this class please see the  documentation of its parent class:
    :class:`fatf.utils.data.discretization.Discretization`.

    Attributes
    ----------
    bins : Dictionary[Index, numpy.array]
        A dictionary mapping indices to numpy array specifying the
        quartile ranges.
    """
    def __init__(self,
                 dataset: np.ndarray,
                 categorical_indices: Optional[List[Index]] = None,
                 feature_names: Optional[List[str]] = None,
                 **kwargs):
        """
        Constructs an ``QuartileDiscretization`` abstract class.
        """
        super().__init__(dataset, categorical_indices=categorical_indices,
                         feature_names=feature_names)

        self.bins = {}
        self.feature_value_names = {}
        feature_interval_names = \
            ['%s <= %.2f'] + ['%.2f < %s <= %.2f']*2 + ['%s > %.2f']
        for feature in self.numerical_indices:
            if self.is_structured:
                qts = np.array(np.percentile(dataset[feature], [25, 50, 75]))
            else:
                qts = np.array(np.percentile(dataset[:, feature],
                                             [25, 50, 75]))
            self.bins[feature] = qts
            feature_name = self.feature_names[feature]
            interval_format = [(feature_name, qts[0]),
                               (qts[0], feature_name, qts[1]),
                               (qts[1], feature_name, qts[2]),
                               (feature_name, qts[2])]
            self.feature_value_names[feature] = [
                interval%x for(interval, x) in
                zip(feature_interval_names, interval_format)]

    def discretize(self, data: np.ndarray) -> np.ndarray:
        """
        Discretizes data into quartiles.
        """
        self._validate_discretize_input(data)
        discretized_data = data.copy()

        for feature, values in self.bins.items():
            if self.is_structured or fuav.is_1d_array(data):
                values = np.searchsorted(values, discretized_data[feature])
                discretized_data[feature] = values
            else:
                values = np.searchsorted(values, discretized_data[:, feature])
                discretized_data[:, feature] = values

        return discretized_data
