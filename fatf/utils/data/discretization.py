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

__all__ = []

Index = Union[int, str]


def _validate_input(dataset: np.ndarray,
                    categorical_indices: Optional[List[Index]] = None) -> bool:
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

    is_valid = True
    return is_valid


class Discretizer(abc.ABC):
    """
    Abstract class for all discretisers

     An abstract class that all discretizer classes should inherit from. It 
    contains abstract ``__init__`` and ``discretize`` methods and an input
    validator -- ``_validate_discretize_input`` -- for the ``discretize``
    method. The validation of the input parameter to the initialisation
    method is done via the :func:`fatf.utils.data.discretize._validate_input`
    function.

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
    """
    def __init__(self,
                 dataset: np.ndarray,
                 categorical_indices: Optional[List[Index]]):
        """
        Constructs an ``Discretizer`` abstract class.
        """
        assert _validate_input(
            dataset, 
            categorical_indices=categorical_indices), 'Invalid Input'

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

        self.categorical_indices = sorted(list(categorical_indices))
        self.numerical_indices = sorted(list(numerical_indices))
        self.features_number = len(all_indices)

    @abc.abstractmethod
    def discretize(self,
                   data: np.ndarray) -> np.ndarray:
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

    def _validate_discretize_input(self,
                                   data:np.ndarray) -> bool:
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
                    self.dataset, np.array(data), strict_comparison=True)
        if not are_similar:
            raise TypeError('The dtype of the data is different to '
                            'the dtype of the data array used to '
                            'initialise this class.')
        if fuav.is_1d_like(data):
            features_number = data.shape[0]
        elif fuav.is_2d_array(data):
            features_number = data.shape[1]
        else:
            raise IncorrectShapeError(
                'data must be a 1-dimensional array, 2-dimensional array or '
                'void object for structured rows.')
        if not self.is_structured:
            if features_number != self.dataset.shape[1]:
                raise IncorrectShapeError(
                    'The data must contain the same number of features as '
                    'the dataset used to initialise this class.')

        is_valid = True
        return is_valid
