"""
.. versionadded:: 0.0.2

The :mod:`fatf.utils.data.discretisation` module implements data
discretisation approaches.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
#         Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

from typing import Dict  # pylint: disable=unused-import
from typing import List, Optional, Union

import abc
import warnings

import numpy as np

import fatf.utils.array.tools as fuat
import fatf.utils.array.validation as fuav

from fatf.exceptions import IncorrectShapeError

__all__ = ['Discretiser', 'QuartileDiscretiser']

Index = Union[int, str]


def _validate_input_discretiser(
        dataset: np.ndarray,
        categorical_indices: Optional[List[Index]] = None,
        feature_names: Optional[List[str]] = None) -> bool:
    """
    Validates the input parameters of an arbitrary discretiser class.

    Parameters
    ----------
    dataset : numpy.ndarray
        A 2-dimensional numpy array with a dataset to be discretised.
    categorical_indices : List[column indices], optional (default=None)
        A list of column indices that should be treat as categorical features.
    feature_names : List[strings], optional (default=None)
        A list of feature names in order they appear in the ``dataset`` array.

    Raises
    ------
    IncorrectShapeError
        The input ``dataset`` is not a 2-dimensional numpy array.
    IndexError
        Some of the column indices given in the ``categorical_indices`` list
        are invalid for the input ``dataset``.
    TypeError
        The ``dataset`` is not of a base (numerical and/or string) type.
        The ``categorical_indices`` is neither a Python list nor ``None``.
        The ``feature_names`` is neither a Python list nor ``None`` or one of
        its elements (if it is a list) is not a string.
    ValueError
        The length of the ``feature_names`` list is different than the number
        of columns (features) in the input ``dataset``.

    Returns
    -------
    is_valid : boolean
        ``True`` if the input is valid, ``False`` otherwise.
    """
    # pylint: disable=too-many-branches
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
                                 'input dataset: {}.'.format(
                                     invalid_indices.tolist()))
        else:
            raise TypeError('The categorical_indices parameter must be a '
                            'Python list or None.')

    if feature_names is not None:
        if isinstance(feature_names, list):
            if fuav.is_structured_array(dataset):
                features_number = len(dataset.dtype.names)
            else:
                features_number = dataset.shape[1]
            if len(feature_names) != features_number:
                raise ValueError('The length of feature_names list must be '
                                 'equal to the number of features (columns) '
                                 'in the input dataset.')

            for name in feature_names:
                if not isinstance(name, str):
                    raise TypeError('All of the feature_names must be '
                                    'strings. The *{}* feature name is not a '
                                    'string.'.format(name))
        else:
            raise TypeError('The feature_names parameter must be a Python '
                            'list or None.')

    is_valid = True
    return is_valid


class Discretiser(abc.ABC):
    """
    An abstract class that all discretiser implementations should inherit from.

    .. versionadded:: 0.0.2

    The validation of the initialiser input parameters is done via the
    ``fatf.utils.data.discretise._validate_input_discretiser`` function.
    This abstract class also contains an abstract ``discretise`` method and its
    input validator ``_validate_input_discretise``. The ``discretise`` method
    should be overwritten in the children classes and the
    ``_validate_input_discretise`` methods should be called therein to validate
    their input.

    If you need extra initialisation capabilities, you may overwrite the
    ``__init__`` method in which case please remember to call
    ``super().__init__()`` at its top to make sure that all of the abstract
    class attributes are validated and initialised.

    .. warning::
       The ``feature_value_names`` and ``feature_bin_boundaries`` class
       attributes must be overwritten by every child class. The first attribute
       is of ``Dictionary[Column Index, Dictionary[integer, string]]`` type
       where the outer dictionary is mapping a column (feature) index of the
       input ``dataset`` to a dictionary with keys being discretised bin ids
       for that feature and values being these bins (string) descriptions, for
       example, if we discretised a feature vector into quartiles the inner
       dictionary would be {0: 'feature < q1', 1: 'q1 < feature < q2', 2: 'q2 <
       feature < q3', 3: 'feature > q3'}, where q1, q2 and q3 are quartile
       boundaries.

       The ``feature_bin_boundaries`` attribute should be overwritten with a
       dictionary which keys are column (feature) indices and values are numpy
       arrays holding bin boundaries for each feature. Using the above example
       this would be ``numpy.array([q1, q2, q3])``. (By default the upper bin
       boundary should be inclusive.)

    Parameters
    ----------
    dataset : numpy.ndarray
        A 2-dimensional numpy array with a dataset to be discretised.
    categorical_indices : List[column indices], optional (default=None)
        A list of column indices that should be treat as categorical features.
        If ``None`` is given, this will be inferred from the ``dataset`` array:
        string-based columns will be treated as categorical features and
        numerical columns will be treated as numerical features.
    feature_names : List[strings], optional (default=None)
        A list of feature names in order they appear in the ``dataset`` array.
        If ``None``, this will be extracted from the ``dataset`` array. For
        structured arrays these will be the column names extracted from the
        dtype; for classic arrays these will be numbers indicating the column
        index in the array.

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
        Some of the column indices given in the ``categorical_indices`` list
        are invalid for the input ``dataset``.
    TypeError
        The ``dataset`` is not of a base (numerical and/or string) type.
        The ``categorical_indices`` is neither a Python list nor ``None``.
        The ``feature_names`` is neither a Python list nor ``None`` or one of
        its elements (if it is a list) is not a string.
    ValueError
        The length of the ``feature_names`` list is different than the number
        of columns (features) in the input ``dataset``.

    Attributes
    ----------
    dataset_dtype : numpy.dtype
        The dtype of the input ``dataset``.
    is_structured : boolean
        ``True`` if the input ``dataset`` is a structured numpy array,
        ``False`` otherwise.
    features_number : integer
        The number of features (columns) in the input ``dataset``.
    categorical_indices : List[Column Indices]
        A list of column indices that should be treat as categorical features.
    numerical_indices : List[Column Indices]
        A list of column indices that should be treat as numerical features.
    feature_names_map : Dict[Column Index, String]
        A dictionary that holds mapping of column (feature) indices to their
        names (feature names). If the ``feature_names`` parameter was not given
        (``None``), the feature names are inferred from the ``dataset``.
    feature_value_names : Dictionary[Index, Dictionary[Integer, String]]
        A dictionary mapping ``dataset`` column (feature) indices to
        dictionaries holding description (value) of each discrete value (key)
        for that feature.
    feature_bin_boundaries : Dictionary[Index, numpy.ndarray]
        A dictionary mapping ``dataset`` column (feature) indices to numpy
        arrays holding bin boundaries (with the upper threshold inclusive) for
        each feature.
    """

    # pylint: disable=too-few-public-methods,too-many-instance-attributes

    def __init__(self,
                 dataset: np.ndarray,
                 categorical_indices: Optional[List[Index]] = None,
                 feature_names: Optional[List[str]] = None) -> None:
        """
        Constructs a ``Discretiser`` abstract class.
        """
        # Must be overwritten in children classes
        self.feature_value_names = {}  # type: Dict[Index, Dict[int, str]]
        self.feature_bin_boundaries = {}  # type: Dict[Index, np.ndarray]

        assert _validate_input_discretiser(
            dataset,
            categorical_indices=categorical_indices,
            feature_names=feature_names), 'Invalid input.'

        self.is_structured = fuav.is_structured_array(dataset)

        self.dataset_dtype = dataset.dtype

        # Sort out column indices
        indices_num, indices_cat = fuat.indices_by_type(dataset)
        num_indices = set(indices_num)
        cat_indices = set(indices_cat)
        all_indices = num_indices.union(cat_indices)

        if categorical_indices is None:
            categorical_indices = cat_indices  # type: ignore
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
                categorical_indices = cat_indices.union(  # type: ignore
                    categorical_indices)
            numerical_indices = all_indices.difference(categorical_indices)
        self.categorical_indices = sorted(
            list(categorical_indices))  # type: ignore
        self.numerical_indices = sorted(list(numerical_indices))

        self.features_number = len(all_indices)

        if self.is_structured:
            indices = self.dataset_dtype.names
        else:
            indices = range(self.features_number)

        if feature_names is None:
            feature_names_map = {x: str(x) for x in indices}
        else:
            feature_names_map = dict(zip(indices, feature_names))
        self.feature_names_map = feature_names_map

    def _validate_input_discretise(
            self, dataset: Union[np.ndarray, np.void]) -> bool:
        """
        Validates the input parameters of the ``discretise`` method.

        This method checks the validity of the input ``dataset``, which can be
        either a 1-D or a 2-D array with *similar* dtype to the data array
        used to initialise this class.

        Parameters
        ----------
        dataset : Union[numpy.ndarray, numpy.void]
            A data point (1-D array) or a data set (2-D array) to be
            discretised.

        Raises
        ------
        IncorrectShapeError
            The input ``dataset`` is neither 1- nor 2-dimensional numpy array.
            The number of features (columns) in the input ``dataset`` is
            different than the number of features in the dataset used to
            initialise this object.
        TypeError
            The dtype of the input ``dataset`` is too different from the dtype
            of the dataset used to initialise this object.

        Returns
        -------
        is_valid : boolean
            ``True`` if the input parameter is valid, ``False`` otherwise.
        """
        is_valid = False

        if not (fuav.is_1d_like(dataset) or fuav.is_2d_array(dataset)):
            raise IncorrectShapeError('The dataset must be either a '
                                      '1-dimensional (a plane numpy array or '
                                      'numpy void for structured '
                                      '1-dimensional arrays) or a '
                                      '2-dimensional array.')

        are_similar = fuav.are_similar_dtype_arrays(
            np.empty((0, ), dtype=self.dataset_dtype),
            np.array(dataset),
            strict_comparison=False)
        if not are_similar:
            raise TypeError('The dtype of the input dataset is too different '
                            'from the dtype of the dataset used to initialise '
                            'this class.')
        # The dimensions of a structured array are automatically compared above
        if not self.is_structured:
            if fuav.is_1d_like(dataset):
                features_number = dataset.shape[0]
            else:
                features_number = dataset.shape[1]

            if features_number != self.features_number:
                raise IncorrectShapeError('The input dataset must contain the '
                                          'same number of features as the '
                                          'dataset used to initialise this '
                                          'class.')

        is_valid = True
        return is_valid

    @abc.abstractmethod
    def discretise(self, dataset: Union[np.ndarray, np.void]) -> np.ndarray:
        """
        Discretises non-categorical (numerical) features in the ``dataset``.

        This is an abstract method that must be implemented for each
        discretiser object that inherits form ``Discretiser``. This method
        should return a numpy.ndarray with all non-categorical columns
        (features) of the input ``dataset`` being discretised.

        .. warning::
           When implementing this method please remember to call
           ``assert self._validate_input_discretise(dataset)`` to validate
           the input parameters.

        Parameters
        ----------
        dataset : Union[numpy.ndarray, numpy.void]
            A data point (1-D) or an array (2-D) of data points to be
            discretised.

        Raises
        ------
        NotImplementedError
            This is an abstract method and has not been implemented.
        IncorrectShapeError
            The input ``dataset`` is neither 1- nor 2-dimensional numpy array.
            The number of features (columns) in the input ``dataset`` is
            different than the number of features in the dataset used to
            initialise this object.
        TypeError
            The dtype of the input ``dataset`` is too different from the dtype
            of the dataset used to initialise this object.

        Returns
        -------
        discretised_data : numpy.ndarray
            A discretised data array.
        """
        assert self._validate_input_discretise(
            dataset), 'Invalid discretise method input.'  # pragma: nocover

        raise NotImplementedError(
            'The discretise method must be overwritten.')  # pragma: nocover


class QuartileDiscretiser(Discretiser):
    """
    Discretises selected numerical features of the ``dataset`` into quartiles.

    .. versionadded:: 0.0.2

    This class discretises numerical columns (features) of the ``dataset`` by
    mapping their values onto quartile ids to which they belong. The quartile
    boundaries are computed based of the ``dataset`` used to initialise this
    class.

    Parameters
    ----------
    dataset : numpy.ndarray
        A 2-dimensional numpy array with a dataset to be discretised.
    categorical_indices : List[column indices], optional (default=None)
        A list of column indices that should be treat as categorical features.
        If ``None`` is given, this will be inferred from the ``dataset`` array:
        string-based columns will be treated as categorical features and
        numerical columns will be treated as numerical features.
    feature_names : List[strings], optional (default=None)
        A list of feature names in order they appear in the ``dataset`` array.
        If ``None``, this will be extracted from the ``dataset`` array. For
        structured arrays these will be the column names extracted from the
        dtype; for classic arrays these will be numbers indicating the column
        index in the array.

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
        Some of the column indices given in the ``categorical_indices`` list
        are invalid for the input ``dataset``.
    TypeError
        The ``dataset`` is not of a base (numerical and/or string) type.
        The ``categorical_indices`` is neither a Python list nor ``None``.
        The ``feature_names`` is neither a Python list nor ``None`` or one of
        its elements (if it is a list) is not a string.
    ValueError
        The length of the ``feature_names`` list is different than the number
        of columns (features) in the input ``dataset``.

    Attributes
    ----------
    dataset_dtype : numpy.dtype
        The dtype of the input ``dataset``.
    is_structured : boolean
        ``True`` if the input ``dataset`` is a structured numpy array,
        ``False`` otherwise.
    features_number : integer
        The number of features (columns) in the input ``dataset``.
    categorical_indices : List[Column Indices]
        A list of column indices that should be treat as categorical features.
    numerical_indices : List[Column Indices]
        A list of column indices that should be treat as numerical features.
    feature_names_map : Dict[Column Index, String]
        A dictionary that holds mapping of column (feature) indices to their
        names (feature names). If the ``feature_names`` parameter was not given
        (``None``), the feature names are inferred from the ``dataset``.
    feature_value_names : Dictionary[Index, Dictionary[Integer, String]]
        A dictionary mapping ``dataset`` column (feature) indices to
        dictionaries holding quartile description (value) of each quartile id
        (key) for that feature.
    feature_bin_boundaries : Dictionary[Index, numpy.ndarray]
        A dictionary mapping ``dataset`` column (feature) indices to numpy
        arrays holding quartile bin boundaries (with the upper boundary
        inclusive) for each feature.
    discretised_dtype : numpy.dtype
        The dtype of the discretised arrays output by the ``discrete`` method.
    """

    # pylint: disable=too-few-public-methods

    def __init__(self,
                 dataset: np.ndarray,
                 categorical_indices: Optional[List[Index]] = None,
                 feature_names: Optional[List[str]] = None) -> None:
        """
        Constructs a ``QuartileDiscretiser`` object.
        """
        super().__init__(
            dataset,
            categorical_indices=categorical_indices,
            feature_names=feature_names)

        # Prepare dtype of the discretised array
        if self.is_structured:
            self.discretised_dtype = []  # type: List[np.dtype]
            for feature in self.dataset_dtype.names:
                if feature in self.numerical_indices:
                    self.discretised_dtype.append((feature, np.int8))
                else:
                    self.discretised_dtype.append(
                        (feature, self.dataset_dtype[feature]))
        else:
            if self.categorical_indices:
                self.discretised_dtype = self.dataset_dtype
            else:
                self.discretised_dtype = np.int8

        percentile_qartile = [25, 50, 75]

        for feature in self.numerical_indices:
            if self.is_structured:
                qts = np.percentile(dataset[feature], percentile_qartile)
            else:
                qts = np.percentile(dataset[:, feature], percentile_qartile)
            self.feature_bin_boundaries[feature] = qts

            feature_name = self.feature_names_map[feature]

            self.feature_value_names[feature] = {
                0: '*{}* <= {:.2f}'.format(feature_name, qts[0]),
                qts.shape[0]: '{:.2f} < *{}*'.format(qts[-1], feature_name)
            }
            for i in range(1, qts.shape[0]):
                bin_name = '{:.2f} < *{}* <= {:.2f}'.format(
                    qts[i - 1], feature_name, qts[i])
                self.feature_value_names[feature][i] = bin_name

    def discretise(self, dataset: Union[np.ndarray, np.void]
                   ) -> Union[np.ndarray, np.void]:
        """
        Discretises numerical features of the ``dataset`` into quartiles.

        Parameters
        ----------
        dataset : Union[numpy.ndarray, numpy.void]
            A data point (1-D) or an array (2-D) of data points to be
            discretised.
        Raises
        ------
        IncorrectShapeError
            The input ``dataset`` is neither 1- nor 2-dimensional numpy array.
            The number of features (columns) in the input ``dataset`` is
            different than the number of features in the dataset used to
            initialise this object.
        TypeError
            The dtype of the input ``dataset`` is too different from the dtype
            of the dataset used to initialise this object.

        Returns
        -------
        discretised_data : Union[numpy.ndarray, numpy.void]
            A discretised data array.
        """
        self._validate_input_discretise(dataset)

        if self.is_structured and fuav.is_1d_like(dataset):
            discretised_dataset = dataset.copy().astype(self.discretised_dtype)
        else:
            discretised_dataset = np.zeros_like(
                dataset, dtype=self.discretised_dtype)

        for feature in self.categorical_indices:
            if self.is_structured or fuav.is_1d_array(dataset):
                discretised_dataset[feature] = dataset[feature]
            else:
                discretised_dataset[:, feature] = dataset[:, feature]

        for feature, boundaries in self.feature_bin_boundaries.items():
            if self.is_structured or fuav.is_1d_array(dataset):
                discretised_dataset[feature] = np.searchsorted(
                    boundaries, dataset[feature])
            else:
                discretised_dataset[:, feature] = np.searchsorted(
                    boundaries, dataset[:, feature])

        return discretised_dataset
