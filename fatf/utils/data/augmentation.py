"""
The :mod:`fatf.utils.data.augmentation` module implements data set augmenters.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
#         Kacper Sokol <k.sokol@bristol.ac.uk>
#         Rafael Poyiadzi <rp13102@bristol.ac.uk>
# License: new BSD

# pylint: disable=too-many-lines

from numbers import Number
from typing import Callable, List, Optional, Tuple, Union
from typing import Set  # pylint: disable=unused-import

import abc
import logging
import warnings

import scipy.stats
import scipy.spatial

import numpy as np

from fatf.exceptions import IncompatibleModelError, IncorrectShapeError

import fatf.utils.array.tools as fuat
import fatf.utils.array.validation as fuav
import fatf.utils.distances as fud
import fatf.utils.validation as fuv

__all__ = ['Augmentation',
           'NormalSampling',
           'TruncatedNormalSampling',
           'Mixup',
           'NormalClassDiscovery',
           'DecisionBoundarySphere',
           'LocalSphere']  # yapf: disable

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

Index = Union[int, str]


def _validate_input(dataset: np.ndarray,
                    ground_truth: Optional[np.ndarray] = None,
                    categorical_indices: Optional[List[Index]] = None,
                    int_to_float: bool = True) -> bool:
    """
    Validates the input parameters of an arbitrary augmentation class.

    For the description of the input parameters and exceptions raised by this
    function, please see the documentation of the
    :class:`fatf.utils.data.augmentation.Augmentation` class.

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

    if ground_truth is not None:
        if not fuav.is_1d_array(ground_truth):
            raise IncorrectShapeError('The ground_truth array must be '
                                      '1-dimensional. (Or None if it is not '
                                      'required.)')
        if not fuav.is_base_array(ground_truth):
            raise TypeError('The ground_truth array must be of a base type.')
        if ground_truth.shape[0] != dataset.shape[0]:
            raise IncorrectShapeError('The number of labels in the '
                                      'ground_truth array is not equal to the '
                                      'number of data points in the dataset '
                                      'array.')

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

    if not isinstance(int_to_float, bool):
        raise TypeError('The int_to_float parameter has to be a boolean.')

    is_valid = True
    return is_valid


class Augmentation(abc.ABC):
    """
    An abstract class for implementing data augmentation methods.

    An abstract class that all augmentation classes should inherit from. It
    contains abstract ``__init__`` and ``sample`` methods and an input
    validator -- ``_validate_sample_input`` -- for the ``sample`` method. The
    validation of the input parameters to the initialisation method is done via
    the ``fatf.utils.data.augmentation._validate_input`` function.

    .. note::
       The ``_validate_sample_input`` method should be called in all
       implementations of the ``sample`` method in the children classes to
       ensure that all the input parameters of this method are valid.

    Parameters
    ----------
    dataset : numpy.ndarray
        A 2-dimensional numpy array with a dataset to be used for sampling.
    ground_truth : numpy.ndarray, optional (default=None)
        A 1-dimensional numpy array with labels for the supplied dataset.
    categorical_indices : List[column indices], optional (default=None)
        A list of column indices that should be treat as categorical features.
        If ``None`` is given this will be inferred from the data array:
        string-based columns will be treated as categorical features and
        numerical columns will be treated as numerical features.
    int_to_float : boolean
        If ``True``, all of the integer dtype columns in the ``dataset`` will
        be generalised to ``numpy.float64`` type. Otherwise, integer type
        columns will remain integer and floating point type columns will remain
        floating point.

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
        The input ``dataset`` is not a 2-dimensional numpy array. The
        ``ground_truth`` array is not a 1-dimensional numpy array. The number
        of ground truth annotation is different than the number of rows in the
        data array.
    IndexError
        Some of the column indices given in the ``categorical_indices``
        parameter are not valid for the input ``dataset``.
    TypeError
        The ``categorical_indices`` parameter is neither a list nor ``None``.
        The ``dataset`` or the ``ground_truth`` array (if not ``None``) are not
        of base (numerical and/or string) type. The ``int_to_float`` parameter
        is not a boolean.

    Attributes
    ----------
    dataset : numpy.ndarray
        A 2-dimensional numpy array with a dataset to be used for sampling.
    data_points_number : integer
        The number of data points in the ``dataset``.
    is_structured : boolean
        ``True`` if the ``dataset`` is a structured numpy array, ``False``
        otherwise.
    ground_truth : Union[numpy.ndarray, None]
        A 1-dimensional numpy array with labels for the supplied dataset.
    categorical_indices : List[column indices]
        A list of column indices that should be treat as categorical features.
    numerical_indices : List[column indices]
        A list of column indices that should be treat as numerical features.
    features_number : integer
        The number of features (columns) in the input ``dataset``.
    sample_dtype : Union[numpy.dtype, List[Tuple[string, numpy.dtype]]
        A dtype with numerical dtypes (in case of a structured data array)
        generalised to support the assignment of sampled values. For example,
        if the dtype of a numerical feature is ``int`` and the sampling
        generates ``float`` this dtype will generalise the type of that column
        to ``float``.
    """

    # pylint: disable=too-few-public-methods,too-many-instance-attributes
    def __init__(self,
                 dataset: np.ndarray,
                 ground_truth: Optional[np.ndarray] = None,
                 categorical_indices: Optional[np.ndarray] = None,
                 int_to_float: bool = True) -> None:
        """
        Constructs an ``Augmentation`` abstract class.
        """
        # pylint: disable=too-many-locals
        assert _validate_input(
            dataset,
            ground_truth=ground_truth,
            categorical_indices=categorical_indices,
            int_to_float=int_to_float), 'Invalid input.'

        self.dataset = dataset
        self.data_points_number = dataset.shape[0]
        self.is_structured = fuav.is_structured_array(dataset)

        self.ground_truth = ground_truth

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

        # Sort out the dtype of the sampled array.
        ntype = np.dtype(np.float64) if int_to_float else np.dtype(np.int64)
        if self.is_structured:
            sample_dtype = []
            for column_name in self.dataset.dtype.names:
                if column_name in self.numerical_indices:
                    new_dtype = fuat.generalise_dtype(
                        self.dataset.dtype[column_name], ntype)
                    sample_dtype.append((column_name, new_dtype))
                elif column_name in self.categorical_indices:
                    sample_dtype.append((column_name,
                                         self.dataset.dtype[column_name]))
                else:
                    assert False, 'Unknown column name.'  # pragma: nocover
        else:
            if fuav.is_numerical_array(self.dataset):
                sample_dtype = fuat.generalise_dtype(self.dataset.dtype, ntype)
            else:
                sample_dtype = self.dataset.dtype
        self.sample_dtype = sample_dtype

    @abc.abstractmethod
    def sample(self,
               data_row: Optional[Union[np.ndarray, np.void]] = None,
               samples_number: int = 50) -> np.ndarray:
        """
        Samples a given number of data points based on the initialisation data.

        This is an abstract method that must be implemented for each child
        object. This method should provide two modes of operation:

        - if ``data_row`` is ``None``, the sample should be from the
          distribution of the whole dataset that was used to initialise this
          class; and

        - if ``data_row`` is a numpy array with a data point, the sample should
          be from the vicinity of this data point.

        Parameters
        ----------
        data_row : Union[numpy.ndarray, numpy.void], optional (default=None)
            A data point. If given, the sample will be generated around that
            point.
        samples_number : integer, optional (default=50)
            The number of samples to be generated.

        Raises
        ------
        NotImplementedError
            This is an abstract method and has not been implemented.

        Returns
        -------
        samples : numpy.ndarray
            Sampled data.
        """
        assert self._validate_sample_input(  # pragma: nocover
            data_row, samples_number), 'Invalid sample method input.'

        raise NotImplementedError(  # pragma: nocover
            'sample method needs to be overwritten.')

    def _validate_sample_input(self,
                               data_row: Union[None, np.ndarray, np.void],
                               samples_number: int) -> bool:
        """
        Validates input parameters of the ``sample`` method.

        This function checks the validity of ``data_row`` and
        ``samples_number`` parameters.

        Raises
        ------
        IncorrectShapeError
            The ``data_row`` is not a 1-dimensional numpy array-like object.
            The number of features (columns) in the ``data_row`` is different
            to the number of features in the data array used to initialise this
            object.
        TypeError
            The dtype of the ``data_row`` is different than the dtype of the
            data array used to initialise this object. The ``samples_number``
            parameter is not an integer.
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

            are_similar = fuav.are_similar_dtype_arrays(
                self.dataset, np.array([data_row]), strict_comparison=True)
            if not are_similar:
                raise TypeError('The dtype of the data_row is different to '
                                'the dtype of the data array used to '
                                'initialise this class.')

            # If the dataset is structured and the data_row has a different
            # number of features this will be caught by the above dtype check.
            # For classic numpy arrays this has to be done separately.
            if not self.is_structured:
                if data_row.shape[0] != self.dataset.shape[1]:
                    raise IncorrectShapeError('The data_row must contain the '
                                              'same number of features as the '
                                              'dataset used to initialise '
                                              'this class.')

        if isinstance(samples_number, int):
            if samples_number < 1:
                raise ValueError('The samples_number parameter must be a '
                                 'positive integer.')
        else:
            raise TypeError('The samples_number parameter must be an integer.')

        is_valid = True
        return is_valid


class NormalSampling(Augmentation):
    """
    Sampling data from a normal distribution.

    This class allows to sample data according to a normal distribution. The
    sampling can be performed either around a particular data point (by
    supplying the ``data_row`` parameter to the ``sample`` method) or around
    the mean of the whole ``dataset`` (if ``data_row`` is not given when
    calling the ``sample`` method). In both cases, the standard deviation
    of each numerical feature calculated for the whole dataset is used. For
    categorical features, the values are sampled with replacement with the
    probability for each unique value calculated based on the frequency of its
    appearance in the dataset.

    For additional parameters, attributes, warnings and exceptions raised by
    this class please see the documentation of its parent class:
    :class:`fatf.utils.data.augmentation.Augmentation`.

    Attributes
    ----------
    numerical_sampling_values : Dictionary[column index, Tuple[number, number]]
        Dictionary mapping numerical column feature indices to tuples of two
        numbers: column's *mean* and its *standard deviation*.
    categorical_sampling_values : Dictionary[column index, \
Tuple[numpy.ndarray, numpy.ndarray]]
        Dictionary mapping categorical column feature indices to tuples
        consisting of two 1-dimensional numpy arrays: one with unique values
        for that column and the other one with their normalised (summing up to
        1) frequencies.
    """

    # pylint: disable=too-few-public-methods
    def __init__(self,
                 dataset: np.ndarray,
                 categorical_indices: Optional[List[Index]] = None,
                 int_to_float: bool = True) -> None:
        """
        Constructs a ``NormalSampling`` data augmentation class.
        """
        # pylint: disable=too-many-locals,too-many-branches
        super().__init__(
            dataset,
            categorical_indices=categorical_indices,
            int_to_float=int_to_float)

        # Get sampling parameters for numerical features.
        numerical_sampling_values = dict()
        if self.numerical_indices:
            if self.is_structured:
                num_features_array = fuat.as_unstructured(
                    self.dataset[self.numerical_indices])
            else:
                num_features_array = self.dataset[:, self.numerical_indices]

            num_features_mean = num_features_array.mean(axis=0)
            num_features_std = num_features_array.std(axis=0)

            for i, index in enumerate(self.numerical_indices):
                numerical_sampling_values[index] = (num_features_mean[i],
                                                    num_features_std[i])
        self.numerical_sampling_values = numerical_sampling_values

        # Get sampling parameters for categorical features.
        categorical_sampling_values = dict()
        for column_name in self.categorical_indices:
            if self.is_structured:
                feature_column = self.dataset[column_name]
            else:
                feature_column = self.dataset[:, column_name]

            feature_values, values_counts = np.unique(
                feature_column, return_counts=True)
            values_frequencies = values_counts / values_counts.sum()

            categorical_sampling_values[column_name] = (feature_values,
                                                        values_frequencies)
        self.categorical_sampling_values = categorical_sampling_values

    def sample(self,
               data_row: Optional[Union[np.ndarray, np.void]] = None,
               samples_number: int = 50) -> np.ndarray:
        """
        Samples new data from a normal distribution.

        If ``data_row`` parameter is given, the sample will be centered around
        that data point. Otherwise, when the ``data_row`` parameter is
        ``None``, the sample will be generated around the mean of the dataset
        used to initialise this class.

        Numerical features are sampled around their corresponding values in the
        ``data_row`` parameter or the mean of that feature in the dataset using
        the standard deviation calculated from the dataset. Categorical
        features are sampled by choosing with replacement all the possible
        values of that feature with the probability of sampling each value
        corresponding to this value's frequency in the dataset. (This means
        that any particular value of a categorical feature in a ``data_row`` is
        ignored.)

        For the documentation of parameters, warnings and errors please see the
        description of the
        :func:`~fatf.utils.data.augmentation.Augmentation.sample` method in the
        parent :class:`fatf.utils.data.augmentation.Augmentation` class.
        """
        assert self._validate_sample_input(data_row,
                                           samples_number), 'Invalid input.'

        # Create an array to hold the samples.
        if self.is_structured:
            shape = (samples_number, )  # type: Tuple[int, ...]
        else:
            shape = (samples_number, self.features_number)
        samples = np.zeros(shape, dtype=self.sample_dtype)

        # Sample categorical features.
        for index in self.categorical_indices:
            sample_values = np.random.choice(
                self.categorical_sampling_values[index][0],
                size=samples_number,
                replace=True,
                p=self.categorical_sampling_values[index][1])
            if self.is_structured:
                samples[index] = sample_values
            else:
                samples[:, index] = sample_values

        # Sample numerical features.
        for index in self.numerical_indices:
            # Fetch mean ans standard deviation
            sampling_parameters = self.numerical_sampling_values[index]
            std = sampling_parameters[1]
            # If a data row is given sample around that value, otherwise
            # sample around data mean.
            if data_row is None:
                mean = sampling_parameters[0]
            else:
                mean = data_row[index]

            sample_values = np.random.normal(0, 1, samples_number) * std + mean

            if self.is_structured:
                samples[index] = sample_values
            else:
                samples[:, index] = sample_values

        return samples


class TruncatedNormalSampling(Augmentation):
    """
    Sampling data from a truncated normal distribution.

    .. versionadded:: 0.0.2

    This class allows to sample data according to the
    `truncated normal distribution`_. The sampling can be performed either
    around a particular data point (by supplying the ``data_row`` parameter to
    the ``sample`` method) or around the mean of the whole ``dataset`` (if
    ``data_row`` is not given when calling the ``sample`` method). In both
    cases, the standard deviation of each numerical feature calculated for the
    whole ``dataset`` is used. The minimum and maximum of each numerical
    feature are also used as the bounds for the truncated normal distribution.
    For categorical features, the values are sampled with replacement with the
    probability for each unique value calculated based on the frequency of
    their appearance in the dataset.

    For additional parameters, attributes, warnings and exceptions raised by
    this class please see the documentation of its parent class:
    :class:`fatf.utils.data.augmentation.Augmentation`.

    .. _`truncated normal distribution`: https://en.wikipedia.org/wiki/
       Truncated_normal_distribution

    Attributes
    ----------
    numerical_sampling_values : Dictionary[column index, \
Tuple[number, number, number, number]]
        Dictionary mapping numerical column feature indices to tuples of four
        numbers: column's *mean*, *standard deviation*, its *minimum* and
        *maximum* value.
    categorical_sampling_values : Dictionary[column index, \
Tuple[numpy.ndarray, numpy.ndarray]]
        Dictionary mapping categorical column feature indices to tuples
        consisting of two 1-dimensional numpy arrays: one with unique values
        for that column and the other one with their normalised (summing up to
        1) frequencies.
    """

    # pylint: disable=too-few-public-methods

    def __init__(self,
                 dataset: np.ndarray,
                 categorical_indices: Optional[List[Index]] = None,
                 int_to_float: bool = True) -> None:
        """
        Constructs a ``TruncatedNormalSampling`` data augmentation class.
        """
        # pylint: disable=too-many-locals
        super().__init__(
            dataset=dataset,
            categorical_indices=categorical_indices,
            int_to_float=int_to_float)

        # Get sampling parameters for numerical features.
        numerical_sampling_values = dict()
        if self.numerical_indices:
            if self.is_structured:
                num_features_array = fuat.as_unstructured(
                    self.dataset[self.numerical_indices])
            else:
                num_features_array = self.dataset[:, self.numerical_indices]

            num_features_mean = num_features_array.mean(axis=0)
            num_features_std = num_features_array.std(axis=0)
            num_features_min = num_features_array.min(axis=0)
            num_features_max = num_features_array.max(axis=0)

            for i, index in enumerate(self.numerical_indices):
                numerical_sampling_values[index] = (num_features_mean[i],
                                                    num_features_std[i],
                                                    num_features_min[i],
                                                    num_features_max[i])
        self.numerical_sampling_values = numerical_sampling_values

        # Get sampling parameters for categorical features.
        categorical_sampling_values = dict()
        for column_name in self.categorical_indices:
            if self.is_structured:
                feature_column = self.dataset[column_name]
            else:
                feature_column = self.dataset[:, column_name]

            feature_values, values_counts = np.unique(
                feature_column, return_counts=True)
            values_frequencies = values_counts / values_counts.sum()

            categorical_sampling_values[column_name] = (feature_values,
                                                        values_frequencies)
        self.categorical_sampling_values = categorical_sampling_values

    def sample(self,
               data_row: Optional[Union[np.ndarray, np.void]] = None,
               samples_number: int = 50) -> np.ndarray:
        """
        Samples new data from a truncated normal distribution.

        If ``data_row`` parameter is given, the sample will be centered around
        that data point. Otherwise, when the ``data_row`` parameter is
        ``None``, the sample will be generated around the mean of the dataset
        used to initialise this class.

        Numerical features are sampled around their corresponding values in the
        ``data_row`` parameter or the mean of that feature in the dataset using
        the standard deviation, minimum and maximum values calculated from the
        dataset. Categorical features are sampled by choosing with replacement
        all the possible values of that feature with the probability of
        sampling each value corresponding to this value's frequency in the
        dataset. (This means that any particular value of a categorical feature
        in a ``data_row`` is ignored.)

        For the documentation of parameters, warnings and errors please see the
        description of the
        :func:`fatf.utils.data.augmentation.Augmentation.sample` method in the
        parent :class:`fatf.utils.data.augmentation.Augmentation` class.
        """
        assert self._validate_sample_input(data_row,
                                           samples_number), 'Invalid input.'
        # Create an array to hold the samples.
        if self.is_structured:
            shape = (samples_number, )  # type: Tuple[int, ...]
        else:
            shape = (samples_number, self.features_number)
        samples = np.zeros(shape, dtype=self.sample_dtype)

        # Sample categorical features.
        for index in self.categorical_indices:
            sample_values = np.random.choice(
                self.categorical_sampling_values[index][0],
                size=samples_number,
                replace=True,
                p=self.categorical_sampling_values[index][1])
            if self.is_structured:
                samples[index] = sample_values
            else:
                samples[:, index] = sample_values

        # Sample numerical features.
        for index in self.numerical_indices:
            sampling_parameters = self.numerical_sampling_values[index]
            mean, std, minimum, maximum = sampling_parameters
            if data_row is not None:
                mean = data_row[index]

            sample_values = scipy.stats.truncnorm.rvs(
                (minimum - mean) / std, (maximum - mean) / std,
                loc=mean,
                scale=std,
                size=samples_number)

            if self.is_structured:
                samples[index] = sample_values
            else:
                samples[:, index] = sample_values

        return samples


def _validate_input_mixup(
        beta_parameters: Union[None, Tuple[float, float]]) -> bool:
    """
    Validates :class:``.Mixup`` class-specific input parameters.

    Parameters
    ----------
    beta_parameters : Union[Tuple[number, number], None]
        Either ``None`` (for the default values) or a pair of numbers that will
        be used as beta distribution parameters.

    Raises
    ------
    TypeError
        The ``beta_parameters`` parameter is neither ``None`` nor a tuple. One
        of the values in the ``beta_parameters`` tuple is not a number.
    ValueError
        The ``beta_parameters`` tuple is not a pair (2-tuple). One of the
        numbers in the ``beta_parameters`` tuple is not positive.

    Returns
    -------
    is_valid : boolean
        ``True`` if input is valid, ``False`` otherwise.
    """
    is_valid = False

    # Check beta parameters
    if beta_parameters is None:
        pass
    elif isinstance(beta_parameters, tuple):
        if len(beta_parameters) != 2:
            raise ValueError('The beta_parameters parameter has to be a '
                             '2-tuple (a pair) of numbers.')
        for index, name in enumerate(['first', 'second']):
            if isinstance(beta_parameters[index], Number):
                if beta_parameters[index] <= 0:
                    raise ValueError('The {} beta parameter cannot be a '
                                     'negative number.'.format(name))
            else:
                raise TypeError('The {} beta parameter has to be a '
                                'numerical type.'.format(name))
    else:
        raise TypeError('The beta_parameters parameter has to be a tuple '
                        'with two numbers or None to use the default '
                        'parameters value.')

    is_valid = True
    return is_valid


class Mixup(Augmentation):
    """
    Sampling data with the Mixup method.

    This object implements the Mixup method introduced by [ZHANG2018MIXUP]_.
    For a specific data point it select points at random from the ``dataset``
    (making sure that the sample is stratified when the ``ground_truth``
    parameter is given), then it draws samples from a Beta distribution and it
    forms new data points (samples) according to the convex combination of the
    original data pint and the randomly sampled dataset points.

    .. note::
       Sampling from the ``dataset`` mean is not yet implemented.

    For additional parameters, attributes, warnings and exceptions raised by
    this class please see the documentation of its parent class:
    :class:`fatf.utils.data.augmentation.Augmentation` and the function that
    validates the input parameters
    ``fatf.utils.data.augmentation._validate_input_mixup``.

    .. [ZHANG2018MIXUP] Zhang, H., Cisse, M., Dauphin, Y. N. and Lopez-Paz, D.,
       2018. mixup: Beyond Empirical Risk Minimization. International
       Conference on Learning Representations (ICLR 2018).

    Parameters
    ----------
    beta_parameters : Tuple[number, number]], optional (default=None)
        A pair of numerical parameters used with the Beta distribution. If
        ``None``, the beta parameters will be set to ``(2, 5)``.

    Raises
    ------
    TypeError
        The ``beta_parameters`` parameter is neither ``None`` nor a tuple. One
        of the values in the ``beta_parameters`` tuple is not a number.
    ValueError
        The ``beta_parameters`` tuple is not a pair (2-tuple). One of the
        numbers in the ``beta_parameters`` tuple is not positive.

    Attributes
    ----------
    threshold : number
        A threshold used for mixing the random sample from the ``dataset`` with
        the instance used to generate a sample. The threshold value is 0.5.
    beta_parameters : Tuple[number, number]
        A pair of numbers used with the Beta distribution sampling.
    ground_truth_unique : np.ndarray
        A sorted array holding all the unique values of the ground truth.
    ground_truth_frequencies : np.ndarray
        An array holding frequencies of all the unique values in the ground
        truth array. The order of the frequencies correspond with the order of
        the unique values. The frequencies are normalised and they sum up to 1.
    indices_per_label : List[np.ndarray]
        A list of arrays holding (``dataset``) row indices corresponding to
        each of the unique ground truth values. The order of this list
        corresponds with the order of the unique values.
    ground_truth_probabilities : np.ndarray
        A numpy array of [number of dataset instances, number of unique ground
        truth values] shape that holds one-hot encoding (pseudo-probabilities)
        of the ground truth labels. The column ordering of this array
        corresponds with the order of the unique values.
    """

    # pylint: disable=too-few-public-methods
    def __init__(self,
                 dataset: np.ndarray,
                 ground_truth: Optional[np.ndarray] = None,
                 categorical_indices: Optional[np.ndarray] = None,
                 beta_parameters: Optional[Tuple[float, float]] = None,
                 int_to_float: bool = True) -> None:
        """
        Constructs a ``Mixup`` data augmentation class.
        """
        # pylint: disable=too-many-arguments
        super().__init__(
            dataset,
            ground_truth=ground_truth,
            categorical_indices=categorical_indices,
            int_to_float=int_to_float)
        assert _validate_input_mixup(beta_parameters), 'Invalid Mixup input.'

        self.threshold = 0.50

        # Get the distribution of the ground truth and collect row indices per
        # label
        if ground_truth is None:
            ground_truth_unique = None
            ground_truth_frequencies = None
            indices_per_label = None
            ground_truth_probabilities = None
        else:
            ground_truth_unique, counts = np.unique(
                self.ground_truth, return_counts=True)
            ground_truth_frequencies = counts / counts.sum()
            indices_per_label = [
                np.where(self.ground_truth == label)[0]
                for label in ground_truth_unique
            ]

            # Get pseudo-probabilities per instance, i.e. 1 indicates the label
            ground_truth_probabilities = np.zeros(
                (self.data_points_number, ground_truth_unique.shape[0]),
                dtype=np.int8)  # np.int8 suffices since these are 0s and 1s
            for i, indices in enumerate(indices_per_label):
                ground_truth_probabilities[indices, i] = 1
        self.ground_truth_unique = ground_truth_unique
        self.ground_truth_frequencies = ground_truth_frequencies
        self.indices_per_label = indices_per_label
        self.ground_truth_probabilities = ground_truth_probabilities

        # Check beta parameters
        if beta_parameters is None:
            beta_parameters = (2, 5)
        self.beta_parameters = beta_parameters

    def _validate_sample_input_mixup(
            self, data_row_target: Union[float, str, None],
            with_replacement: bool, return_probabilities: bool) -> bool:
        """
        Validates ``sample`` method input parameters for the ``Mixup`` class.

        This function checks the validity of ``data_row_target``,
        ``with_replacement`` and ``return_probabilities`` parameters.

        Parameters
        ----------
        data_row_target : Union[number, string, None]
            Either ``None`` or a label (class) of the data row to sample new
            data around.
        with_replacement : boolean
            A boolean parameter that indicates whether the ``dataset`` row
            indices should be sampled with replacements (``True``) or not
            (``False``).
        return_probabilities : boolean
            A boolean parameter that indicates whether the sampled target array
            should a class probability matrix (``True``) or a 1-dimensional
            array with the labels (``False``).

        Warns
        -----
        UserWarning
            The user is warned when the ``data_row_target`` parameter is given
            but the ``Mixup`` class was initialised without the ground truth
            for the ``dataset``, therefore sampling target values is not
            possible and the ``data_row_target`` parameter will be ignored.

        Raises
        ------
        TypeError
            The ``return_probabilities`` or ``with_replacement`` parameters are
            not booleans. The ``data_row_target`` parameter is neither a number
            not a string.
        ValueError
            The ``data_row_target`` parameter has a value that does not appear
            in the ground truth vector used to initialise this class.

        Returns
        -------
        is_valid : boolean
            ``True`` if input parameters are valid, ``False`` otherwise.
        """
        is_valid = False

        if data_row_target is None:
            pass
        elif isinstance(data_row_target, (Number, str)):
            if self.ground_truth_unique is None:
                msg = ('This Mixup class has not been initialised with a '
                       'ground truth vector. The value of the data_row_target '
                       'parameter will be ignored, therefore target values '
                       'samples will not be returned.')
                warnings.warn(msg, UserWarning)
            else:
                if data_row_target not in self.ground_truth_unique:
                    raise ValueError('The value of the data_row_target '
                                     'parameter is not present in the ground '
                                     'truth labels used to initialise this '
                                     'class. The data row target value is not '
                                     'recognised.')
        else:
            raise TypeError('The data_row_target parameter should either be '
                            'None or a string/number indicating the target '
                            'class.')

        if not isinstance(with_replacement, bool):
            raise TypeError('with_replacement parameter has to be boolean.')

        if not isinstance(return_probabilities, bool):
            raise TypeError('return_probabilities parameter has to be '
                            'boolean.')

        is_valid = True
        return is_valid

    def _get_stratified_indices(self, samples_number: int,
                                with_replacement: bool) -> np.ndarray:
        """
        Selects random row indices from the ``dataset``.

        Selects ``samples_number`` number of row indices at random either with
        replacements or not (depending on the value of the ``with_replacement``
        parameter). The indices selection is stratified according to the ground
        truth distribution if ground truth vector was given when this class
        was initialised. Otherwise, the indices are generated at random.

        Parameters
        ----------
        samples_number : integer
            The number of data points to be sampled.
        with_replacement : boolean
            A boolean parameter that indicates whether the ``dataset`` row
            indices should be sampled with replacements (``True``) or not
            (``False``).

        Warns
        -----
        UserWarning
            The user is warned that the random row indices will not be
            stratified according to the ground truth distribution if ground
            truth vector was not given when this class was initialised.

        Returns
        -------
        random_indices : numpy.ndarray
            A 1-dimensional numpy array of shape [samples_number, ] that holds
            randomly selected row indices from the ``dataset``.
        """
        assert isinstance(samples_number, int), 'Has to be an integer.'
        assert samples_number > 0, 'Has to be positive.'
        #
        assert isinstance(with_replacement, bool), 'Has to be boolean.'

        if self.ground_truth_frequencies is None:
            msg = ('Since the ground truth vector was not provided while '
                   'initialising the Mixup class it is not possible to get a '
                   'stratified sample of data points. Instead, Mixup will '
                   'choose data points at random, which is equivalent to '
                   'assuming that the class distribution is balanced.')
            warnings.warn(msg, UserWarning)

            random_indices = np.random.choice(
                self.data_points_number,
                samples_number,
                replace=with_replacement)
        else:
            # Get sample quantities per class -- stratified
            samples_per_label = [
                int(freq * samples_number)
                for freq in self.ground_truth_frequencies
            ]

            # Due to integer casting there may be a sub- or under-sampling
            # happening. This gets corrected for below.
            samples_per_label_len = len(samples_per_label)
            diff = samples_number - sum(samples_per_label)
            diff_val = 1 if diff >= 0 else -1
            for _ in range(diff):
                random_index = np.random.randint(0, samples_per_label_len)
                samples_per_label[random_index] += diff_val
            assert samples_number == sum(samples_per_label), 'Wrong quantity.'

            # Get a sample representative of the original label distribution
            random_indices = []
            for i, label_sample_quantity in enumerate(samples_per_label):
                random_indices_label = np.random.choice(
                    self.indices_per_label[i],  # type: ignore
                    label_sample_quantity,
                    replace=with_replacement)
                random_indices.append(random_indices_label)
            random_indices = np.concatenate(random_indices)

        return random_indices

    def _get_sample_targets(self, data_row_target: Union[float, str],
                            return_probabilities: bool,
                            random_draws_lambda: np.ndarray,
                            random_draws_lambda_1: np.ndarray,
                            random_indices: np.ndarray) -> np.ndarray:
        """
        Samples target values for the sampled data instance.

        The target values can either be represented as a class probability
        matrix (``return_probabilities`` set to ``True``) or an array with a
        single label per instance selected based on the highest probability
        (``return_probabilities`` set to ``False``).

        Parameters
        ----------
        data_row_target : Union[number, string]
            A label (class) of the data row to sample new data around.
        return_probabilities : boolean,
            A boolean parameter that indicates whether the sampled target array
            should a class probability matrix (``True``) or a 1-dimensional
            array with the labels (``False``).
        random_draws_lambda : numpy.ndarray,
            A numpy array with the Beta distribution sample.
        random_draws_lambda_1 : numpy.ndarray,
            A numpy array with *1 -* the Beta distribution sample.
        random_indices : numpy.ndarray
            A 1-dimensional numpy array of shape [samples_number, ] that holds
            randomly selected row indices from the ``dataset``.

        Returns
        -------
        samples_target : numpy.ndarray
            Either a numpy array of shape [samples_number, number of unique
            labels (classes)] holding the class probabilities for the sampled
            data or a 1-dimensional numpy array with labels for the sampled
            data.
        """
        # pylint: disable=too-many-arguments
        assert isinstance(data_row_target, (Number, str)), 'Invalid label.'
        assert isinstance(return_probabilities, bool), 'Must be boolean.'
        #
        assert self.ground_truth_unique is not None, 'Missing ground truth.'
        assert self.ground_truth_probabilities is not None, 'Missing labels.'

        # Encode the target as a probability vector (one-hot encoding)
        encoded_data_row_target = np.zeros(
            (1, self.ground_truth_unique.shape[0]), dtype=np.int8)
        target_index_mask = self.ground_truth_unique == data_row_target
        encoded_data_row_target[0, target_index_mask] = 1
        assert encoded_data_row_target.sum() == 1, 'Invalid probability array.'

        # Sort out labels -- this will be probability vectors
        st1 = np.apply_along_axis(
            np.multiply, 0, self.ground_truth_probabilities[random_indices],
            random_draws_lambda_1)
        st2 = np.apply_along_axis(np.multiply, 0, encoded_data_row_target,
                                  random_draws_lambda)
        samples_target = st1 + st2

        # Sort out labels -- this will be numbers
        # samples_target = (
        #     random_draws_lambda_1 * self.ground_truth[random_indices]
        #     + random_draws_lambda * data_row_target)

        # If the user wants labels rather than probabilities...
        if not return_probabilities:
            tmap = np.vectorize(lambda index: self.ground_truth_unique[index])
            target_index = samples_target.argmax(axis=1)
            samples_target = tmap(target_index)

        return samples_target

    def sample(  # type: ignore
            self,
            data_row: Optional[Union[np.ndarray, np.void]] = None,
            data_row_target: Optional[Union[float, str]] = None,
            samples_number: int = 50,
            with_replacement: bool = True,
            return_probabilities: bool = False) -> Tuple[np.ndarray, ...]:
        """
        Samples new data around the provided ``data_row`` using Mixup method.

        If ``data_row_target`` is ``None``, only sampled data will be returned.
        Otherwise, if ``data_row_target`` is provided, the ``Mixup`` class will
        also attempt to sample labels. In this case the labels can either be
        an array of class probabilities when the ``return_probabilities``
        parameter is set to ``True``, or an array with a single label per
        instance selected based on the highest probability when the
        ``return_probabilities`` parameter is set to ``False``.

        .. note::
           Sampling from the ``dataset`` mean is not yet implemented.

        For the documentation of extra parameters, warnings and errors please
        see the description of the
        :func:`~fatf.utils.data.augmentation.Augmentation.sample` method in the
        parent :class:`fatf.utils.data.augmentation.Augmentation` class.

        Parameters
        ----------
        data_row_target : Union[number, string], optional (default=None)
            A label (class) of the provided ``data_row``. If ``None`` the
            function will only return sampled data, otherwise it will also
            return targets for the sampled data.
        with_replacement : boolean, optional (default=True)
            If ``True`` data points are sampled with replacements from the
            original ``dataset``.
        return_probabilities : boolean, optional (default=False)
            If ``True`` the target (class) samples for the sampled data points
            are in form of a class probability matrix, otherwise they are a
            flat array with the target labels.

        Warns
        -----
        UserWarning
            The user is warned when the ``data_row_target`` parameter is given
            but the ``Mixup`` class was initialised without the ground truth
            for the ``dataset``, therefore sampling target values is not
            possible and the ``data_row_target`` parameter will be ignored.
            The user is also warned that the random row indices will not be
            stratified according to the ground truth distribution if ground
            truth vector was not given when this class was initialised.

        Raises
        ------
        NotImplementedError
            Raised when the user is trying to sample around the mean of the
            ``dataset`` -- this functionality is not yet implemented.
        TypeError
            The ``return_probabilities`` or ``with_replacement`` parameters are
            not booleans. The ``data_row_target`` parameter is neither a number
            not a string.
        ValueError
            The ``data_row_target`` parameter has a value that does not appear
            in the ground truth vector used to initialise this class.

        Returns
        -------
        samples : numpy.ndarray
            A numpy array of shape [``samples_number``, number of features]
            that holds the sampled data.
        samples_target : numpy.ndarray, optional (returned when the \
``data_row_target`` parameter is not ``None``)
            Either a numpy array of shape [samples_number, number of unique
            labels (classes)] holding the class probabilities for the sampled
            data or a 1-dimensional numpy array with labels for the sampled
            data.
        """
        # pylint: disable=arguments-differ,too-many-locals,too-many-arguments
        assert self._validate_sample_input(
            data_row, samples_number), 'Invalid mixup sampling input.'
        assert self._validate_sample_input_mixup(
            data_row_target, with_replacement,
            return_probabilities), 'Invalid mixup sampling input.'

        if data_row is None:
            raise NotImplementedError('Sampling around the data mean is not '
                                      'yet implemented for the Mixup class.')

        # Get stratified random row indices of the original data set
        random_indices = self._get_stratified_indices(samples_number,
                                                      with_replacement)
        random_data_points = self.dataset[random_indices]

        random_draws_lambda = np.random.beta(*self.beta_parameters,
                                             samples_number)
        random_draws_lambda_1 = 1 - random_draws_lambda
        mask = random_draws_lambda <= self.threshold

        # Create an array to hold the samples.
        if self.is_structured:
            shape = (samples_number, )  # type: Tuple[int, ...]
        else:
            shape = (samples_number, self.features_number)
        samples = np.zeros(shape, dtype=self.sample_dtype)

        # Sort out numerical features
        # yapf: disable
        for index in self.numerical_indices:
            if self.is_structured:
                samples[index] = (
                    random_draws_lambda_1 * random_data_points[index]
                    + random_draws_lambda * data_row[index])
            else:
                samples[:, index] = (
                    random_draws_lambda_1 * random_data_points[:, index]
                    + random_draws_lambda * data_row[index])
        # yapf: enable

        # Sort out categorical features
        for index in self.categorical_indices:
            if self.is_structured:
                samples[index][mask] = data_row[index]
                samples[index][~mask] = random_data_points[index][~mask]
            else:
                samples[mask, index] = data_row[index]
                samples[~mask, index] = random_data_points[~mask, index]

        # Get target values/probabilities sample if requested
        if self.ground_truth_unique is None or data_row_target is None:
            to_return = samples
        else:
            samples_target = self._get_sample_targets(
                data_row_target, return_probabilities, random_draws_lambda,
                random_draws_lambda_1, random_indices)
            to_return = samples, samples_target
        return to_return


def _validate_input_normalclassdiscovery(
        predictive_function: Callable[[np.ndarray], np.ndarray],
        classes_number: Union[None, int], class_proportion_threshold: float,
        standard_deviation_init: float,
        standard_deviation_increment: float) -> bool:
    """
    Validates the input parameters of the ``NormalClassDiscovery`` class.

    This function validates input parameters of the
    :class:`fatf.utils.data.augmentation.NormalClassDiscovery` class. For the
    description of the input parameters and errors please see the documentation
    of the :class:`fatf.utils.data.augmentation.NormalClassDiscovery` class.

    Returns
    -------
    is_valid : boolean
        ``True`` if the input is valid, ``False`` otherwise.
    """
    # pylint: disable=too-many-branches
    is_valid = False

    if callable(predictive_function):
        prms_n = fuv.get_required_parameters_number(predictive_function)
        if prms_n != 1:
            raise IncompatibleModelError('The predictive function must take '
                                         'exactly *one* required parameter: '
                                         'a data array to be predicted.')
    else:
        raise TypeError('The predictive_function should be a Python '
                        'callable, e.g., a Python function.')

    if classes_number is not None:
        if isinstance(classes_number, int):
            if classes_number < 2:
                raise ValueError('The classes_number parameter has to be an '
                                 'integer larger than 1 (at least a binary '
                                 'classification problem).')
        else:
            raise TypeError('The classes_number parameter is neither None nor '
                            'an integer.')

    if isinstance(class_proportion_threshold, Number):
        if class_proportion_threshold >= 1 or class_proportion_threshold <= 0:
            raise ValueError('The class_proportion_threshold parameter must '
                             'be a number between 0 and 1 (not inclusive).')
    else:
        raise TypeError('The class_proportion_threshold parameter is not a '
                        'number.')

    if isinstance(standard_deviation_init, Number):
        if standard_deviation_init <= 0:
            raise ValueError('The standard_deviation_init parameter must be a '
                             'positive number (greater than 0).')
    else:
        raise TypeError('The standard_deviation_init parameter is not a '
                        'number.')

    if isinstance(standard_deviation_increment, Number):
        if standard_deviation_increment <= 0:
            raise ValueError('The standard_deviation_increment parameter must '
                             'be a positive number (greater than 0).')
    else:
        raise TypeError('The standard_deviation_increment parameter is not a '
                        'number.')

    is_valid = True
    return is_valid


class NormalClassDiscovery(Augmentation):
    """
    Sampling data to discover instances spanning all the possible classes.

    .. versionadded:: 0.0.2

    This augmenter ensures that the generated sample has at least a predefined
    proportion (cf. ``class_proportion_threshold`` parameter) of every possible
    class. For a specific data point, it samples with a normal distribution
    centered around this point, incrementally increasing the standard deviation
    of the sample until the proportion of the samples of a class different
    (assigned by the predictive function) than the one of the specified data
    point is reached. Next, one of the data points found to be in another class
    is used as the centre of the normal distribution sampling to discover
    another class. These steps are repeated until all of the classes (with
    satisfying proportion) are in the sampled data set. If the ``sample``
    method is called without a ``data_row``, the starting point for the
    sampling procedure is the mean of the ``dataset``. For categorical
    features in the dataset, the values are sampled with replacement with the
    probability for each unique value calculated based on the frequency of
    their appearance in the dataset.

    .. note:: The number of classes when using a *classifier*.

       Consider using the ``classes_number`` parameter when using a
       non-probabilistic ``predictive_function``. For more details please see
       the description of the ``classes_number`` parameter.

       (When initialising this class without user-defined number of classes --
       via the ``classes_number`` parameter -- it will log the number of
       discovered target classes when the ``predictive_function`` is a
       *classifier*.)

    For additional parameters, attributes, warnings and exceptions raised by
    this class please see the documentation of its parent class:
    :class:`fatf.utils.data.augmentation.Augmentation`.

    This augmentation approach is similar to the *Growing Spheres* technique
    introduced by [LAUGEL2018INVERSE]_.

    .. [LAUGEL2018INVERSE] Laugel, T., Lesot, M.J., Marsala, C., Renard, X. and
       Detyniecki, M., 2017. Inverse Classification for Comparison-based
       Interpretability in Machine Learning. arXiv preprint arXiv:1712.08443.

    Parameters
    ----------
    predictive_function : Callable[[numpy.ndarray], numpy.ndarray]
        A Python callable, e.g., a function, that is either a *classifier* or a
        *probabilistic* predictor. This function is used to compute the class
        of the sampled data, which is used to ensure meeting the
        ``class_proportion_threshold``. A probabilistic function is expected to
        output a 2-dimensional numpy array with the assigned class being the
        one with maximum probability. A classifier function is expected to
        output a 1-dimensional numpy array with class assignment. The
        ``predictive_function`` should require exactly one input parameter --
        a data array to be predicted.
    classes_number : integer, optional (default=None)
        The number of classes (target values) modelled by the
        ``predictive_function``. If the ``predictive_function`` is
        probabilistic, the number of classes is inferred from the width of the
        probabilities output by the ``predictive_function``. If the
        ``predictive_function`` is a classifier, it is applied to the input
        ``dataset`` and the number of classes is computed based on the unique
        number of elements in this predictions array. **Since the latter case**
        **may result in not all of the classes being discovered, it is**
        **advised to specify the number of classes using this parameter.**
    class_proportion_threshold : float, optional (default=0.05)
        The minimum proportion of data points assigned to a different class
        by the ``predictive_function`` when sampling for each data point as per
        the procedure described above.

        .. warning:: Setting the ``class_proportion_threshold`` parameter.

           This augmenter samples a cloud of points for each discovered class
           with each cloud having 1 / ``classes_number`` number of points.
           This means that the value of the ``class_proportion_threshold`` has
           to be smaller than this number for the sampling to be successful.
           For example, for 2 classes and 100 sampled points, 2 clouds of 50
           data points each will be generated. By setting the
           ``class_proportion_threshold`` parameter to ``0.6``, at least 60
           point of each class are expected, which cannot be achieved.

    standard_deviation_init : float, optional (default=1)
        The standard deviation of the normal distribution used for initial
        sampling around each selected data point.
    standard_deviation_increment : float, optional (default=0.1)
        The increment used to increase the standard deviation every time the
        sample does not satisfy the specified ``class_proportion_threshold``
        or at least one data point of yet unseen class is not discovered.

    Raises
    ------
    IncompatibleModelError
        The ``predictive_function`` does not require exactly one input
        parameter.
    RuntimeError
        The class initialisation was unable to identify the number of classes
        using the input ``dataset`` and the provided ``predictive_function``.
        The value of the ``class_proportion_threshold`` parameter is too large
        for the given number of classes (please see the warning in the
        ``class_proportion_threshold`` parameter description for more
        information).
    TypeError
        The ``predictive_function`` is not a Python callable. The
        ``classes_number`` is neither ``None`` nor an integer.
        The ``class_proportion_threshold`` is not a float. Either
        ``standard_deviation_init`` or ``standard_deviation_increment`` is not
        a number.
    ValueError
        The ``classes_number`` parameter is smaller than 2.
        The ``class_proportion_threshold`` parameter is outside of the (0, 1)
        range (non-inclusive). The ``standard_deviation_init`` or
        ``standard_deviation_increment`` parameter is not a positive number.

    Attributes
    ----------
    predictive_function : Callable[[numpy.ndarray], numpy.ndarray]
        The predictive function used to initialise this class.
    is_probabilistic : boolean
        ``True`` if the ``predictive_function`` is probabilistic, ``False``
        otherwise. This attribute is set based on the shape of the numpy array
        output by the ``predictive_function``: if it is a 2-dimensional
        array, the ``predictive_function`` is assumed to be probabilistic, if
        it is a 1-dimensional array, the ``predictive_function`` is assumed to
        be a classifier.
    classes_number : integer
        The number of classes modelled by the ``predictive_function``, either
        defined by the user when initialising this class or inferred from the
        output of the ``predictive_function``.
    standard_deviation_init : float
        The initial value of the standard deviation used to initialise this
        class.
    standard_deviation_increment : float
        The standard deviation increment value used to initialise this class.
    class_proportion_threshold : float
        The value of the smallest proportion of a different class for sampling
        used to initialise this class.
    categorical_sampling_values : Dictionary[column index, \
Tuple[numpy.ndarray, numpy.ndarray]]
        Dictionary mapping categorical column feature indices to tuples
        consisting of two 1-dimensional numpy arrays: one with unique values
        for that column and the other one with their normalised (summing up to
        1) frequencies.
    """

    # pylint: disable=too-few-public-methods

    def __init__(self,
                 dataset: np.ndarray,
                 predictive_function: Callable[[np.ndarray], np.ndarray],
                 categorical_indices: Optional[np.ndarray] = None,
                 int_to_float: bool = True,
                 classes_number: Optional[int] = None,
                 class_proportion_threshold: float = 0.05,
                 standard_deviation_init: float = 1.0,
                 standard_deviation_increment: float = 0.1) -> None:
        """
        Constructs a ``NormalClassDiscovery`` data augmentation class.
        """
        # pylint: disable=too-many-arguments,too-many-locals
        super().__init__(
            dataset,
            categorical_indices=categorical_indices,
            int_to_float=int_to_float)
        assert _validate_input_normalclassdiscovery(
            predictive_function, classes_number, class_proportion_threshold,
            standard_deviation_init,
            standard_deviation_increment), 'Invalid input.'

        self.predictive_function = predictive_function

        # Check whether the function is probabilistic or a plane classifier
        predictions = self.predictive_function(dataset[[0]])
        assert not fuav.is_structured_array(predictions), 'Not plain numpy.'
        assert (fuav.is_2d_array(predictions)
                or fuav.is_1d_array(predictions)), 'Can only be 1-D or 2-D.'
        self.is_probabilistic = fuav.is_2d_array(predictions)

        # Try to infer the number of classes, otherwise prompt the user to
        # provide the number of classes.
        if classes_number is None:
            if self.is_probabilistic:
                classes_number = predictions.shape[1]
            else:
                predictions = self.predictive_function(dataset)
                unique_predictions = np.unique(predictions)
                assert fuav.is_1d_array(unique_predictions), 'Not a 1-D array.'
                if unique_predictions.shape[0] < 2:
                    raise RuntimeError('For the specified (classification) '
                                       'predictive function, classifying the '
                                       'input dataset provided only one '
                                       'target class. To use this augmenter '
                                       'please initialise it with the '
                                       'classes_number parameter.')
                classes_number = unique_predictions.shape[0]
                logger.info(
                    'The number of classes was not specified by the user. '
                    'Based on *classification* of the input dataset %d '
                    'classes were found.', classes_number)
        self.classes_number = classes_number

        self.standard_deviation_init = standard_deviation_init
        self.standard_deviation_increment = standard_deviation_increment
        self.class_proportion_threshold = class_proportion_threshold

        # If expected class_proportion_threshold is equal or larger than
        # 1/the number of classes, sampling *cannot* be successful since a new
        # class will never be discovered.
        if self.class_proportion_threshold >= 1 / self.classes_number:
            raise RuntimeError('The lower bound on the proportion of each '
                               'class must be smaller than 1/(the number of '
                               'classes) for this sampling implementation. '
                               '(Please see the documentation of the '
                               'NormalClassDiscovery augmenter for more '
                               'information.')

        # Get sampling parameters for categorical features.
        categorical_sampling_values = dict()
        for column_name in self.categorical_indices:
            if self.is_structured:
                feature_column = self.dataset[column_name]
            else:
                feature_column = self.dataset[:, column_name]

            feature_values, values_counts = np.unique(
                feature_column, return_counts=True)
            values_frequencies = values_counts / values_counts.sum()

            categorical_sampling_values[column_name] = (feature_values,
                                                        values_frequencies)
        self.categorical_sampling_values = categorical_sampling_values

    def sample(self,
               data_row: Optional[Union[np.ndarray, np.void]] = None,
               samples_number: int = 50,
               max_iter: int = 1000) -> np.ndarray:
        """
        Samples data using normal distribution class discovery process.

        For the additional documentation of parameters, warnings and errors
        please see the description of the
        :func:`fatf.utils.data.augmentation.Augmentation.sample` method in the
        parent :class:`fatf.utils.data.augmentation.Augmentation` class.

        Parameters
        ----------
        max_iter : integer, optional (default=1000)
            The maximum number of iterations for the iterative normal sampling
            procedure. If the limit is reached and the
            ``class_proportion_threshold`` is not satisfied in addition to
            discovering at least one data point of yet unseen class a
            ``RuntimeError`` is raised. If this is the case you may want to
            consider initialising the class with a smaller
            ``class_proportion_threshold`` parameter or larger
            ``standard_deviation_init`` and ``standard_deviation_increment``
            parameters. Alternatively, increasing the ``max_iter`` may help to
            discover all of the classes with the other parameters fixed.

        Raises
        ------
        RuntimeError
            The maximum number of iterations was reached without discovering
            samples from every class (with the specified proportion).
        TypeError
            The ``max_iter`` parameter is not an integer.
        ValueError
            The ``max_iter`` parameter is not a positive number
            (greater than 0).

        Returns
        -------
        samples : numpy.ndarray
            A numpy array of [``samples_number``, number of features] shape
            holding the sampled data.
        """
        # pylint: disable=arguments-differ,too-many-locals,too-many-branches
        # pylint: disable=too-many-statements
        assert self._validate_sample_input(data_row, samples_number)
        if isinstance(max_iter, int):
            if max_iter <= 0:
                raise ValueError('The max_iter parameter must be a positive '
                                 'number.')
        else:
            raise TypeError('The max_iter parameter is not a positive '
                            'integer.')

        # Sample from the mean of the dataset if a data_row is not given
        if data_row is None:
            data_row = np.zeros_like(self.dataset[0])
            # Get the most frequent value for each categorical feature
            for index in self.categorical_indices:
                max_freq_idx = (
                    self.categorical_sampling_values[index][1].argmax())
                # Most frequent value
                data_row[index] = (
                    self.categorical_sampling_values[index][0][max_freq_idx])
            # Get the mean of each numerical feature
            for index in self.numerical_indices:
                data_row[index] = self.dataset[index].mean(axis=0)

        # Prepare the row to be used with the predictive_function
        # and get the output array shape
        if self.is_structured:
            row = data_row.reshape(-1)
        else:
            row = data_row.reshape(1, -1)

        row_labels = self.predictive_function(row)
        assert row_labels.shape[0] == 1, 'Only 1 data point predicted.'
        if self.is_probabilistic:
            assert fuav.is_2d_array(row_labels), 'Probabilistic outputs 2-D.'
            row_label = row_labels[0].argmax()
        else:
            assert fuav.is_1d_array(row_labels), 'Classifier outputs 1-D.'
            row_label = row_labels[0]

        # Get the number of (almost) equal samples per (class) normal
        # distribution
        min_nomal_smaples = int(samples_number / self.classes_number)
        samples_per_normal = self.classes_number * [min_nomal_smaples]
        #
        missing_normal_smaples = (
            samples_number - self.classes_number * min_nomal_smaples)
        for i in range(missing_normal_smaples):
            samples_per_normal[i] += 1
        assert sum(samples_per_normal) - samples_number == 0, 'Wrong samples #'

        samples_list = []
        normal_dist_counter = 0
        # Labels seen in the correct proportion
        seen_labels = set()  # type: Set[Union[int, str]]
        #
        current_std = self.standard_deviation_init
        current_data_row = data_row
        current_label = row_label
        if self.is_structured:
            iter_shape = (samples_per_normal[normal_dist_counter],
                          )  # type: Tuple[int, ...]
        else:
            iter_shape = (samples_per_normal[normal_dist_counter],
                          self.features_number)

        for _ in range(max_iter):
            # Create an array to hold the samples.
            samples_iter = np.zeros(iter_shape, dtype=self.sample_dtype)

            # Sample categorical features.
            for index in self.categorical_indices:
                sample_values = np.random.choice(
                    self.categorical_sampling_values[index][0],
                    size=samples_per_normal[normal_dist_counter],
                    replace=True,
                    p=self.categorical_sampling_values[index][1])
                if self.is_structured:
                    samples_iter[index] = sample_values
                else:
                    samples_iter[:, index] = sample_values

            # Sample numerical features.
            for index in self.numerical_indices:
                mean = current_data_row[index]
                sample_values = np.random.normal(
                    0, 1, samples_per_normal[normal_dist_counter])
                sample_values = sample_values * current_std + mean
                if self.is_structured:
                    samples_iter[index] = sample_values
                else:
                    samples_iter[:, index] = sample_values

            # Get predictions for the sampled data
            predictions = self.predictive_function(samples_iter)
            if self.is_probabilistic:
                predictions = predictions.argmax(axis=1)

            current_label_count = np.where(
                predictions == current_label)[0].shape[0]
            expected_proportion = (
                self.class_proportion_threshold * samples_number)

            # At least one unseen class different than the current label
            unique_labels = np.unique(predictions)
            new_label = False
            for label in unique_labels:
                # A class different to the current one has been discovered...
                if label != current_label:
                    # ...and it is either the last one or an unseen one
                    if (normal_dist_counter + 1 == self.classes_number
                            or label not in seen_labels):
                        new_label = True
                        # Get a random data point of unseen label
                        new_label_data_row_index = np.random.choice(
                            np.where(predictions == label)[0])
                        new_label_data_row = samples_iter[[
                            new_label_data_row_index
                        ]]
                        break

            # If the proportion of the current label is satisfied
            # and there is at least one data point of an unseen label...
            if current_label_count >= expected_proportion and new_label:
                # Add sampled array to the samples_list
                samples_list.append(samples_iter)
                # Add the current_label to the seen_labels
                seen_labels.add(current_label)
                # Have we seen all of the classes
                if len(seen_labels) == self.classes_number:
                    break

                # Increment normal_dist_counter
                normal_dist_counter += 1
                # Update iter_shape
                if self.is_structured:
                    iter_shape = (samples_per_normal[normal_dist_counter], )
                else:
                    iter_shape = (samples_per_normal[normal_dist_counter],
                                  self.features_number)
                # Reset the current_std
                current_std = self.standard_deviation_init

                # Pick a new current_data_row of a different class
                current_data_row = new_label_data_row[0]
                # Save its class to current_label
                current_labels = self.predictive_function(new_label_data_row)
                if self.is_probabilistic:
                    current_label = current_labels[0].argmax()
                else:
                    current_label = current_labels[0]
            else:
                current_std += self.standard_deviation_increment
        else:
            raise RuntimeError('The maximum number of iterations was reached '
                               'without sampling enough data points for each '
                               'class. Please try increasing the max_iter '
                               'parameter or decreasing the '
                               'class_proportion_threshold parameter. '
                               'Increasing the standard_deviation_init and '
                               'standard_deviation_increment parameters '
                               'may also help.')

        samples = np.concatenate(samples_list)
        return samples


def _validate_input_decisionboundarysphere(
        predictive_function: Callable[[np.ndarray], np.ndarray],
        radius_init: float, radius_increment: float) -> bool:
    """
    Validates input parameters of the ``DecisionBoundarySphere`` augmenter.

    For the description of the input parameters, errors and exceptions please
    see the documentation of the
    :class:`fatf.utils.data.augmentation.DecisionBoundarySphere` class.

    Returns
    -------
    is_valid : boolean
        ``True`` if the input is valid, ``False`` otherwise.
    """
    is_valid = False

    if callable(predictive_function):
        prms_n = fuv.get_required_parameters_number(predictive_function)
        if prms_n != 1:
            raise IncompatibleModelError('The predictive function must take '
                                         'exactly *one* required parameter: '
                                         'a data array to be predicted.')
    else:
        raise TypeError('The predictive_function should be a Python '
                        'callable, e.g., a Python function.')

    if isinstance(radius_init, Number):
        if radius_init <= 0:
            raise ValueError('The radius_init parameter must be a positive '
                             'number (greater than 0).')
    else:
        raise TypeError('The radius_init parameter is not a number.')

    if isinstance(radius_increment, Number):
        if radius_increment <= 0:
            raise ValueError('The radius_increment parameter is not a '
                             'positive number (greater than 0).')
    else:
        raise TypeError('The radius_increment parameter is not a number.')

    is_valid = True
    return is_valid


class DecisionBoundarySphere(Augmentation):
    """
    Sampling data in a hyper-sphere around the closest decision boundary.

    .. versionadded:: 0.0.2

    ``DecisionBoundarySphere`` implements an adapted version of the local
    surrogate sampling introduced by [LAUGEL2018DEFINING]_. A hyper-sphere is
    grown around the specified data point until a decision boundary is found,
    then from a point on this decision boundary data points are sampled
    uniformly in an l-2 hyper-sphere with a user-predefined radius.

    .. note:: Categorical features.

       This augmenter does not currently support data sets with categorical
       features.

    For additional parameters, attributes, warnings and exceptions raised by
    this class please see the documentation of its parent class:
    :class:`fatf.utils.data.augmentation.Augmentation`.

    .. [LAUGEL2018DEFINING] Laugel, T., Renard, X., Lesot, M. J., Marsala,
       C., & Detyniecki, M. (2018). Defining locality for surrogates in
       post-hoc interpretablity. Workshop on Human Interpretability for
       Machine Learning (WHI) -- International Conference on Machine Learning,
       2018.

    Parameters
    ----------
    predictive_function : Callable[[numpy.ndarray], numpy.ndarray]
        A Python callable, e.g., a function, that is either a *classifier* or a
        *probabilistic* predictor. This function is used to compute the class
        of the sampled data, which is used to identify a decision boundary.
        A probabilistic function is expected to output a 2-dimensional numpy
        array with the assigned class being the one with maximum probability. A
        classifier function is expected to output a 1-dimensional numpy array
        with class assignment. The ``predictive_function`` should require
        exactly one input parameter -- a data array to be predicted.
    radius_init : float, optional (default=0.01)
        The initial radius of the specified data point around which a
        hyper-sphere will be placed to discover a decision boundary.
    radius_increment : float, optional (default=0.01)
        The additive increment to the initial hyper-sphere radius by which it
        will be incremented (in every iteration of the sampling procedure) if
        no decision boundary has been discovered.

    Raises
    ------
    IncompatibleModelError
        The ``predictive_function`` does not require exactly one input
        parameter.
    NotImplementedError
        Some of the features in the data set are categorical -- this feature
        type is not supported at present.
    TypeError
        The ``predictive_function`` parameter is not a Python callable. Either
        the ``radius_init`` or ``radius_increment`` parameter is not a number.
    ValueError
        Either ``radius_init`` or ``radius_increment`` parameter is less or
        equal to 0.

    Attributes
    ----------
    predictive_function : Callable[[numpy.ndarray], numpy.ndarray]
        The predictive function used to initialise this class.
    is_probabilistic : boolean
        ``True`` if the ``predictive_function`` is probabilistic, ``False``
        otherwise. This is set based on the shape of the numpy array output
        by the ``predictive_function``: if it is a 2-dimensional array, the
        ``predictive_function`` is assumed to be probabilistic, if it is a
        1-dimensional array, the ``predictive_function`` is assumed to be a
        classifier.
    radius_init : float
        The initial radius of a hyper-sphere placed around the specified data
        point within which new data points will be sampled to discover a
        decision boundary.
    radius_increment : float
        The additive increment to the initial hyper-sphere radius by which it
        will be incremented (in every iteration of the sampling procedure) if
        no decision boundary has been discovered.
    """

    # pylint: disable=too-few-public-methods

    def __init__(self,
                 dataset: np.ndarray,
                 predictive_function: Callable[[np.ndarray], np.ndarray],
                 categorical_indices: Optional[List[Index]] = None,
                 int_to_float: bool = True,
                 radius_init: float = 0.01,
                 radius_increment: float = 0.01) -> None:
        """
        Constructs a ``DecisionBoundarySphere`` data augmentation class.
        """
        # pylint: disable=too-many-arguments
        super().__init__(
            dataset=dataset,
            categorical_indices=categorical_indices,
            int_to_float=int_to_float)
        assert _validate_input_decisionboundarysphere(
            predictive_function, radius_init, radius_increment)
        if self.categorical_indices:
            raise NotImplementedError('The DecisionBoundarySphere augmenter '
                                      'does not currently support data sets '
                                      'with categorical features.')

        self.predictive_function = predictive_function
        self.radius_init = radius_init
        self.radius_increment = radius_increment

        # Check whether the function is probabilistic or a plane classifier
        predictions = self.predictive_function(dataset[[0]])
        assert not fuav.is_structured_array(predictions), 'Not plain numpy.'
        assert (fuav.is_2d_array(predictions)
                or fuav.is_1d_array(predictions)), 'Can only be 1-D or 2-D.'
        self.is_probabilistic = fuav.is_2d_array(predictions)

    def _validate_sample_input(  # type: ignore
            self, data_row: Union[np.ndarray, np.void], sphere_radius: float,
            samples_number: int, discover_samples_number: int,
            max_iter: int) -> bool:
        """
        Validates input parameters for the ``sample`` method.

        For additional description of the input parameters, warnings and
        errors please see the documentation of the
        :func:`fatf.utils.data.augmentation.DecisionBoundarySphere.sample`
        method.

        Returns
        -------
        is_valid : boolean
            ``True`` if the input is valid, ``False`` otherwise.
        """
        # pylint: disable=arguments-differ,too-many-arguments
        is_valid = False
        assert super()._validate_sample_input(data_row, samples_number)

        if isinstance(sphere_radius, Number):
            if sphere_radius <= 0:
                raise ValueError('The sphere_radius parameter must be a '
                                 'positive number (greater than 0).')
        else:
            raise TypeError('The sphere_radius parameter must be a number.')

        if isinstance(discover_samples_number, int):
            if discover_samples_number <= 0:
                raise ValueError('The discover_samples_number parameter must '
                                 'be a positive integer (greater than 0).')
        else:
            raise TypeError('The discover_samples_number parameter must be an '
                            'integer.')

        if isinstance(max_iter, int):
            if max_iter <= 0:
                raise ValueError('The max_iter parameter must be a positive '
                                 'integer (greater than 0).')
        else:
            raise TypeError('The max_iter parameter must be an integer.')

        is_valid = True
        return is_valid

    def sample(  # type: ignore
            self,
            data_row: Union[np.ndarray, np.void],
            sphere_radius: float = 0.05,
            samples_number: int = 50,
            discover_samples_number: int = 100,
            max_iter: int = 1000) -> np.ndarray:
        """
        Samples data around the closest decision boundary to the ``data_row``.

        For the additional documentation of the input parameters, warnings and
        errors please see the description of the
        :func:`fatf.utils.data.augmentation.Augmentation.sample` method in the
        parent :class:`fatf.utils.data.augmentation.Augmentation` class.

        Parameters
        ----------
        sphere_radius : float, optional (default=0.05)
            Radius of the hyper-sphere around the closest decision boundary to
            ``data_row`` within which new data points will be sampled.
        discover_samples_number : integer, optional (default=100)
            Number of samples generated at each iteration of the sampling
            procedure that are used to discover the nearest decision boundary
            around the ``data_row``.
        max_iter : integer, optional (default=1000)
            The maximum number of iterations for the iterative hyper-sphere
            growing (around the ``data_row``) procedure. If the limit is
            reached and a decision boundary has not been found a
            ``RuntimeError`` is raised. If this is the case you may want to
            consider initialising the class with a larger ``radius_init`` or
            ``radius_increment`` parameter. Alternatively, increasing the
            ``discover_samples_number`` or ``max_iter`` parameter may help to
            discover the nearest boundary with all the other parameters fixed.

        Raises
        ------
        NotImplementedError
            The ``data_row`` is ``None`` -- sampling from the mean of the
            ``dataset`` used to initialise this class is not yet implemented.
        RuntimeError
            The maximum number of iterations was reached without the algorithm
            discovering a decision boundary.
        TypeError
            The ``sphere_radius`` parameter is not a number. The
            ``discover_samples_number`` or ``max_iter`` parameter is not
            an integer.
        ValueError
            The ``sphere_radius``, ``discover_samples_number`` or ``max_iter``
            parameter is not a positive number (greater than 0).

        Returns
        -------
        samples : numpy.ndarray
            A numpy array of shape [``samples_number``, number of features]
            that holds the sampled data.
        """
        # pylint: disable=arguments-differ,too-many-arguments
        # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        assert self._validate_sample_input(
            data_row, sphere_radius, samples_number, discover_samples_number,
            max_iter), 'Invalid input.'
        if data_row is None:
            raise NotImplementedError('Sampling around the mean of the '
                                      'initialisation dataset is not '
                                      'currently supported by the '
                                      'DecisionBoundarySphere augmenter.')

        if self.is_structured:
            shape_ds = (discover_samples_number, )  # type: Tuple[int, ...]
            shape_sample = (samples_number, )  # type: Tuple[int, ...]
            row = data_row.reshape(-1)
        else:
            shape_ds = (discover_samples_number, self.features_number)
            shape_sample = (samples_number, self.features_number)
            row = data_row.reshape(1, -1)

        current_radius = self.radius_init

        row_labels = self.predictive_function(row)
        assert row_labels.shape[0] == 1, 'Only 1 data point predicted.'
        if self.is_probabilistic:
            assert fuav.is_2d_array(row_labels), 'Probabilistic outputs 2-D.'
            row_label = row_labels[0].argmax()
        else:
            assert fuav.is_1d_array(row_labels), 'Classifier outputs 1-D.'
            row_label = row_labels[0]

        for _ in range(max_iter):
            discover_samples = np.zeros(shape_ds, dtype=self.sample_dtype)

            uniform = np.random.uniform(
                0, current_radius, size=(discover_samples_number, 1))
            normal = np.random.normal(
                0, 1, (discover_samples_number, self.features_number))
            normal_norm = np.linalg.norm(normal, ord=2, axis=1)
            normal_norm = np.expand_dims(normal_norm, 1)
            directional_vectors = uniform * normal / normal_norm

            for i, index in enumerate(self.numerical_indices):
                if self.is_structured:
                    discover_samples[index] = (
                        data_row[index] + directional_vectors[:, i])
                else:
                    discover_samples[:, index] = (
                        data_row[i] + directional_vectors[:, i])

            # Get predictions for the sampled data
            predictions_sampled = self.predictive_function(discover_samples)
            if self.is_probabilistic:
                predictions_sampled = predictions_sampled.argmax(axis=1)
            unseen_predictions = np.where(predictions_sampled != row_label)[0]

            if unseen_predictions.size:
                # Get one of the samples on (or past) the decision boundary
                unclassified_samples = discover_samples[unseen_predictions]
                if self.is_structured:
                    distances = fud.euclidean_array_distance(
                        row, unclassified_samples)
                else:
                    distances = scipy.spatial.distance.cdist(
                        row, unclassified_samples, metric='euclidean')
                boundary_sample = unclassified_samples[np.argmin(distances)]
                break
            else:
                current_radius += self.radius_increment
        else:
            raise RuntimeError('The maximum number of iterations was reached '
                               'without discovering a decision boundary. '
                               'Please try increasing the max_iter or '
                               'discover_samples_number parameter. '
                               'Alternatively, initialise this class with a '
                               'larger radius_init or radius_increment '
                               'parameter.')

        # Uniformly sample in an l-2 hyper-sphere around the decision boundary
        samples = np.zeros(shape_sample, dtype=self.sample_dtype)

        uniform = np.random.uniform(0, sphere_radius, size=(samples_number, 1))
        normal = np.random.normal(0, 1, (samples_number, self.features_number))
        normal_norm = np.linalg.norm(normal, ord=2, axis=1)
        normal_norm = np.expand_dims(normal_norm, 1)
        directional_vectors = uniform * normal / normal_norm

        for i, index in enumerate(self.numerical_indices):
            if self.is_structured:
                samples[index] = (
                    boundary_sample[index] + directional_vectors[:, i])
            else:
                samples[:, index] = (
                    boundary_sample[i] + directional_vectors[:, i])

        return samples


class LocalSphere(Augmentation):
    """
    Sampling data in a hyper-sphere around the selected data point.

    .. versionadded:: 0.0.2

    ``LocalSphere`` implements an adapted version of the local fidelity
    sampling method introduced by [LAUGEL2018DEFINING]_. For a specific data
    point, it samples uniformly within a hyper-sphere, which radius corresponds
    to a specified percentage of the maximum l-2 distance between the specified
    data point and all the other instances in the input ``dataset``.

    .. note:: Categorical features.

       This augmenter does not currently support data sets with categorical
       features.

    For additional parameters, attributes, warnings and exceptions raised by
    this class please see the documentation of its parent class:
    :class:`fatf.utils.data.augmentation.Augmentation`.

    .. [LAUGEL2018DEFINING] Laugel, T., Renard, X., Lesot, M. J., Marsala,
       C., & Detyniecki, M. (2018). Defining locality for surrogates in
       post-hoc interpretablity. Workshop on Human Interpretability for
       Machine Learning (WHI) -- International Conference on Machine Learning,
       2018.

    Raises
    ------
    NotImplementedError
        Some of the features in the data set are categorical -- this feature
        type is not supported at present.
    """

    # pylint: disable=too-few-public-methods

    def __init__(self,
                 dataset: np.ndarray,
                 categorical_indices: Optional[List[Index]] = None,
                 int_to_float: bool = True) -> None:
        """
        Constructs a ``LocalSphere`` data augmentation class.
        """
        super().__init__(
            dataset=dataset,
            categorical_indices=categorical_indices,
            int_to_float=int_to_float)

        if self.categorical_indices:
            raise NotImplementedError('The LocalSphere augmenter does not '
                                      'currently support data sets with '
                                      'categorical features.')

    def sample(  # type: ignore
            self,
            data_row: Union[np.ndarray, np.void],
            fidelity_radius_percentage: int = 5,
            samples_number: int = 50) -> np.ndarray:
        """
        Samples new data in a hyper-sphere around the selected data point.

        For the additional description of the parameters, warnings and errors
        please see the documentation of the
        :func:`fatf.utils.data.augmentation.Augmentation.sample` method in the
        parent :class:`fatf.utils.data.augmentation.Augmentation` class.

        Parameters
        ----------
        fidelity_radius_percentage : integer, optional (default=5)
            The percentage of the maximum distance between the input
            ``data_row`` and all of the points in the ``dataset`` (provided
            when initialising this class), which will determine the radius of
            the hyper-sphere used for sampling uniformly around the
            ``data_row``.

        Raises
        ------
        NotImplementedError
            The ``data_row`` is ``None`` -- sampling from the mean of the
            ``dataset`` used to initialise this class is not yet implemented.
        TypeError
            The ``fidelity_radius_percentage`` parameter is not an integer.
        ValueError
            The ``fidelity_radius_percentage`` parameter is not a positive
            integer.

        Returns
        -------
        samples : numpy.ndarray
            A numpy array of shape [``samples_number``, number of features]
            holding the sampled data.
        """
        # pylint: disable=arguments-differ
        assert self._validate_sample_input(data_row,
                                           samples_number), 'Invalid input.'
        if data_row is None:
            raise NotImplementedError('Sampling around the mean of the '
                                      'initialisation dataset is not '
                                      'currently supported by the LocalSphere '
                                      'augmenter.')
        if isinstance(fidelity_radius_percentage, int):
            if fidelity_radius_percentage <= 0:
                raise ValueError('The fidelity_radius_percentage parameter '
                                 'must be a positive integer (greater than '
                                 '0).')
        else:
            raise TypeError('The fidelity_radius_percentage parameter must be '
                            'an integer.')

        if self.is_structured:
            shape = (samples_number, )  # type: Tuple[int, ...]
            distances = fud.euclidean_array_distance(
                np.expand_dims(data_row, 0), self.dataset)
        else:
            shape = (samples_number, self.features_number)
            distances = scipy.spatial.distance.cdist(
                np.expand_dims(data_row, 0), self.dataset, metric='euclidean')
        assert np.all(distances >= 0), 'Distances cannot be negative.'

        # Get max radius
        radius = fidelity_radius_percentage / 100 * distances.max()

        # Get radii
        uniform = np.random.uniform(0, radius, size=(samples_number, 1))
        # Get random directions for the radii
        normal = np.random.normal(0, 1, (samples_number, self.features_number))
        # Get scaling of the random directions to preserve the radii
        normal_norm = np.linalg.norm(normal, ord=2, axis=1)
        normal_norm = np.expand_dims(normal_norm, 1)
        # Compute the directional vectors
        directional_vectors = uniform * normal / normal_norm

        samples = np.zeros(shape, dtype=self.sample_dtype)
        for i, index in enumerate(self.numerical_indices):
            if self.is_structured:
                samples[index] = data_row[index] + directional_vectors[:, i]
            else:
                samples[:, index] = data_row[i] + directional_vectors[:, i]
        return samples
