"""
Data augmentation classes.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
#         Kacper Sokol <k.sokol@bristol.ac.uk>
#         Rafael Poyiadzi <rp13102@bristol.ac.uk>
# License: new BSD

import abc
import warnings

from numbers import Number
from typing import List, Optional, Tuple, Union

import numpy as np

from fatf.exceptions import IncorrectShapeError

import fatf.utils.array.tools as fuat
import fatf.utils.array.validation as fuav

__all__ = ['NormalSampling']

Index = Union[int, str]


def _validate_input(dataset: np.ndarray,
                    ground_truth: Optional[np.ndarray] = None,
                    categorical_indices: Optional[List[Index]] = None) -> bool:
    """
    Validates the input parameters of an arbitrary augmentation class.

    Parameters
    ----------
    dataset : numpy.ndarray
        A 2-dimensional numpy array with a dataset to be used for sampling.
    ground_truth : numpy.ndarray, optional (default=None)
        A 1-dimensional numpy array with labels for the supplied dataset.
    categorical_indices : List[column indices], optional (default=None)
        A list of column indices that should be treat as categorical features.

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
        of base (numerical and/or string) type.

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

    is_valid = True
    return is_valid


class Augmentation(abc.ABC):
    """
    An abstract class for implementing data augmentation methods.

    An abstract class that all augmentation classes should inherit from. It
    contains abstract ``__init__`` and ``sample`` methods and an input
    validator -- ``_validate_sample_input`` -- for the ``sample`` method. The
    validation of the input parameter to the initialisation method is done via
    the :func:`fatf.utils.data.augmentation._validate_input` function.

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
        of base (numerical and/or string) type.

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
    """

    # pylint: disable=too-few-public-methods
    def __init__(self,
                 dataset: np.ndarray,
                 ground_truth: Optional[np.ndarray] = None,
                 categorical_indices: Optional[np.ndarray] = None) -> None:
        """
        Constructs an ``Augmentation`` abstract class.
        """
        assert _validate_input(
            dataset,
            ground_truth=ground_truth,
            categorical_indices=categorical_indices), 'Invalid input.'

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
            ``True`` if input parameter are valid, ``False`` otherwise.
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
    this class please see the  documentation of its parent class:
    :class:`fatf.utils.data.augmentation.Augmentation`.

    Attributes
    ----------
    numerical_sampling_values : Dictionary[column index, Tuple[number, number]]
        Dictionary mapping numerical column feature indices to tuples of two
        numbers: column's *mean* and its *standard deviation*.
    categorical_sampling_values :
    Dictionary[column index, Tuple[numpy.ndarray, numpy.ndarray]]
        Dictionary mapping categorical column feature indices to tuples two
        1-dimensional numpy arrays: one with unique values for that column
        and the other one with their normalised (sum up to 1) frequencies.
    sample_dtype : Union[numpy.dtype, List[Tuple[string, numpy.dtype]]
        A dtype with numerical dtypes (in case of a structured data array)
        generalised to support the assignment of sampled values. For example,
        if the dtype of a numerical feature is ``int`` and the sampling
        generates ``float`` this dtype will generalise the type of that column
        to ``float``.
    """

    # pylint: disable=too-few-public-methods
    def __init__(self,
                 dataset: np.ndarray,
                 categorical_indices: Optional[List[Index]] = None) -> None:
        """
        Constructs a ``NormalSampling`` data augmentation class.
        """
        # pylint: disable=too-many-locals,too-many-branches
        super().__init__(dataset, categorical_indices=categorical_indices)

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

        # Sort out the dtype of the sampled array.
        if self.is_structured:
            sample_dtype = []
            for column_name in self.dataset.dtype.names:
                if column_name in self.numerical_indices:
                    new_dtype = fuat.generalise_dtype(
                        self.dataset.dtype[column_name], np.dtype(np.float64))
                    sample_dtype.append((column_name, new_dtype))
                elif column_name in self.categorical_indices:
                    sample_dtype.append((column_name,
                                         self.dataset.dtype[column_name]))
                else:
                    assert False, 'Unknown column name.'  # pragma: nocover
        else:
            if fuav.is_numerical_array(self.dataset):
                sample_dtype = fuat.generalise_dtype(self.dataset.dtype,
                                                     np.dtype(np.float64))
            else:
                sample_dtype = self.dataset.dtype
        self.sample_dtype = sample_dtype

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
            smaple_values = np.random.choice(
                self.categorical_sampling_values[index][0],
                size=samples_number,
                replace=True,
                p=self.categorical_sampling_values[index][1])
            if self.is_structured:
                samples[index] = smaple_values
            else:
                samples[:, index] = smaple_values

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


def _validate_input_mixup(
        beta_parameters: Union[None, Tuple[Number, Number]]) -> bool:
    """
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
    Object to perform data generation following the Mixup method.

    For a specific point, select points frm the dataset at random, then draw
    samples frm Beta distribution, and form new points according to the convex
    combinations of the points.

    Attributes
    ----------
    threshold : number
        Pass.
    beta_parameters : Tuple[number, number]
        Pass.
    ground_truth_unique : np.ndarray
        Pass.
    ground_truth_frequencies : np.ndarray
        Pass.
    indices_per_label : List[np.ndarray]
        Pass.
    ground_truth_probabilities : np.ndarray
        [number of instances, number of unique classes]
        Pass.
    """
    def __init__(self,
                 dataset: np.ndarray,
                 ground_truth: Optional[np.ndarray] = None,
                 categorical_indices: Optional[np.ndarray] = None,
                 beta_parameters: Optional[Tuple[Number, Number]] = None
                 ) -> None:
        """
        Constructs a ``Mixup`` data augmentation class.
        """
        super().__init__(
            dataset,
            ground_truth=ground_truth,
            categorical_indices=categorical_indices)
        assert _validate_input_mixup(beta_parameters), 'Invalid mixup input.'

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
            indices_per_label = [np.where(self.ground_truth == label)[0]
                                 for label in ground_truth_unique]

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

    def _validate_sample_input_mixup(self,
                                     data_row_target: Union[Number, str, None],
                                     with_replacement: bool,
                                     return_probabilities: bool) -> bool:
        """
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
            samples_per_label = [int(freq * samples_number)
                                 for freq in self.ground_truth_frequencies]

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
                    self.indices_per_label[i],
                    label_sample_quantity,
                    replace=with_replacement)
                random_indices.append(random_indices_label)
            random_indices = np.concatenate(random_indices)

        return random_indices

    def _get_sample_targets(self,
                            data_row_target: Union[Number, str],
                            return_probabilities: bool,
                            random_draws_lambda: np.ndarray,
                            random_draws_lambda_1: np.ndarray,
                            random_indices: np.ndarray) -> np.ndarray:
        """
        """
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
        samples_target = (
            np.apply_along_axis(
                np.multiply,
                0,
                self.ground_truth_probabilities[random_indices],
                random_draws_lambda_1)
            + np.apply_along_axis(
                np.multiply,
                0,
                encoded_data_row_target,
                random_draws_lambda))

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

    def sample(self,
               data_row: Optional[Union[np.ndarray, np.void]] = None,
               data_row_target: Optional[Union[Number, str]] = None,
               samples_number: int = 50,
               with_replacement: bool = True,
               return_probabilities: bool = False) -> Tuple[np.ndarray, ...]:
        """
        Pass.
        """
        assert self._validate_sample_input(
            data_row, samples_number), 'Invalid mixup sampling input.'
        assert self._validate_sample_input_mixup(
            data_row_target,
            with_replacement,
            return_probabilities), 'Invalid mixup sampling input.'

        if data_row is None:
            raise NotImplementedError('Sampling around the data mean is not '
                                      'yet implemented for the Mixup class.')

        # Get stratified random row indices of the original data set
        random_indices = self._get_stratified_indices(
            samples_number, with_replacement)
        random_data_points = self.dataset[random_indices]

        random_draws_lambda = np.random.beta(*self.beta_parameters,
                                             samples_number)
        random_draws_lambda_1 = 1 - random_draws_lambda
        mask = random_draws_lambda <= self.threshold

        # Create an array to hold the samples.
        if self.is_structured:
            shape = (samples_number, )
        else:
            shape = (samples_number, self.features_number)
        samples = np.zeros(shape, dtype=self.dataset.dtype)

        # Sort out numerical features
        for index in self.numerical_indices:
            column_values = (random_draws_lambda_1 * random_data_points[index]
                             + random_draws_lambda * data_row[index])
            if self.is_structured:
                samples[index] = column_values
            else:
                samples[:, index] = column_values

        # Sort out categorical features
        for index in self.categorical_indices:
            if self.is_structured:
                samples[index][mask] = data_row[index]
                samples[index][~mask] = random_data_points[index][~mask]
            else:
                samples[mask, index] = data_row[index]
                samples[~mask, index] = random_data_points[~mask, index]

        # Get target values/probabilities sample if requested
        if data_row_target is None or self.ground_truth_unique is None:
            return samples
        else:
            samples_target = self._get_sample_targets(
                data_row_target, return_probabilities, random_draws_lambda,
                random_draws_lambda_1, random_indices)
            return samples, samples_target
