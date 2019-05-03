"""
Data augmentation methods
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
#         Alex Hepburn <ah13558@bristol.ac.uk>
# License: new BSD

import collections
from abc import ABCMeta, abstractmethod
from typing import Optional
import warnings

import numpy as np

import fatf.utils.array.tools as fuat
import fatf.utils.array.validation as fuav

from fatf.exceptions import IncorrectShapeError

__all__ = ['NormalSampling']


class Augmentation(metaclass=ABCMeta):
    """
    Super class for data augmentation.

    A super class that all augmentation classes should inherit from. Contains
    abstract method for `__init__` and `sample`. Also contains methods for
    validating the input dataset and indices as well as a separate method for
    validating the inputs for `sample` function. `_validate_sample_input`
    should be called from all child classes inside the `sample` function.

    Parameters
    ----------
    dataset : numpy.ndarray
        A 2-dimensional numerical numpy array with a dataset to be used.
    categorical_indices : numpy.ndarray
        An array of indices that should be treat as categorical variables.

    Warns
    -----
    UserWarning
        If index of a string type column is not in `categorical_indices` it
        it will be added.
    
    Raises
    ------
    IncorrectShapeError
        The input data is not a 2-dimensional array. The categorical indices
        list/array (``categorical_features``) is not 1-dimensional.
    TypeError
        `categorical_indices` is not eiher a numpy.ndarray or None. `dataset`
        is not a numpy.ndarray, or `categorical_indices` is None so will be
        inferred by looking at the data type.
    
    Attributes
    ----------
    dataset : numpy.ndarray
        A 2-dimensional numerical numpy array with a dataset to be used.
    categorical_indices : numpy.ndarray
        An array of indices that should be treat as categorical variables.
    """
    @abstractmethod
    def __init__(self,
                 dataset: np.ndarray,
                 categorical_indices: np.ndarray = None) -> None:
        """
        Constructor for Augmentation super class.
        """
        assert self._validate_input(dataset, categorical_indices), \
            'Input is invalid'
        self.dataset = dataset

        # If string based columns found, add them to categorical_indices.
        numerical_indices, non_numerical_indices = \
            fuat.indices_by_type(dataset)

        if categorical_indices is None:
            if non_numerical_indices.size > 0:
                categorical_indices = non_numerical_indices
                msg = ('No categorical_indcies were provided. The categorical '
                    'columns will be inferred by looking at the type of values '
                    'in the dataset.')
                warnings.warn(msg, UserWarning)
            else:
                categorical_indices = np.array([])
        else:
            if not set(non_numerical_indices).issubset(set(categorical_indices)):
                msg = ('String based indices were found in dataset but not '
                       'given as categorical_indices. String based columns '
                       'will automatically be treated as categorical columns.')
                warnings.warn(msg, UserWarning)
                categorical_indices = np.unique(
                    np.concatenate([categorical_indices, 
                                    non_numerical_indices]))
        self.num_features = numerical_indices.shape[0] + \
            non_numerical_indices.shape[0]
        self.categorical_indices = categorical_indices
        # Find indices that aren't in self.categorical_indices
        if fuav.is_structured_array(self.dataset):
            indices = np.array(self.dataset.dtype.names)
        else:
            indices = np.arange(0, self.dataset.shape[1], 1)
        self.non_categorical_indices = np.setdiff1d(indices,
                                                    self.categorical_indices)


    def _validate_input(self,
                        dataset: np.ndarray,
                        categorical_indices: np.ndarray = None) -> bool:
        """
        Validates input array and categorical_indices.

        Parameters
        ----------
        dataset : numpy.ndarray
            A 2-dimensional numerical numpy array with a dataset to be used.
        categorical_indices : numpy.ndarray
            An array of indices that should be treat as categorical variables.
        
        Raises
        ------
        IncorrectShapeError
            The input data is not a 2-dimensional array. The categorical indices
            list/array (``categorical_features``) is not 1-dimensional.
        TypeError
            `categorical_indices` is not eiher a numpy.ndarray or None. `dataset`
            is not a numpy.ndarray.
        
        Returns
        -------
        is_input_ok : boolean
            If input is valid.
        """
        is_input_ok = False

        if not isinstance(dataset, np.ndarray):
            raise TypeError('dataset must be a numpy.ndarray.')

        if (categorical_indices is not None and not 
                isinstance(categorical_indices, np.ndarray)):
            raise TypeError('categorical_indices must be a numpy.ndarray or '
                            'None')
        
        if not fuav.is_2d_array(dataset):
            raise IncorrectShapeError('The input dataset must be a '
                                      '2-dimensional array.')
        
        if categorical_indices is not None:
            invalid_indices = fuat.get_invalid_indices(dataset, 
                                                    categorical_indices)
            if invalid_indices.size > 0:
                raise IndexError('Indices {} are not valid for input '
                                 'dataset.'.format(invalid_indices))

        is_input_ok = True
    
        return is_input_ok


    def _validate_sample_input(self,
                               data_row: Optional[np.ndarray] = None,
                               num_samples: int = 10):
        """
        Validates sample input for all `sample` functions.
        
        This function checks the validaty of `data_row` with respect to the
        dataset provided in the constructor for the augmentor.

        Raises
        ------
        ValueError

        Returns
        -------
        is_input_ok : boolean
            If input is valid.
        """
        is_input_ok = False
        if (not isinstance(data_row, np.ndarray) and not
                isinstance(data_row, np.void)):
            raise TypeError('data_row must be numpy.ndarray.')

        if not isinstance(num_samples, int):
            raise TypeError('num_samples must be an integer.')

        if num_samples < 1:
            raise ValueError('num_samples must be an integer greater than 0.')

        if not fuav.is_1d_like(data_row):
            raise IncorrectShapeError('data_row must be a 1-dimensional '
                                      'array.')

        if not fuav.are_similar_dtype_arrays(
                self.dataset, np.array(data_row, dtype=data_row.dtype),
                strict_comparison=True):
            raise ValueError('data_row provided is not of the same dtype as '
                             'the dataset used to initialise this class. '
                             'Please ensure that the dataset and data_row '
                             'dtypes are identical.')

        # If structured and different number of features, would be caught
        # in the previous if statement.
        if not fuav.is_structured_array(self.dataset):
            if data_row.shape[0] != self.dataset.shape[1]:
                raise ValueError('data_row must contain the same number of '
                                'features as the dataset used in the class '
                                'constructor.')

        is_input_ok = True

        return is_input_ok


    @abstractmethod
    def sample(self,
               data_row: Optional[np.ndarray] = None,
               num_samples: Optional[int] = 10) -> np.ndarray:
        """
        Abstract method that must be implemented for each child class.

        Must sample around data_row or if data_row is not given then should
        generate samples from the whole dataset.
        """
        pass


class NormalSampling(Augmentation):
    """
    Class that samples using a normal distribution (0, 1).

    For additional parameteres, attributes, warnings and exceptions raised
    please see the  documentation of the :func`fatf.utils.data.Augmentation`
    class.

    Attributes
    ----------
    non_categorical_sampling_values : dictionary[Union[str, int], Tuple[float, float, float]]
        Dictionary mapping non-categorical feature indices to tuples containing
        (mean, variance, std) for each non-categorical feature.
    categorical_sampling_values : dictionary[Union[str, int], Tuple[List[Union[str, int]], np.ndarray]]
        Dcitionary mapping categorical feature indices to tuples containing
        (list of values, frequencies of values) for each categorical feature.
    """
    def __init__(self,
                 dataset: np.ndarray,
                 categorical_indices: np.ndarray = None) -> None:
        super(NormalSampling, self).__init__(dataset, categorical_indices)
        numerical_features = \
            fuat.as_unstructured(self.dataset[self.non_categorical_indices])

        non_categorical_sampling_values = dict()
        # In case numerical_features is empty array
        if numerical_features.size > 0:
            features_mean = np.mean(numerical_features,axis=1)
            features_var = np.var(numerical_features, axis=1)
            features_std = np.sqrt(features_var)
            for cf, mean, var, std in zip(
                    self.non_categorical_indices, features_mean, features_var, 
                    features_std):
                non_categorical_sampling_values[cf] = (mean, var, std)
        categorical__sampling_values = dict()
        for cf in self.categorical_indices:
            feature_vector = self.dataset[cf]
            feature_counter = collections.Counter(feature_vector)
            values = list(feature_counter.keys())
            frequencies = np.array(list(feature_counter.values()),
                                   dtype=np.float)
            frequencies /= np.sum(frequencies)
            categorical__sampling_values[cf] = (values, frequencies)

        self.non_categorical_sampling_values = non_categorical_sampling_values
        self.categorical_sampling_values = categorical__sampling_values



    def sample(self,
               data_row: Optional[np.ndarray] = None,
               num_samples: Optional[int] = 10) -> np.ndarray:
        """
        Samles from normal around `data_row` or around mean.

        If `data_row` is None, then samples will be generated around feature
        means using `self.feature_means` attribute.
        """
        self._validate_sample_input(data_row, num_samples)
        pass
'''
def lime(num_samples,
         data_set,
         data_row=None):
    """Generates a neighborhood around a prediction.

    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to
    the means and stds in the training data. For categorical features,
    perturb by sampling according to the training distribution, and making
    a binary feature that is 1 when the value is the same as the instance
    being explained.

    Args:
        data_row: 1d numpy array, corresponding to a row
        num_samples: size of the neighborhood to learn the linear model

    Returns:
        A tuple (data, inverse), where:
            data: dense num_samples * K matrix, where categorical features
            are encoded with either 0 (not equal to the corresponding value
            in data_row) or 1. The first row is the original instance.
            inverse: same as data, except the categorical features are not
            binary, but categorical (as the original data)
    """

    # TODO: Check input array is 2D; data row is 1D
    # TODO: num samples has to be a positive integer

    # TODO
    categorical_features, numerical_features = TODO_get_feature_types()

    features_mean = np.mean(data_set, axis=1)
    features_var = np.var(data_set, axis=1)
    features_std = np.sqrt(features_var)

    feature_frequencies = dict()
    feature_values = dict()
    for cf in categorical_features:
        feature_vector = data_set[cf]
        feature_counter = collections.Counter(feature_vector)

        feature_values[cf] = list(feature_counter.keys())

        feature_frequencies[cf] = np.array(list(feature_counter.values()))
        feature_frequencies[cf] /= np.sum(feature_frequencies[cf])

    # If data point is not given take a global sample (based on the mean of all features)
    data_row = features_mean if data_row is None else data_row

    num_features = data_set.shape[0]
    data = np.zeros((num_samples, num_features))

    data = np.random.normal(
            0, 1, num_samples * num_features).reshape(
            num_samples, num_features)
    data = data * features_std + data_row

    data[0] = data_row.copy()
    for column in categorical_features:
        vals = feature_values[column]
        freqs = feature_frequencies[column]

        new_column = np.random.choice(vals, size=num_samples,
                                          replace=True, p=freqs)
        new_column[0] = data[0, column]

        data[:, column] = new_column
    data[0] = data_row.copy()

    return data


def lime_data(generated_data, original_data):
    binary_data = generated_data.copy()
    for column in categorical_features:
        binary_column = np.array([1 if x == data_row[column]
                                  else 0 for x in inverse_column])
        binary_column[0] = 1
        data[:, column] = binary_column
    pass
'''
