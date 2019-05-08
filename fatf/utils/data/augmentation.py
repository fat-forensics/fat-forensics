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
    non_categorical_indices : numpy.ndarray
        An array of indices that are not categorical variables.
    is_structured : boolean
        If the input dataset is structured or not.

    Warns
    -----
    UserWarning
        If index of a string type column is not in `categorical_indices` it
        it will be added or if no `categorical_indices` were provided, whether
        or not a feature is categorical will be inferred based on the data
        type.
    
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
        self.is_structured = fuav.is_structured_array(dataset)

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
        if self.is_structured:
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
            `categorical_indices` is not eiher a numpy.ndarray or None. 
            `dataset` is not a numpy.ndarray.
        IndexError
            Indices in `categorical_indices` are not valid for `dataset`.

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
                               num_samples: Optional[int] = 10):
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

        if not isinstance(num_samples, int):
            raise TypeError('num_samples must be an integer.')

        if num_samples < 1:
            raise ValueError('num_samples must be an integer greater than 0.')

        if data_row is not None:
            if (not isinstance(data_row, np.ndarray) and not
                    isinstance(data_row, np.void)):
                raise TypeError('data_row must be numpy.ndarray.')
            if not fuav.is_1d_like(data_row):
                raise IncorrectShapeError('data_row must be a 1-dimensional '
                                        'array.')

            if not fuav.are_similar_dtype_arrays(
                    self.dataset, np.array(data_row, dtype=data_row.dtype),
                    strict_comparison=True):
                raise ValueError('data_row provided is not of the same dtype '
                                 'as the dataset used to initialise this '
                                 'class. Please ensure that the dataset and '
                                 'data_row dtypes are identical.')

            # If structured and different number of features, would be caught
            # in the previous if statement.
            if not self.is_structured:
                if data_row.shape[0] != self.dataset.shape[1]:
                    raise ValueError('data_row must contain the same number of '
                                    'features as the dataset used in the class '
                                    'constructor.')

        is_input_ok = True

        return is_input_ok


    @abstractmethod
    def sample(self,
               data_row: Optional[np.ndarray] = None,
               num_samples: Optional[int] = None) -> np.ndarray: 
        """
        Abstract method that must be implemented for each child class.

        Must sample around data_row or if data_row is not given then should
        generate samples from the whole dataset.

        Parameters
        ----------
        data_row : Optional[numpy.ndarray], default = None
            Row of data to sample around.
        num_samples : Optional[integer], default = None
            Number of samples to generate.
        
        Returns
        -------
        data : numpy.ndarray
            Sampled data.
        """
        raise NotImplementedError('Sample method needs to be overwritten.')


class NormalSampling(Augmentation):
    """
    Class that samples using a normal distribution.

    Sampling can be performed either using the mean of the dataset or a normal
    centered on a certain data point. In both cases, the standard deviation
    of each feature in the dataset is used. For categorical features, the
    probability for each unique value is calculated based off how many times
    the value appears in the dataset and the values are sample from this.

    For additional parameteres, attributes, warnings and exceptions raised
    please see the  documentation of the :func`fatf.utils.data.Augmentation`
    class.

    Attributes
    ----------
    non_categorical_sampling_values : dictionary[Union[string, integer],
            Tuple[float, float, float]]
        Dictionary mapping non-categorical feature indices to tuples containing
        (mean, variance, std) for each non-categorical feature.
    categorical_sampling_values : dictionary[Union[string, integer], 
            Tuple[List[Union[string, integer]], numpy.ndarray]]
        Dcitionary mapping categorical feature indices to tuples containing
        (list of values, frequencies of values) for each categorical feature.
    dataset_generalised : numpy.ndarray
        Dataset attribute with generalised dtype in case sampling does not
        agree with base dtype. For example, if dtype of a feature is `int` and
        sampling will genereate `float` values.
    """
    def __init__(self,
                 dataset: np.ndarray,
                 categorical_indices: np.ndarray = None) -> None:
        super(NormalSampling, self).__init__(dataset, categorical_indices)

        if self.non_categorical_indices.size > 0:
            if self.is_structured:
                numerical_features = fuat.as_unstructured(
                    self.dataset[self.non_categorical_indices])
            else:
                numerical_features = self.dataset[:, 
                                                  self.non_categorical_indices]
        else:
            numerical_features = np.array([])
        
        # Check if non-categorical indices are numerical and need 
        # generalising for sampling (could try to combine with 
        # fatf.transparency.models.feature_influence._generalise_dataset_type)
        if self.is_structured:
            new_dtypes = []
            for cf in self.dataset.dtype.names:
                if cf in self.non_categorical_indices:
                    dtype = fuat.generalise_dtype(self.dataset.dtype[cf],
                                                  np.dtype(np.float64))
                    new_dtypes.append((cf, dtype))
                else:
                    new_dtypes.append((cf, self.dataset.dtype[cf]))
            self.dataset_generalised = self.dataset.astype(new_dtypes)
        else:
            dtype = fuat.generalise_dtype(self.dataset.dtype,
                                          np.dtype(np.float64))
            self.dataset_generalised = dataset.astype(dtype)

        non_categorical_sampling_values = dict()
        # In case numerical_features is empty array
        if numerical_features.size > 0:
            features_mean = np.mean(numerical_features,axis=0)
            features_var = np.var(numerical_features, axis=0)
            features_std = np.sqrt(features_var)
            for cf, mean, var, std in zip(
                    self.non_categorical_indices, features_mean, features_var, 
                    features_std):
                non_categorical_sampling_values[cf] = (mean, var, std)

        categorical_sampling_values = dict()
        for cf in self.categorical_indices:
            if self.is_structured:
                feature_vector = self.dataset[cf]
            else:
                feature_vector = self.dataset[:, cf]
            feature_counter = collections.Counter(feature_vector)
            values = list(feature_counter.keys())
            frequencies = np.array(list(feature_counter.values()),
                                   dtype=np.float)
            frequencies /= np.sum(frequencies)
            categorical_sampling_values[cf] = (values, frequencies)

        self.non_categorical_sampling_values = non_categorical_sampling_values
        self.categorical_sampling_values = categorical_sampling_values



    def sample(self,
               data_row: Optional[np.ndarray] = None,
               num_samples: Optional[int] = 50) -> np.ndarray:
        """
        Samles from normal around `data_row` or around mean.

        Generates data around instance `data_row` or around the mean of the
        dataset. For numerical data, features are sampled around the value in
        `data_row` or from the mean of the dataset, with standard deviation
        of the feature in the dataset. For categorical features, a distribution
        is constructed over the values in the dataset and randomly sampled
        from, ignoring the categorical value in `data_row`.

        If `data_row` is None, then samples will be generated around feature
        means using `self.feature_means` attribute.

        For parameters please see the  documentation of the 
        :func`fatf.utils.data.Augmentation.sample` function.
        """
        assert self._validate_sample_input(data_row, num_samples), \
            'Input is invald'
        
        is_global = data_row is None

        num_features = (len(self.categorical_indices) +
                        len(self.non_categorical_indices))
        shape = (num_samples,) if self.is_structured else \
            (num_samples, num_features)
        data = np.zeros(shape, dtype=self.dataset_generalised.dtype)

        if self.categorical_indices.size > 0:
            for cf in self.categorical_indices:
                sampling_values = self.categorical_sampling_values[cf]
                values = np.random.choice(sampling_values[0], size=num_samples,
                                        replace=True, p=sampling_values[1])
                if self.is_structured:
                    data[cf] = values
                else:
                    data[:, cf] = values

        if self.non_categorical_indices.size > 0:
            features = np.random.normal(
                0, 1, num_samples * len(self.non_categorical_indices))

            if fuav.is_structured_array(data):
                data[self.non_categorical_indices] = \
                    list(zip(*features.reshape(
                    len(self.non_categorical_indices), num_samples)))
            else:
                data[:, self.non_categorical_indices] = features.reshape(
                    num_samples, len(self.non_categorical_indices))

            for ncf in self.non_categorical_indices:
                sampling_values = self.non_categorical_sampling_values[ncf]
                mean, std = sampling_values[0], sampling_values[2]
                mean = data_row[ncf] if not is_global else mean
                if self.is_structured:
                    data[ncf] = (data[ncf] * std + mean)
                else:
                    data[:, ncf] = (data[:, ncf] * std + mean)

        return data
