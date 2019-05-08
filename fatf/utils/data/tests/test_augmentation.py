"""
Functions for testing data augmentation classes.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
#         Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import pytest
import collections

import numpy as np

import fatf
from fatf.exceptions import IncorrectShapeError
import fatf.utils.data.augmentation as fuda
from fatf.utils.testing.arrays import (BASE_NP_ARRAY, BASE_STRUCTURED_ARRAY,
                                       NOT_BASE_NP_ARRAY)

ONE_D_ARRAY = np.array([0, 4, 3, 0])

NUMERICAL_NP_ARRAY = np.array([
    [0, 0, 0.08, 0.69],
    [1, 0, 0.03, 0.29],
    [0, 1, 0.99, 0.82],
    [2, 1, 0.73, 0.48],
    [1, 0, 0.36, 0.89],
    [0, 1, 0.07, 0.21]])
NUMERICAL_STRUCT_ARRAY = np.array(
    [(0, 0, 0.08, 0.69),
     (1, 0, 0.03, 0.29),
     (0, 1, 0.99, 0.82),
     (2, 1, 0.73, 0.48),
     (1, 0, 0.36, 0.89),
     (0, 1, 0.07, 0.21)],
    dtype=[('a', 'i'), ('b', 'i'), ('c', 'f'), ('d', 'f')])
CATEGORICAL_NP_ARRAY = np.array([
    ['a', 'b', 'c'],
    ['a', 'f', 'g'],
    ['b', 'c', 'c'],
    ['b', 'f', 'c'],
    ['a', 'f', 'c'],
    ['a', 'b', 'g']])
CATEGORICAL_STRUCT_ARRAY = np.array(
    [('a', 'b', 'c'),
     ('a', 'f', 'g'),
     ('b', 'c', 'c'),
     ('b', 'f', 'c'),
     ('a', 'f', 'c'),
     ('a', 'b', 'g')],
    dtype=[('a', 'U1'), ('b', 'U1'), ('c', 'U1')])
MIXED_ARRAY = np.array(
    [(0, 'a', 0.08, 'a'),
     (0, 'f', 0.03, 'bb'),
     (1, 'c', 0.99, 'aa'),
     (1, 'a', 0.73, 'a'),
     (0, 'c', 0.36, 'b'),
     (1, 'f', 0.07, 'bb')],
    dtype=[('a', 'i'), ('b', 'U1'), ('c', 'f'), ('d', 'U2')])

NUMERICAL_NP_RESULTS = np.array([
    [0.370, -0.069, 0.317, 1.081],
    [-0.175, -0.117, 0.658, 0.887],
    [-0.350, 0.271, -0.090, 0.570]])
NUMERICAL_STRUCT_RESULTS = np.array([
    (0.180, -0.281, -0.252, 0.632),
    (-1.426, -0.506, -0.437, 0.707),
    (-1.286, 0.157, 0.616, 0.324)],
    dtype=[('a', 'f'), ('b', 'f'), ('c', 'f'), ('d', 'f')])
NUMERICAL_NP_CAT_RESULTS = np.array([
    [0., 0.267, 0.609, 0.350],
    [0., 0.257, 0.526, 0.868],
    [0., 0.576, -0.119, 1.007]])
NUMERICAL_STRUCT_CAT_RESULTS = np.array([
    (0, -0.362, 0.149, 0.623),
    (1, -0.351, 0.458, 0.702),
    (2, -0.047, -0.244, 0.860)],
    dtype=[('a', 'i'), ('b', 'f'), ('c', 'f'), ('d', 'f')])
CATEGORICAL_NP_RESULTS = np.array([
    ['a', 'f', 'c'],
    ['a', 'b', 'c'],
    ['a', 'b', 'c']])
CATEGORICAL_STRUCT_RESULTS = np.array([
    ('a', 'f', 'c'),
    ('b', 'b', 'g'),
    ('a', 'f', 'g')],
    dtype=[('a', 'U1'), ('b', 'U1'), ('c', 'U1')])
MIXED_RESULTS = np.array([
    (0.254, 'a', 0.429, 'a'),
    (0.071, 'f', -0.310, 'bb',),
    (0.481, 'c', 0.180, 'b')],
    dtype=[('a', '<f8'), ('b', 'U1'), ('c', '<f8'), ('d', 'U2')])


class BaseAugmentor(fuda.Augmentation):
    """
    Dummy class to test :func`fatf.utils.data.augmentation.Augmentation.
    _validate_input` and :func`fatf.utils.data.augmentation.Augmentation.
    _validate_sample_input`.

    """
    def __init__(self, dataset, categorical_indices=None):
        super(BaseAugmentor, self).__init__(dataset, categorical_indices)
    
    def sample(self, data_row=None, num_samples=10):
        self._validate_sample_input(data_row, num_samples)
        return True


class BrokenAugmentor(fuda.Augmentation):
    """
    Class with no `sample` function defined.
    """
    def __init__(self, dataset, categorical_indices=None):
        super(BrokenAugmentor, self).__init__(dataset, categorical_indices)


def test_Augmentation():
    """
    tests :class`fatf.utils.data.augmentation.Augmentation`
    """
    msg = ('Can\'t instantiate abstract class BrokenAugmentor with abstract '
           'methods sample')
    with pytest.raises(TypeError) as exin:
        augmentor = BrokenAugmentor(NUMERICAL_NP_ARRAY)
    assert str(exin.value) == msg

    msg = ('dataset must be a numpy.ndarray.')
    with pytest.raises(TypeError) as exin:
        augmentor = BaseAugmentor(0)
    assert str(exin.value) == msg

    msg = ('categorical_indices must be a numpy.ndarray or None')
    with pytest.raises(TypeError) as exin:
        augmentor = BaseAugmentor(NUMERICAL_NP_ARRAY, categorical_indices=0)
    assert str(exin.value) == msg

    msg = ('The input dataset must be a 2-dimensional array.')
    with pytest.raises(IncorrectShapeError) as exin:
        augmentor = BaseAugmentor(ONE_D_ARRAY, np.array([0]))
    assert str(exin.value) == msg

    msg = ('Indices {} are not valid for input dataset.')
    with pytest.raises(IndexError) as exin:
        augmentor = BaseAugmentor(NUMERICAL_NP_ARRAY, np.array([10]))
    assert str(exin.value) == msg.format(np.array([10]))
    with pytest.raises(IndexError) as exin:
        augmentor = BaseAugmentor(NUMERICAL_NP_ARRAY, np.array(['a']))
    assert str(exin.value) == msg.format(np.array(['a']))
    with pytest.raises(IndexError) as exin:
        augmentor = BaseAugmentor(NUMERICAL_STRUCT_ARRAY, np.array(['l']))
    assert str(exin.value) == msg.format(np.array(['l']))

    msg = ('No categorical_indcies were provided. The categorical columns '
           'will be inferred by looking at the type of values in the dataset.')
    with pytest.warns(UserWarning) as warning:
        augmentor = BaseAugmentor(CATEGORICAL_NP_ARRAY)
    assert len(warning) == 1
    assert str(warning[0].message) == msg
    assert np.array_equal(augmentor.categorical_indices, np.array([0, 1, 2]))
    with pytest.warns(UserWarning) as warning:
        augmentor = BaseAugmentor(CATEGORICAL_STRUCT_ARRAY)
    assert len(warning) == 1
    assert str(warning[0].message) == msg
    assert np.array_equal(augmentor.categorical_indices, 
                          np.array(['a', 'b', 'c']))
    with pytest.warns(UserWarning) as warning:
        augmentor = BaseAugmentor(MIXED_ARRAY)
    assert len(warning) == 1
    assert str(warning[0].message) == msg
    assert np.array_equal(augmentor.categorical_indices, np.array(['b', 'd']))

    msg = ('String based indices were found in dataset but not given as '
           'categorical_indices. String based columns will automatically be '
           'treated as categorical columns.')
    with pytest.warns(UserWarning) as warning:
        augmentor = BaseAugmentor(CATEGORICAL_NP_ARRAY, np.array([0]))
    assert len(warning) == 1
    assert str(warning[0].message) == msg
    assert np.array_equal(augmentor.categorical_indices, np.array([0, 1, 2]))
    with pytest.warns(UserWarning) as warning:
        augmentor = BaseAugmentor(CATEGORICAL_STRUCT_ARRAY, np.array(['a']))
    assert len(warning) == 1
    assert str(warning[0].message) == msg
    assert np.array_equal(augmentor.categorical_indices, 
                          np.array(['a', 'b', 'c']))
    with pytest.warns(UserWarning) as warning:
        augmentor = BaseAugmentor(MIXED_ARRAY, np.array(['b']))
    assert len(warning) == 1
    assert str(warning[0].message) == msg
    assert np.array_equal(augmentor.categorical_indices, np.array(['b', 'd']))

    # Validate sample input rows
    augmentor = BaseAugmentor(NUMERICAL_NP_ARRAY)
    msg = ('data_row must be numpy.ndarray.')
    with pytest.raises(TypeError) as exin:
        sample = augmentor.sample(1)
    assert str(exin.value) == msg

    msg = ('num_samples must be an integer.')
    with pytest.raises(TypeError) as exin:
        sample = augmentor.sample(np.array([]), 'a')
    assert str(exin.value) == msg

    msg = ('num_samples must be an integer greater than 0.')
    with pytest.raises(ValueError) as exin:
        sample = augmentor.sample(np.array([]), -1)
    assert str(exin.value) == msg

    msg = ('data_row provided is not of the same dtype as the dataset used to '
           'initialise this class. Please ensure that the dataset and data_row '
           'dtypes are identical.')
    with pytest.raises(ValueError) as exin:
        sample = augmentor.sample(np.array(['a', 'b', 'c', 'd']))
    assert str(exin.value) == msg
    with pytest.raises(ValueError) as exin:
        sample = augmentor.sample(MIXED_ARRAY[0])
    assert str(exin.value) == msg
    augmentor = BaseAugmentor(CATEGORICAL_STRUCT_ARRAY, 
                              np.array(['a', 'b', 'c']))
    with pytest.raises(ValueError) as exin:
        sample = augmentor.sample(np.array([0.1]))
    assert str(exin.value) == msg
    with pytest.raises(ValueError) as exin:
        sample = augmentor.sample(MIXED_ARRAY[0][['a', 'b']])
    assert str(exin.value) == msg

    augmentor = BaseAugmentor(NUMERICAL_NP_ARRAY)
    msg = ('data_row must contain the same number of features as the dataset '
           'used in the class constructor.')
    with pytest.raises(ValueError) as exin:
        sample = augmentor.sample(np.array([0.1]))
    assert str(exin.value) == msg
    augmentor = BaseAugmentor(CATEGORICAL_NP_ARRAY, np.array([0, 1, 2]))
    with pytest.raises(ValueError) as exin:
        sample = augmentor.sample(np.array(['a']))
    assert str(exin.value) == msg

    msg = ('data_row must be a 1-dimensional array.')
    with pytest.raises(IncorrectShapeError) as exin:
        sample = augmentor.sample(NUMERICAL_NP_ARRAY)
    assert str(exin.value) == msg
    augmentor = BaseAugmentor(NUMERICAL_STRUCT_ARRAY)
    with pytest.raises(IncorrectShapeError) as exin:
        sample = augmentor.sample(NUMERICAL_STRUCT_ARRAY)
    assert str(exin.value) == msg

    # All ok
    augmentor = BaseAugmentor(NUMERICAL_NP_ARRAY)
    assert augmentor.sample(NUMERICAL_NP_ARRAY[0, :])
    assert augmentor.sample(num_samples=100)
    assert augmentor.sample(NUMERICAL_NP_ARRAY[0, :], num_samples=100)

    augmentor = BaseAugmentor(CATEGORICAL_STRUCT_ARRAY, 
                              categorical_indices=np.array(['a' ,'b' ,'c']))
    assert augmentor.sample(CATEGORICAL_STRUCT_ARRAY[0])
    assert augmentor.sample(num_samples=100)
    assert augmentor.sample(CATEGORICAL_STRUCT_ARRAY[0], num_samples=100)



def test_NormalSampling():
    """
    tests :func`fatf.utils.data.augmentation.NormalSampling`
    """
    fatf.setup_random_seed()

    # Test class inheritence and calcuating non_categorical_indices
    # and categorical_indices
    augmentor = fuda.NormalSampling(NUMERICAL_NP_ARRAY, np.array([0]))
    assert augmentor.__class__.__bases__[0].__name__ == 'Augmentation'
    assert np.array_equal(augmentor.categorical_indices, np.array([0]))
    assert np.array_equal(augmentor.non_categorical_indices,
                          np.array([1, 2, 3]))

    augmentor = fuda.NormalSampling(NUMERICAL_NP_ARRAY)
    assert np.array_equal(augmentor.categorical_indices, np.array([]))
    assert np.array_equal(augmentor.non_categorical_indices,
                          np.array([0, 1, 2, 3]))

    augmentor = fuda.NormalSampling(NUMERICAL_STRUCT_ARRAY, np.array(['a']))
    assert np.array_equal(augmentor.categorical_indices, np.array(['a']))
    assert np.array_equal(augmentor.non_categorical_indices,
                          np.array(['b', 'c', 'd']))

    msg = ('No categorical_indcies were provided. The categorical columns '
           'will be inferred by looking at the type of values in the dataset.')
    with pytest.warns(UserWarning) as warning:
        augmentor = fuda.NormalSampling(CATEGORICAL_NP_ARRAY)
    assert len(warning) == 1
    assert str(warning[0].message) == msg
    assert np.array_equal(augmentor.categorical_indices,
                          np.array([0, 1, 2]))
    assert np.array_equal(augmentor.non_categorical_indices, np.array([]))

    # Pure numerical sampling
    augmentor = fuda.NormalSampling(NUMERICAL_NP_ARRAY)
    augmentor_struct = fuda.NormalSampling(NUMERICAL_STRUCT_ARRAY)
    samples = augmentor.sample(NUMERICAL_NP_ARRAY[0, :], num_samples=3)
    samples_struct = augmentor_struct.sample(NUMERICAL_STRUCT_ARRAY[0],
                                             num_samples=3)
    assert np.allclose(samples, NUMERICAL_NP_RESULTS, atol=1e-2)
    assert all([np.allclose(samples_struct[x], NUMERICAL_STRUCT_RESULTS[x],
                atol=1e-1) for x in samples_struct.dtype.names])

    samples = augmentor.sample(NUMERICAL_NP_ARRAY[0, :], num_samples=1000)
    samples_struct = augmentor_struct.sample(NUMERICAL_STRUCT_ARRAY[0],
                                             num_samples=1000)
    assert np.allclose(samples.mean(axis=0), NUMERICAL_NP_ARRAY[0, :],
                       atol=1e-1)
    assert np.allclose(samples.std(axis=0), NUMERICAL_NP_ARRAY.std(axis=0),
                       atol=1e-1)
    assert all([np.allclose(np.mean(samples_struct[x]),
                NUMERICAL_STRUCT_ARRAY[0][x], atol=1e-1) for x in 
                samples_struct.dtype.names])
    assert all([np.allclose(np.std(samples_struct[x]),
                np.std(NUMERICAL_STRUCT_ARRAY[x]), atol=1e-1) for x in 
                samples_struct.dtype.names])
    # Mean of dataset
    samples = augmentor.sample(num_samples=1000)
    samples_struct = augmentor_struct.sample(num_samples=1000)
    assert np.allclose(samples.mean(axis=0), NUMERICAL_NP_ARRAY.mean(axis=0),
                       atol=1e-1)
    assert np.allclose(samples.std(axis=0), NUMERICAL_NP_ARRAY.std(axis=0),
                       atol=1e-1)
    assert all([np.allclose(np.mean(samples_struct[x]),
                np.mean(NUMERICAL_STRUCT_ARRAY[x]), atol=1e-1) for x in 
                samples_struct.dtype.names])
    assert all([np.allclose(np.std(samples_struct[x]),
                np.std(NUMERICAL_STRUCT_ARRAY[x]), atol=1e-1) for x in 
                samples_struct.dtype.names])

    # Numerical sampling with one categorical_indices defined
    augmentor = fuda.NormalSampling(NUMERICAL_NP_ARRAY, np.array([0]))
    augmentor_struct = fuda.NormalSampling(NUMERICAL_STRUCT_ARRAY,
                                           np.array(['a']))
    samples = augmentor.sample(NUMERICAL_NP_ARRAY[0, :], num_samples=3)
    samples_struct = augmentor_struct.sample(NUMERICAL_STRUCT_ARRAY[0],
                                             num_samples=3)
    assert np.allclose(samples, NUMERICAL_NP_CAT_RESULTS, atol=1e-2)
    assert all([np.allclose(samples_struct[x], NUMERICAL_STRUCT_CAT_RESULTS[x],
                atol=1e-1) for x in samples_struct.dtype.names])
    
    samples = augmentor.sample(NUMERICAL_NP_ARRAY[0, :], num_samples=100)
    samples_struct = augmentor_struct.sample(NUMERICAL_STRUCT_ARRAY[0],
                                             num_samples=100)
    assert np.allclose(samples.mean(axis=0)[1:], NUMERICAL_NP_ARRAY[0, 1:],
                       atol=1e-1)
    assert np.allclose(samples.std(axis=0)[1:],
                       NUMERICAL_NP_ARRAY.std(axis=0)[1:], atol=1e-1)
    assert all([np.allclose(np.mean(samples_struct[x]),
                NUMERICAL_STRUCT_ARRAY[0][x], atol=1e-1) for x in 
                samples_struct.dtype.names[1:]])
    assert all([np.allclose(np.std(samples_struct[x]),
                np.std(NUMERICAL_STRUCT_ARRAY[x]), atol=1e-1) for x in 
                samples_struct.dtype.names[1:]])
    # Test the categorical feature
    collection = collections.Counter(samples[:, 0])
    collection_struct = collections.Counter(samples_struct['a'])
    freq = np.array([collection[k] for k in [0, 1, 2]], dtype=np.float)
    freq_struct = np.array([collection_struct[k] for k in [0, 1, 2]],
                           dtype=np.float)
    freq, freq_struct = freq / np.sum(freq), freq_struct / np.sum(freq_struct)
    assert np.allclose(freq, np.array([0.5, 0.3, 0.2]), atol=1e-1)
    assert np.allclose(freq_struct, np.array([0.5, 0.3, 0.2]), atol=1e-1)
     # Mean of dataset
    samples = augmentor.sample(num_samples=1000)
    samples_struct = augmentor_struct.sample(num_samples=1000)
    assert np.allclose(samples.mean(axis=0)[1:], 
                       NUMERICAL_NP_ARRAY.mean(axis=0)[1:], atol=1e-1)
    assert all([np.allclose(np.mean(samples_struct[x]),
                np.mean(NUMERICAL_STRUCT_ARRAY[x]), atol=1e-1) for x in 
                samples_struct.dtype.names[1:]])
    assert all([np.allclose(np.std(samples_struct[x]),
                np.std(NUMERICAL_STRUCT_ARRAY[x]), atol=1e-1) for x in 
                samples_struct.dtype.names[1:]])
    # Test the categorical feature
    collection = collections.Counter(samples[:, 0])
    collection_struct = collections.Counter(samples_struct['a'])
    freq = np.array([collection[k] for k in [0, 1, 2]], dtype=np.float)
    freq_struct = np.array([collection_struct[k] for k in [0, 1, 2]],
                           dtype=np.float)
    freq, freq_struct = freq / np.sum(freq), freq_struct / np.sum(freq_struct)
    assert np.allclose(freq, np.array([0.5, 0.3, 0.2]), atol=1e-1)
    assert np.allclose(freq_struct, np.array([0.5, 0.3, 0.2]), atol=1e-1)

    # Pure categorical array
    augmentor = fuda.NormalSampling(CATEGORICAL_NP_ARRAY, np.array([0, 1, 2]))
    augmentor_struct = fuda.NormalSampling(CATEGORICAL_STRUCT_ARRAY,
                                           np.array(['a', 'b', 'c']))
    samples = augmentor.sample(CATEGORICAL_NP_ARRAY[0], num_samples=3)
    samples_struct = augmentor_struct.sample(CATEGORICAL_STRUCT_ARRAY[0],
                                             num_samples=3)
    assert np.array_equal(samples, CATEGORICAL_NP_RESULTS)
    assert np.array_equal(samples_struct, CATEGORICAL_STRUCT_RESULTS)

    samples = augmentor.sample(CATEGORICAL_NP_ARRAY[0], num_samples=100)
    samples_struct = augmentor_struct.sample(CATEGORICAL_STRUCT_ARRAY[0],
                                             num_samples=100)
    # Check proportions of categorical features
    freqs, freqs_struct = [], []
    vals = [['a', 'b'], ['b', 'f', 'c'], ['c', 'g']]
    proportions = [np.array([0.66, 0.33]), np.array([0.33, 0.5, 0.16]),
                   np.array([0.66, 0.33])]
    for cf, cfs, val in zip([0, 1, 2], ['a', 'b', 'c'], vals):
        collection = collections.Counter(samples[:, cf])
        collection_struct = collections.Counter(samples_struct[cfs])
        freq = np.array([collection[k] for k in val], dtype=np.float)
        freq_struct = np.array([collection_struct[k] for k in val],
                                dtype=np.float)
        freqs.append(freq / np.sum(freq))
        freqs_struct.append(freq_struct / np.sum(freq_struct))
    # Check for all features that unstructured and structured have right
    # value proportions
    for f, fs, p in zip(freqs, freqs_struct, proportions):
        assert np.allclose(f, p, atol=1e-1)
        assert np.allclose(fs, p, atol=1e-1)
    # No need to check for mean of dataset since categorical features are 
    # sampeld from the distribution of the entire dataset and not centered on
    # the data_row

    # Mixed array with categorical_indices defined when feature is string
    augmentor = fuda.NormalSampling(MIXED_ARRAY, np.array(['b', 'd']))
    samples = augmentor.sample(MIXED_ARRAY[0], num_samples=3)
    assert all([np.allclose(samples[x], MIXED_RESULTS[x], atol=1e-2)
                for x in ['a', 'c']])
    assert np.array_equal(samples[['b', 'd']], MIXED_RESULTS[['b', 'd']])

    samples = augmentor.sample(MIXED_ARRAY[0], num_samples=1000)
    assert all([np.allclose(np.mean(samples[x]), MIXED_ARRAY[0][x], 
                atol=1e-1) for x in ['a', 'c']])
    assert all([np.allclose(np.std(samples[x]), np.std(MIXED_ARRAY[x]), 
                atol=1e-1) for x in ['a', 'c']])
    # Check proportions of categorical features
    freqs, freqs_struct = [], []
    vals = [['a', 'f', 'c'], ['a', 'aa', 'b', 'bb']]
    proportions = [np.array([0.33, 0.33, 0.33]),
                   np.array([0.33, 0.16, 0.16, 0.33])]
    for cf, val in zip(['b', 'd'], vals):
        collection = collections.Counter(samples[cf])
        freq = np.array([collection[k] for k in val], dtype=np.float)
        freqs.append(freq / np.sum(freq))
    # Check for all features that unstructured and structured have right
    # value proportions
    assert all([np.allclose(f, p, atol=1e-1) for f,p in 
                zip(freqs, proportions)])
    # Mean of dataset
    samples = augmentor.sample(num_samples=1000)
    assert all([np.allclose(np.mean(samples[x]), np.mean(MIXED_ARRAY[x]), 
                atol=1e-1) for x in ['a', 'c']])
    assert all([np.allclose(np.std(samples[x]), np.std(MIXED_ARRAY[x]), 
                atol=1e-1) for x in ['a', 'c']])
    # Check proportions of categorical features
    freqs, freqs_struct = [], []
    vals = [['a', 'f', 'c'], ['a', 'aa', 'b', 'bb']]
    proportions = [np.array([0.33, 0.33, 0.33]),
                   np.array([0.33, 0.16, 0.16, 0.33])]
    for cf, val in zip(['b', 'd'], vals):
        collection = collections.Counter(samples[cf])
        freq = np.array([collection[k] for k in val], dtype=np.float)
        freqs.append(freq / np.sum(freq))
    # Check for all features that unstructured and structured have right
    # value proportions
    assert all([np.allclose(f, p, atol=1e-1) for f,p in 
                zip(freqs, proportions)])
