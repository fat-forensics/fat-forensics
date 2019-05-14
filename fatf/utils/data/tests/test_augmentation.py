"""
Functions for testing data augmentation classes.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
#         Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import pytest

import numpy as np

import fatf

from fatf.exceptions import IncorrectShapeError

import fatf.utils.data.augmentation as fuda

# yapf: disable
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
    [0.370, 0.762, 0.658, 0.829],
    [-0.103, -0.117, 0.361, 0.571],
    [0.483, -0.117, -0.092, 0.570]])
NUMERICAL_STRUCT_RESULTS = np.array(
    [(0.180, -0.281, -0.252, 0.632),
     (-1.426, -0.506, -0.437, 0.707),
     (-1.286, 0.157, 0.616, 0.324)],
    dtype=[('a', 'f'), ('b', 'f'), ('c', 'f'), ('d', 'f')])
NUMERICAL_NP_CAT_RESULTS = np.array([
    [0., 0.267, 0.268, 0.986],
    [0., 0.723, 0.526, 0.551],
    [0., -0.662, 0.334, 1.007]])
NUMERICAL_STRUCT_CAT_RESULTS = np.array(
    [(0, -0.362, 0.149, 0.623),
     (1, -0.351, 0.458, 0.702),
     (2, -0.047, -0.244, 0.860)],
    dtype=[('a', 'i'), ('b', 'f'), ('c', 'f'), ('d', 'f')])
CATEGORICAL_NP_RESULTS = np.array([
    ['a', 'f', 'c'],
    ['a', 'b', 'c'],
    ['a', 'b', 'c']])
CATEGORICAL_STRUCT_RESULTS = np.array(
    [('a', 'c', 'c'),
     ('b', 'b', 'g'),
     ('a', 'f', 'g')],
    dtype=[('a', 'U1'), ('b', 'U1'), ('c', 'U1')])
MIXED_RESULTS = np.array(
    [(0.254, 'a', 0.429, 'a'),
     (0.071, 'c', -0.310, 'aa',),
     (0.481, 'f', 0.180, 'bb')],
    dtype=[('a', '<f8'), ('b', 'U1'), ('c', '<f8'), ('d', 'U2')])
NUMERICAL_NP_0_CAT_VAL = np.array([0, 1, 2])
NUMERICAL_NP_0_CAT_FREQ = np.array([0.5, 0.3, 0.2])
# yapf: enable


def test_validate_input():
    """
    Tests :func:`fatf.utils.data.augmentation._validate_input` function.
    """
    incorrect_shape_data = ('The input dataset must be a 2-dimensional numpy '
                            'array.')
    type_error_data = 'The input dataset must be of a base type.'
    incorrect_shape_gt = ('The ground_truth array must be 1-dimensional. (Or '
                          'None if it is not required.)')
    type_error_gt = 'The ground_truth array must be of a base type.'
    incorrect_shape_instances = ('The number of labels in the ground_truth '
                                 'array is not equal to the number of data '
                                 'points in the dataset array.')
    index_error_cidx = ('The following indices are invalid for the input '
                        'dataset: {}.')
    type_error_cidx = ('The categorical_indices parameter must be a Python '
                       'list or None.')

    with pytest.raises(IncorrectShapeError) as exin:
        fuda._validate_input(np.array([0, 4, 3, 0]))
    assert str(exin.value) == incorrect_shape_data

    with pytest.raises(TypeError) as exin:
        fuda._validate_input(np.array([[0, 4], [None, 0]]))
    assert str(exin.value) == type_error_data

    #

    with pytest.raises(IncorrectShapeError) as exin:
        fuda._validate_input(MIXED_ARRAY, MIXED_ARRAY)
    assert str(exin.value) == incorrect_shape_gt

    with pytest.raises(TypeError) as exin:
        fuda._validate_input(MIXED_ARRAY, np.array([1, 2, 3, None, 4, 5]))
    assert str(exin.value) == type_error_gt

    with pytest.raises(IncorrectShapeError) as exin:
        fuda._validate_input(MIXED_ARRAY, np.array([1, 2, 3]))
    assert str(exin.value) == incorrect_shape_instances

    #

    with pytest.raises(TypeError) as exin:
        fuda._validate_input(NUMERICAL_NP_ARRAY, categorical_indices=0)
    assert str(exin.value) == type_error_cidx

    with pytest.raises(IndexError) as exin:
        fuda._validate_input(MIXED_ARRAY, categorical_indices=['f'])
    assert str(exin.value) == index_error_cidx.format(['f'])
    with pytest.raises(IndexError) as exin:
        fuda._validate_input(MIXED_ARRAY, categorical_indices=[1])
    assert str(exin.value) == index_error_cidx.format([1])

    #

    assert fuda._validate_input(
        MIXED_ARRAY,
        categorical_indices=['a', 'b'],
        ground_truth=np.array([1, 2, 3, 4, 5, 6]))


class TestAugmentation(object):
    """
    Tests :class:`fatf.utils.data.augmentation.Augmentation` abstract class.
    """

    class BrokenAugmentor1(fuda.Augmentation):
        """
        A broken data augmentation implementation.

        This class does not have a ``sample`` method.
        """

        def __init__(self, dataset, categorical_indices=None):
            """
            Dummy init method.
            """
            super().__init__(dataset, categorical_indices=categorical_indices)

    class BrokenAugmentor2(fuda.Augmentation):
        """
        A broken data augmentation implementation.

        This class does not have a ``sample`` method.
        """

    class BaseAugmentor(fuda.Augmentation):
        """
        A dummy data augmentation implementation.

        For :func:`fatf.utils.data.augmentation._validate_input` and
        :func:`~fatf.utils.data.augmentation.Augmentation._validate_sample_input`
        testing.
        """

        def __init__(self, dataset, categorical_indices=None):
            """
            Dummy init method.
            """
            super().__init__(dataset, categorical_indices=categorical_indices)

        def sample(self, data_row=None, samples_number=10):
            """
            Dummy sample method.
            """
            self._validate_sample_input(data_row, samples_number)
            return np.ones((samples_number, self.features_number))

    def test_augmentation_class_init(self):
        """
        Tests :class:`fatf.utils.data.augmentation.Augmentation` class init.
        """
        abstract_method_error = ("Can't instantiate abstract class "
                                 '{} with abstract methods sample')
        user_warning = (
            'Some of the string-based columns in the input dataset were not '
            'selected as categorical features via the categorical_indices '
            'parameter. String-based columns cannot be treated as numerical '
            'features, therefore they will be also treated as categorical '
            'features (in addition to the ones selected with the '
            'categorical_indices parameter).')

        with pytest.raises(TypeError) as exin:
            self.BrokenAugmentor1(NUMERICAL_NP_ARRAY)
        msg = abstract_method_error.format('BrokenAugmentor1')
        assert str(exin.value) == msg

        with pytest.raises(TypeError) as exin:
            self.BrokenAugmentor2(NUMERICAL_NP_ARRAY)
        msg = abstract_method_error.format('BrokenAugmentor2')
        assert str(exin.value) == msg

        with pytest.raises(TypeError) as exin:
            fuda.Augmentation(NUMERICAL_NP_ARRAY)
        assert str(exin.value) == abstract_method_error.format('Augmentation')

        # Test for a categorical index warning
        with pytest.warns(UserWarning) as warning:
            augmentor = self.BaseAugmentor(CATEGORICAL_NP_ARRAY, [0])
        assert len(warning) == 1
        assert str(warning[0].message) == user_warning
        assert np.array_equal(augmentor.categorical_indices, [0, 1, 2])
        #
        with pytest.warns(UserWarning) as warning:
            augmentor = self.BaseAugmentor(CATEGORICAL_STRUCT_ARRAY, ['a'])
        assert len(warning) == 1
        assert str(warning[0].message) == user_warning
        assert np.array_equal(augmentor.categorical_indices,
                              np.array(['a', 'b', 'c']))
        #
        with pytest.warns(UserWarning) as warning:
            augmentor = self.BaseAugmentor(MIXED_ARRAY, ['b'])
        assert len(warning) == 1
        assert str(warning[0].message) == user_warning
        assert np.array_equal(augmentor.categorical_indices, ['b', 'd'])

        # Validate internal variables
        categorical_np_augmentor = self.BaseAugmentor(CATEGORICAL_NP_ARRAY)
        assert np.array_equal(categorical_np_augmentor.dataset,
                              CATEGORICAL_NP_ARRAY)
        assert not categorical_np_augmentor.is_structured
        assert categorical_np_augmentor.ground_truth is None
        assert categorical_np_augmentor.categorical_indices == [0, 1, 2]
        assert categorical_np_augmentor.numerical_indices == []
        assert categorical_np_augmentor.features_number == 3

        categorical_struct_augmentor = self.BaseAugmentor(
            CATEGORICAL_STRUCT_ARRAY)
        assert np.array_equal(categorical_struct_augmentor.dataset,
                              CATEGORICAL_STRUCT_ARRAY)
        assert categorical_struct_augmentor.is_structured
        assert categorical_struct_augmentor.ground_truth is None
        assert (categorical_struct_augmentor.categorical_indices
                == ['a', 'b', 'c'])  # yapf: disable
        assert categorical_struct_augmentor.numerical_indices == []
        assert categorical_struct_augmentor.features_number == 3

        mixed_augmentor = self.BaseAugmentor(MIXED_ARRAY)
        assert np.array_equal(mixed_augmentor.dataset, MIXED_ARRAY)
        assert mixed_augmentor.is_structured
        assert mixed_augmentor.ground_truth is None
        assert mixed_augmentor.categorical_indices == ['b', 'd']
        assert mixed_augmentor.numerical_indices == ['a', 'c']
        assert mixed_augmentor.features_number == 4

        numerical_np_augmentor = self.BaseAugmentor(NUMERICAL_NP_ARRAY, [0, 1])
        assert np.array_equal(numerical_np_augmentor.dataset,
                              NUMERICAL_NP_ARRAY)
        assert not numerical_np_augmentor.is_structured
        assert numerical_np_augmentor.ground_truth is None
        assert numerical_np_augmentor.categorical_indices == [0, 1]
        assert numerical_np_augmentor.numerical_indices == [2, 3]
        assert numerical_np_augmentor.features_number == 4

    def test_augmentation_sample_validation(self):
        """
        Tests :func:`~fatf.utils.data.augmentation.Augmentation.sample` method.

        This function test validation of input for the ``sample`` method.
        """
        incorrect_shape_data_row = ('The data_row must either be a '
                                    '1-dimensional numpy array or numpy void '
                                    'object for structured rows.')
        type_error_data_row = ('The dtype of the data_row is different to the '
                               'dtype of the data array used to initialise '
                               'this class.')
        incorrect_shape_features = ('The data_row must contain the same '
                                    'number of features as the dataset used '
                                    'to initialise this class.')
        #
        value_error_samples_number = ('The samples_number parameter must be a '
                                      'positive integer.')
        type_error_samples_number = ('The samples_number parameter must be an '
                                     'integer.')

        # Validate sample input rows
        numerical_np_augmentor = self.BaseAugmentor(NUMERICAL_NP_ARRAY)
        categorical_np_augmentor = self.BaseAugmentor(CATEGORICAL_NP_ARRAY)
        numerical_struct_augmentor = self.BaseAugmentor(NUMERICAL_STRUCT_ARRAY)
        categorical_struct_augmentor = self.BaseAugmentor(
            CATEGORICAL_STRUCT_ARRAY, categorical_indices=['a', 'b', 'c'])

        # data_row shape
        with pytest.raises(IncorrectShapeError) as exin:
            numerical_np_augmentor.sample(NUMERICAL_NP_ARRAY)
        assert str(exin.value) == incorrect_shape_data_row
        with pytest.raises(IncorrectShapeError) as exin:
            numerical_struct_augmentor.sample(NUMERICAL_STRUCT_ARRAY)
        assert str(exin.value) == incorrect_shape_data_row

        # data_row type
        with pytest.raises(TypeError) as exin:
            numerical_np_augmentor.sample(np.array(['a', 'b', 'c', 'd']))
        assert str(exin.value) == type_error_data_row
        with pytest.raises(TypeError) as exin:
            numerical_struct_augmentor.sample(MIXED_ARRAY[0])
        assert str(exin.value) == type_error_data_row
        with pytest.raises(TypeError) as exin:
            categorical_np_augmentor.sample(np.array([0.1]))
        assert str(exin.value) == type_error_data_row
        # Structured too short
        with pytest.raises(TypeError) as exin:
            numerical_struct_augmentor.sample(MIXED_ARRAY[['a', 'b']][0])
        assert str(exin.value) == type_error_data_row

        # data_row features number
        with pytest.raises(IncorrectShapeError) as exin:
            numerical_np_augmentor.sample(np.array([0.1, 1, 2]))
        assert str(exin.value) == incorrect_shape_features
        with pytest.raises(IncorrectShapeError) as exin:
            categorical_np_augmentor.sample(np.array(['a', 'b']))
        assert str(exin.value) == incorrect_shape_features

        # samples_number type
        with pytest.raises(TypeError) as exin:
            numerical_np_augmentor.sample(np.array([0, 0, 0.08, 0.69]), 'a')
        assert str(exin.value) == type_error_samples_number
        with pytest.raises(TypeError) as exin:
            numerical_np_augmentor.sample(np.array([0, 0, 0.08, 0.69]), 5.5)
        assert str(exin.value) == type_error_samples_number

        # samples_number value
        with pytest.raises(ValueError) as exin:
            numerical_np_augmentor.sample(np.array([0, 0, 0.08, 0.69]), -1)
        assert str(exin.value) == value_error_samples_number
        with pytest.raises(ValueError) as exin:
            numerical_np_augmentor.sample(np.array([0, 0, 0.08, 0.69]), 0)
        assert str(exin.value) == value_error_samples_number

        # All OK
        ones_30 = np.ones((10, 3))
        ones_40 = np.ones((10, 4))
        ones_300 = np.ones((100, 3))
        ones_400 = np.ones((100, 4))

        assert np.array_equal(
            numerical_np_augmentor.sample(NUMERICAL_NP_ARRAY[0, :]), ones_40)
        assert np.array_equal(
            numerical_np_augmentor.sample(samples_number=100), ones_400)
        assert np.array_equal(
            numerical_np_augmentor.sample(
                NUMERICAL_NP_ARRAY[0, :], samples_number=100), ones_400)

        assert np.array_equal(
            categorical_struct_augmentor.sample(CATEGORICAL_STRUCT_ARRAY[0]),
            ones_30)
        assert np.array_equal(
            categorical_struct_augmentor.sample(samples_number=100), ones_300)
        assert np.array_equal(
            categorical_struct_augmentor.sample(
                CATEGORICAL_STRUCT_ARRAY[0], samples_number=100), ones_300)


class TestNormalSampling(object):
    """
    Tests :class:`fatf.utils.data.augmentation.NormalSampling` class.
    """
    fatf.setup_random_seed()

    numerical_np_0_augmentor = fuda.NormalSampling(NUMERICAL_NP_ARRAY, [0])
    numerical_np_augmentor = fuda.NormalSampling(NUMERICAL_NP_ARRAY)
    numerical_struct_a_augmentor = fuda.NormalSampling(NUMERICAL_STRUCT_ARRAY,
                                                       ['a'])
    numerical_struct_augmentor = fuda.NormalSampling(NUMERICAL_STRUCT_ARRAY)
    categorical_np_augmentor = fuda.NormalSampling(CATEGORICAL_NP_ARRAY)
    categorical_np_012_augmentor = fuda.NormalSampling(CATEGORICAL_NP_ARRAY,
                                                       [0, 1, 2])
    categorical_struct_abc_augmentor = fuda.NormalSampling(
        CATEGORICAL_STRUCT_ARRAY, ['a', 'b', 'c'])
    mixed_augmentor = fuda.NormalSampling(MIXED_ARRAY, ['b', 'd'])

    def test_init(self):
        """
        Tests :class:`fatf.utils.data.augmentation.NormalSampling` class init.
        """
        # Test class inheritance
        assert (self.numerical_np_0_augmentor.__class__.__bases__[0].__name__
                == 'Augmentation')

        # Test calculating numerical and categorical indices
        assert self.numerical_np_0_augmentor.categorical_indices == [0]
        assert self.numerical_np_0_augmentor.numerical_indices == [1, 2, 3]
        #
        assert self.numerical_np_augmentor.categorical_indices == []
        assert self.numerical_np_augmentor.numerical_indices == [0, 1, 2, 3]
        #
        assert self.numerical_struct_a_augmentor.categorical_indices == ['a']
        assert (self.numerical_struct_a_augmentor.numerical_indices
                == ['b', 'c', 'd'])  # yapf: disable
        #
        assert self.categorical_np_augmentor.categorical_indices == [0, 1, 2]
        assert self.categorical_np_augmentor.numerical_indices == []

        # Test attributes unique to NormalSampling
        csv = self.numerical_np_0_augmentor.categorical_sampling_values
        nsv = self.numerical_np_0_augmentor.numerical_sampling_values
        dtype = self.numerical_np_0_augmentor.sample_dtype
        #
        assert len(csv) == 1
        assert 0 in csv
        assert len(csv[0]) == 2
        assert np.array_equal(csv[0][0], np.array([0, 1, 2]))
        assert np.allclose(
            csv[0][1], np.array([3 / 6, 2 / 6, 1 / 6]), atol=1e-3)
        #
        assert len(nsv) == 3
        assert 1 in nsv and 2 in nsv and 3 in nsv
        assert len(nsv[1]) == 2 and len(nsv[2]) == 2 and len(nsv[3]) == 2
        assert nsv[1][0] == pytest.approx(.5, abs=1e-3)
        assert nsv[1][1] == pytest.approx(.5, abs=1e-3)
        assert nsv[2][0] == pytest.approx(.377, abs=1e-3)
        assert nsv[2][1] == pytest.approx(.366, abs=1e-3)
        assert nsv[3][0] == pytest.approx(.563, abs=1e-3)
        assert nsv[3][1] == pytest.approx(.257, abs=1e-3)
        #
        assert dtype == np.float64

        # Test type generalisation
        dtype = self.mixed_augmentor.sample_dtype
        assert len(dtype) == 4
        assert len(dtype[0]) == 2
        assert dtype[0][0] == 'a'
        assert dtype[0][1] == np.float64
        assert len(dtype[1]) == 2
        assert dtype[1][0] == 'b'
        assert dtype[1][1] == 'U1'
        assert len(dtype[2]) == 2
        assert dtype[2][0] == 'c'
        assert dtype[2][1] == np.float64
        assert len(dtype[3]) == 2
        assert dtype[3][0] == 'd'
        assert dtype[3][1] == 'U2'

    def test_sample(self):
        """
        Tests :func:`~fatf.utils.data.augmentation.NormalSampling.sample`.
        """
        # Pure numerical sampling of a data point
        # ...numpy array results
        samples = self.numerical_np_augmentor.sample(
            NUMERICAL_NP_ARRAY[0, :], samples_number=3)
        assert np.allclose(samples, NUMERICAL_NP_RESULTS, atol=1e-3)

        # ...structured array results
        samples_struct = self.numerical_struct_augmentor.sample(
            NUMERICAL_STRUCT_ARRAY[0], samples_number=3)
        for i in samples_struct.dtype.names:
            assert np.allclose(
                samples_struct[i], NUMERICAL_STRUCT_RESULTS[i], atol=1e-3)

        # ...numpy array results mean
        samples = self.numerical_np_augmentor.sample(
            NUMERICAL_NP_ARRAY[0, :], samples_number=1000)
        assert np.allclose(
            samples.mean(axis=0), NUMERICAL_NP_ARRAY[0, :], atol=1e-1)
        assert np.allclose(
            samples.std(axis=0), NUMERICAL_NP_ARRAY.std(axis=0), atol=1e-1)

        # ...structured array results mean
        samples_struct = self.numerical_struct_augmentor.sample(
            NUMERICAL_STRUCT_ARRAY[0], samples_number=1000)
        for i in samples_struct.dtype.names:
            assert np.allclose(
                np.mean(samples_struct[i]),
                NUMERICAL_STRUCT_ARRAY[0][i],
                atol=1e-1)
            assert np.allclose(
                np.std(samples_struct[i]),
                np.std(NUMERICAL_STRUCT_ARRAY[i]),
                atol=1e-1)

        # Pure numerical sampling of the mean of the data
        # ...numpy array mean
        samples = self.numerical_np_augmentor.sample(samples_number=1000)
        assert np.allclose(
            samples.mean(axis=0), NUMERICAL_NP_ARRAY.mean(axis=0), atol=1e-1)
        assert np.allclose(
            samples.std(axis=0), NUMERICAL_NP_ARRAY.std(axis=0), atol=1e-1)

        # ...structured array mean
        samples_struct = self.numerical_struct_augmentor.sample(
            samples_number=1000)
        for i in samples_struct.dtype.names:
            assert np.allclose(
                np.mean(samples_struct[i]),
                np.mean(NUMERICAL_STRUCT_ARRAY[i]),
                atol=1e-1)
            assert np.allclose(
                np.std(samples_struct[i]),
                np.std(NUMERICAL_STRUCT_ARRAY[i]),
                atol=1e-1)

        #######################################################################

        # Numerical sampling with one categorical index defined
        # ...numpy array results
        samples = self.numerical_np_0_augmentor.sample(
            NUMERICAL_NP_ARRAY[0, :], samples_number=3)
        assert np.allclose(samples, NUMERICAL_NP_CAT_RESULTS, atol=1e-3)

        # ...structured array results
        samples_struct = self.numerical_struct_a_augmentor.sample(
            NUMERICAL_STRUCT_ARRAY[0], samples_number=3)
        for i in samples_struct.dtype.names:
            assert np.allclose(
                samples_struct[i], NUMERICAL_STRUCT_CAT_RESULTS[i], atol=1e-3)

        # ...numpy array results mean
        samples = self.numerical_np_0_augmentor.sample(
            NUMERICAL_NP_ARRAY[0, :], samples_number=100)
        # ......numerical
        assert np.allclose(
            samples.mean(axis=0)[1:], NUMERICAL_NP_ARRAY[0, 1:], atol=1e-1)
        assert np.allclose(
            samples.std(axis=0)[1:],
            NUMERICAL_NP_ARRAY.std(axis=0)[1:],
            atol=1e-1)
        # ......categorical
        val, freq = np.unique(samples[:, 0], return_counts=True)
        freq = freq / freq.sum()
        assert np.array_equal(val, NUMERICAL_NP_0_CAT_VAL)
        assert np.allclose(freq, NUMERICAL_NP_0_CAT_FREQ, atol=1e-1)

        # ...structured array results mean
        samples_struct = self.numerical_struct_a_augmentor.sample(
            NUMERICAL_STRUCT_ARRAY[0], samples_number=100)
        # ......numerical
        for i in samples_struct.dtype.names[1:]:
            assert np.allclose(
                np.mean(samples_struct[i]),
                NUMERICAL_STRUCT_ARRAY[0][i],
                atol=1e-1)
            assert np.allclose(
                np.std(samples_struct[i]),
                np.std(NUMERICAL_STRUCT_ARRAY[i]),
                atol=1e-1)
        # ......categorical
        val_struct, freq_struct = np.unique(
            samples_struct['a'], return_counts=True)
        freq_struct = freq_struct / freq_struct.sum()
        assert np.array_equal(val_struct, NUMERICAL_NP_0_CAT_VAL)
        assert np.allclose(freq_struct, NUMERICAL_NP_0_CAT_FREQ, atol=1e-1)

        # ...numpy array mean
        samples = self.numerical_np_0_augmentor.sample(samples_number=1000)
        # ......numerical
        assert np.allclose(
            samples.mean(axis=0)[1:],
            NUMERICAL_NP_ARRAY.mean(axis=0)[1:],
            atol=1e-1)
        # ......categorical
        val, freq = np.unique(samples[:, 0], return_counts=True)
        freq = freq / freq.sum()
        assert np.array_equal(val, NUMERICAL_NP_0_CAT_VAL)
        assert np.allclose(freq, NUMERICAL_NP_0_CAT_FREQ, atol=1e-1)

        # ...structured array mean
        samples_struct = self.numerical_struct_a_augmentor.sample(
            samples_number=1000)
        # ......numerical
        for i in samples_struct.dtype.names[1:]:
            assert np.allclose(
                np.mean(samples_struct[i]),
                np.mean(NUMERICAL_STRUCT_ARRAY[i]),
                atol=1e-1)
            assert np.allclose(
                np.std(samples_struct[i]),
                np.std(NUMERICAL_STRUCT_ARRAY[i]),
                atol=1e-1)
        # ......categorical
        val_struct, freq_struct = np.unique(
            samples_struct['a'], return_counts=True)
        freq_struct = freq_struct / freq_struct.sum()
        assert np.array_equal(val_struct, NUMERICAL_NP_0_CAT_VAL)
        assert np.allclose(freq_struct, NUMERICAL_NP_0_CAT_FREQ, atol=1e-1)

        #######################################################################
        #######################################################################

        # Pure categorical sampling
        # ...numpy array
        samples = self.categorical_np_012_augmentor.sample(
            CATEGORICAL_NP_ARRAY[0], samples_number=3)
        assert np.array_equal(samples, CATEGORICAL_NP_RESULTS)

        # ...structured array
        samples_struct = self.categorical_struct_abc_augmentor.sample(
            CATEGORICAL_STRUCT_ARRAY[0], samples_number=3)
        assert np.array_equal(samples_struct, CATEGORICAL_STRUCT_RESULTS)

        vals = [['a', 'b'], ['b', 'c', 'f'], ['c', 'g']]
        # ...numpy array proportions and values
        samples = self.categorical_np_012_augmentor.sample(
            CATEGORICAL_NP_ARRAY[0], samples_number=100)
        #
        proportions = [
            np.array([0.62, 0.38]),
            np.array([0.31, 0.17, 0.52]),
            np.array([0.63, 0.37])
        ]
        for i, index in enumerate([0, 1, 2]):
            val, freq = np.unique(samples[:, index], return_counts=True)
            freq = freq / freq.sum()
            assert np.array_equal(val, vals[i])
            assert np.allclose(freq, proportions[i], atol=1e-2)

        # ...structured array proportions and values
        samples_struct = self.categorical_struct_abc_augmentor.sample(
            CATEGORICAL_STRUCT_ARRAY[0], samples_number=100)
        #
        proportions = [
            np.array([0.74, 0.26]),
            np.array([0.38, 0.12, 0.50]),
            np.array([0.63, 0.37])
        ]
        for i, index in enumerate(['a', 'b', 'c']):
            val, freq = np.unique(samples_struct[index], return_counts=True)
            freq = freq / freq.sum()
            assert np.array_equal(val, vals[i])
            assert np.allclose(freq, proportions[i], atol=1e-2)

        # No need to check for mean of dataset since categorical features are
        # sampled from the distribution of the entire dataset and not centered
        # on the data_row.

        #######################################################################
        #######################################################################

        # Mixed array with categorical indices auto-discovered
        vals = [['a', 'c', 'f'], ['a', 'aa', 'b', 'bb']]
        proportions = [
            np.array([0.33, 0.33, 0.33]),
            np.array([0.33, 0.16, 0.16, 0.33])
        ]
        # Instance
        samples = self.mixed_augmentor.sample(MIXED_ARRAY[0], samples_number=3)
        # ...categorical
        assert np.array_equal(samples[['b', 'd']], MIXED_RESULTS[['b', 'd']])
        # ...numerical
        for i in ['a', 'c']:
            assert np.allclose(samples[i], MIXED_RESULTS[i], atol=1e-3)

        # Instance mean
        samples = self.mixed_augmentor.sample(
            MIXED_ARRAY[0], samples_number=1000)
        # ...numerical
        for i in ['a', 'c']:
            assert np.allclose(
                np.mean(samples[i]), MIXED_ARRAY[0][i], atol=1e-1)
            assert np.allclose(
                np.std(samples[i]), np.std(MIXED_ARRAY[i]), atol=1e-1)
        # ...categorical
        for i, index in enumerate(['b', 'd']):
            val, freq = np.unique(samples[index], return_counts=True)
            freq = freq / freq.sum()
            assert np.array_equal(val, vals[i])
            assert np.allclose(freq, proportions[i], atol=1e-1)

        # Dataset mean
        samples = self.mixed_augmentor.sample(samples_number=1000)
        # ...numerical
        for i in ['a', 'c']:
            assert np.allclose(
                np.mean(samples[i]), np.mean(MIXED_ARRAY[i]), atol=1e-1)
            assert np.allclose(
                np.std(samples[i]), np.std(MIXED_ARRAY[i]), atol=1e-1)
        # ...categorical
        for i, index in enumerate(['b', 'd']):
            val, freq = np.unique(samples[index], return_counts=True)
            freq = freq / freq.sum()
            assert np.array_equal(val, vals[i])
            assert np.allclose(freq, proportions[i], atol=1e-1)
