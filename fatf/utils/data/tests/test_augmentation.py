"""
Functions and classes for testing data augmentation approaches.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
#         Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import scipy
import scipy.stats

import pytest

import numpy as np

import fatf

from fatf.exceptions import IncompatibleModelError, IncorrectShapeError

import fatf.utils.array.tools as fuat
import fatf.utils.data.augmentation as fuda
import fatf.utils.distances as fud
import fatf.utils.models as fum

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
    type_error_itf = 'The int_to_float parameter has to be a boolean.'

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

    with pytest.raises(TypeError) as exin:
        fuda._validate_input(NUMERICAL_NP_ARRAY, int_to_float='True')
    assert str(exin.value) == type_error_itf

    #

    assert fuda._validate_input(
        MIXED_ARRAY,
        categorical_indices=['a', 'b'],
        ground_truth=np.array([1, 2, 3, 4, 5, 6]),
        int_to_float=False)


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
            super().__init__(  # pragma: nocover
                dataset,
                categorical_indices=categorical_indices)

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

        def __init__(self,
                     dataset,
                     categorical_indices=None,
                     int_to_float=True):
            """
            Dummy init method.
            """
            super().__init__(
                dataset,
                categorical_indices=categorical_indices,
                int_to_float=int_to_float)

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

        # Test type generalisation
        assert numerical_np_augmentor.sample_dtype == np.float64
        #
        dtype = mixed_augmentor.sample_dtype
        assert len(dtype) == 4
        for i in range(4):
            assert len(dtype[i]) == 2
        assert dtype[0][0] == 'a'
        assert dtype[0][1] == np.float64
        assert dtype[1][0] == 'b'
        assert dtype[1][1] == 'U1'
        assert dtype[2][0] == 'c'
        assert dtype[2][1] == np.float64
        assert dtype[3][0] == 'd'
        assert dtype[3][1] == 'U2'
        #
        # Test type generalisation
        numerical_struct_augmentor_i2f = self.BaseAugmentor(
            NUMERICAL_STRUCT_ARRAY, int_to_float=True)
        dtype = numerical_struct_augmentor_i2f.sample_dtype
        assert len(dtype) == 4
        for i, name in enumerate(['a', 'b', 'c', 'd']):
            assert len(dtype[i]) == 2
            assert dtype[i][0] == name
            assert dtype[i][1] == np.float64
        #
        numerical_struct_augmentor = self.BaseAugmentor(
            NUMERICAL_STRUCT_ARRAY, int_to_float=False)
        dtype = numerical_struct_augmentor.sample_dtype
        assert len(dtype) == 4
        for i in range(4):
            assert len(dtype[i]) == 2
        assert dtype[0][0] == 'a'
        assert dtype[0][1] == np.int64
        assert dtype[1][0] == 'b'
        assert dtype[1][1] == np.int64
        assert dtype[2][0] == 'c'
        assert dtype[2][1] == np.float64
        assert dtype[3][0] == 'd'
        assert dtype[3][1] == np.float64

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
    numerical_np_0_augmentor = fuda.NormalSampling(NUMERICAL_NP_ARRAY, [0])
    numerical_np_augmentor = fuda.NormalSampling(NUMERICAL_NP_ARRAY)
    numerical_struct_a_augmentor = fuda.NormalSampling(NUMERICAL_STRUCT_ARRAY,
                                                       ['a'])
    numerical_struct_augmentor = fuda.NormalSampling(NUMERICAL_STRUCT_ARRAY)
    numerical_struct_augmentor_f = fuda.NormalSampling(
        NUMERICAL_STRUCT_ARRAY, int_to_float=False)
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

    def test_sample(self):
        """
        Tests :func:`~fatf.utils.data.augmentation.NormalSampling.sample`.
        """
        fatf.setup_random_seed()

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

        #######################################################################

        # Sample without float cast
        samples = self.numerical_struct_augmentor_f.sample(samples_number=5)
        samples_answer = np.array(
            [(-1, 0, 0.172, 0.624),
             (1, 1, 0.343, 0.480),
             (0, 0, 0.649, 0.374),
             (0, 0, 0.256, 0.429),
             (0, 0, 0.457, 0.743)],
            dtype=NUMERICAL_STRUCT_ARRAY.dtype)  # yapf: disable
        for i in ['a', 'b', 'c', 'd']:
            assert np.allclose(samples[i], samples_answer[i], atol=1e-3)

        # Cast to float on in the tests to compare (this ouput was generated
        # with self.numerical_struct_augmentor)
        samples = self.numerical_struct_augmentor_f.sample(samples_number=5)
        samples_answer = np.array(
            [(1.250, 0.264, 0.381, 0.479),
             (-0.181, 1.600, 0.602, 0.345),
             (0.472, 0.609, -0.001, 1.026),
             (0.105, 1.091, 0.384, 0.263),
             (1.263, -0.007, 0.762, 0.603)],
            dtype=NUMERICAL_STRUCT_ARRAY.dtype)  # yapf: disable
        for i in ['a', 'b', 'c', 'd']:
            assert np.allclose(samples[i], samples_answer[i], atol=1e-3)


def test_validate_input_mixup():
    """
    Tests :func:`fatf.utils.data.augmentation._validate_input_mixup` function.
    """
    type_error_out = ('The beta_parameters parameter has to be a tuple with '
                      'two numbers or None to use the default parameters '
                      'value.')
    type_error_in = 'The {} beta parameter has to be a numerical type.'
    value_error_out = ('The beta_parameters parameter has to be a 2-tuple '
                       '(a pair) of numbers.')
    value_error_in = 'The {} beta parameter cannot be a negative number.'

    with pytest.raises(TypeError) as exin:
        fuda._validate_input_mixup('tuple')
    assert str(exin.value) == type_error_out

    with pytest.raises(ValueError) as exin:
        fuda._validate_input_mixup(('tuple', ))
    assert str(exin.value) == value_error_out

    with pytest.raises(TypeError) as exin:
        fuda._validate_input_mixup(('1', 2))
    assert str(exin.value) == type_error_in.format('first')
    with pytest.raises(TypeError) as exin:
        fuda._validate_input_mixup((1, '2'))
    assert str(exin.value) == type_error_in.format('second')

    with pytest.raises(ValueError) as exin:
        fuda._validate_input_mixup((0, 0))
    assert str(exin.value) == value_error_in.format('first')
    with pytest.raises(ValueError) as exin:
        fuda._validate_input_mixup((0.1, 0))
    assert str(exin.value) == value_error_in.format('second')

    assert fuda._validate_input_mixup(None)
    assert fuda._validate_input_mixup((.1, .1))


class TestMixup(object):
    """
    Tests :class:`fatf.utils.data.augmentation.Mixup` class.
    """
    numerical_labels = np.array([0, 1, 0, 0, 0, 1])
    categorical_labels = np.array(['b', 'a', 'a', 'a', 'a', 'b'])

    numerical_np_augmentor = fuda.Mixup(NUMERICAL_NP_ARRAY, int_to_float=False)
    numerical_struct_augmentor = fuda.Mixup(
        NUMERICAL_STRUCT_ARRAY,
        categorical_labels,
        beta_parameters=(3, 6),
        int_to_float=False)
    categorical_np_augmentor = fuda.Mixup(
        CATEGORICAL_NP_ARRAY, int_to_float=False)
    categorical_struct_augmentor = fuda.Mixup(
        CATEGORICAL_STRUCT_ARRAY,
        categorical_indices=['a', 'b', 'c'],
        int_to_float=False)
    mixed_augmentor = fuda.Mixup(
        MIXED_ARRAY, numerical_labels, int_to_float=False)
    mixed_augmentor_i2f = fuda.Mixup(MIXED_ARRAY, numerical_labels)

    def test_init(self):
        """
        Tests :class:`fatf.utils.data.augmentation.Mixup` class initialisation.
        """
        # Test class inheritance
        assert (self.numerical_np_augmentor.__class__.__bases__[0].__name__
                == 'Augmentation')  # yapf: disable

        # Check threshold
        assert self.mixed_augmentor.threshold == 0.5

        # Check beta parameters
        assert self.numerical_struct_augmentor.beta_parameters == (3, 6)
        assert self.mixed_augmentor.beta_parameters == (2, 5)

        # Check ground_truth_unique, ground_truth_frequencies,
        # indices_per_label and ground_truth_probabilities
        assert self.numerical_np_augmentor.ground_truth is None
        assert self.numerical_np_augmentor.ground_truth_unique is None
        assert self.numerical_np_augmentor.ground_truth_frequencies is None
        assert self.numerical_np_augmentor.ground_truth_probabilities is None
        assert self.numerical_np_augmentor.indices_per_label is None
        #
        assert np.array_equal(self.numerical_struct_augmentor.ground_truth,
                              self.categorical_labels)
        assert np.array_equal(
            self.numerical_struct_augmentor.ground_truth_unique,
            np.array(['a', 'b']))
        assert np.array_equal(
            self.numerical_struct_augmentor.ground_truth_frequencies,
            np.array([4 / 6, 2 / 6]))
        assert np.array_equal(
            self.numerical_struct_augmentor.ground_truth_probabilities,
            np.array([[0, 1], [1, 0], [1, 0], [1, 0], [1, 0], [0, 1]]))
        assert len(self.numerical_struct_augmentor.indices_per_label) == 2
        assert np.array_equal(
            self.numerical_struct_augmentor.indices_per_label[0],
            np.array([1, 2, 3, 4]))
        assert np.array_equal(
            self.numerical_struct_augmentor.indices_per_label[1],
            np.array([0, 5]))

    def test_sample_errors(self):
        """
        Tests for errors in :func:`~fatf.utils.data.augmentation.Mixup.sample`.
        """
        not_implemented_error = ('Sampling around the data mean is not yet '
                                 'implemented for the Mixup class.')
        type_error_probs = 'return_probabilities parameter has to be boolean.'
        type_error_replace = 'with_replacement parameter has to be boolean.'
        type_error_target = ('The data_row_target parameter should either be '
                             'None or a string/number indicating the target '
                             'class.')
        value_error_target = ('The value of the data_row_target parameter is '
                              'not present in the ground truth labels used to '
                              'initialise this class. The data row target '
                              'value is not recognised.')
        user_warning = ('This Mixup class has not been initialised with a '
                        'ground truth vector. The value of the '
                        'data_row_target parameter will be ignored, therefore '
                        'target values samples will not be returned.')

        with pytest.raises(TypeError) as exin:
            self.numerical_np_augmentor.sample(data_row_target=('4', '2'))
        assert str(exin.value) == type_error_target

        with pytest.raises(ValueError) as exin:
            self.numerical_struct_augmentor.sample(data_row_target='1')
        assert str(exin.value) == value_error_target

        with pytest.warns(UserWarning) as warning:
            with pytest.raises(NotImplementedError) as exin:
                self.numerical_np_augmentor.sample(data_row_target='1')
            assert str(exin.value) == not_implemented_error
        assert len(warning) == 1
        assert str(warning[0].message) == user_warning

        with pytest.raises(TypeError) as exin:
            self.numerical_np_augmentor.sample(return_probabilities=1)
        assert str(exin.value) == type_error_probs

        with pytest.raises(TypeError) as exin:
            self.numerical_np_augmentor.sample(with_replacement=1)
        assert str(exin.value) == type_error_replace

        with pytest.raises(NotImplementedError) as exin:
            self.numerical_np_augmentor.sample()
        assert str(exin.value) == not_implemented_error

    def test_sample(self):
        """
        Tests :func:`~fatf.utils.data.augmentation.Mixup.sample` method.
        """
        user_warning_gt = (
            'This Mixup class has not been initialised with a ground truth '
            'vector. The value of the data_row_target parameter will be '
            'ignored, therefore target values samples will not be returned.')
        user_warning_strat = (
            'Since the ground truth vector was not provided while '
            'initialising the Mixup class it is not possible to get a '
            'stratified sample of data points. Instead, Mixup will choose '
            'data points at random, which is equivalent to assuming that the '
            'class distribution is balanced.')
        fatf.setup_random_seed()

        # Mixed array with ground truth and probabilities
        samples = self.mixed_augmentor_i2f.sample(
            MIXED_ARRAY[0], 0, 5, return_probabilities=True)
        assert len(samples) == 2
        answer_sample = np.array(
            [(0.000, 'a', 0.332, 'a'),
             (0.000, 'a', 0.080, 'a'),
             (0.780, 'a', 0.587, 'a'),
             (0.992, 'a', 0.725, 'a'),
             (0.734, 'a', 0.073, 'a')],
            dtype=[('a', '<f4'), ('b', '<U1'),
                   ('c', '<f4'), ('d', '<U2')])  # yapf: disable
        answer_sample_gt = np.array([[1, 0], [1, 0], [1, 0], [1, 0],
                                     [0.266, 0.734]])
        assert np.allclose(samples[1], answer_sample_gt, atol=1e-3)
        for i in ['a', 'c']:
            assert np.allclose(samples[0][i], answer_sample[i], atol=1e-3)
        for i in ['b', 'd']:
            assert np.array_equal(samples[0][i], answer_sample[i])

        # Mixed array with ground truth and probabilities
        samples = self.mixed_augmentor.sample(
            MIXED_ARRAY[0], 1, 5, return_probabilities=True)
        assert len(samples) == 2
        answer_sample = np.array(
            [(0, 'a', 0.829, 'a'),
             (0, 'a', 0.601, 'a'),
             (0, 'a', 0.255, 'a'),
             (0, 'a', 0.377, 'a'),
             (0, 'a', 0.071, 'a')],
            dtype=[('a', '<i4'), ('b', '<U1'),
                   ('c', '<f4'), ('d', '<U2')])  # yapf: disable
        answer_sample_gt = np.array([[0.823, 0.177], [0.802, 0.198],
                                     [0.624, 0.376], [0.457, 0.543], [0, 1]])
        assert np.allclose(samples[1], answer_sample_gt, atol=1e-3)
        for i in ['a', 'c']:
            assert np.allclose(samples[0][i], answer_sample[i], atol=1e-3)
        for i in ['b', 'd']:
            assert np.array_equal(samples[0][i], answer_sample[i])

        # Numpy array without ground truth -- categorical
        with pytest.warns(UserWarning) as warning:
            samples = self.categorical_np_augmentor.sample(
                CATEGORICAL_NP_ARRAY[0], samples_number=5)
        assert len(warning) == 1
        assert str(warning[0].message) == user_warning_strat
        #
        answer_sample = np.array([['a', 'b', 'c'], ['a', 'b', 'c'],
                                  ['a', 'b', 'c'], ['a', 'b', 'c'],
                                  ['a', 'b', 'c']])
        assert np.array_equal(samples, answer_sample)

        # Numpy array without ground truth -- numerical -- test for warning
        with pytest.warns(UserWarning) as warning:
            samples = self.numerical_np_augmentor.sample(
                NUMERICAL_NP_ARRAY[0], data_row_target=1, samples_number=5)
        assert len(warning) == 2
        assert str(warning[0].message) == user_warning_gt
        assert str(warning[1].message) == user_warning_strat
        #
        answer_sample = np.array([[0.792, 0.000, 0.040, 0.373],
                                  [0.000, 0.000, 0.080, 0.690],
                                  [1.220, 0.610, 0.476, 0.562],
                                  [0.000, 0.000, 0.080, 0.690],
                                  [1.389, 0.694, 0.531, 0.544]])
        assert np.allclose(samples, answer_sample, atol=1e-3)

        # Structured array with ground truth -- numerical -- no probabilities
        samples = self.numerical_struct_augmentor.sample(
            NUMERICAL_STRUCT_ARRAY[0], samples_number=5, data_row_target='b')
        assert len(samples) == 2
        answer_sample = np.array(
            [(0, 0, 0.039, 0.358),
             (1, 0, 0.544, 0.540),
             (1, 0, 0.419, 0.580),
             (0, 0, 0.080, 0.690),
             (0, 0, 0.080, 0.690)],
            dtype=[('a', '<i4'), ('b', '<i4'),
                   ('c', '<f4'), ('d', '<f4')])  # yapf: disable
        answer_sample_gt = np.array(['a', 'a', 'a', 'b', 'b'])
        assert np.array_equal(samples[1], answer_sample_gt)
        for index in ['a', 'b', 'c', 'd']:
            assert np.allclose(
                samples[0][index], answer_sample[index], atol=1e-3)


def get_truncated_mean_std(minimum, maximum, original_mean, original_std):
    """
    Computes the theoretical mean and standard deviation of a truncated
    normal distribution from its initialisation parameters: the original
    normal mean and standard deviation, and the minimum and maximum within
    which values are truncated.

    Equations for calculating these -- implemented by this function -- can
    be found here_.

    .. _here: https://en.wikipedia.org/wiki/Truncated_normal_distribution
    """

    def cdf(epsilon):
        return (1 / 2) * (1 + scipy.special.erf(epsilon / np.sqrt(2)))

    def norm(episilon):
        return 1 / np.sqrt(2 * np.pi) * np.exp(-1 / 2 * episilon**2)

    alpha = (minimum - original_mean) / original_std
    beta = (maximum - original_mean) / original_std
    z_phi = cdf(beta) - cdf(alpha)

    n_ab = norm(alpha) - norm(beta)

    computed_mean = original_mean + (n_ab / z_phi) * original_std
    computed_var = (original_std**2
                    * (1 + (alpha * norm(alpha) - beta * norm(beta)) / z_phi
                       - (n_ab / z_phi)**2
                       )
                    )  # yapf: disable
    computed_std = np.sqrt(computed_var)

    return computed_mean, computed_std


class TestTruncatedNormalSampling(object):
    """
    Tests :class:`fatf.utils.data.augmentation.TruncatedNormalSampling` class.
    """
    numerical_np_augmentor = fuda.TruncatedNormalSampling(NUMERICAL_NP_ARRAY)
    numerical_np_0_augmentor = fuda.TruncatedNormalSampling(
        NUMERICAL_NP_ARRAY, [0])

    numerical_struct_augmentor = fuda.TruncatedNormalSampling(
        NUMERICAL_STRUCT_ARRAY)
    numerical_struct_a_augmentor = fuda.TruncatedNormalSampling(
        NUMERICAL_STRUCT_ARRAY, ['a'])
    numerical_struct_augmentor_f = fuda.TruncatedNormalSampling(
        NUMERICAL_STRUCT_ARRAY, int_to_float=False)

    categorical_np_augmentor = fuda.TruncatedNormalSampling(
        CATEGORICAL_NP_ARRAY)
    categorical_np_012_augmentor = fuda.TruncatedNormalSampling(
        CATEGORICAL_NP_ARRAY, [0, 1, 2])

    categorical_struct_abc_augmentor = fuda.TruncatedNormalSampling(
        CATEGORICAL_STRUCT_ARRAY, ['a', 'b', 'c'])

    mixed_augmentor = fuda.TruncatedNormalSampling(MIXED_ARRAY, ['b', 'd'])

    def test_init(self):
        """
        Tests ``TruncatedNormalSampling`` class initialisation.
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

        # Test attributes unique to TruncatedNormalSampling
        csv = self.numerical_np_0_augmentor.categorical_sampling_values
        nsv = self.numerical_np_0_augmentor.numerical_sampling_values
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
        assert len(nsv[1]) == 4 and len(nsv[2]) == 4 and len(nsv[3]) == 4
        assert nsv[1][0] == pytest.approx(.5, abs=1e-3)
        assert nsv[1][1] == pytest.approx(.5, abs=1e-3)
        assert nsv[1][2] == pytest.approx(0., abs=1e-3)
        assert nsv[1][3] == pytest.approx(1., abs=1e-3)

        assert nsv[2][0] == pytest.approx(.377, abs=1e-3)
        assert nsv[2][1] == pytest.approx(.366, abs=1e-3)
        assert nsv[2][2] == pytest.approx(0.03, abs=1e-3)
        assert nsv[2][3] == pytest.approx(0.99, abs=1e-3)

        assert nsv[3][0] == pytest.approx(.563, abs=1e-3)
        assert nsv[3][1] == pytest.approx(.257, abs=1e-3)
        assert nsv[3][2] == pytest.approx(0.21, abs=1e-3)
        assert nsv[3][3] == pytest.approx(0.89, abs=1e-3)

    def test_sample(self):
        """
        Tests ``sample`` for the ``TruncatedNormalSampling`` augmenter.
        """
        fatf.setup_random_seed()

        # yapf: disable
        numerical_np_truncated_results = np.array([
            [0.361, 0.396, 0.0593, 0.731],
            [1.423, 0.094, 0.595, 0.258],
            [0.816, 0.094, 0.356, 0.871]])
        numerical_struct_truncated_results = np.array(
            [(1.014, 0.111, 0.254, 0.408),
             (0.199, 0.186, 0.178, 0.517),
             (0.170, 0.338, 0.364, 0.560)],
            dtype=[('a', 'f'), ('b', 'f'), ('c', 'f'), ('d', 'f')])
        numerical_np_truncated_cat_results = np.array([
            [1., 0.531, 0.269, 0.587],
            [1., 0.154, 0.136, 0.751],
            [1., 0.696, 0.594, 0.653]])
        numerical_struct_truncated_cat_results = np.array(
            [(0, 0.243, 0.048, 0.697),
             (1, 0.066, 0.591, 0.842),
             (1, 0.728, 0.214, 0.418)],
            dtype=[('a', 'i'), ('b', 'f'), ('c', 'f'), ('d', 'f')])
        categorical_np_results = np.array([
            ['a', 'f', 'c'],
            ['a', 'f', 'c'],
            ['b', 'f', 'g']])
        categorical_struct_results = np.array(
            [('a', 'b', 'g'),
             ('a', 'f', 'c'),
             ('a', 'f', 'c')],
            dtype=[('a', 'U1'), ('b', 'U1'), ('c', 'U1')])
        mixed_results = np.array(
            [(0.668, 'a', 0.522, 'bb'),
             (0.195, 'c', 0.075, 'a'),
             (0.266, 'f', 0.586, 'b')],
            dtype=[('a', '<f8'), ('b', 'U1'), ('c', '<f8'), ('d', 'U2')])
        # yapf: enable

        # Calculate what the mean and std of truncated normals should be
        min_ = NUMERICAL_NP_ARRAY.min(axis=0)
        max_ = NUMERICAL_NP_ARRAY.max(axis=0)
        mean = NUMERICAL_NP_ARRAY.mean(axis=0)
        std = NUMERICAL_NP_ARRAY.std(axis=0)

        nt_results_mean, nt_results_std = get_truncated_mean_std(
            min_, max_, NUMERICAL_NP_ARRAY[0], std)
        nt_results_data_mean, nt_results_data_std = get_truncated_mean_std(
            min_, max_, mean, std)
        mixed_numerical_values = fuat.structured_to_unstructured(
            MIXED_ARRAY[['a', 'c']])

        min_ = mixed_numerical_values.min(axis=0)
        max_ = mixed_numerical_values.max(axis=0)
        std = mixed_numerical_values.std(axis=0)
        mean = mixed_numerical_values.mean(axis=0)

        nt_mixed_results_mean, nt_mixed_results_std = get_truncated_mean_std(
            min_, max_, mixed_numerical_values[0], std)
        nt_mixed_results_data = get_truncated_mean_std(min_, max_, mean, std)
        nt_mixed_results_data_mean = nt_mixed_results_data[0]
        nt_mixed_results_data_std = nt_mixed_results_data[1]

        # Pure numerical sampling of a data point
        # ...numpy array results
        samples = self.numerical_np_augmentor.sample(
            NUMERICAL_NP_ARRAY[0, :], samples_number=3)
        assert np.allclose(samples, numerical_np_truncated_results, atol=1e-3)

        # ...structured array results
        samples_struct = self.numerical_struct_augmentor.sample(
            NUMERICAL_STRUCT_ARRAY[0], samples_number=3)
        for i in samples_struct.dtype.names:
            assert np.allclose(
                samples_struct[i],
                numerical_struct_truncated_results[i],
                atol=1e-3)

        # ...numpy array results mean
        samples = self.numerical_np_augmentor.sample(
            NUMERICAL_NP_ARRAY[0, :], samples_number=1000)
        assert np.allclose(samples.mean(axis=0), nt_results_mean, atol=1e-1)
        assert np.allclose(samples.std(axis=0), nt_results_std, atol=1e-1)

        # ...structured array results mean
        samples_struct = self.numerical_struct_augmentor.sample(
            NUMERICAL_STRUCT_ARRAY[0], samples_number=1000)
        for i, name in enumerate(samples_struct.dtype.names):
            assert np.allclose(
                np.mean(samples_struct[name]), nt_results_mean[i], atol=1e-1)
            assert np.allclose(
                np.std(samples_struct[name]), nt_results_std[i], atol=1e-1)

        # Pure numerical sampling from the mean of the data
        # ...numpy array mean
        samples = self.numerical_np_augmentor.sample(samples_number=1000)
        assert np.allclose(
            samples.mean(axis=0), nt_results_data_mean, atol=1e-1)
        assert np.allclose(samples.std(axis=0), nt_results_data_std, atol=1e-1)

        # ...structured array mean
        samples_struct = self.numerical_struct_augmentor.sample(
            samples_number=1000)
        for i, name in enumerate(samples_struct.dtype.names):
            assert np.allclose(
                np.mean(samples_struct[name]),
                nt_results_data_mean[i],
                atol=1e-1)
            assert np.allclose(
                np.std(samples_struct[name]),
                nt_results_data_std[i],
                atol=1e-1)

        #######################################################################

        # Numerical sampling with one categorical index defined
        # ...numpy array results
        samples = self.numerical_np_0_augmentor.sample(
            NUMERICAL_NP_ARRAY[0, :], samples_number=3)
        assert np.allclose(
            samples, numerical_np_truncated_cat_results, atol=1e-3)

        # ...structured array results
        samples_struct = self.numerical_struct_a_augmentor.sample(
            NUMERICAL_STRUCT_ARRAY[0], samples_number=3)
        for i in samples_struct.dtype.names:
            assert np.allclose(
                samples_struct[i],
                numerical_struct_truncated_cat_results[i],
                atol=1e-3)

        # ...numpy array results mean
        # ......numerical
        samples = self.numerical_np_0_augmentor.sample(
            NUMERICAL_NP_ARRAY[0, :], samples_number=100)
        assert np.allclose(
            samples.mean(axis=0)[1:], nt_results_mean[1:], atol=1e-1)
        assert np.allclose(
            samples.std(axis=0)[1:], nt_results_std[1:], atol=1e-1)
        # ......categorical
        val, freq = np.unique(samples[:, 0], return_counts=True)
        freq = freq / freq.sum()
        assert np.array_equal(val, NUMERICAL_NP_0_CAT_VAL)
        assert np.allclose(freq, NUMERICAL_NP_0_CAT_FREQ, atol=1e-1)

        # ...structured array results mean
        samples_struct = self.numerical_struct_a_augmentor.sample(
            NUMERICAL_STRUCT_ARRAY[0], samples_number=100)
        # ......numerical
        for i, name in enumerate(samples_struct.dtype.names[1:]):
            assert np.allclose(
                np.mean(samples_struct[name]),
                nt_results_mean[1:][i],
                atol=1e-1)
            assert np.allclose(
                np.std(samples_struct[name]), nt_results_std[1:][i], atol=1e-1)
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
            samples.mean(axis=0)[1:], nt_results_data_mean[1:], atol=1e-1)
        assert np.allclose(
            samples.std(axis=0)[1:], nt_results_data_std[1:], atol=1e-1)
        # ......categorical
        val, freq = np.unique(samples[:, 0], return_counts=True)
        freq = freq / freq.sum()
        assert np.array_equal(val, NUMERICAL_NP_0_CAT_VAL)
        assert np.allclose(freq, NUMERICAL_NP_0_CAT_FREQ, atol=1e-1)

        # ...structured array mean
        samples_struct = self.numerical_struct_a_augmentor.sample(
            samples_number=1000)
        # ......numerical
        for i, name in enumerate(samples_struct.dtype.names[1:]):
            assert np.allclose(
                np.mean(samples_struct[name]),
                nt_results_data_mean[1:][i],
                atol=1e-1)
            assert np.allclose(
                np.std(samples_struct[name]),
                nt_results_data_std[1:][i],
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
        assert np.array_equal(samples, categorical_np_results)

        # ...structured array
        samples_struct = self.categorical_struct_abc_augmentor.sample(
            CATEGORICAL_STRUCT_ARRAY[0], samples_number=3)
        assert np.array_equal(samples_struct, categorical_struct_results)

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
        for i, index in enumerate(range(CATEGORICAL_NP_ARRAY.shape[1])):
            val, freq = np.unique(samples[:, index], return_counts=True)
            freq = freq / freq.sum()
            assert np.array_equal(val, vals[i])
            assert np.allclose(freq, proportions[i], atol=1e-1)

        # ...structured array proportions and values
        samples_struct = self.categorical_struct_abc_augmentor.sample(
            CATEGORICAL_STRUCT_ARRAY[0], samples_number=100)
        #
        proportions = [
            np.array([0.74, 0.26]),
            np.array([0.38, 0.12, 0.50]),
            np.array([0.63, 0.37])
        ]
        for i, index in enumerate(CATEGORICAL_STRUCT_ARRAY.dtype.names):
            val, freq = np.unique(samples_struct[index], return_counts=True)
            freq = freq / freq.sum()
            assert np.array_equal(val, vals[i])
            assert np.allclose(freq, proportions[i], atol=1e-1)

        # No need to check for mean of the datasets since categorical features
        # are sampled from the distribution of the entire dataset and not
        # centered on the data_row.

        #######################################################################
        #######################################################################

        # Mixed array with categorical indices auto-discovered
        vals = [['a', 'c', 'f'], ['a', 'aa', 'b', 'bb']]
        proportions = [
            np.array([0.33, 0.33, 0.33]),
            np.array([0.33, 0.16, 0.16, 0.33])
        ]
        mixed_cat, mixed_num = ['b', 'd'], ['a', 'c']
        # Instance
        samples = self.mixed_augmentor.sample(MIXED_ARRAY[0], samples_number=3)
        # ...categorical
        assert np.array_equal(samples[mixed_cat], mixed_results[mixed_cat])
        # ...numerical
        for i in mixed_num:
            assert np.allclose(samples[i], mixed_results[i], atol=1e-3)

        # Instance mean
        samples = self.mixed_augmentor.sample(
            MIXED_ARRAY[0], samples_number=1000)
        # ...numerical
        for i, name in enumerate(mixed_num):
            assert np.allclose(
                np.mean(samples[name]), nt_mixed_results_mean[i], atol=1e-1)
            assert np.allclose(
                np.std(samples[name]), nt_mixed_results_std[i], atol=1e-1)
        # ...categorical
        for i, index in enumerate(mixed_cat):
            val, freq = np.unique(samples[index], return_counts=True)
            freq = freq / freq.sum()
            assert np.array_equal(val, vals[i])
            assert np.allclose(freq, proportions[i], atol=1e-1)

        # Dataset mean
        samples = self.mixed_augmentor.sample(samples_number=1000)
        # ...numerical
        for i, name in enumerate(mixed_num):
            assert np.allclose(
                np.mean(samples[name]),
                nt_mixed_results_data_mean[i],
                atol=1e-1)
            assert np.allclose(
                np.std(samples[name]), nt_mixed_results_data_std[i], atol=1e-1)
        # ...categorical
        for i, index in enumerate(mixed_cat):
            val, freq = np.unique(samples[index], return_counts=True)
            freq = freq / freq.sum()
            assert np.array_equal(val, vals[i])
            assert np.allclose(freq, proportions[i], atol=1e-1)

        #######################################################################

        # Sample without float cast
        samples = self.numerical_struct_augmentor_f.sample(samples_number=5)
        samples_answer = np.array(
            [(0, 0, 0.345, 0.442),
             (1, 0, 0.311, 0.338),
             (1, 0, 0.040, 0.553),
             (0, 0, 0.886, 0.822),
             (0, 0, 0.164, 0.315)],
            dtype=NUMERICAL_STRUCT_ARRAY.dtype)  # yapf: disable
        for i in NUMERICAL_STRUCT_ARRAY.dtype.names:
            assert np.allclose(samples[i], samples_answer[i], atol=1e-3)

        # Compare with the same augmentation but with int_to_float=False and
        # casted to integers afterwards (generated with
        # self.numerical_struct_augmentor).
        samples = self.numerical_struct_augmentor_f.sample(samples_number=5)
        samples_answer = np.array([(0.718, 0.476, 0.449, 0.615),
                                   (0.047, 0.883, 0.205, 0.329),
                                   (1.255, 0.422, 0.302, 0.627),
                                   (1.024, 0.512, 0.122, 0.790),
                                   (1.123, 0.670, 0.386, 0.471)],
                                  dtype=NUMERICAL_STRUCT_ARRAY.dtype)
        for i in NUMERICAL_STRUCT_ARRAY.dtype.names:
            assert np.allclose(samples[i], samples_answer[i], atol=1e-3)


def test_validate_input_normalclassdiscovery():
    """
    Tests the ``_validate_input_normalclassdiscovery`` function.

    Tests the
    :func:`fatf.utils.data.augmentation._validate_input_normalclassdiscovery`
    function.
    """
    predictive_function_model = ('The predictive function must take exactly '
                                 '*one* required parameter: a data array to '
                                 'be predicted.')
    predictive_function_type = ('The predictive_function should be a Python '
                                'callable, e.g., a Python function.')
    classes_number_type = ('The classes_number parameter is neither None nor '
                           'an integer.')
    classes_number_value = ('The classes_number parameter has to be an '
                            'integer larger than 1 (at least a binary '
                            'classification problem).')
    class_proportion_type = ('The class_proportion_threshold parameter is not '
                             'a number.')
    class_proportion_value = ('The class_proportion_threshold parameter must '
                              'be a number between 0 and 1 (not inclusive).')
    standard_deviation_init_type = ('The standard_deviation_init parameter is '
                                    'not a number.')
    standard_deviation_init_value = ('The standard_deviation_init parameter '
                                     'must be a positive number (greater than '
                                     '0).')
    standard_deviation_increment_type = ('The standard_deviation_increment '
                                         'parameter is not a number.')
    standard_deviation_increment_value = ('The standard_deviation_increment '
                                          'parameter must be a positive '
                                          'number (greater than 0).')

    def invalid_predict_proba(self, x, y):
        pass  # pragma: no cover

    model = fum.KNN(k=3)

    with pytest.raises(TypeError) as exin:
        fuda._validate_input_normalclassdiscovery(None, None, None, None, None)
    assert str(exin.value) == predictive_function_type
    with pytest.raises(IncompatibleModelError) as exin:
        fuda._validate_input_normalclassdiscovery(invalid_predict_proba, None,
                                                  None, None, None)
    assert str(exin.value) == predictive_function_model

    with pytest.raises(TypeError) as exin:
        fuda._validate_input_normalclassdiscovery(model.predict, '1', None,
                                                  None, None)
    assert str(exin.value) == classes_number_type
    with pytest.raises(ValueError) as exin:
        fuda._validate_input_normalclassdiscovery(model.predict, 1, None, None,
                                                  None)
    assert str(exin.value) == classes_number_value

    with pytest.raises(TypeError) as exin:
        fuda._validate_input_normalclassdiscovery(model.predict, 2, None, None,
                                                  None)
    assert str(exin.value) == class_proportion_type
    with pytest.raises(ValueError) as exin:
        fuda._validate_input_normalclassdiscovery(model.predict, None, 0, None,
                                                  None)
    assert str(exin.value) == class_proportion_value
    with pytest.raises(ValueError) as exin:
        fuda._validate_input_normalclassdiscovery(model.predict, None, 1.0,
                                                  None, None)
    assert str(exin.value) == class_proportion_value

    with pytest.raises(TypeError) as exin:
        fuda._validate_input_normalclassdiscovery(model.predict_proba, 3, 0.9,
                                                  None, None)
    assert str(exin.value) == standard_deviation_init_type
    with pytest.raises(ValueError) as exin:
        fuda._validate_input_normalclassdiscovery(model.predict, None, 0.1, 0,
                                                  None)
    assert str(exin.value) == standard_deviation_init_value
    with pytest.raises(ValueError) as exin:
        fuda._validate_input_normalclassdiscovery(model.predict, None, 0.1, -5,
                                                  None)
    assert str(exin.value) == standard_deviation_init_value

    with pytest.raises(TypeError) as exin:
        fuda._validate_input_normalclassdiscovery(model.predict_proba, None,
                                                  0.5, 6, None)
    assert str(exin.value) == standard_deviation_increment_type
    with pytest.raises(ValueError) as exin:
        fuda._validate_input_normalclassdiscovery(model.predict_proba, None,
                                                  0.5, 6, 0)
    assert str(exin.value) == standard_deviation_increment_value
    with pytest.raises(ValueError) as exin:
        fuda._validate_input_normalclassdiscovery(model.predict_proba, None,
                                                  0.5, 6, -0.5)
    assert str(exin.value) == standard_deviation_increment_value


class TestNormalClassDiscovery(object):
    """
    Tests :class:`fatf.utils.data.augmentation.NormalClassDiscovery` class.
    """
    numerical_labels = np.array([0, 1, 0, 1, 1, 0])

    numerical_classifier = fum.KNN(k=3)
    numerical_classifier.fit(NUMERICAL_NP_ARRAY, numerical_labels)
    #
    numerical_np_augmentor = fuda.NormalClassDiscovery(
        NUMERICAL_NP_ARRAY, numerical_classifier.predict_proba)
    numerical_np_0_augmentor = fuda.NormalClassDiscovery(
        NUMERICAL_NP_ARRAY, numerical_classifier.predict, [0])

    numerical_struct_classifier = fum.KNN(k=3)
    numerical_struct_classifier.fit(NUMERICAL_STRUCT_ARRAY, numerical_labels)
    #
    numerical_struct_augmentor = fuda.NormalClassDiscovery(
        NUMERICAL_STRUCT_ARRAY,
        numerical_struct_classifier.predict_proba,
        standard_deviation_init=0.5,
        standard_deviation_increment=0.2,
        class_proportion_threshold=0.1)
    numerical_struct_augmentor_f = fuda.NormalClassDiscovery(
        NUMERICAL_STRUCT_ARRAY,
        numerical_struct_classifier.predict,
        int_to_float=False,
        classes_number=2)
    numerical_struct_a_augmentor = fuda.NormalClassDiscovery(
        NUMERICAL_STRUCT_ARRAY,
        numerical_struct_classifier.predict, ['a'],
        classes_number=2)

    categorical_classifier = fum.KNN(k=3)
    categorical_classifier.fit(CATEGORICAL_NP_ARRAY, numerical_labels)
    #
    categorical_np_augmentor = fuda.NormalClassDiscovery(
        CATEGORICAL_NP_ARRAY, categorical_classifier.predict, classes_number=2)
    categorical_np_012_augmentor = fuda.NormalClassDiscovery(
        CATEGORICAL_NP_ARRAY,
        categorical_classifier.predict, [0, 1, 2],
        classes_number=2)

    categorical_struct_classifier = fum.KNN(k=3)
    categorical_struct_classifier.fit(CATEGORICAL_STRUCT_ARRAY,
                                      numerical_labels)
    #
    categorical_struct_abc_augmentor = fuda.NormalClassDiscovery(
        CATEGORICAL_STRUCT_ARRAY,
        categorical_struct_classifier.predict, ['a', 'b', 'c'],
        classes_number=2)

    mixed_classifier = fum.KNN(k=3)
    mixed_classifier.fit(MIXED_ARRAY, numerical_labels)
    #
    mixed_augmentor = fuda.NormalClassDiscovery(
        MIXED_ARRAY, mixed_classifier.predict, ['b', 'd'], classes_number=2)

    def test_init(self, caplog):
        """
        Tests ``NormalClassDiscovery`` class initialisation.
        """
        runtime_error_class_n = ('For the specified (classification) '
                                 'predictive function, classifying the input '
                                 'dataset provided only one target class. To '
                                 'use this augmenter please initialise it '
                                 'with the classes_number parameter.')
        logger_info = ('The number of classes was not specified by the user. '
                       'Based on *classification* of the input dataset {} '
                       'classes were found.')
        runtime_error_prop = ('The lower bound on the proportion of each '
                              'class must be smaller than 1/(the number of '
                              'classes) for this sampling implementation. '
                              '(Please see the documentation of the '
                              'NormalClassDiscovery augmenter for more '
                              'information.')

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

        # Test for non-probabilistic
        # ...successful class number inference logging
        assert len(caplog.records) == 0
        _ = fuda.NormalClassDiscovery(NUMERICAL_NP_ARRAY,
                                      self.numerical_classifier.predict)
        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == 'INFO'
        assert caplog.records[0].getMessage() == logger_info.format(2)
        # ...failed class number inference
        with pytest.raises(RuntimeError) as exin:
            fuda.NormalClassDiscovery(
                np.array([[2, 1, 0.73, 0.48], [1, 0, 0.36, 0.89]]),
                self.numerical_classifier.predict)
        assert str(exin.value) == runtime_error_class_n

        # Impossible class proportion threshold
        with pytest.raises(RuntimeError) as exin:
            fuda.NormalClassDiscovery(
                NUMERICAL_NP_ARRAY,
                self.numerical_classifier.predict,
                class_proportion_threshold=0.5)
        assert str(exin.value) == runtime_error_prop

        # Test attributes unique to NormalClassDiscovery...
        # ...numpy probabilistic
        assert (self.numerical_np_augmentor.predictive_function
                == self.numerical_classifier.predict_proba)  # yapf: disable
        assert self.numerical_np_augmentor.is_probabilistic is True
        assert self.numerical_np_augmentor.classes_number == 2
        assert self.numerical_np_augmentor.standard_deviation_init == 1
        assert self.numerical_np_augmentor.standard_deviation_increment == 0.1
        assert self.numerical_np_augmentor.class_proportion_threshold == 0.05
        assert not self.numerical_np_augmentor.categorical_sampling_values

        # ...numpy classifier
        assert (self.numerical_np_0_augmentor.predictive_function
                == self.numerical_classifier.predict)  # yapf: disable
        assert self.numerical_np_0_augmentor.is_probabilistic is False
        assert self.numerical_np_0_augmentor.classes_number == 2
        assert self.numerical_np_0_augmentor.standard_deviation_init == 1
        assert self.numerical_np_0_augmentor.standard_deviation_increment == .1
        assert self.numerical_np_0_augmentor.class_proportion_threshold == .05
        #
        csv = self.numerical_np_0_augmentor.categorical_sampling_values
        assert len(csv) == 1
        assert 0 in csv
        assert len(csv[0]) == 2
        assert np.array_equal(csv[0][0], np.array([0, 1, 2]))
        assert np.allclose(
            csv[0][1], np.array([3 / 6, 2 / 6, 1 / 6]), atol=1e-3)

        # ...structured probabilistic
        assert (
            self.numerical_struct_augmentor.predictive_function
            == self.numerical_struct_classifier.predict_proba
        )  # yapf: disable
        assert self.numerical_struct_augmentor.is_probabilistic is True
        assert self.numerical_struct_augmentor.classes_number == 2
        assert (self.numerical_struct_augmentor.standard_deviation_init
                == 0.5)  # yapf: disable
        assert (self.numerical_struct_augmentor.standard_deviation_increment
                == 0.2)  # yapf: disable
        assert (self.numerical_struct_augmentor.class_proportion_threshold
                == 0.1)  # yapf: disable
        assert not self.numerical_struct_augmentor.categorical_sampling_values

        # ...structured classifier
        assert (self.categorical_struct_abc_augmentor.predictive_function
                == self.categorical_struct_classifier.predict)  # yapf: disable
        assert self.categorical_struct_abc_augmentor.is_probabilistic is False
        assert self.categorical_struct_abc_augmentor.classes_number == 2
        assert (
            self.categorical_struct_abc_augmentor.standard_deviation_init == 1)
        assert (
            self.categorical_struct_abc_augmentor.standard_deviation_increment
            == 0.1)  # yapf: disable
        assert (
            self.categorical_struct_abc_augmentor.class_proportion_threshold
            == 0.05)  # yapf: disable
        csv = self.categorical_struct_abc_augmentor.categorical_sampling_values
        assert len(csv) == 3
        assert 'a' in csv and 'b' in csv and 'c' in csv
        #
        assert len(csv['a']) == 2
        assert np.array_equal(csv['a'][0], np.array(['a', 'b']))
        assert np.allclose(csv['a'][1], np.array([4 / 6, 2 / 6]), atol=1e-3)
        #
        assert len(csv['b']) == 2
        assert np.array_equal(csv['b'][0], np.array(['b', 'c', 'f']))
        assert np.allclose(
            csv['b'][1], np.array([2 / 6, 1 / 6, 3 / 6]), atol=1e-3)
        #
        assert len(csv['c']) == 2
        assert np.array_equal(csv['c'][0], np.array(['c', 'g']))
        assert np.allclose(csv['c'][1], np.array([4 / 6, 2 / 6]), atol=1e-3)

    def test_sample(self):
        """
        Tests :func:`fatf.utils.data.augmentation.NormalClassDiscovery.sample`.
        """
        fatf.setup_random_seed()
        max_iter_type = 'The max_iter parameter is not a positive integer.'
        max_iter_value = 'The max_iter parameter must be a positive number.'
        runtime_msg = ('The maximum number of iterations was reached '
                       'without sampling enough data points for each '
                       'class. Please try increasing the max_iter '
                       'parameter or decreasing the '
                       'class_proportion_threshold parameter. '
                       'Increasing the standard_deviation_init and '
                       'standard_deviation_increment parameters may also '
                       'help.')

        with pytest.raises(TypeError) as exin:
            self.numerical_np_0_augmentor.sample(
                NUMERICAL_NP_ARRAY[0], max_iter='a')
        assert str(exin.value) == max_iter_type
        with pytest.raises(ValueError) as exin:
            self.numerical_np_0_augmentor.sample(
                NUMERICAL_NP_ARRAY[0], max_iter=-1)
        assert str(exin.value) == max_iter_value

        # yapf: disable
        numerical_samples = np.array([
            [0.088, 0.024, -0.505, 0.934],
            [-0.175, -0.471, -0.049, -0.155],
            [-2.289, -1.651, -0.110, -2.343],
            [0.8346353, -1.189, -0.435, -1.269]])

        numerical_0_samples = np.array([
            [1, -0.196, 1.378, 1.608],
            [0, -0.451, -0.908, 1.016],
            [0, 1.588, 0.976, 1.275],
            [2, 0.033, 2.253, 1.130]])

        numerical_struct_samples = np.array(
            [(0.637, -1.328, -0.118, 0.916),
             (-0.146, 0.173, -0.065, 0.607),
             (1.396, -1.405, 1.552, 1.498),
             (-2.150, -2.201, 2.599, 2.582)],
            dtype=[('a', 'f'), ('b', 'f'), ('c', 'f'), ('d', 'f')])

        numerical_struct_0_samples = np.array(
            [(0, -1.461, -0.580, -2.496),
             (1, -0.728, 1.033, 1.372),
             (1, -1.509, -0.972, -0.833),
             (0, -0.943, -0.142, -3.236)],
            dtype=[('a', 'i'), ('b', 'f'), ('c', 'f'), ('d', 'f')])

        categorical_samples = np.array([
            ['a', 'b', 'g'],
            ['a', 'f', 'g'],
            ['b', 'b', 'g'],
            ['a', 'f', 'c']])

        categorical_012_samples = np.array([
            ['a', 'c', 'c'],
            ['a', 'f', 'g'],
            ['a', 'c', 'c'],
            ['a', 'f', 'g']])

        categorical_struct_samples = np.array(
            [('a', 'f', 'c'),
             ('b', 'b', 'c'),
             ('a', 'b', 'g'),
             ('a', 'f', 'g')],
            dtype=CATEGORICAL_STRUCT_ARRAY.dtype)

        mixed_samples = np.array(
            [(0.690, 'a', -1.944, 'b'),
             (0.124, 'a', -1.102, 'bb'),
             (1.445, 'c', -1.224, 'bb'),
             (2.122, 'c', -0.028, 'aa')],
            dtype=[('a', '<f8'), ('b', 'U1'), ('c', '<f8'), ('d', 'U2')])

        numerical_samples_mean = np.array([
            [-0.299, 1.016, -1.442, 0.611],
            [0.159, -1.271, -0.347, 0.698],
            [1.402, -2.630, 0.346, 0.754],
            [-1.389, -0.431, 0.716, -0.882]])

        numerical_0_samples_mean = np.array([
            [0, 0.220, 0.733, 0.239],
            [2, 0.325, 0.180, 3.161],
            [0, 0.795, -0.818, 3.386],
            [1, 0.907, -1.070, 2.265]])

        numerical_struct_samples_mean = np.array(
            [(0.215, -0.429, 0.723, 0.341),
             (-0.808, 0.515, 0.586, 0.570),
             (-0.920, 0.673, 0.546, -0.382),
             (0.359, 0.131, -0.254, 1.302)],
            dtype=[('a', 'f'), ('b', 'f'), ('c', 'f'), ('d', 'f')])

        numerical_struct_0_samples_mean = np.array(
            [(0, -0.146, -0.832, 1.089),
             (0, 0.462, -0.683, 2.174),
             (2, 0.439, -1.637, 1.484),
             (1, 1.292, -1.461, 4.102)],
            dtype=[('a', 'i'), ('b', 'f'), ('c', 'f'), ('d', 'f')])

        categorical_samples_mean = np.array([
            ['b', 'f', 'c'],
            ['b', 'b', 'c'],
            ['a', 'f', 'g'],
            ['a', 'b', 'c']])

        categorical_012_samples_mean = np.array([
            ['a', 'c', 'c'],
            ['b', 'f', 'c'],
            ['b', 'f', 'g'],
            ['a', 'c', 'c']])

        categorical_struct_samples_mean = np.array(
            [('a', 'f', 'c'),
             ('a', 'b', 'c'),
             ('a', 'b', 'c'),
             ('a', 'f', 'c')],
            dtype=CATEGORICAL_STRUCT_ARRAY.dtype)

        mixed_samples_mean = np.array(
            [(-1.250, 'c', 2.623, 'bb'),
             (2.352, 'c', -1.269, 'a'),
             (0.489, 'f', -0.604, 'bb'),
             (3.556, 'a', -0.741, 'bb')],
            dtype=[('a', '<f8'), ('b', 'U1'), ('c', '<f8'), ('d', 'U2')])
        # yapf: enable

        samples = self.numerical_np_augmentor.sample(
            NUMERICAL_NP_ARRAY[0], samples_number=4)
        assert np.allclose(samples, numerical_samples, atol=1e-3)

        samples = self.numerical_np_0_augmentor.sample(
            NUMERICAL_NP_ARRAY[0], samples_number=4)
        assert np.allclose(samples, numerical_0_samples, atol=1e-3)

        samples = self.numerical_struct_augmentor.sample(
            NUMERICAL_STRUCT_ARRAY[0], samples_number=4)
        for i in samples.dtype.names:
            assert np.allclose(
                samples[i], numerical_struct_samples[i], atol=1e-3)

        samples = self.numerical_struct_a_augmentor.sample(
            NUMERICAL_STRUCT_ARRAY[0], samples_number=4)
        for i in samples.dtype.names:
            assert np.allclose(
                samples[i], numerical_struct_0_samples[i], atol=1e-3)

        samples = self.categorical_np_augmentor.sample(
            CATEGORICAL_NP_ARRAY[0], samples_number=4)
        assert np.array_equal(samples, categorical_samples)

        samples = self.categorical_np_012_augmentor.sample(
            CATEGORICAL_NP_ARRAY[0], samples_number=4)
        assert np.array_equal(samples, categorical_012_samples)

        samples = self.categorical_struct_abc_augmentor.sample(
            CATEGORICAL_STRUCT_ARRAY[0], samples_number=4)
        for i in samples.dtype.names:
            assert np.array_equal(samples[i], categorical_struct_samples[i])

        samples = self.mixed_augmentor.sample(MIXED_ARRAY[0], samples_number=4)
        assert np.array_equal(samples[['b', 'd']], mixed_samples[['b', 'd']])
        for i in ['a', 'c']:
            assert np.allclose(samples[i], mixed_samples[i], atol=1e-3)

        # Test if minimum_per_class works
        samples = self.numerical_np_augmentor.sample(
            NUMERICAL_NP_ARRAY[0], samples_number=1000)
        predictions = self.numerical_classifier.predict(samples)
        _, counts = np.unique(predictions, return_counts=True)
        assert np.all(counts >= 0.05 * 1000)

        samples = self.numerical_np_0_augmentor.sample(
            NUMERICAL_NP_ARRAY[0], samples_number=1000)
        predictions = self.numerical_classifier.predict(samples)
        _, counts = np.unique(predictions, return_counts=True)
        assert np.all(counts > 0.05 * 1000)

        # Initialised with higher rate
        samples = self.numerical_struct_augmentor.sample(
            NUMERICAL_STRUCT_ARRAY[0], samples_number=1000)
        predictions = self.numerical_struct_classifier.predict(samples)
        _, counts = np.unique(predictions, return_counts=True)
        assert np.all(counts > 0.1 * 1000)

        samples = self.numerical_struct_a_augmentor.sample(
            NUMERICAL_STRUCT_ARRAY[0], samples_number=1000)
        predictions = self.numerical_struct_classifier.predict(samples)
        _, counts = np.unique(predictions, return_counts=True)
        assert np.all(counts > 0.05 * 1000)

        #######################################################################

        # Get averages and proportions
        vals = [['a', 'b'], ['b', 'c', 'f'], ['c', 'g']]
        proportions = [
            np.array([0.676, 0.324]),
            np.array([0.333, 0.151, 0.516]),
            np.array([0.667, 0.333])
        ]
        ###
        samples = self.categorical_np_augmentor.sample(
            CATEGORICAL_NP_ARRAY[0], samples_number=1000)
        predictions = self.categorical_classifier.predict(samples)
        _, counts = np.unique(predictions, return_counts=True)
        assert np.all(counts > 0.05 * 1000)
        for i, index in enumerate(range(CATEGORICAL_NP_ARRAY.shape[1])):
            val, freq = np.unique(samples[:, index], return_counts=True)
            freq = freq / freq.sum()
            assert np.array_equal(val, vals[i])
            assert np.allclose(freq, proportions[i], atol=1e-3)
        ###
        proportions = [
            np.array([0.665, 0.335]),
            np.array([0.357, 0.158, 0.485]),
            np.array([0.645, 0.355])
        ]
        samples = self.categorical_np_012_augmentor.sample(
            CATEGORICAL_NP_ARRAY[0], samples_number=1000)
        predictions = self.categorical_classifier.predict(samples)
        _, counts = np.unique(predictions, return_counts=True)
        assert np.all(counts > 0.05 * 1000)
        for i, index in enumerate(range(CATEGORICAL_NP_ARRAY.shape[1])):
            val, freq = np.unique(samples[:, index], return_counts=True)
            freq = freq / freq.sum()
            assert np.array_equal(val, vals[i])
            assert np.allclose(freq, proportions[i], atol=1e-3)
        ###
        proportions = [
            np.array([0.632, 0.368]),
            np.array([0.322, 0.172, 0.506]),
            np.array([0.675, 0.325])
        ]
        samples = self.categorical_struct_abc_augmentor.sample(
            CATEGORICAL_STRUCT_ARRAY[0], samples_number=1000)
        predictions = self.categorical_struct_classifier.predict(samples)
        _, counts = np.unique(predictions, return_counts=True)
        assert np.all(counts > 0.05 * 1000)
        for i, index in enumerate(CATEGORICAL_STRUCT_ARRAY.dtype.names):
            val, freq = np.unique(samples[index], return_counts=True)
            freq = freq / freq.sum()
            assert np.array_equal(val, vals[i])
            assert np.allclose(freq, proportions[i], atol=1e-3)

        #######################################################################

        # Sample without float cast
        samples = self.numerical_struct_augmentor_f.sample(
            NUMERICAL_STRUCT_ARRAY[0], samples_number=5)
        samples_answer = np.array(
            [(1, 0, 0.196, -0.434),
             (-1, 0, -0.980, 1.238),
             (1, -1, 1.104, 0.609),
             (0, -1, -3.930, -0.454),
             (0, 0, -0.483, 1.122)],
            dtype=NUMERICAL_STRUCT_ARRAY.dtype)  # yapf: disable
        for i in NUMERICAL_STRUCT_ARRAY.dtype.names:
            assert np.allclose(samples[i], samples_answer[i], atol=1e-3)

        #######################################################################

        # Test if max_iter is too low to find all classes
        with pytest.raises(RuntimeError) as exin:
            self.numerical_np_0_augmentor.sample(
                NUMERICAL_NP_ARRAY[0], max_iter=1)
        assert str(exin.value) == runtime_msg

        #######################################################################

        # Test with mean of dataset as starting point
        samples = self.numerical_np_augmentor.sample(samples_number=4)
        assert np.allclose(samples, numerical_samples_mean, atol=1e-3)

        samples = self.numerical_np_0_augmentor.sample(samples_number=4)
        assert np.allclose(samples, numerical_0_samples_mean, atol=1e-3)

        samples = self.numerical_struct_augmentor.sample(samples_number=4)
        for i in samples.dtype.names:
            assert np.allclose(
                samples[i], numerical_struct_samples_mean[i], atol=1e-3)

        samples = self.numerical_struct_a_augmentor.sample(samples_number=4)
        for i in samples.dtype.names:
            assert np.allclose(
                samples[i], numerical_struct_0_samples_mean[i], atol=1e-3)

        samples = self.categorical_np_augmentor.sample(samples_number=4)
        assert np.array_equal(samples, categorical_samples_mean)

        samples = self.categorical_np_012_augmentor.sample(samples_number=4)
        assert np.array_equal(samples, categorical_012_samples_mean)

        samples = self.categorical_struct_abc_augmentor.sample(
            samples_number=4)
        for i in samples.dtype.names:
            assert np.array_equal(samples[i],
                                  categorical_struct_samples_mean[i])

        samples = self.mixed_augmentor.sample(samples_number=4)
        assert np.array_equal(samples[['b', 'd']],
                              mixed_samples_mean[['b', 'd']])
        for i in ['a', 'c']:
            assert np.allclose(samples[i], mixed_samples_mean[i], atol=1e-3)

        # Test if minimum_per_class works with mean of dataset
        samples = self.numerical_np_augmentor.sample(samples_number=1000)
        predictions = self.numerical_classifier.predict(samples)
        _, counts = np.unique(predictions, return_counts=True)
        assert np.all(counts > 0.05 * 1000)

        samples = self.numerical_np_0_augmentor.sample(samples_number=1000)
        predictions = self.numerical_classifier.predict(samples)
        _, counts = np.unique(predictions, return_counts=True)
        assert np.all(counts > 0.05 * 1000)

        samples = self.numerical_struct_augmentor.sample(samples_number=1000)
        predictions = self.numerical_struct_classifier.predict(samples)
        _, counts = np.unique(predictions, return_counts=True)
        assert np.all(counts > 0.1 * 1000)

        samples = self.numerical_struct_a_augmentor.sample(samples_number=1000)
        predictions = self.numerical_struct_classifier.predict(samples)
        _, counts = np.unique(predictions, return_counts=True)
        assert np.all(counts > 0.05 * 1000)

        #######################################################################

        vals = [['a', 'b'], ['b', 'c', 'f'], ['c', 'g']]
        proportions = [
            np.array([0.7, 0.3]),
            np.array([0.3, 0.2, 0.5]),
            np.array([0.7, 0.3])
        ]

        samples = self.categorical_np_augmentor.sample(samples_number=1000)
        predictions = self.categorical_classifier.predict(samples)
        _, counts = np.unique(predictions, return_counts=True)
        assert np.all(counts > 0.05 * 1000)
        for i, index in enumerate(range(3)):
            val, freq = np.unique(samples[:, index], return_counts=True)
            freq = freq / freq.sum()
            assert np.array_equal(val, vals[i])
            assert np.allclose(freq, proportions[i], atol=1e-1)

        samples = self.categorical_np_012_augmentor.sample(samples_number=1000)
        predictions = self.categorical_classifier.predict(samples)
        _, counts = np.unique(predictions, return_counts=True)
        assert np.all(counts > 0.05 * 1000)
        for i, index in enumerate(range(3)):
            val, freq = np.unique(samples[:, index], return_counts=True)
            freq = freq / freq.sum()
            assert np.array_equal(val, vals[i])
            assert np.allclose(freq, proportions[i], atol=1e-1)

        samples = self.categorical_struct_abc_augmentor.sample(
            samples_number=1000)
        predictions = self.categorical_struct_classifier.predict(samples)
        _, counts = np.unique(predictions, return_counts=True)
        assert np.all(counts > 0.05 * 1000)
        for i, index in enumerate(['a', 'b', 'c']):
            val, freq = np.unique(samples[index], return_counts=True)
            freq = freq / freq.sum()
            assert np.array_equal(val, vals[i])
            assert np.allclose(freq, proportions[i], atol=1e-1)


def test_validate_input_decisionboundarysphere():
    """
    Tests ``_validate_input_decisionboundarysphere`` function.

    Tests
    :func:`fatf.utils.data.augmentation._validate_input_decisionboundarysphere`
    function.
    """
    predictive_function_type = ('The predictive_function should be a Python '
                                'callable, e.g., a Python function.')
    predictive_function_model = ('The predictive function must take exactly '
                                 '*one* required parameter: a data array to '
                                 'be predicted.')
    starting_std_type = 'The radius_init parameter is not a number.'
    starting_std_value = ('The radius_init parameter must be a positive '
                          'number (greater than 0).')
    increment_std_type = 'The radius_increment parameter is not a number.'
    increment_std_value = ('The radius_increment parameter is not a '
                           'positive number (greater than 0).')

    def predict(x, y=None):
        pass  # pragma: no cover

    def invalid_predict(x, y):
        pass  # pragma: no cover

    model = fum.KNN(k=3)

    with pytest.raises(TypeError) as exin:
        fuda._validate_input_decisionboundarysphere(None, None, None)
    assert str(exin.value) == predictive_function_type
    with pytest.raises(IncompatibleModelError) as exin:
        fuda._validate_input_decisionboundarysphere(
            invalid_predict, None, None)  # yapf: disable
    assert str(exin.value) == predictive_function_model

    with pytest.raises(TypeError) as exin:
        fuda._validate_input_decisionboundarysphere(model.predict, 'a', None)
    assert str(exin.value) == starting_std_type
    with pytest.raises(ValueError) as exin:
        fuda._validate_input_decisionboundarysphere(predict, -0.1, 0.1)
    assert str(exin.value) == starting_std_value

    with pytest.raises(TypeError) as exin:
        fuda._validate_input_decisionboundarysphere(
            model.predict_proba, 0.1, 'a')  # yapf: disable
    assert str(exin.value) == increment_std_type
    with pytest.raises(ValueError) as exin:
        fuda._validate_input_decisionboundarysphere(predict, 0.1, -0.1)
    assert str(exin.value) == increment_std_value


class TestDecisionBoundarySphere():
    """
    Tests :class:`fatf.utils.data.augmentation.DecisionBoundarySphere` class.
    """
    numerical_labels = np.array([0, 1, 0, 1, 1, 0])
    numerical_easy_labels = np.array([0, 0, 1, 1])
    numerical_array_easy = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])

    numerical_easy_classifier = fum.KNN(k=1)
    numerical_easy_classifier.fit(numerical_array_easy, numerical_easy_labels)
    #
    numerical_np_easy_augmentor = fuda.DecisionBoundarySphere(
        numerical_array_easy, numerical_easy_classifier.predict_proba)

    numerical_classifier = fum.KNN(k=3)
    numerical_classifier.fit(NUMERICAL_NP_ARRAY, numerical_labels)
    #
    numerical_np_augmentor = fuda.DecisionBoundarySphere(
        NUMERICAL_NP_ARRAY, numerical_classifier.predict)
    numerical_np_augmentor_runtime = fuda.DecisionBoundarySphere(
        NUMERICAL_NP_ARRAY, numerical_classifier.predict, radius_init=0.0001)

    numerical_struct_classifier = fum.KNN(k=3)
    numerical_struct_classifier.fit(NUMERICAL_STRUCT_ARRAY, numerical_labels)
    #
    numerical_struct_augmentor = fuda.DecisionBoundarySphere(
        NUMERICAL_STRUCT_ARRAY, numerical_struct_classifier.predict)
    numerical_struct_augmentor_f = fuda.DecisionBoundarySphere(
        NUMERICAL_STRUCT_ARRAY,
        numerical_struct_classifier.predict,
        int_to_float=False)

    def test_init(self):
        """
        Tests ``DecisionBoundarySphere`` class initialisation.

        Tests :class:`fatf.utils.data.augmentation.DecisionBoundarySphere`
        class initialisation.
        """
        cat_err = ('The DecisionBoundarySphere augmenter does not currently '
                   'support data sets with categorical features.')

        # Test class inheritance
        assert (self.numerical_np_augmentor.__class__.__bases__[0].__name__
                == 'Augmentation')  # yapf: disable

        # Test calculating numerical and categorical indices
        assert self.numerical_np_augmentor.categorical_indices == []
        assert self.numerical_np_augmentor.numerical_indices == [0, 1, 2, 3]
        #
        assert (self.numerical_struct_augmentor.numerical_indices
                == ['a', 'b', 'c', 'd'])  # yapf: disable

        categorical_classifier = fum.KNN(k=3)
        categorical_classifier.fit(CATEGORICAL_NP_ARRAY, self.numerical_labels)
        with pytest.raises(NotImplementedError) as exin:
            fuda.DecisionBoundarySphere(CATEGORICAL_NP_ARRAY,
                                        categorical_classifier.predict)
        assert str(exin.value) == cat_err

        # Test attributes unique to DecisionBoundarySphere...
        # ...classifier
        assert (self.numerical_np_augmentor.predictive_function
                == self.numerical_classifier.predict)  # yapf: disable
        assert self.numerical_np_augmentor.is_probabilistic is False
        assert self.numerical_np_augmentor.radius_init == 0.01
        assert self.numerical_np_augmentor.radius_increment == 0.01
        # ...probabilistic
        assert (
            self.numerical_np_easy_augmentor.predictive_function
            == self.numerical_easy_classifier.predict_proba
        )  # yapf: disable
        assert self.numerical_np_easy_augmentor.is_probabilistic is True
        assert self.numerical_np_easy_augmentor.radius_init == 0.01
        assert self.numerical_np_easy_augmentor.radius_increment == 0.01

    def test_validate_sample_input(self):
        """
        Tests ``_validate_sample_input`` method.

        Tests :func:`fatf.utils.data.augmentation.DecisionBoundarySphere.\
_validate_sample_input` method. Most errors and exceptions are caught by
        :func:`fatf.utils.data.augmentation.Augmentation.\
_validate_sample_input` method (which has already been tested).
        """
        sphere_radius_type = 'The sphere_radius parameter must be a number.'
        sphere_radius_value = ('The sphere_radius parameter must be a '
                               'positive number (greater than 0).')
        samples_type = ('The discover_samples_number parameter must be an '
                        'integer.')
        samples_val = ('The discover_samples_number parameter must be a '
                       'positive integer (greater than 0).')
        max_iter_type = 'The max_iter parameter must be an integer.'
        max_iter_val = ('The max_iter parameter must be a positive integer '
                        '(greater than 0).')

        with pytest.raises(TypeError) as exin:
            self.numerical_np_augmentor._validate_sample_input(
                NUMERICAL_NP_ARRAY[0], 'a', 10, 10, 10)
        assert str(exin.value) == sphere_radius_type
        with pytest.raises(ValueError) as exin:
            self.numerical_np_augmentor._validate_sample_input(
                NUMERICAL_NP_ARRAY[0], 0.0, 10, 10, 10)
        assert str(exin.value) == sphere_radius_value
        with pytest.raises(ValueError) as exin:
            self.numerical_np_augmentor._validate_sample_input(
                NUMERICAL_NP_ARRAY[0], -0.05, 10, 10, 10)
        assert str(exin.value) == sphere_radius_value

        with pytest.raises(TypeError) as exin:
            self.numerical_np_augmentor._validate_sample_input(
                NUMERICAL_NP_ARRAY[0], 0.05, 10, 'a', 10)
        assert str(exin.value) == samples_type
        with pytest.raises(ValueError) as exin:
            self.numerical_np_augmentor._validate_sample_input(
                NUMERICAL_NP_ARRAY[0], 0.05, 10, -1, 10)
        assert str(exin.value) == samples_val

        with pytest.raises(TypeError) as exin:
            self.numerical_np_augmentor._validate_sample_input(
                NUMERICAL_NP_ARRAY[0], 0.05, 10, 10, 'a')
        assert str(exin.value) == max_iter_type
        with pytest.raises(ValueError) as exin:
            self.numerical_np_augmentor._validate_sample_input(
                NUMERICAL_NP_ARRAY[0], 0.05, 10, 10, -1)
        assert str(exin.value) == max_iter_val

    def test_sample(self):
        """
        Tests ``sample`` method.

        Tests
        :func:`fatf.utils.data.augmentation.DecisionBoundarySphere.sample`
        method.
        """
        runtime_msg = ('The maximum number of iterations was reached without '
                       'discovering a decision boundary. Please try '
                       'increasing the max_iter or discover_samples_number '
                       'parameter. Alternatively, initialise this class with '
                       'a larger radius_init or radius_increment parameter.')
        mean_msg = ('Sampling around the mean of the initialisation dataset '
                    'is not currently supported by the DecisionBoundarySphere '
                    'augmenter.')

        with pytest.raises(NotImplementedError) as exin:
            self.numerical_np_augmentor.sample(None)
        assert str(exin.value) == mean_msg

        with pytest.raises(RuntimeError) as exin:
            self.numerical_np_augmentor_runtime.sample(
                NUMERICAL_NP_ARRAY[0], max_iter=1)
        assert str(exin.value) == runtime_msg

        numerical_results = np.array([[0.126, 0.010, 0.148, 0.923],
                                      [-0.134, 0.129, -0.240, 0.850],
                                      [0.125, 0.087, 0.018, 0.840],
                                      [0.042, 0.053, 0.039, 0.671]])
        numerical_struct_results = np.array([(-0.018, 0.003, 0.098, 0.658),
                                             (-0.010, 0.158, 0.311, 0.930),
                                             (0.115, 0.079, 0.128, 0.700),
                                             (-0.013, 0.137, 0.088, 0.548)],
                                            dtype=[('a', 'f'), ('b', 'f'),
                                                   ('c', 'f'), ('d', 'f')])
        numerical_struct_results_f = np.array(
            [(0, 0, 0.239, 0.747),
             (0, 0, 0.086, 0.584),
             (0, 0, -0.097, 0.722),
             (0, 0, 0.101, 0.678)],
            dtype=NUMERICAL_STRUCT_ARRAY.dtype)  # yapf: disable

        sphere_radius = 0.4

        fatf.setup_random_seed()

        # Easy example of (0,0), (0,1), (1,0) and (1,1)
        samples = self.numerical_np_easy_augmentor.sample(
            self.numerical_array_easy[3],
            sphere_radius=sphere_radius,
            samples_number=100,
            discover_samples_number=1000)
        max_dist = fud.euclidean_array_distance(samples, samples).max()
        assert np.allclose(samples.mean(axis=0), np.array([0.5, 1]), atol=0.1)
        assert np.isclose(max_dist, 2 * sphere_radius, atol=0.1)

        samples = self.numerical_np_augmentor.sample(
            NUMERICAL_NP_ARRAY[0],
            sphere_radius=sphere_radius,
            samples_number=4)
        assert np.allclose(samples, numerical_results, atol=1e-3)

        samples = self.numerical_np_augmentor.sample(
            NUMERICAL_NP_ARRAY[0],
            sphere_radius=sphere_radius,
            samples_number=100,
            discover_samples_number=1000)
        decision_boundary = np.array([-0.01, 0.01, 0.07, 0.69])
        assert np.allclose(samples.mean(axis=0), decision_boundary, atol=0.1)
        max_dist = fud.euclidean_array_distance(samples, samples).max()
        assert np.isclose(max_dist, 2 * sphere_radius, atol=0.1)

        samples = self.numerical_struct_augmentor.sample(
            NUMERICAL_STRUCT_ARRAY[0],
            sphere_radius=sphere_radius,
            samples_number=4)
        for i in samples.dtype.names:
            assert np.allclose(
                samples[i], numerical_struct_results[i], atol=1e-3)

        samples = self.numerical_struct_augmentor.sample(
            NUMERICAL_STRUCT_ARRAY[0],
            sphere_radius=sphere_radius,
            samples_number=100,
            discover_samples_number=1000)
        decision_boundary = [-0.01, 0.01, 0.07, 0.69]
        for i, bound in zip(samples.dtype.names, decision_boundary):
            assert np.isclose(samples[i].mean(), bound, atol=0.1)
        max_dist = fud.euclidean_array_distance(samples, samples).max()
        assert np.isclose(max_dist, 2 * sphere_radius, atol=0.1)

        samples = self.numerical_struct_augmentor_f.sample(
            NUMERICAL_STRUCT_ARRAY[0],
            sphere_radius=sphere_radius,
            samples_number=4)
        for i in samples.dtype.names:
            assert np.allclose(
                samples[i], numerical_struct_results_f[i], atol=1e-3)

        samples = self.numerical_struct_augmentor_f.sample(
            NUMERICAL_STRUCT_ARRAY[0],
            sphere_radius=sphere_radius,
            samples_number=100,
            discover_samples_number=1000)
        decision_boundary = [0, 0, 0.05, 0.6]
        for i, bound in zip(samples.dtype.names, decision_boundary):
            assert np.isclose(samples[i].mean(), bound, atol=0.1)
        max_dist = fud.euclidean_array_distance(samples, samples).max()
        assert np.isclose(max_dist, 2 * sphere_radius, atol=0.1)


class TestLocalSphere(object):
    """
    Tests :class:`fatf.utils.data.augmentation.LocalSphere` class.
    """
    numerical_np_augmentor = fuda.LocalSphere(NUMERICAL_NP_ARRAY)
    numerical_struct_augmentor = fuda.LocalSphere(NUMERICAL_STRUCT_ARRAY)
    numerical_struct_augmentor_f = fuda.LocalSphere(
        NUMERICAL_STRUCT_ARRAY, int_to_float=False)

    def test_init(self):
        """
        Tests :class:`fatf.utils.data.augmentation.LocalSphere` class init.
        """
        cat_err = ('The LocalSphere augmenter does not currently support data '
                   'sets with categorical features.')

        # Test class inheritance
        assert (self.numerical_np_augmentor.__class__.__bases__[0].__name__
                == 'Augmentation')  # yapf: disable

        # Test calculating numerical and categorical indices
        assert self.numerical_np_augmentor.categorical_indices == []
        assert self.numerical_np_augmentor.numerical_indices == [0, 1, 2, 3]
        #
        assert (self.numerical_struct_augmentor.numerical_indices
                == ['a', 'b', 'c', 'd'])  # yapf: disable

        with pytest.raises(NotImplementedError) as exin:
            fuda.LocalSphere(CATEGORICAL_NP_ARRAY)
        assert str(exin.value) == cat_err

    def test_sample(self):
        """
        Tests :func:`fatf.utils.data.augmentation.LocalSphere.sample`.
        """
        type_msg = ('The fidelity_radius_percentage parameter must be an '
                    'integer.')
        value_msg = ('The fidelity_radius_percentage parameter must be a '
                     'positive integer (greater than 0).')
        mean_msg = ('Sampling around the mean of the initialisation dataset '
                    'is not currently supported by the LocalSphere augmenter.')

        with pytest.raises(NotImplementedError) as exin:
            self.numerical_np_augmentor.sample(None)
        assert str(exin.value) == mean_msg

        with pytest.raises(TypeError) as exin:
            self.numerical_np_augmentor.sample(
                NUMERICAL_NP_ARRAY[0], fidelity_radius_percentage='a')
        assert str(exin.value) == type_msg
        with pytest.raises(TypeError) as exin:
            self.numerical_np_augmentor.sample(
                NUMERICAL_NP_ARRAY[0], fidelity_radius_percentage=0.0)
        assert str(exin.value) == type_msg

        with pytest.raises(ValueError) as exin:
            self.numerical_np_augmentor.sample(
                NUMERICAL_NP_ARRAY[0], fidelity_radius_percentage=0)
        assert str(exin.value) == value_msg
        with pytest.raises(ValueError) as exin:
            self.numerical_np_augmentor.sample(
                NUMERICAL_NP_ARRAY[0], fidelity_radius_percentage=-5)
        assert str(exin.value) == value_msg

        numerical_results = np.array([[-0.057, -0.057, 0.467, 0.878],
                                      [-0.536, 0.620, -0.449, 0.158],
                                      [0.078, -0.618, -0.477, 0.508],
                                      [-0.357, 0.111, -0.240, 0.192]])
        numerical_struct_results = np.array([(-0.329, 0.119, 0.161, 0.681),
                                             (-0.025, -0.013, 0.075, 0.661),
                                             (-0.156, 0.138, 0.382, 0.601),
                                             (0.209, -0.528, 0.262, 0.468)],
                                            dtype=[('a', 'f'), ('b', 'f'),
                                                   ('c', 'f'), ('d', 'f')])
        numerical_struct_results_f = np.array(
            [(0, 0, 0.122, 0.606),
             (0, 0, -0.024, 0.706),
             (0, 0, 0.105, 0.816),
             (0, 0, -0.151, 0.898)],
            dtype=NUMERICAL_STRUCT_ARRAY.dtype)  # yapf: disable

        fatf.setup_random_seed()

        max_distance_dataset = 2.34

        # Numerical non-structured
        samples = self.numerical_np_augmentor.sample(
            NUMERICAL_NP_ARRAY[0],
            fidelity_radius_percentage=50,
            samples_number=4)
        assert np.allclose(numerical_results, samples, atol=1e-3)
        # 1000 samples
        samples = self.numerical_np_augmentor.sample(
            NUMERICAL_NP_ARRAY[0],
            fidelity_radius_percentage=20,
            samples_number=1000)
        max_dist = fud.euclidean_array_distance(
            np.expand_dims(NUMERICAL_NP_ARRAY[0], 0), samples).max()
        assert np.isclose(max_dist, 0.2 * max_distance_dataset, atol=1e-1)
        assert np.allclose(
            samples.mean(axis=0), NUMERICAL_NP_ARRAY[0], atol=1e-1)
        # Assert uniformity with Kolmogorov-Smirnov test for goodness of fit
        for i in range(samples.shape[1]):
            feature = samples[:, i]
            scale = 1 / (feature.max() - feature.min())
            sc = scale * feature - feature.min() * scale
            result = scipy.stats.kstest(sc, 'uniform').statistic
            assert result < 0.25

        # Numerical structured
        samples = self.numerical_struct_augmentor.sample(
            NUMERICAL_STRUCT_ARRAY[0],
            fidelity_radius_percentage=50,
            samples_number=4)
        for i in samples.dtype.names:
            assert np.allclose(
                samples[i], numerical_struct_results[i], atol=1e-3)
        # 1000 samples
        samples = self.numerical_struct_augmentor.sample(
            NUMERICAL_STRUCT_ARRAY[0],
            fidelity_radius_percentage=20,
            samples_number=1000)
        max_dist = fud.euclidean_array_distance(
            np.expand_dims(NUMERICAL_STRUCT_ARRAY[0], 0), samples).max()
        assert np.isclose(max_dist, 0.2 * max_distance_dataset, atol=1e-1)
        for i in samples.dtype.names:
            assert np.isclose(
                samples[i].mean(), NUMERICAL_STRUCT_ARRAY[0][i], atol=0.1)
        # Assert uniformity with Kolmogorov-Smirnov test for goodness of fit
        for i in samples.dtype.names:
            feature = samples[i]
            scale = 1 / (feature.max() - feature.min())
            sc = scale * feature - feature.min() * scale
            result = scipy.stats.kstest(sc, 'uniform').statistic
            assert result < 0.25

        # Numerical structured without cast
        samples = self.numerical_struct_augmentor_f.sample(
            NUMERICAL_STRUCT_ARRAY[0],
            fidelity_radius_percentage=50,
            samples_number=4)
        for i in samples.dtype.names:
            assert np.allclose(
                samples[i], numerical_struct_results_f[i], atol=1e-3)
        # 1000 samples
        samples = self.numerical_struct_augmentor_f.sample(
            NUMERICAL_STRUCT_ARRAY[0],
            fidelity_radius_percentage=20,
            samples_number=1000)
        max_dist = fud.euclidean_array_distance(
            np.expand_dims(NUMERICAL_STRUCT_ARRAY[0], 0), samples).max()
        assert np.isclose(max_dist, 0.2 * max_distance_dataset, atol=1e-1)
        for i in samples.dtype.names:
            assert np.isclose(
                samples[i].mean(), NUMERICAL_STRUCT_ARRAY[0][i], atol=0.1)
        for i in ['c', 'd']:
            feature = samples[i]
            scale = 1 / (feature.max() - feature.min())
            sc = scale * feature - feature.min() * scale
            result = scipy.stats.kstest(sc, 'uniform').statistic
            assert result < 0.25
