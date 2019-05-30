"""
Functions for testing data augmentation classes.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
#         Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import pytest

import numpy as np
import scipy

import fatf

from fatf.exceptions import IncorrectShapeError, IncompatibleModelError

import fatf.utils.models as fum
import fatf.utils.data.augmentation as fuda
import fatf.utils.array.tools as fuat

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
        Function for getting the theoretical truncated normal mean and std
        from the original normal mean and std, and the minimum and maximum
        for which values are truncated.

        Equations for calculating these can be found at:
        https://en.wikipedia.org/wiki/Truncated_normal_distribution
        """
        def cdf(epsilon):
            return 1/2 * (1 + scipy.special.erf(epsilon/np.sqrt(2)))

        def norm(episilon):
            return (1/np.sqrt(2*np.pi) * np.exp(-1/2 * episilon**2))
        
        alpha = (minimum-original_mean) / original_std
        beta = (maximum-original_mean) / original_std
        z_phi = cdf(beta) - cdf(alpha)
        mean = original_mean + ((norm(alpha) - norm(beta)) / z_phi) * \
               original_std
        std = np.sqrt(original_std ** 2 * (1 + (alpha * norm(alpha) - beta * \
              norm(beta)) / z_phi - ((norm(alpha) - norm(beta)) / z_phi) ** 2 ))
        return mean, std


class TestTruncatedNormal(object):
    """
    Tests :class:`fatf.utils.data.augmentation.TruncatedNormal` class.
    """
    numerical_np_0_augmentor = fuda.TruncatedNormal(NUMERICAL_NP_ARRAY, [0])
    numerical_np_augmentor = fuda.TruncatedNormal(NUMERICAL_NP_ARRAY)
    numerical_struct_a_augmentor = fuda.TruncatedNormal(NUMERICAL_STRUCT_ARRAY,
                                                       ['a'])
    numerical_struct_augmentor = fuda.TruncatedNormal(NUMERICAL_STRUCT_ARRAY)
    numerical_struct_augmentor_f = fuda.TruncatedNormal(
        NUMERICAL_STRUCT_ARRAY, int_to_float=False)
    categorical_np_augmentor = fuda.TruncatedNormal(CATEGORICAL_NP_ARRAY)
    categorical_np_012_augmentor = fuda.TruncatedNormal(CATEGORICAL_NP_ARRAY,
                                                       [0, 1, 2])
    categorical_struct_abc_augmentor = fuda.TruncatedNormal(
        CATEGORICAL_STRUCT_ARRAY, ['a', 'b', 'c'])
    mixed_augmentor = fuda.TruncatedNormal(MIXED_ARRAY, ['b', 'd'])

    def test_init(self):
        """
        Tests :class:`fatf.utils.data.augmentation.TruncatedNormal` class init.
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

        # Test attributes unique to TruncatedNormal
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
        Tests :func:`~fatf.utils.data.augmentation.TruncatedNormal.sample`.
        """
        fatf.setup_random_seed()

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
        # Calculate what mean and std of truncated normals should be
        mini = NUMERICAL_NP_ARRAY.min(axis=0)
        maxi = NUMERICAL_NP_ARRAY.max(axis=0)
        std = NUMERICAL_NP_ARRAY.std(axis=0)
        mean = NUMERICAL_NP_ARRAY.mean(axis=0)
        numerical_truncated_results_mean, numerical_truncated_results_std = \
            get_truncated_mean_std(mini, maxi, NUMERICAL_NP_ARRAY[0], std)
        (numerical_truncated_results_dataset_mean, 
            numerical_truncated_results_dataset_std) = \
                get_truncated_mean_std(mini, maxi, mean, std)
        mixed_numerical_values = fuat.structured_to_unstructured(
            MIXED_ARRAY[['a', 'c']])
        mini = mixed_numerical_values.min(axis=0)
        maxi = mixed_numerical_values.max(axis=0)
        std = mixed_numerical_values.std(axis=0)
        mean = mixed_numerical_values.mean(axis=0)
        mixed_truncated_results_mean, mixed_truncated_results_std = \
            get_truncated_mean_std(mini, maxi, mixed_numerical_values[0],
                                        std)
        (mixed_truncated_results_dataset_mean,
            mixed_truncated_results_dataset_std) = \
                get_truncated_mean_std(mini, maxi, mean, std)

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
                samples_struct[i], numerical_struct_truncated_results[i], atol=1e-3)
        
        # ...numpy array results mean
        samples = self.numerical_np_augmentor.sample(
            NUMERICAL_NP_ARRAY[0, :], samples_number=1000)
        assert np.allclose(
            samples.mean(axis=0), numerical_truncated_results_mean, atol=1e-1)
        assert np.allclose(
            samples.std(axis=0), numerical_truncated_results_std, atol=1e-1)

        # ...structured array results mean
        samples_struct = self.numerical_struct_augmentor.sample(
            NUMERICAL_STRUCT_ARRAY[0], samples_number=1000)
        for i, name in enumerate(samples_struct.dtype.names):
            assert np.allclose(
                np.mean(samples_struct[name]),
                numerical_truncated_results_mean[i],
                atol=1e-1)
            assert np.allclose(
                np.std(samples_struct[name]),
                numerical_truncated_results_std[i],
                atol=1e-1)

        # Pure numerical sampling of the mean of the data
        # ...numpy array mean
        samples = self.numerical_np_augmentor.sample(samples_number=1000)
        assert np.allclose(
            samples.mean(axis=0), numerical_truncated_results_dataset_mean,
            atol=1e-1)
        assert np.allclose(
            samples.std(axis=0), numerical_truncated_results_dataset_std,
            atol=1e-1)

        # ...structured array mean
        samples_struct = self.numerical_struct_augmentor.sample(
            samples_number=1000)
        for i, name in enumerate(samples_struct.dtype.names):
            assert np.allclose(
                np.mean(samples_struct[name]),
                numerical_truncated_results_dataset_mean[i],
                atol=1e-1)
            assert np.allclose(
                np.std(samples_struct[name]),
                numerical_truncated_results_dataset_std[i],
                atol=1e-1)

        #######################################################################

        # Numerical sampling with one categorical index defined
        # ...numpy array results
        samples = self.numerical_np_0_augmentor.sample(
            NUMERICAL_NP_ARRAY[0, :], samples_number=3)
        assert np.allclose(samples, numerical_np_truncated_cat_results,
                           atol=1e-3)

        # ...structured array results
        samples_struct = self.numerical_struct_a_augmentor.sample(
            NUMERICAL_STRUCT_ARRAY[0], samples_number=3)
        for i in samples_struct.dtype.names:
            assert np.allclose(
                samples_struct[i], numerical_struct_truncated_cat_results[i],
                atol=1e-3)

        # ...numpy array results mean
        samples = self.numerical_np_0_augmentor.sample(
            NUMERICAL_NP_ARRAY[0, :], samples_number=100)
        # ......numerical
        assert np.allclose(
            samples.mean(axis=0)[1:], numerical_truncated_results_mean[1:],
                         atol=1e-1)
        assert np.allclose(
            samples.std(axis=0)[1:],
            numerical_truncated_results_std[1:],
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
        for i, name in enumerate(samples_struct.dtype.names[1:]):
            assert np.allclose(
                np.mean(samples_struct[name]),
                numerical_truncated_results_mean[1:][i],
                atol=1e-1)
            assert np.allclose(
                np.std(samples_struct[name]),
                numerical_truncated_results_std[1:][i],
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
            numerical_truncated_results_dataset_mean[1:],
            atol=1e-1)
        assert np.allclose(
            samples.std(axis=0)[1:],
            numerical_truncated_results_dataset_std[1:],
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
        for i, name in enumerate(samples_struct.dtype.names[1:]):
            assert np.allclose(
                np.mean(samples_struct[name]),
                numerical_truncated_results_dataset_mean[1:][i],
                atol=1e-1)
            assert np.allclose(
                np.std(samples_struct[name]),
                numerical_truncated_results_dataset_std[1:][i],
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
        for i, index in enumerate([0, 1, 2]):
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
        for i, index in enumerate(['a', 'b', 'c']):
            val, freq = np.unique(samples_struct[index], return_counts=True)
            freq = freq / freq.sum()
            assert np.array_equal(val, vals[i])
            assert np.allclose(freq, proportions[i], atol=1e-1)

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
        assert np.array_equal(samples[['b', 'd']], mixed_results[['b', 'd']])
        # ...numerical
        for i in ['a', 'c']:
            assert np.allclose(samples[i], mixed_results[i], atol=1e-3)

        # Instance mean
        samples = self.mixed_augmentor.sample(
            MIXED_ARRAY[0], samples_number=1000)
        # ...numerical
        for i, name in enumerate(['a', 'c']):
            assert np.allclose(
                np.mean(samples[name]), mixed_truncated_results_mean[i],
                        atol=1e-1)
            assert np.allclose(
                np.std(samples[name]), mixed_truncated_results_std[i],
                       atol=1e-1)
        # ...categorical
        for i, index in enumerate(['b', 'd']):
            val, freq = np.unique(samples[index], return_counts=True)
            freq = freq / freq.sum()
            assert np.array_equal(val, vals[i])
            assert np.allclose(freq, proportions[i], atol=1e-1)

        # Dataset mean
        samples = self.mixed_augmentor.sample(samples_number=1000)
        # ...numerical
        for i, name in enumerate(['a', 'c']):
            assert np.allclose(
                np.mean(samples[name]), mixed_truncated_results_dataset_mean[i],
                atol=1e-1)
            assert np.allclose(
                np.std(samples[name]), mixed_truncated_results_dataset_std[i],
                atol=1e-1)
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
            [(0, 0, 0.345, 0.442),
             (1, 0, 0.311, 0.338),
             (1, 0, 0.040, 0.553),
             (0, 0, 0.886, 0.822),
             (0, 0, 0.164, 0.315)],
            dtype=NUMERICAL_STRUCT_ARRAY.dtype)  # yapf: disable
        for i in ['a', 'b', 'c', 'd']:
            assert np.allclose(samples[i], samples_answer[i], atol=1e-3)

        # Cast to float on in the tests to compare (this ouput was generated
        # with self.numerical_struct_augmentor)
        samples = self.numerical_struct_augmentor_f.sample(samples_number=5)
        samples_answer = np.array(
            [(0.718, 0.476, 0.449, 0.615),
             (0.047, 0.883, 0.205, 0.329),
             (1.255, 0.422, 0.302, 0.627),
             (1.024, 0.512, 0.122, 0.790),
             (1.123, 0.670, 0.386, 0.471)],
            dtype=NUMERICAL_STRUCT_ARRAY.dtype)  # yapf: disable
        for i in ['a', 'b', 'c', 'd']:
            assert np.allclose(samples[i], samples_answer[i], atol=1e-3)


def test_validate_input_growingspheres():
    """
    Tests :func:`fatf.utils.data.augmentation._validate_input_growingspheres`.
    """
    incompatible_model_msg = ('This functionality requires the model '
                              'to be capable of outputting predicted '
                              'class via predict method.')
    starting_std_msg = ('starting_std is not a float.')
    increment_std_msg = ('increment_std is not a float.')
    minimum_per_class_msg = ('minimum_per_class is not a float.')
    minimum_per_class_value_msg = ('minimum_per_class must be a float '
                                   'between 0 and 1.')
    starting_std_value_msg = ('starting_std must be a positive float greater '
                              'than 0.')
    increment_std_value_msg = ('increment_std must be a positive float '
                               'greater than 0.')

    class invalid_model():
        def __init__(self): pass
        def predict_proba(self, x, y): pass
    model = invalid_model()
    with pytest.raises(IncompatibleModelError) as exin:
        fuda._validate_input_growingspheres(model, None, None, None)
    assert str(exin.value) == incompatible_model_msg
    
    model = fum.KNN(k=3)
    with pytest.raises(TypeError) as exin:
        fuda._validate_input_growingspheres(model, 'a', None, None)
    assert str(exin.value) == starting_std_msg

    with pytest.raises(TypeError) as exin:
        fuda._validate_input_growingspheres(model, 0.1, 'a', None)
    assert str(exin.value) == increment_std_msg

    with pytest.raises(TypeError) as exin:
        fuda._validate_input_growingspheres(model, 0.1, 0.1, 'a')
    assert str(exin.value) == minimum_per_class_msg

    with pytest.raises(ValueError) as exin:
        fuda._validate_input_growingspheres(model, 0.1, 0.1, -0.1)
    assert str(exin.value) == minimum_per_class_value_msg

    with pytest.raises(ValueError) as exin:
        fuda._validate_input_growingspheres(model, 0.1, 0.1, 1.1)
    assert str(exin.value) == minimum_per_class_value_msg

    with pytest.raises(ValueError) as exin:
        fuda._validate_input_growingspheres(model, -0.1, 0.1, 0.1)
    assert str(exin.value) == starting_std_value_msg

    with pytest.raises(ValueError) as exin:
        fuda._validate_input_growingspheres(model, 0.1, -0.1, 0.1)
    assert str(exin.value) == increment_std_value_msg
    

class TestGrowingSpheres(object):
    """
    Tests :class:`fatf.utils.data.augmentation.GrowingSpheres` class.
    """
    # Construct new dataset made up of 3 Gaussian clouds in 2-dimensions
    #dataset = np.vstack([
    #    np.random.normal(0, 1, (10, 2)), np.random.normal(2, 1, (10, 2)),
    #    np.random.normal(4, 1, (10, 2))])
    numerical_labels = np.array([0, 1, 0, 1, 1, 0])

    numerical_classifier = fum.KNN(k=3)
    numerical_classifier.fit(NUMERICAL_NP_ARRAY, numerical_labels)
    numerical_struct_classifier = fum.KNN(k=3)
    numerical_struct_classifier.fit(NUMERICAL_STRUCT_ARRAY, numerical_labels)
    categorical_classifier = fum.KNN(k=3)
    categorical_classifier.fit(CATEGORICAL_NP_ARRAY, numerical_labels)
    categorical_struct_classifier = fum.KNN(k=3)
    categorical_struct_classifier.fit(CATEGORICAL_STRUCT_ARRAY,
                                      numerical_labels)
    mixed_classifier = fum.KNN(k=3)
    mixed_classifier.fit(MIXED_ARRAY, numerical_labels)
    numerical_np_0_augmentor = fuda.GrowingSpheres(
        NUMERICAL_NP_ARRAY, numerical_classifier, [0])
    numerical_np_augmentor = fuda.GrowingSpheres(
        NUMERICAL_NP_ARRAY, numerical_classifier)
    numerical_struct_a_augmentor = fuda.GrowingSpheres(
        NUMERICAL_STRUCT_ARRAY, numerical_struct_classifier, ['a'])
    numerical_struct_augmentor = fuda.GrowingSpheres(
        NUMERICAL_STRUCT_ARRAY, numerical_struct_classifier)
    numerical_struct_augmentor_f = fuda.GrowingSpheres(
        NUMERICAL_STRUCT_ARRAY, numerical_struct_classifier, int_to_float=False)
    categorical_np_augmentor = fuda.GrowingSpheres(
        CATEGORICAL_NP_ARRAY, categorical_classifier)
    categorical_np_012_augmentor = fuda.GrowingSpheres(
        CATEGORICAL_NP_ARRAY, categorical_classifier, [0, 1, 2])
    categorical_struct_abc_augmentor = fuda.GrowingSpheres(
        CATEGORICAL_STRUCT_ARRAY, categorical_struct_classifier, ['a', 'b', 'c'])
    mixed_augmentor = fuda.GrowingSpheres(
        MIXED_ARRAY, mixed_classifier, ['b', 'd'])

    def test_init(self):
        """
        Tests :class:`fatf.utils.data.augmentation.GrowingSpheres` class init.
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
        
        # Test attributes unique to GrowingSpheres
        assert self.numerical_np_0_augmentor

    def test_sample(self):
        """
        Tests :func:`~fatf.utils.data.augmentation.GrowingSpheres.sample`.
        """
        fatf.setup_random_seed()
        max_iter_msg = 'max_iter is not a positive integer.'
        runtime_msg = ('Maximum iterations reached. Try increasing '
                       'max_iter or decreasing minimum_per_class.')

        with pytest.raises(TypeError) as exin:
            self.numerical_np_0_augmentor.sample(NUMERICAL_NP_ARRAY[0],
                                                 max_iter='a')
        assert str(exin.value) == max_iter_msg

        with pytest.raises(TypeError) as exin:
            self.numerical_np_0_augmentor.sample(NUMERICAL_NP_ARRAY[0],
                                                 max_iter=-1)
        assert str(exin.value) == max_iter_msg

        numerical_samples = np.array([
            [-0.014, 0.022, 0.071, 0.680],
            [ 0.004, -0.003, 0.082, 0.685],
            [-0.013, 0.001, 0.062, 0.680],
            [-0.020, -0.021, 0.086, 0.722]])

        numerical_0_samples = np.array([
            [ 2., 0.002, 0.079, 0.711],
            [ 0., 0.062, 0.049, 0.665],
            [ 0., -0.005, 0.109, 0.728],
            [ 0., 0.008, 0.099, 0.652]])

        numerical_struct_samples = np.array(
            [(-0.041, 0.015, 0.110, 0.664),
             (-0.028, 0.119, 0.100, 0.682),
             (0.015, 0.017, 0.069, 0.673),
             (0.015, 0.035, 0.104, 0.692)],
            dtype=[('a', 'f'), ('b', 'f'), ('c', 'f'), ('d', 'f')])

        numerical_struct_0_samples = np.array(
            [(0, 0.057, 0.067, 0.640),
             (1, 0.003, 0.128, 0.724),
             (1, 0.100, 0.112, 0.743),
             (1, 0.086, 0.148, 0.600)],
             dtype=[('a', 'i'), ('b', 'f'), ('c', 'f'), ('d', 'f')])

        categorical_samples = np.array([
            ['a', 'f', 'g'],
            ['a', 'f', 'g'],
            ['b', 'c', 'c'],
            ['a', 'f', 'g']])

        categorical_012_samples = np.array([
            ['a', 'f', 'c'],
            ['a', 'c', 'c'],
            ['a', 'f', 'c'],
            ['a', 'b', 'c']])
        
        categorical_struct_samples = np.array(
            [('a', 'b', 'c'), 
             ('a', 'c', 'g'),
             ('a', 'f', 'c'),
             ('a', 'f', 'c')],
             dtype=CATEGORICAL_STRUCT_ARRAY.dtype)
        
        mixed_samples = np.array(
            [(-0.002, 'f', 0.098, 'bb'),
             (-0.023, 'f', 0.077, 'bb'),
             (-0.015, 'a', 0.086, 'a'),
             (0.013, 'a', 0.083, 'a')],
            dtype=[('a', '<f8'), ('b', 'U1'), ('c', '<f8'), ('d', 'U2')])

        numerical_samples_mean = np.array([
            [0.445, 0.073, 0.411, 1.256],
            [0.106, 0.315, 0.913, 0.601],
            [0.111, 0.369, 0.611, 1.284],
            [0.302, 0.248, 0.618, 1.393]])

        numerical_0_samples_mean = np.array([
            [0., 0.318, 0.712, 1.059],
            [1., 0.336, 0.705, 1.082],
            [0., 0.329, 0.698, 1.066],
            [0., 0.332, 0.690, 1.030]])

        numerical_struct_samples_mean = np.array(
            [(0.001, -0.003, 0.381, 0.529),
             (0.005, 0.028, 0.374, 0.539),
             (-0.002, 0.009, 0.358, 0.594),
             (0.021, -0.030, 0.334, 0.554)],
            dtype=[('a', 'f'), ('b', 'f'), ('c', 'f'), ('d', 'f')])

        numerical_struct_0_samples_mean = np.array(
            [(0, 0.015, 0.334, 0.581),
             (2, 0.035, 0.380, 0.502),
             (0, 0.024, 0.485, 0.510),
             (0, 0.004, 0.396, 0.565)],
             dtype=[('a', 'i'), ('b', 'f'), ('c', 'f'), ('d', 'f')])

        categorical_samples_mean = np.array([
            ['a', 'b', 'c'],
            ['b', 'f', 'g'],
            ['b', 'c', 'c'],
            ['b', 'c', 'g']])

        categorical_012_samples_mean = np.array([
            ['a', 'c', 'g'],
            ['a', 'b', 'c'],
            ['b', 'f', 'g'],
            ['a', 'b', 'c']])
        
        categorical_struct_samples_mean = np.array(
            [('b', 'f', 'c'), 
             ('a', 'b', 'c'),
             ('a', 'f', 'c'),
             ('a', 'b', 'g')],
             dtype=CATEGORICAL_STRUCT_ARRAY.dtype)
        
        mixed_samples_mean = np.array(
            [(-0.002, 'a', 0.387, 'aa'),
             (-0.010, 'c', 0.373, 'b'),
             (0.003, 'c', 0.362, 'b'),
             (0.003, 'f', 0.374, 'aa')],
            dtype=[('a', '<f8'), ('b', 'U1'), ('c', '<f8'), ('d', 'U2')])

        samples = self.numerical_np_augmentor.sample(
            NUMERICAL_NP_ARRAY[0], samples_number=4)
        assert np.allclose(samples, numerical_samples, atol=1e-2)

        samples = self.numerical_np_0_augmentor.sample(
            NUMERICAL_NP_ARRAY[0], samples_number=4)
        assert np.allclose(samples, numerical_0_samples, atol=1e-2)

        samples = self.numerical_struct_augmentor.sample(
            NUMERICAL_STRUCT_ARRAY[0], samples_number=4)
        for i in samples.dtype.names:
            assert np.allclose(
                samples[i], numerical_struct_samples[i], atol=1e-2)

        samples = self.numerical_struct_a_augmentor.sample(
            NUMERICAL_STRUCT_ARRAY[0], samples_number=4)
        for i in samples.dtype.names:
            assert np.allclose(
                samples[i], numerical_struct_0_samples[i], atol=1e-2)

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

        samples = self.mixed_augmentor.sample(
            MIXED_ARRAY[0], samples_number=4)
        assert np.array_equal(samples[['b', 'd']], mixed_samples[['b', 'd']])
        for i in ['a', 'c']:
            assert np.allclose(samples[i], mixed_samples[i], atol=1e-2)

        # Test if minimum_per_class works
        samples = self.numerical_np_augmentor.sample(
            NUMERICAL_NP_ARRAY[0], samples_number=1000)
        predictions = self.numerical_classifier.predict(samples)
        unique, counts = np.unique(predictions, return_counts=True)
        assert np.all(counts > 0.05*1000)

        samples = self.numerical_np_0_augmentor.sample(
            NUMERICAL_NP_ARRAY[0], samples_number=1000)
        predictions = self.numerical_classifier.predict(samples)
        unique, counts = np.unique(predictions, return_counts=True)
        assert np.all(counts > 0.05*1000)

        samples = self.numerical_struct_augmentor.sample(
            NUMERICAL_STRUCT_ARRAY[0], samples_number=1000)
        predictions = self.numerical_struct_classifier.predict(samples)
        unique, counts = np.unique(predictions, return_counts=True)
        assert np.all(counts > 0.05*1000)

        samples = self.numerical_struct_a_augmentor.sample(
            NUMERICAL_STRUCT_ARRAY[0], samples_number=1000)
        predictions = self.numerical_struct_classifier.predict(samples)
        unique, counts = np.unique(predictions, return_counts=True)
        assert np.all(counts > 0.05*1000)

        samples = self.categorical_np_augmentor.sample(
            CATEGORICAL_NP_ARRAY[0], samples_number=1000)
        predictions = self.categorical_classifier.predict(samples)
        unique, counts = np.unique(predictions, return_counts=True)
        assert np.all(counts > 0.05*1000)
        vals = [['a', 'b'], ['b', 'c', 'f'], ['c', 'g']]
        proportions = [
            np.array([0.62, 0.38]),
            np.array([0.31, 0.17, 0.52]),
            np.array([0.63, 0.37])
        ]
        for i, index in enumerate([0, 1, 2]):
            val, freq = np.unique(samples[:, index], return_counts=True)
            freq = freq / freq.sum()
            assert np.array_equal(val, vals[i])
            assert np.allclose(freq, proportions[i], atol=1e-1)

        samples = self.categorical_np_012_augmentor.sample(
            CATEGORICAL_NP_ARRAY[0], samples_number=1000)
        predictions = self.categorical_classifier.predict(samples)
        unique, counts = np.unique(predictions, return_counts=True)
        assert np.all(counts > 0.05*1000)
        for i, index in enumerate([0, 1, 2]):
            val, freq = np.unique(samples[:, index], return_counts=True)
            freq = freq / freq.sum()
            assert np.array_equal(val, vals[i])
            assert np.allclose(freq, proportions[i], atol=1e-1)

        samples = self.categorical_struct_abc_augmentor.sample(
            CATEGORICAL_STRUCT_ARRAY[0], samples_number=1000)
        predictions = self.categorical_struct_classifier.predict(samples)
        unique, counts = np.unique(predictions, return_counts=True)
        assert np.all(counts > 0.05*1000)
        proportions = [
            np.array([0.74, 0.26]),
            np.array([0.38, 0.12, 0.50]),
            np.array([0.63, 0.37])
        ]
        for i, index in enumerate(['a', 'b', 'c']):
            val, freq = np.unique(samples[index], return_counts=True)
            freq = freq / freq.sum()
            assert np.array_equal(val, vals[i])
            assert np.allclose(freq, proportions[i], atol=1e-1)

        #######################################################################

        # Sample without float cast
        samples = self.numerical_struct_augmentor_f.sample(
            NUMERICAL_STRUCT_ARRAY[0], samples_number=5)
        samples_answer = np.array(
            [(0, 0, 0.087, 0.708),
             (0, 0, -0.022, 0.624),
             (0, 0, 0.140, 0.789),
             (0, 0, 0.052, 0.695),
             (0, 0, 0.081, 0.676)],
            dtype=NUMERICAL_STRUCT_ARRAY.dtype)  # yapf: disable
        for i in ['a', 'b', 'c', 'd']:
            assert np.allclose(samples[i], samples_answer[i], atol=1e-3)

        # Test if max_iter is too low to find all classes
        with pytest.raises(RuntimeError) as exin:
            self.numerical_np_0_augmentor.sample(NUMERICAL_NP_ARRAY[0],
                                                 max_iter=1)
        assert str(exin.value) == runtime_msg

        #######################################################################

        # Test with mean of dataset as starting point
        samples = self.numerical_np_augmentor.sample(samples_number=4)
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        assert np.allclose(samples, numerical_samples_mean, atol=1e-2)

        samples = self.numerical_np_0_augmentor.sample(samples_number=4)
        assert np.allclose(samples, numerical_0_samples_mean, atol=1e-2)

        samples = self.numerical_struct_augmentor.sample(samples_number=4)
        for i in samples.dtype.names:
            assert np.allclose(
                samples[i], numerical_struct_samples_mean[i], atol=1e-2)

        samples = self.numerical_struct_a_augmentor.sample(samples_number=4)
        for i in samples.dtype.names:
            assert np.allclose(
                samples[i], numerical_struct_0_samples_mean[i], atol=1e-2)

        samples = self.categorical_np_augmentor.sample(samples_number=4)
        assert np.array_equal(samples, categorical_samples_mean)
        
        samples = self.categorical_np_012_augmentor.sample(samples_number=4)
        assert np.array_equal(samples, categorical_012_samples_mean)

        samples = self.categorical_struct_abc_augmentor.sample(samples_number=4)
        for i in samples.dtype.names:
            assert np.array_equal(
                samples[i], categorical_struct_samples_mean[i])

        samples = self.mixed_augmentor.sample(samples_number=4)
        assert np.array_equal(samples[['b', 'd']], 
                              mixed_samples_mean[['b', 'd']])
        for i in ['a', 'c']:
            assert np.allclose(samples[i], mixed_samples_mean[i], atol=1e-2)
        
        # Test if minimum_per_class works with mean of dataset
        samples = self.numerical_np_augmentor.sample(samples_number=1000)
        predictions = self.numerical_classifier.predict(samples)
        unique, counts = np.unique(predictions, return_counts=True)
        assert np.all(counts > 0.05*1000)

        samples = self.numerical_np_0_augmentor.sample(samples_number=1000)
        predictions = self.numerical_classifier.predict(samples)
        unique, counts = np.unique(predictions, return_counts=True)
        assert np.all(counts > 0.05*1000)

        samples = self.numerical_struct_augmentor.sample(samples_number=1000)
        predictions = self.numerical_struct_classifier.predict(samples)
        unique, counts = np.unique(predictions, return_counts=True)
        assert np.all(counts > 0.05*1000)

        samples = self.numerical_struct_a_augmentor.sample(samples_number=1000)
        predictions = self.numerical_struct_classifier.predict(samples)
        unique, counts = np.unique(predictions, return_counts=True)
        assert np.all(counts > 0.05*1000)

        samples = self.categorical_np_augmentor.sample(samples_number=1000)
        predictions = self.categorical_classifier.predict(samples)
        unique, counts = np.unique(predictions, return_counts=True)
        assert np.all(counts > 0.05*1000)
        vals = [['a', 'b'], ['b', 'c', 'f'], ['c', 'g']]
        proportions = [
            np.array([0.62, 0.38]),
            np.array([0.31, 0.17, 0.52]),
            np.array([0.63, 0.37])
        ]
        for i, index in enumerate([0, 1, 2]):
            val, freq = np.unique(samples[:, index], return_counts=True)
            freq = freq / freq.sum()
            assert np.array_equal(val, vals[i])
            assert np.allclose(freq, proportions[i], atol=1e-1)

        samples = self.categorical_np_012_augmentor.sample(samples_number=1000)
        predictions = self.categorical_classifier.predict(samples)
        unique, counts = np.unique(predictions, return_counts=True)
        assert np.all(counts > 0.05*1000)
        for i, index in enumerate([0, 1, 2]):
            val, freq = np.unique(samples[:, index], return_counts=True)
            freq = freq / freq.sum()
            assert np.array_equal(val, vals[i])
            assert np.allclose(freq, proportions[i], atol=1e-1)

        samples = self.categorical_struct_abc_augmentor.sample(
            samples_number=1000)
        predictions = self.categorical_struct_classifier.predict(samples)
        unique, counts = np.unique(predictions, return_counts=True)
        assert np.all(counts > 0.05*1000)
        proportions = [
            np.array([0.74, 0.26]),
            np.array([0.38, 0.12, 0.50]),
            np.array([0.63, 0.37])
        ]
        for i, index in enumerate(['a', 'b', 'c']):
            val, freq = np.unique(samples[index], return_counts=True)
            freq = freq / freq.sum()
            assert np.array_equal(val, vals[i])
            assert np.allclose(freq, proportions[i], atol=1e-1)
