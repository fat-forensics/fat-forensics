"""
Tests the data discretisation module.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
#         Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import pytest

import numpy as np

import fatf.utils.data.discretisation as fudd

from fatf.exceptions import IncorrectShapeError

# yapf: disable
NUMERICAL_NP_ARRAY = np.array([
    [0, 0, 0.08, 0.69],
    [1, 0, 0.03, 0.29],
    [0, 1, 0.99, 0.82],
    [2, 1, 0.73, 0.48],
    [1, 0, 0.36, 0.89],
    [0, 1, 0.07, 0.21]])
NUMERICAL_3D_ARRAY = np.ones((2, 2, 2))
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
NUMERICAL_NP_CAT_DISCRETISED = np.array([
    [0, 0, 1, 2],
    [1, 0, 0, 0],
    [0, 2, 3, 3],
    [2, 2, 3, 1],
    [1, 0, 2, 3],
    [0, 2, 0, 0]])
NUMERICAL_STRUCT_CAT_DISCRETISED = np.array([
    (0, 0, 1, 2),
    (1, 0, 0, 0),
    (0, 2, 3, 3),
    (2, 2, 3, 1),
    (1, 0, 2, 3),
    (0, 2, 0, 0)],
    dtype=[('a', 'i'), ('b', 'i1'), ('c', 'i1'), ('d', 'i1')])
MIXED_DISCRETISED = np.array(
    [(0, 'a', 1, 'a'),
     (0, 'f', 0, 'bb'),
     (2, 'c', 3, 'aa'),
     (2, 'a', 3, 'a'),
     (0, 'c', 2, 'b'),
     (2, 'f', 0, 'bb')],
    dtype=[('a', 'i1'), ('b', 'U1'), ('c', 'i1'), ('d', 'U2')])
# yapf: enable


def test_validate_input_discretiser():
    """
    Tests :func:`fatf.utils.data.discretisation._validate_input_discretiser`.
    """
    data_incorrect_shape = ('The input dataset must be a 2-dimensional numpy '
                            'array.')
    data_type_error = 'The input dataset must be of a base type.'
    #
    cidx_index_error = ('The following indices are invalid for the input '
                        'dataset: {}.')
    cidx_type_error = ('The categorical_indices parameter must be a Python '
                       'list or None.')
    #
    feature_names_type_error = ('The feature_names parameter must be a Python '
                                'list or None.')
    feature_name_type_error = ('All of the feature_names must be strings. The '
                               '*{}* feature name is not a string.')
    feature_names_incorrect_shape = ('The length of feature_names list must '
                                     'be equal to the number of features '
                                     '(columns) in the input dataset.')

    with pytest.raises(IncorrectShapeError) as exin:
        fudd._validate_input_discretiser(np.array([0, 4, 3, 0]))
    assert str(exin.value) == data_incorrect_shape
    #
    with pytest.raises(TypeError) as exin:
        fudd._validate_input_discretiser(np.array([[0, 4], [None, 0]]))
    assert str(exin.value) == data_type_error

    with pytest.raises(TypeError) as exin:
        fudd._validate_input_discretiser(
            NUMERICAL_NP_ARRAY, categorical_indices=0)
    assert str(exin.value) == cidx_type_error
    #
    with pytest.raises(IndexError) as exin:
        fudd._validate_input_discretiser(
            MIXED_ARRAY, categorical_indices=['f'])
    assert str(exin.value) == cidx_index_error.format(['f'])
    with pytest.raises(IndexError) as exin:
        fudd._validate_input_discretiser(MIXED_ARRAY, categorical_indices=[1])
    assert str(exin.value) == cidx_index_error.format([1])

    with pytest.raises(TypeError) as exin:
        fudd._validate_input_discretiser(NUMERICAL_NP_ARRAY, feature_names='a')
    assert str(exin.value) == feature_names_type_error
    #
    with pytest.raises(ValueError) as exin:
        fudd._validate_input_discretiser(NUMERICAL_NP_ARRAY, feature_names=[1])
    assert str(exin.value) == feature_names_incorrect_shape
    #
    with pytest.raises(TypeError) as exin:
        fudd._validate_input_discretiser(
            NUMERICAL_NP_ARRAY, feature_names=[0, 1, 2, 'a'])
    assert str(exin.value) == feature_name_type_error.format(0)
    #
    with pytest.raises(TypeError) as exin:
        fudd._validate_input_discretiser(
            MIXED_ARRAY, feature_names=['a', 'b', 3, 'd'])
    assert str(exin.value) == feature_name_type_error.format(3)

    assert fudd._validate_input_discretiser(
        MIXED_ARRAY, categorical_indices=['a', 'b'])


class TestDiscretiser():
    """
    Tests :class:`fatf.utils.data.discretisation.Discretiser` abstract class.
    """

    class BrokenDiscretiser(fudd.Discretiser):
        """
        A broken -- no ``sample`` method -- data discretiser.
        """

    class BaseDiscretiser(fudd.Discretiser):
        """
        A dummy data discretiser implementation.

        Tests
        :func:`fatf.utils.data.discretisation._validate_input_discretiser`
        function and
        :func:`fatf.utils.data.discretisation.Discretiser.\
_validate_input_discretise` method.
        """

        def discretise(self, data):
            """
            Dummy discretise method.
            """
            self._validate_input_discretise(data)
            return np.ones(data.shape)

    def test_discretiser_class_init(self):
        """
        Tests :class:`fatf.utils.data.discretisation.Discretiser`
        class init.
        """
        abstract_method_error = ("Can't instantiate abstract class "
                                 '{} with abstract methods discretise')
        user_warning = (
            'Some of the string-based columns in the input dataset were not '
            'selected as categorical features via the categorical_indices '
            'parameter. String-based columns cannot be treated as numerical '
            'features, therefore they will be also treated as categorical '
            'features (in addition to the ones selected with the '
            'categorical_indices parameter).')

        with pytest.raises(TypeError) as exin:
            fudd.Discretiser(NUMERICAL_NP_ARRAY)
        assert str(exin.value) == abstract_method_error.format('Discretiser')

        with pytest.raises(TypeError) as exin:
            self.BrokenDiscretiser(NUMERICAL_NP_ARRAY)
        msg = abstract_method_error.format('BrokenDiscretiser')
        assert str(exin.value) == msg

        # Test for a categorical index warning
        with pytest.warns(UserWarning) as warning:
            discretiser = self.BaseDiscretiser(CATEGORICAL_NP_ARRAY, [0])
        assert len(warning) == 1
        assert str(warning[0].message) == user_warning
        assert np.array_equal(discretiser.categorical_indices, [0, 1, 2])
        #
        with pytest.warns(UserWarning) as warning:
            discretiser = self.BaseDiscretiser(CATEGORICAL_STRUCT_ARRAY, ['a'])
        assert len(warning) == 1
        assert str(warning[0].message) == user_warning
        assert np.array_equal(discretiser.categorical_indices,
                              np.array(['a', 'b', 'c']))
        #
        with pytest.warns(UserWarning) as warning:
            discretiser = self.BaseDiscretiser(MIXED_ARRAY, ['b'])
        assert len(warning) == 1
        assert str(warning[0].message) == user_warning
        assert np.array_equal(discretiser.categorical_indices, ['b', 'd'])

        # Validate internal variables
        categorical_np_discretiser = self.BaseDiscretiser(
            CATEGORICAL_NP_ARRAY, feature_names=['aa', 'bb', 'cc'])
        assert categorical_np_discretiser.dataset_dtype == np.dtype('<U1')
        assert not categorical_np_discretiser.is_structured
        assert categorical_np_discretiser.features_number == 3
        assert categorical_np_discretiser.categorical_indices == [0, 1, 2]
        assert categorical_np_discretiser.numerical_indices == []
        assert (categorical_np_discretiser.feature_names_map
                == dict(zip([0, 1, 2], ['aa', 'bb', 'cc'])))  # yapf: disable
        assert categorical_np_discretiser.feature_value_names == {}
        assert categorical_np_discretiser.feature_bin_boundaries == {}

        categorical_struct_discretiser = self.BaseDiscretiser(
            CATEGORICAL_STRUCT_ARRAY)
        assert (
            categorical_struct_discretiser.dataset_dtype
            == np.dtype([('a', 'U1'), ('b', 'U1'), ('c', 'U1')])
        )  # yapf: disable
        assert categorical_struct_discretiser.is_structured
        assert categorical_struct_discretiser.features_number == 3
        assert (categorical_struct_discretiser.categorical_indices
                == ['a', 'b', 'c'])  # yapf: disable
        assert categorical_struct_discretiser.numerical_indices == []
        assert (
            categorical_struct_discretiser.feature_names_map
            == dict(zip(['a', 'b', 'c'], ['a', 'b', 'c']))
        )  # yapf: disable
        assert categorical_struct_discretiser.feature_value_names == {}
        assert categorical_struct_discretiser.feature_bin_boundaries == {}

        mixed_discretiser = self.BaseDiscretiser(MIXED_ARRAY)
        assert (
            mixed_discretiser.dataset_dtype
            == np.dtype([('a', 'i'), ('b', 'U1'), ('c', 'f'), ('d', 'U2')])
        )  # yapf: disable
        assert mixed_discretiser.is_structured
        assert mixed_discretiser.features_number == 4
        assert mixed_discretiser.categorical_indices == ['b', 'd']
        assert mixed_discretiser.numerical_indices == ['a', 'c']
        assert (
            mixed_discretiser.feature_names_map
            == dict(zip(['a', 'b', 'c', 'd'], ['a', 'b', 'c', 'd']))
        )  # yapf: disable
        assert mixed_discretiser.feature_value_names == {}
        assert mixed_discretiser.feature_bin_boundaries == {}

        numerical_np_discretiser = self.BaseDiscretiser(
            NUMERICAL_NP_ARRAY, [0, 1], feature_names=['a', 'b', 'c', 'd'])
        assert numerical_np_discretiser.dataset_dtype == np.dtype('float64')
        assert not numerical_np_discretiser.is_structured
        assert numerical_np_discretiser.features_number == 4
        assert numerical_np_discretiser.categorical_indices == [0, 1]
        assert numerical_np_discretiser.numerical_indices == [2, 3]
        assert (
            numerical_np_discretiser.feature_names_map
            == dict(zip([0, 1, 2, 3], ['a', 'b', 'c', 'd']))
        )  # yapf: disable
        assert numerical_np_discretiser.feature_value_names == {}
        assert numerical_np_discretiser.feature_bin_boundaries == {}

    def test_validate_input_discretise(self):
        """
        Tests ``_validate_input_discretise`` method.

        This function test validation of the input for the
        :func:`fatf.utils.data.discretisation.Discretiser.\
_validate_input_discretise` method.
        """
        incorrect_shape_data_row = ('The dataset must be either a '
                                    '1-dimensional (a plane numpy array or '
                                    'numpy void for structured 1-dimensional '
                                    'arrays) or a 2-dimensional array.')
        type_error_data_row = ('The dtype of the input dataset is too '
                               'different from the dtype of the dataset used '
                               'to initialise this class.')
        incorrect_shape_features = ('The input dataset must contain the same '
                                    'number of features as the dataset used '
                                    'to initialise this class.')

        # Validate sample input rows
        numerical_np_discretiser = self.BaseDiscretiser(NUMERICAL_NP_ARRAY)
        categorical_np_discretiser = self.BaseDiscretiser(CATEGORICAL_NP_ARRAY)
        numerical_struct_discretiser = self.BaseDiscretiser(
            NUMERICAL_STRUCT_ARRAY)

        # Data row shape
        with pytest.raises(IncorrectShapeError) as exin:
            numerical_np_discretiser.discretise(NUMERICAL_3D_ARRAY)
        assert str(exin.value) == incorrect_shape_data_row

        # Data type
        with pytest.raises(TypeError) as exin:
            numerical_np_discretiser.discretise(np.array(['a', 'b', 'c', 'd']))
        assert str(exin.value) == type_error_data_row
        #
        with pytest.raises(TypeError) as exin:
            numerical_struct_discretiser.discretise(MIXED_ARRAY[0])
        assert str(exin.value) == type_error_data_row
        #
        with pytest.raises(TypeError) as exin:
            categorical_np_discretiser.discretise(np.array([0.1]))
        assert str(exin.value) == type_error_data_row
        # Structured too short
        with pytest.raises(TypeError) as exin:
            numerical_struct_discretiser.discretise(MIXED_ARRAY[['a', 'b']][0])
        assert str(exin.value) == type_error_data_row

        # Data features number
        with pytest.raises(IncorrectShapeError) as exin:
            numerical_np_discretiser.discretise(np.array([0.1, 1, 2]))
        assert str(exin.value) == incorrect_shape_features
        #
        with pytest.raises(IncorrectShapeError) as exin:
            categorical_np_discretiser.discretise(np.array(['a', 'b']))
        assert str(exin.value) == incorrect_shape_features
        #
        with pytest.raises(IncorrectShapeError) as exin:
            numerical_np_discretiser.discretise(
                np.array([[0.1, 1, 2], [0.1, 1, 2]]))
        assert str(exin.value) == incorrect_shape_features
        #
        with pytest.raises(IncorrectShapeError) as exin:
            categorical_np_discretiser.discretise(
                np.array([['a', 'b'], ['a', 'b']]))
        assert str(exin.value) == incorrect_shape_features

        # All OK
        assert np.array_equal(
            numerical_np_discretiser.discretise(NUMERICAL_NP_ARRAY[0, :]),
            np.ones(NUMERICAL_NP_ARRAY[0, :].shape))
        assert np.array_equal(
            numerical_np_discretiser.discretise(NUMERICAL_NP_ARRAY),
            np.ones(NUMERICAL_NP_ARRAY.shape))
        assert np.array_equal(
            categorical_np_discretiser.discretise(CATEGORICAL_NP_ARRAY[0, :]),
            np.ones(CATEGORICAL_NP_ARRAY[0, :].shape))
        assert np.array_equal(
            categorical_np_discretiser.discretise(CATEGORICAL_NP_ARRAY),
            np.ones(CATEGORICAL_NP_ARRAY.shape))


class TestQuartileDiscretiser(object):
    """
    Tests :class:`fatf.utils.data.discretisation.QuartileDiscretiser` class.
    """
    numerical_np_discretiser = fudd.QuartileDiscretiser(
        NUMERICAL_NP_ARRAY, [0])
    numerical_np_discretiser_full = fudd.QuartileDiscretiser(
        NUMERICAL_NP_ARRAY[:, 1:], [])
    numerical_struct_discretiser = fudd.QuartileDiscretiser(
        NUMERICAL_STRUCT_ARRAY, ['a'])

    categorical_np_discretiser = fudd.QuartileDiscretiser(
        CATEGORICAL_NP_ARRAY, [0, 1, 2])
    categorical_struct_discretiser = fudd.QuartileDiscretiser(
        CATEGORICAL_STRUCT_ARRAY, ['a', 'b', 'c'])

    mixed_struct_discretiser = fudd.QuartileDiscretiser(
        MIXED_ARRAY, ['b', 'd'])

    def test_init(self):
        """
        Tests initialisation of the ``QuartileDiscretiser`` class.

        This method tests for
        :class:`fatf.utils.data.discretisation.QuartileDiscretiser` class
        initialisation.
        """
        correct_feature_names = {
            1: {
                0: '*1* <= 0.00',
                1: '0.00 < *1* <= 0.50',
                2: '0.50 < *1* <= 1.00',
                3: '1.00 < *1*'
            },
            2: {
                0: '*2* <= 0.07',
                1: '0.07 < *2* <= 0.22',
                2: '0.22 < *2* <= 0.64',
                3: '0.64 < *2*'
            },
            3: {
                0: '*3* <= 0.34',
                1: '0.34 < *3* <= 0.58',
                2: '0.58 < *3* <= 0.79',
                3: '0.79 < *3*'
            }
        }
        correct_feature_names_full = {
            0: {
                0: '*0* <= 0.00',
                1: '0.00 < *0* <= 0.50',
                2: '0.50 < *0* <= 1.00',
                3: '1.00 < *0*'
            },
            1: {
                0: '*1* <= 0.07',
                1: '0.07 < *1* <= 0.22',
                2: '0.22 < *1* <= 0.64',
                3: '0.64 < *1*'
            },
            2: {
                0: '*2* <= 0.34',
                1: '0.34 < *2* <= 0.58',
                2: '0.58 < *2* <= 0.79',
                3: '0.79 < *2*'
            }
        }
        correct_feature_names_struct = {
            'b': {
                0: '*b* <= 0.00',
                1: '0.00 < *b* <= 0.50',
                2: '0.50 < *b* <= 1.00',
                3: '1.00 < *b*'
            },
            'c': {
                0: '*c* <= 0.07',
                1: '0.07 < *c* <= 0.22',
                2: '0.22 < *c* <= 0.64',
                3: '0.64 < *c*'
            },
            'd': {
                0: '*d* <= 0.34',
                1: '0.34 < *d* <= 0.58',
                2: '0.58 < *d* <= 0.79',
                3: '0.79 < *d*'
            }
        }
        correct_feature_names_mixed = {
            'a': {
                0: '*a* <= 0.00',
                1: '0.00 < *a* <= 0.50',
                2: '0.50 < *a* <= 1.00',
                3: '1.00 < *a*'
            },
            'c': {
                0: '*c* <= 0.07',
                1: '0.07 < *c* <= 0.22',
                2: '0.22 < *c* <= 0.64',
                3: '0.64 < *c*'
            }
        }
        bins = [
            np.array([0, 0.5, 1]),
            np.array([0.07, 0.22, 0.64]),
            np.array([0.34, 0.58, 0.79])
        ]

        # Test calculating numerical and categorical indices and overwriting
        # the dictionaries
        assert self.numerical_np_discretiser.categorical_indices == [0]
        assert self.numerical_np_discretiser.numerical_indices == [1, 2, 3]
        assert (self.numerical_np_discretiser.feature_value_names
                == correct_feature_names)  # yapf: disable
        keys = sorted(correct_feature_names.keys())
        for i, key in enumerate(keys):
            assert np.allclose(
                self.numerical_np_discretiser.feature_bin_boundaries[key],
                bins[i],
                atol=1e-2)
        assert (self.numerical_np_discretiser.discretised_dtype
                == np.dtype('float64'))  # yapf: disable

        assert self.numerical_struct_discretiser.categorical_indices == ['a']
        assert (self.numerical_struct_discretiser.numerical_indices
                == ['b', 'c', 'd'])  # yapf: disable
        assert (self.numerical_struct_discretiser.feature_value_names
                == correct_feature_names_struct)  # yapf: disable
        keys = sorted(correct_feature_names_struct.keys())
        for i, key in enumerate(keys):
            assert np.allclose(
                self.numerical_struct_discretiser.feature_bin_boundaries[key],
                bins[i],
                atol=1e-2)
        assert (
            self.numerical_struct_discretiser.discretised_dtype
            == np.dtype([('a', 'i'), ('b', 'i1'), ('c', 'i1'), ('d', 'i1')])
        )  # yapf: disable

        assert (self.categorical_np_discretiser.categorical_indices
                == [0, 1, 2])  # yapf: disable
        assert self.categorical_np_discretiser.numerical_indices == []
        assert self.categorical_np_discretiser.feature_value_names == {}
        assert self.categorical_np_discretiser.feature_bin_boundaries == {}
        assert (self.categorical_np_discretiser.discretised_dtype
                == np.dtype('U1'))  # yapf: disable

        assert self.numerical_np_discretiser_full.categorical_indices == []
        assert (self.numerical_np_discretiser_full.numerical_indices
                == [0, 1, 2])  # yapf: disable
        assert (self.numerical_np_discretiser_full.feature_value_names
                == correct_feature_names_full)  # yapf: disable
        keys = sorted(correct_feature_names.keys())
        for i, key in enumerate(keys):
            fky = key - 1
            assert np.allclose(
                self.numerical_np_discretiser_full.feature_bin_boundaries[fky],
                bins[i],
                atol=1e-2)
        assert (self.numerical_np_discretiser_full.discretised_dtype
                == np.dtype('int8'))  # yapf: disable

        assert (self.categorical_struct_discretiser.categorical_indices
                == ['a', 'b', 'c'])  # yapf: disable
        assert self.categorical_struct_discretiser.numerical_indices == []
        assert self.categorical_struct_discretiser.feature_value_names == {}
        assert self.categorical_struct_discretiser.feature_bin_boundaries == {}
        assert (
            self.categorical_struct_discretiser.discretised_dtype
            == np.dtype([('a', 'U1'), ('b', 'U1'), ('c', 'U1')])
        )  # yapf: disable

        assert self.mixed_struct_discretiser.categorical_indices == ['b', 'd']
        assert self.mixed_struct_discretiser.numerical_indices == ['a', 'c']
        assert (self.mixed_struct_discretiser.feature_value_names
                == correct_feature_names_mixed)  # yapf: disable
        keys = sorted(correct_feature_names_mixed.keys())
        for i, key in enumerate(keys):
            assert np.allclose(
                self.mixed_struct_discretiser.feature_bin_boundaries[key],
                bins[i],
                atol=1e-2)
        assert (self.mixed_struct_discretiser.discretised_dtype == np.dtype(
            [('a', 'i1'), ('b', 'U1'), ('c', 'i1'), ('d', 'U2')]))

    def test_discretise(self):
        """
        Tests ``QuartileDiscretiser``\\ 's ``discretise`` method.

        This function tests
        :func:`fatf.utils.data.discretisation.QuartileDiscretiser.discretise`
        method.
        """
        discretised = self.numerical_np_discretiser.discretise(
            NUMERICAL_NP_ARRAY)
        assert np.array_equal(discretised, NUMERICAL_NP_CAT_DISCRETISED)
        discretised = self.numerical_np_discretiser.discretise(
            NUMERICAL_NP_ARRAY[0])
        assert np.array_equal(discretised, NUMERICAL_NP_CAT_DISCRETISED[0])

        discretised = self.numerical_struct_discretiser.discretise(
            NUMERICAL_STRUCT_ARRAY)
        assert np.array_equal(discretised, NUMERICAL_STRUCT_CAT_DISCRETISED)
        discretised = self.numerical_struct_discretiser.discretise(
            NUMERICAL_STRUCT_ARRAY[0])
        assert np.array_equal(discretised, NUMERICAL_STRUCT_CAT_DISCRETISED[0])
        assert isinstance(discretised, np.void)

        discretised = self.categorical_np_discretiser.discretise(
            CATEGORICAL_NP_ARRAY)
        assert np.array_equal(discretised, CATEGORICAL_NP_ARRAY)
        discretised = self.categorical_np_discretiser.discretise(
            CATEGORICAL_NP_ARRAY[0])
        assert np.array_equal(discretised, CATEGORICAL_NP_ARRAY[0])

        discretised = self.numerical_np_discretiser_full.discretise(
            NUMERICAL_NP_ARRAY[:, 1:])
        assert np.array_equal(discretised, NUMERICAL_NP_CAT_DISCRETISED[:, 1:])
        discretised = self.numerical_np_discretiser_full.discretise(
            NUMERICAL_NP_ARRAY[0, 1:])
        assert np.array_equal(discretised, NUMERICAL_NP_CAT_DISCRETISED[0, 1:])

        discretised = self.categorical_struct_discretiser.discretise(
            CATEGORICAL_STRUCT_ARRAY)
        assert np.array_equal(discretised, CATEGORICAL_STRUCT_ARRAY)
        discretised = self.categorical_struct_discretiser.discretise(
            CATEGORICAL_STRUCT_ARRAY[0])
        assert np.array_equal(discretised, CATEGORICAL_STRUCT_ARRAY[0])
        assert isinstance(discretised, np.void)

        discretised = self.mixed_struct_discretiser.discretise(MIXED_ARRAY)
        assert np.array_equal(discretised, MIXED_DISCRETISED)
