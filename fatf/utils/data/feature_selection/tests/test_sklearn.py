"""
This module tests scikit-learn-based feature selection functions.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
#         Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import pytest

try:
    import sklearn
except ImportError:  # pragma: no cover
    pytest.skip(
        'Skipping tests of scikit-learn-based feature selection functions '
        '-- scikit-learn is not installed.',
        allow_module_level=True)
else:
    del sklearn

import importlib
import sys

import numpy as np

import fatf
import fatf.utils.array.validation as fuav
import fatf.utils.data.feature_selection.sklearn as fudfs
import fatf.utils.testing.imports as futi

from fatf.exceptions import IncorrectShapeError

# yapf: disable
ONE_D_ARRAY = np.array([0, 4, 3, 0])
NUMERICAL_NP_ARRAY_TARGET = np.array([2, 0, 1, 1, 0, 2])
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
# yapf: enable

FEATURE_INDICES_WARNING = ('The selected number of features is larger '
                           'than the total number of features in the '
                           'dataset array. All of the features are being '
                           'selected.')
FEATURE_PERCENTAGE_LOG = (
    'Since the number of features to be extracted was not given 24% of '
    'features will be used. This percentage translates to 0 features, '
    'therefore the number of features to be used is overwritten to 1. To '
    'prevent this from happening, you should either explicitly set the '
    'number of features via the features_number parameter or increase the '
    'value of the features_percentage parameter.')


def test_sklearn_import():
    """
    Tests scikit-learn import in the module header.

    Tests ``sklearn`` import in the
    :mod:`fatf.utils.data.feature_selection.sklearn` module.
    """
    # When present
    assert 'fatf.utils.data.feature_selection.sklearn' in sys.modules
    with futi.module_import_tester('sklearn', when_missing=False):
        importlib.reload(fatf.utils.data.feature_selection.sklearn)
    assert 'fatf.utils.data.feature_selection.sklearn' in sys.modules

    # When missing
    assert 'fatf.utils.data.feature_selection.sklearn' in sys.modules
    exception_msg = (
        'scikit-learn (sklearn) Python module is not installed on your '
        'system. You must install it in order to use '
        'fatf.utils.data.feature_selection.sklearn functionality. '
        'One possibility is to install scikit-learn alongside this package '
        'via machine learning dependencies with: pip install '
        'fat-forensics[ml].')
    with futi.module_import_tester('sklearn', when_missing=True):
        with pytest.raises(ImportError) as exin:
            importlib.reload(fatf.utils.data.feature_selection.sklearn)
        assert str(exin.value) == exception_msg
    assert 'fatf.utils.data.feature_selection.sklearn' in sys.modules


def test_validate_input_lasso_path():
    """
    Tests ``_validate_input_lasso_path`` function.

    This function tests the
    :func:`fatf.utils.data.feature_choice.sklearn._validate_input_lasso_path`
    function.
    """
    dataset_shape_msg = 'The input data set must be a 2-dimensional array.'
    dataset_type_msg = ('The input data set must be purely numerical. (The '
                        'lasso path feature selection is based on '
                        'sklearn.linear_model.lars_path function.)')
    #
    target_shape_msg = 'The target array must be a 1-dimensional array.'
    target_shape_2_msg = ('The number of labels in the target array must '
                          'agree with the number of samples in the data set.')
    target_type_msg = ('The target array must be numerical since this feature '
                       'selection method is based on Lasso regression.')
    #
    weights_shape_msg = 'The weights array must 1-dimensional.'
    weights_shape_2_msg = ('The number of weights in the weights array must '
                           'be the same as the number of samples in the input '
                           'data set.')
    weights_type_msg = 'The weights array must be purely numerical.'
    #
    features_number_type_msg = ('The features_number parameter must be an '
                                'integer.')
    features_number_value_msg = ('The features_number parameter must be a '
                                 'positive integer.')
    #
    features_percentage_type_msg = ('The feature_percentage parameter must be '
                                    'an integer.')
    features_percentage_value_msg = ('The feature_percentage parameter must '
                                     'be between 0 and 100 (inclusive).')

    with pytest.raises(IncorrectShapeError) as exin:
        fudfs._validate_input_lasso_path(ONE_D_ARRAY, None, None, None, None)
    assert str(exin.value) == dataset_shape_msg
    #
    with pytest.raises(TypeError) as exin:
        fudfs._validate_input_lasso_path(
            np.array([[None, 0], [0, 1]]), None, None, None, None)
    assert str(exin.value) == dataset_type_msg
    #
    with pytest.raises(TypeError) as exin:
        fudfs._validate_input_lasso_path(CATEGORICAL_NP_ARRAY, None, None,
                                         None, None)
    assert str(exin.value) == dataset_type_msg

    with pytest.raises(IncorrectShapeError) as exin:
        fudfs._validate_input_lasso_path(NUMERICAL_NP_ARRAY,
                                         NUMERICAL_NP_ARRAY, None, None, None)
    assert str(exin.value) == target_shape_msg
    #
    with pytest.raises(TypeError) as exin:
        fudfs._validate_input_lasso_path(NUMERICAL_NP_ARRAY,
                                         np.array(['1', 2, 3, 4]), None, None,
                                         None)
    assert str(exin.value) == target_type_msg
    #
    with pytest.raises(IncorrectShapeError) as exin:
        fudfs._validate_input_lasso_path(
            NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY_TARGET[0:4], None, None, 0)
    assert str(exin.value) == target_shape_2_msg

    with pytest.raises(IncorrectShapeError) as exin:
        fudfs._validate_input_lasso_path(NUMERICAL_NP_ARRAY,
                                         NUMERICAL_NP_ARRAY_TARGET,
                                         NUMERICAL_NP_ARRAY, None, None)
    assert str(exin.value) == weights_shape_msg
    #
    with pytest.raises(TypeError) as exin:
        fudfs._validate_input_lasso_path(
            NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY_TARGET,
            np.array([None, 'b', 3, 'c']), None, None)
    assert str(exin.value) == weights_type_msg
    #
    with pytest.raises(IncorrectShapeError) as exin:
        fudfs._validate_input_lasso_path(
            NUMERICAL_STRUCT_ARRAY, NUMERICAL_NP_ARRAY_TARGET,
            NUMERICAL_NP_ARRAY_TARGET[0:4], None, None)
    assert str(exin.value) == weights_shape_2_msg

    with pytest.raises(TypeError) as exin:
        fudfs._validate_input_lasso_path(NUMERICAL_NP_ARRAY,
                                         NUMERICAL_NP_ARRAY_TARGET,
                                         NUMERICAL_NP_ARRAY_TARGET, 'a', None)
    assert str(exin.value) == features_number_type_msg
    #
    with pytest.raises(ValueError) as exin:
        fudfs._validate_input_lasso_path(NUMERICAL_NP_ARRAY,
                                         NUMERICAL_NP_ARRAY_TARGET,
                                         NUMERICAL_NP_ARRAY_TARGET, 0, None)
    assert str(exin.value) == features_number_value_msg

    with pytest.raises(TypeError) as exin:
        fudfs._validate_input_lasso_path(NUMERICAL_NP_ARRAY,
                                         NUMERICAL_NP_ARRAY_TARGET,
                                         NUMERICAL_NP_ARRAY_TARGET, 2, None)
    assert str(exin.value) == features_percentage_type_msg
    #
    with pytest.raises(TypeError) as exin:
        fudfs._validate_input_lasso_path(NUMERICAL_NP_ARRAY,
                                         NUMERICAL_NP_ARRAY_TARGET,
                                         NUMERICAL_NP_ARRAY_TARGET, 2, 5.5)
    assert str(exin.value) == features_percentage_type_msg
    #
    with pytest.raises(ValueError) as exin:
        fudfs._validate_input_lasso_path(NUMERICAL_NP_ARRAY,
                                         NUMERICAL_NP_ARRAY_TARGET,
                                         NUMERICAL_NP_ARRAY_TARGET, 2, -1)
    assert str(exin.value) == features_percentage_value_msg
    #
    with pytest.raises(ValueError) as exin:
        fudfs._validate_input_lasso_path(NUMERICAL_NP_ARRAY,
                                         NUMERICAL_NP_ARRAY_TARGET,
                                         NUMERICAL_NP_ARRAY_TARGET, 2, 101)
    assert str(exin.value) == features_percentage_value_msg

    # All good
    fudfs._validate_input_lasso_path(NUMERICAL_NP_ARRAY,
                                     NUMERICAL_NP_ARRAY_TARGET,
                                     NUMERICAL_NP_ARRAY_TARGET, 2, 0)
    fudfs._validate_input_lasso_path(NUMERICAL_STRUCT_ARRAY,
                                     NUMERICAL_NP_ARRAY_TARGET,
                                     NUMERICAL_NP_ARRAY_TARGET, 3, 100)


def test_lasso_path(caplog):
    """
    Tests :func:`fatf.utils.data.feature_choice.sklearn.lasso_path` function.
    """
    no_lasso_log = ('The lasso path feature selection could not pick any '
                    'feature subset. All of the features were selected.')
    less_lasso_log = ('The lasso path feature selection could not pick {} '
                      'features. Only {} were selected.')

    assert len(caplog.records) == 0
    fatf.setup_random_seed()
    assert len(caplog.records) == 2
    assert caplog.records[0].levelname == 'INFO'
    assert caplog.records[0].getMessage().startswith('Seeding RNGs ')
    assert caplog.records[1].levelname == 'INFO'
    assert caplog.records[1].getMessage() == 'Seeding RNGs with 42.'

    # Weights and no-weights
    weights = np.ones((NUMERICAL_NP_ARRAY.shape[0], ))
    # Classic array -- weights
    features = fudfs.lasso_path(NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY_TARGET,
                                weights, 2)
    assert fuav.is_1d_array(features)
    assert np.array_equal(features, np.array([0, 1]))
    # Structured array -- no-weights
    features = fudfs.lasso_path(
        NUMERICAL_STRUCT_ARRAY, NUMERICAL_NP_ARRAY_TARGET, features_number=2)
    assert fuav.is_1d_array(features)
    assert np.array_equal(features, np.array(['a', 'b']))
    #
    # Selecting exactly 4 features -- no need for Lasso
    features = fudfs.lasso_path(NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY_TARGET,
                                weights, 4)
    assert fuav.is_1d_array(features)
    assert np.array_equal(features, np.array([0, 1, 2, 3]))
    # Selecting more than 4 features
    with pytest.warns(UserWarning) as warning:
        features = fudfs.lasso_path(NUMERICAL_STRUCT_ARRAY,
                                    NUMERICAL_NP_ARRAY_TARGET, weights, 5)
    assert len(warning) == 1
    assert str(warning[0].message) == FEATURE_INDICES_WARNING
    assert fuav.is_1d_array(features)
    assert np.array_equal(features, np.array(['a', 'b', 'c', 'd']))
    #
    # No features number -- just percentage
    features = fudfs.lasso_path(
        NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY_TARGET, features_percentage=50)
    assert fuav.is_1d_array(features)
    assert np.array_equal(features, np.array([0, 1]))
    # No features number -- just percentage -- too small no features selected
    assert len(caplog.records) == 2
    features = fudfs.lasso_path(
        NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY_TARGET, features_percentage=24)
    assert fuav.is_1d_array(features)
    assert np.array_equal(features, np.array([0]))
    assert len(caplog.records) == 3
    assert caplog.records[2].levelname == 'WARNING'
    assert caplog.records[2].getMessage() == FEATURE_PERCENTAGE_LOG

    # Weights too small so no path is found -- returns all features
    weights = np.array([1, 1, 100, 1, 1, 1]) * 1e-20
    assert len(caplog.records) == 3
    features = fudfs.lasso_path(NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY_TARGET,
                                weights, 2)
    assert fuav.is_1d_array(features)
    assert np.array_equal(features, np.array([0, 1, 2, 3]))
    assert len(caplog.records) == 4
    assert caplog.records[3].levelname == 'WARNING'
    assert caplog.records[3].getMessage() == no_lasso_log

    # Another selection
    weights = np.array([1, 1, 100, 1, 1, 1])
    features = fudfs.lasso_path(NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY_TARGET,
                                weights, 2)
    assert fuav.is_1d_array(features)
    assert np.array_equal(features, np.array([0, 2]))
    features = fudfs.lasso_path(NUMERICAL_STRUCT_ARRAY,
                                NUMERICAL_NP_ARRAY_TARGET, weights, 2)
    assert fuav.is_1d_array(features)
    assert np.array_equal(features, np.array(['a', 'c']))

    # Lasso with no possibility of reducing the number of features
    assert len(caplog.records) == 4
    features = fudfs.lasso_path(
        np.array([[1, 2, 3], [2, 2, 3], [3, 2, 3], [4, 2, 3]]),
        np.array([1, 2, 3, 4]),
        features_number=2)
    assert fuav.is_1d_array(features)
    assert np.array_equal(features, np.array([0]))
    assert len(caplog.records) == 5
    assert caplog.records[4].levelname == 'WARNING'
    assert caplog.records[4].getMessage() == less_lasso_log.format(2, 1)


def test_forward_selection(caplog):
    """
    Tests :func:`fatf.utils.data.feature_choice.sklearn.forward_selection`.
    """
    assert len(caplog.records) == 0
    fatf.setup_random_seed()
    assert len(caplog.records) == 2
    assert caplog.records[0].levelname == 'INFO'
    assert caplog.records[0].getMessage().startswith('Seeding RNGs ')
    assert caplog.records[1].levelname == 'INFO'
    assert caplog.records[1].getMessage() == 'Seeding RNGs with 42.'

    # Weights and no-weights
    weights = np.ones((NUMERICAL_NP_ARRAY.shape[0], ))
    # Classic array -- weights
    features = fudfs.forward_selection(NUMERICAL_NP_ARRAY,
                                       NUMERICAL_NP_ARRAY_TARGET, weights, 2)
    assert fuav.is_1d_array(features)
    assert np.array_equal(features, np.array([0, 1]))
    # Structured array -- no-weights
    features = fudfs.forward_selection(
        NUMERICAL_STRUCT_ARRAY, NUMERICAL_NP_ARRAY_TARGET, features_number=2)
    assert fuav.is_1d_array(features)
    assert np.array_equal(features, np.array(['a', 'b']))
    #
    # Selecting exactly 4 features -- no need for Lasso
    features = fudfs.forward_selection(NUMERICAL_NP_ARRAY,
                                       NUMERICAL_NP_ARRAY_TARGET, weights, 4)
    assert fuav.is_1d_array(features)
    assert np.array_equal(features, np.array([0, 1, 2, 3]))
    # Selecting more than 4 features
    with pytest.warns(UserWarning) as warning:
        features = fudfs.forward_selection(
            NUMERICAL_STRUCT_ARRAY, NUMERICAL_NP_ARRAY_TARGET, weights, 5)
    assert len(warning) == 1
    assert str(warning[0].message) == FEATURE_INDICES_WARNING
    assert fuav.is_1d_array(features)
    assert np.array_equal(features, np.array(['a', 'b', 'c', 'd']))
    #
    # No features number -- just percentage
    features = fudfs.forward_selection(
        NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY_TARGET, features_percentage=50)
    assert fuav.is_1d_array(features)
    assert np.array_equal(features, np.array([0, 1]))
    # No features number -- just percentage -- too small no features selected
    assert len(caplog.records) == 2
    features = fudfs.forward_selection(
        NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY_TARGET, features_percentage=24)
    assert fuav.is_1d_array(features)
    assert np.array_equal(features, np.array([0]))
    assert len(caplog.records) == 3
    assert caplog.records[2].levelname == 'WARNING'
    assert caplog.records[2].getMessage() == FEATURE_PERCENTAGE_LOG

    # Small weights
    weights = np.array([1, 1, 100, 1, 1, 1]) * 1e-20
    features = fudfs.forward_selection(NUMERICAL_NP_ARRAY,
                                       NUMERICAL_NP_ARRAY_TARGET, weights, 2)
    assert fuav.is_1d_array(features)
    assert np.array_equal(features, np.array([0, 3]))

    # Another selection
    weights = np.array([100, 1, 1, 1, 1, 1])
    features = fudfs.forward_selection(NUMERICAL_NP_ARRAY,
                                       NUMERICAL_NP_ARRAY_TARGET, weights, 2)
    assert fuav.is_1d_array(features)
    assert np.array_equal(features, np.array([0, 2]))
    features = fudfs.forward_selection(NUMERICAL_STRUCT_ARRAY,
                                       NUMERICAL_NP_ARRAY_TARGET, weights, 2)
    assert fuav.is_1d_array(features)
    assert np.array_equal(features, np.array(['a', 'c']))

    # Custom data
    features = fudfs.forward_selection(
        np.array([[1, 2, 3], [2, 2, 3], [3, 2, 3], [4, 2, 3]]),
        np.array([1, 2, 3, 4]),
        features_number=2)
    assert fuav.is_1d_array(features)
    assert np.array_equal(features, np.array([0, 1]))
    assert len(caplog.records) == 3


def test_highest_weights(caplog):
    """
    Tests :func:`fatf.utils.data.feature_choice.sklearn.highest_weights`.
    """
    assert len(caplog.records) == 0
    fatf.setup_random_seed()
    assert len(caplog.records) == 2
    assert caplog.records[0].levelname == 'INFO'
    assert caplog.records[0].getMessage().startswith('Seeding RNGs ')
    assert caplog.records[1].levelname == 'INFO'
    assert caplog.records[1].getMessage() == 'Seeding RNGs with 42.'

    # Weights and no-weights
    weights = np.ones((NUMERICAL_NP_ARRAY.shape[0], ))
    # Classic array -- weights
    features = fudfs.highest_weights(NUMERICAL_NP_ARRAY,
                                     NUMERICAL_NP_ARRAY_TARGET, weights, 2)
    assert fuav.is_1d_array(features)
    assert np.array_equal(features, np.array([1, 2]))
    # Structured array -- no-weights
    features = fudfs.highest_weights(
        NUMERICAL_STRUCT_ARRAY, NUMERICAL_NP_ARRAY_TARGET, features_number=2)
    assert fuav.is_1d_array(features)
    assert np.array_equal(features, np.array(['b', 'c']))
    #
    # Selecting exactly 4 features -- no need for Lasso
    features = fudfs.highest_weights(NUMERICAL_NP_ARRAY,
                                     NUMERICAL_NP_ARRAY_TARGET, weights, 4)
    assert fuav.is_1d_array(features)
    assert np.array_equal(features, np.array([0, 1, 2, 3]))
    # Selecting more than 4 features
    with pytest.warns(UserWarning) as warning:
        features = fudfs.highest_weights(NUMERICAL_STRUCT_ARRAY,
                                         NUMERICAL_NP_ARRAY_TARGET, weights, 5)
    assert len(warning) == 1
    assert str(warning[0].message) == FEATURE_INDICES_WARNING
    assert fuav.is_1d_array(features)
    assert np.array_equal(features, np.array(['a', 'b', 'c', 'd']))
    #
    # No features number -- just percentage
    features = fudfs.highest_weights(
        NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY_TARGET, features_percentage=50)
    assert fuav.is_1d_array(features)
    assert np.array_equal(features, np.array([1, 2]))
    # No features number -- just percentage -- too small no features selected
    assert len(caplog.records) == 2
    features = fudfs.highest_weights(
        NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY_TARGET, features_percentage=24)
    assert fuav.is_1d_array(features)
    assert np.array_equal(features, np.array([2]))
    assert len(caplog.records) == 3
    assert caplog.records[2].levelname == 'WARNING'
    assert caplog.records[2].getMessage() == FEATURE_PERCENTAGE_LOG

    # Small weights
    weights = np.array([1, 1, 100, 1, 1, 1]) * 1e-20
    features = fudfs.highest_weights(NUMERICAL_NP_ARRAY,
                                     NUMERICAL_NP_ARRAY_TARGET, weights, 2)
    assert fuav.is_1d_array(features)
    assert np.array_equal(features, np.array([0, 1]))

    # Another selection
    weights = np.array([100, 1, 1, 1, 1, 1])
    features = fudfs.highest_weights(NUMERICAL_NP_ARRAY,
                                     NUMERICAL_NP_ARRAY_TARGET, weights, 2)
    assert fuav.is_1d_array(features)
    assert np.array_equal(features, np.array([2, 3]))
    features = fudfs.highest_weights(NUMERICAL_STRUCT_ARRAY,
                                     NUMERICAL_NP_ARRAY_TARGET, weights, 2)
    assert fuav.is_1d_array(features)
    assert np.array_equal(features, np.array(['c', 'd']))

    # Custom data
    features = fudfs.highest_weights(
        np.array([[1, 2, 3], [2, 2, 3], [3, 2, 3], [4, 2, 3]]),
        np.array([1, 2, 3, 4]),
        features_number=2)
    assert fuav.is_1d_array(features)
    assert np.array_equal(features, np.array([0, 2]))
    assert len(caplog.records) == 3
