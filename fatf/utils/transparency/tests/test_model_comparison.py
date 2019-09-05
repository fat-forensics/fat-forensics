"""
Tests model_comparison functions.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
# License: new BSD

from typing import Dict, Union

import pytest

import numpy as np

import fatf
import fatf.utils.data.augmentation as fuda
import fatf.utils.models.models as fumm
import fatf.utils.metrics.metrics as fummet
import fatf.utils.metrics.tools as fumt
import fatf.utils.transparency.model_comparison as futmc

from fatf.exceptions import IncompatibleModelError, IncorrectShapeError

NUMERICAL_NP_ARRAY = np.array([[0, 0, 0.08, 0.69], [1, 0, 0.03, 0.29],
                               [0, 1, 0.99, 0.82], [2, 1, 0.73, 0.48],
                               [1, 0, 0.36, 0.89], [0, 1, 0.07, 0.21]])
NUMERICAL_STRUCT_ARRAY = np.array([(0, 0, 0.08, 0.69), (1, 0, 0.03, 0.29),
                                   (0, 1, 0.99, 0.82), (2, 1, 0.73, 0.48),
                                   (1, 0, 0.36, 0.89), (0, 1, 0.07, 0.21)],
                                  dtype=[('a', 'i'), ('b', 'i'), ('c', 'f'),
                                         ('d', 'f')])
NUMERICAL_NP_ARRAY_LOCAL = np.array([[0, 0, 0.09, 0.70], [1, 0, 0.06, 0.60],
                                     [0, 0, 0.05, 0.50], [0, 1, 0.99, 0.82]])
NUMERICAL_STRUCT_ARRAY_LOCAL = np.array([(0, 0, 0.09, 0.70), (1, 0, 0.06, 0.60),
                                         (0, 0, 0.05, 0.50), (0, 1, 0.99, 0.82)],
                                         dtype=[('a', 'i'), ('b', 'i'), ('c', 'f8'),
                                                ('d', 'f8')])

NUMERICAL_NP_ARRAY_TARGET = np.array([2, 0, 1, 1, 0, 2])
NUMERICAL_NP_ARRAY_LOCAL_TARGET = np.array([1, 0, 1, 1])


def test_input_is_valid():
    """
    Tests :func:`fatf.utils.transparency.model_comparison._input_is_valid`.
    """
    incorrect_shape_dataset = ('The input dataset must be a 2-dimensional '
                               'numpy array.')
    type_error_dataset = 'The input dataset must be of a base type.'
    incorrect_shape_datarow = ('The data_row must either be a 1-dimensional '
                               'numpy array or numpy void object for '
                               'structured rows.')
    incorrect_dtype_data = ('The dtype of the data_row is different to '
                            'the dtype of the dataset array.')
    features_index_error = ('The following indices are invalid for the input '
                            'dataset: {}.')
    features_type_error = ('The local_features parameter must be a Python '
                           'list or None.')
    datarow_features_error = ('The data_row must contain the same number of '
                              'features as the dataset.')
    global_model_error = ('This functionality requires the global model to '
                          'be capable of outputting predicted class '
                          'probabilities via predict proba method.')
    local_model_error = ('This functionality requires the local model to be '
                         'capable of outputting predicted class via '
                         'predict method.')
    r_fid_type_err = 'r_fid must be a float.'
    r_fid_value_error = 'r_fid must be a positive float.'
    samples_number_value_error = ('The samples_number parameter must be a '
                                  'positive integer.')
    samples_number_type_error = ('The samples_number parameter must be an '
                                 'integer.')
    metric_error = ('The metric function must take only two required '
                    'parameters.')
    global_class_error = ('global_class is larger than the number of classes '
                          'outputted by the global model.')

    with pytest.raises(IncorrectShapeError) as exin:
        futmc._input_is_valid(NUMERICAL_NP_ARRAY[0], None, None, None, None,
                              None, None, None, None)
    assert str(exin.value) == incorrect_shape_dataset

    with pytest.raises(TypeError) as exin:
        futmc._input_is_valid(np.array([[0, 0], [None, 0]]), None, None, None,
                              None, None, None, None, None)
    assert str(exin.value) == type_error_dataset

    with pytest.raises(IncorrectShapeError) as exin:
        futmc._input_is_valid(NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY, None,
                              None, None, None, None, None, None)
    assert str(exin.value) == incorrect_shape_datarow

    with pytest.raises(TypeError) as exin:
        futmc._input_is_valid(NUMERICAL_NP_ARRAY, np.array([0]), None,
                              None, None, None, None, None, None)
    assert str(exin.value) == incorrect_dtype_data

    with pytest.raises(IndexError) as exin:
        futmc._input_is_valid(NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY[0], None,
                              None, [10, 11], None, None, None, None)
    assert str(exin.value) == features_index_error.format(np.array([10, 11]))

    with pytest.raises(TypeError) as exin:
        futmc._input_is_valid(NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY[0], None,
                              None, np.array([10, 11]), None, None, None, None)
    assert str(exin.value) == features_type_error

    with pytest.raises(IncorrectShapeError) as exin:
        futmc._input_is_valid(NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY[0][0:2],
                              None, None, None, None, None, None, None)
    assert str(exin.value) == datarow_features_error

    class valid_model():
        def __init__(self): pass
        def fit(self, x, y): pass
        def predict(self, x): pass
        def predict_proba(self, x): pass

    class invalid_model():
        def __init__(self): pass
        def fit(self, x, y): pass

    class invalid_model2():
        def fit(self, x, y): pass
        def __init__(self): pass
        def predict(self, x): pass

    model1 = valid_model()
    model2 = invalid_model()
    model3 = invalid_model2()

    with pytest.raises(IncompatibleModelError) as exin:
        futmc._input_is_valid(NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY[0], None,
                              model3, None, None, None, None, None)
    assert str(exin.value) == global_model_error

    with pytest.raises(IncompatibleModelError) as exin:
        futmc._input_is_valid(NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY[0],
                              model2, model1, None, None, None,
                              None, None)
    assert str(exin.value) == local_model_error

    with pytest.raises(TypeError) as exin:
        futmc._input_is_valid(NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY[0],
                              model1, model1, None, 'a', None, None, None)
    assert str(exin.value) == r_fid_type_err

    with pytest.raises(ValueError) as exin:
        futmc._input_is_valid(NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY[0],
                              model1, model1, None, -0.1, None, None, None)
    assert str(exin.value) == r_fid_value_error

    with pytest.raises(ValueError) as exin:
        futmc._input_is_valid(NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY[0],
                              model1, model1, None, 0.1, -1, None, None)
    assert str(exin.value) == samples_number_value_error

    with pytest.raises(TypeError) as exin:
        futmc._input_is_valid(NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY[0],
                              model1, model1, None, 0.1, 'a', None, None)
    assert str(exin.value) == samples_number_type_error

    def invalid_metric(x): pass
    def valid_metric(x, y): pass

    with pytest.raises(TypeError) as exin:
        futmc._input_is_valid(NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY[0],
                              model1, model1, None, 0.1, 1, invalid_metric,
                              None)
    assert str(exin.value) == metric_error

    model_real = fumm.KNN(k=3)
    model_real.fit(NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY_TARGET)

    with pytest.raises(ValueError) as exin:
        futmc._input_is_valid(NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY[0],
                              model1, model_real, None, 0.1, 1, valid_metric,
                              10)
    assert str(exin.value) == global_class_error

    # All good
    assert futmc._input_is_valid(NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY[0],
                                 model1, model_real, [0, 1], 0.1, 1,
                                 valid_metric, 0)


def test_local_fidelity_score():
    """
    Tests :func:`fatf.utils.transparency.model_comparison.
    local_fidelity_score`.
    """
    fatf.setup_random_seed()

    global_model = fumm.KNN(k=3)
    local_model = fumm.KNN(k=3, mode='regressor')

    global_model.fit(NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY_TARGET)
    local_model.fit(NUMERICAL_NP_ARRAY_LOCAL, NUMERICAL_NP_ARRAY_LOCAL_TARGET)

    def accuracy(local_predictions, global_predictions):
        global_predictions[global_predictions>=0.5] = 1
        global_predictions[global_predictions<0.5] = 0
        local_predictions[local_predictions>=0.5] = 1
        local_predictions[local_predictions<0.5] = 0
        confusion_matrix = fumt.get_confusion_matrix(
            local_predictions, global_predictions, labels=[0, 1])
        accuracy = fummet.accuracy(confusion_matrix)
        return accuracy

    comparison = futmc.local_fidelity_score(
        NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY[0], local_model, global_model,
        2, accuracy)
    assert np.isclose(comparison, 0.26)

    comparison = futmc.local_fidelity_score(
        NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY[0], global_model, global_model,
        2, accuracy)
    assert np.isclose(comparison, 1.0)

    # Structured array
    global_model = fumm.KNN(k=3)
    local_model = fumm.KNN(k=3, mode='regressor')

    global_model.fit(NUMERICAL_STRUCT_ARRAY, NUMERICAL_NP_ARRAY_TARGET)
    local_model.fit(NUMERICAL_STRUCT_ARRAY_LOCAL[['c', 'd']],
                    NUMERICAL_NP_ARRAY_LOCAL_TARGET)

    comparison = futmc.local_fidelity_score(
        NUMERICAL_STRUCT_ARRAY, NUMERICAL_STRUCT_ARRAY[0], local_model,
        global_model, 0, accuracy, local_features=['c', 'd'])
    assert np.isclose(comparison, 0.78)
