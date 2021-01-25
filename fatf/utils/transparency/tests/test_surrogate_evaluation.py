"""
Tests surrogate evaluation metrics.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
#         Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import pytest

import numpy as np

from fatf.exceptions import IncompatibleModelError, IncorrectShapeError

import fatf
import fatf.utils.metrics.metrics as fummet
import fatf.utils.metrics.tools as fumt
import fatf.utils.models.models as fumm
import fatf.utils.transparency.surrogate_evaluation as futs

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
NUMERICAL_STRUCT_ARRAY_LOCAL = np.array([(0, 0, 0.09, 0.7), (1, 0, 0.06, 0.60),
                                         (0, 0, 0.05, 0.5),
                                         (0, 1, 0.99, 0.82)],
                                        dtype=[('a', 'i'), ('b', 'i'),
                                               ('c', 'f8'), ('d', 'f8')])

NUMERICAL_NP_ARRAY_TARGET = np.array([2, 0, 1, 1, 0, 2])
NUMERICAL_NP_ARRAY_LOCAL_TARGET = np.array([1, 0, 1, 1])


def test_validate_input_local_fidelity():
    """
    Tests the ``_validate_input_local_fidelity`` function.

    This function tests the :func:`fatf.utils.transparency.\
surrogate_evaluation._validate_input_local_fidelity` function.
    """
    incorrect_shape_dataset = ('The input dataset must be a 2-dimensional '
                               'numpy array.')
    type_error_dataset = ('The input dataset must be of a base type -- '
                          'numbers and/or strings.')
    incorrect_shape_datarow = ('The data_row must either be a 1-dimensional '
                               'numpy array or a numpy void object for '
                               'structured data rows.')
    incorrect_dtype_data = ('The dtype of the data_row is too different from '
                            'the dtype of the dataset array.')
    datarow_features_error = ('The data_row must contain the same number of '
                              'features as the dataset.')

    global_model_incompatible = ('The global predictive function must have '
                                 'exactly *one* required parameter to work '
                                 'with this metric.')
    global_model_type = ('The global_predictive_function should be a Python '
                         'callable, e.g., a Python function.')
    local_model_incompatible = ('The local predictive function must have '
                                'exactly *one* required parameter to work '
                                'with this metric.')
    local_model_type = ('The local_predictive_function should be a Python '
                        'callable, e.g., a Python function.')

    metric_param_error = ('The metric_function must take exactly *two* '
                          'required parameters.')
    metric_type_error = ('The metric_function should be a Python callable, '
                         'e.g., a Python function.')

    explained_class_value_error = ('The explained_class_index parameter is '
                                   'negative or larger than the number of '
                                   'classes output by the global '
                                   'probabilistic model.')
    explained_class_type_error = ('For probabilistic global models, i.e., '
                                  'global predictive functions, the '
                                  'explained_class_index parameter has to be '
                                  'an integer or None.')
    explained_class_warning = ('The explained_class_index parameter is not '
                               'None and will be ignored since the global '
                               'model is not probabilistic.')

    features_index_error = ('The following column indices are invalid for '
                            'the input dataset: {}.')
    features_type_error = ('The explained_feature_indices parameter must be '
                           'a Python list or None.')

    fidelity_radius_type_error = ('The fidelity_radius_percentage must be an '
                                  'integer between 1 and 100.')
    fidelity_radius_value_error = ('The fidelity_radius_percentage must be an '
                                   'integer between 1 and 100.')

    samples_number_value_error = ('The samples_number must be a positive '
                                  'integer.')
    samples_number_type_error = 'The samples_number must be an integer.'

    with pytest.raises(IncorrectShapeError) as exin:
        futs._validate_input_local_fidelity(NUMERICAL_NP_ARRAY[0], None, None,
                                            None, None, None, None, None, None)
    assert str(exin.value) == incorrect_shape_dataset

    with pytest.raises(TypeError) as exin:
        futs._validate_input_local_fidelity(
            np.array([[None]]), None, None, None, None, None, None, None, None)
    assert str(exin.value) == type_error_dataset

    with pytest.raises(IncorrectShapeError) as exin:
        futs._validate_input_local_fidelity(NUMERICAL_NP_ARRAY,
                                            NUMERICAL_NP_ARRAY, None, None,
                                            None, None, None, None, None)
    assert str(exin.value) == incorrect_shape_datarow

    with pytest.raises(TypeError) as exin:
        futs._validate_input_local_fidelity(
            NUMERICAL_NP_ARRAY, np.array(['0']), None, None, None, None, None,
            None, None)  # yapf: disable
    assert str(exin.value) == incorrect_dtype_data

    with pytest.raises(IncorrectShapeError) as exin:
        futs._validate_input_local_fidelity(NUMERICAL_NP_ARRAY,
                                            NUMERICAL_NP_ARRAY[0][0:2], None,
                                            None, None, None, None, None, None)
    assert str(exin.value) == datarow_features_error

    def predict(x):
        return np.ones(x.shape[0])

    def predict_invalid(x_1, x_2):
        pass  # pragma: no cover

    def predict_proba(x):
        return np.ones((x.shape[0], 3))

    def predict_proba_invalid():
        pass  # pragma: no cover

    with pytest.raises(TypeError) as exin:
        futs._validate_input_local_fidelity(NUMERICAL_NP_ARRAY,
                                            NUMERICAL_NP_ARRAY[0], None, None,
                                            None, None, None, None, None)
    assert str(exin.value) == global_model_type
    with pytest.raises(IncompatibleModelError) as exin:
        futs._validate_input_local_fidelity(
            NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY[0], predict_invalid, None,
            None, None, None, None, None)
    assert str(exin.value) == global_model_incompatible

    with pytest.raises(TypeError) as exin:
        futs._validate_input_local_fidelity(NUMERICAL_NP_ARRAY,
                                            NUMERICAL_NP_ARRAY[0], predict,
                                            None, None, None, None, None, None)
    assert str(exin.value) == local_model_type
    with pytest.raises(IncompatibleModelError) as exin:
        futs._validate_input_local_fidelity(
            NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY[0], predict_proba,
            predict_proba_invalid, None, None, None, None, None)
    assert str(exin.value) == local_model_incompatible

    def invalid_metric(x):
        pass  # pragma: no cover

    def metric(x_1, x_2):
        pass  # pragma: no cover

    with pytest.raises(TypeError) as exin:
        futs._validate_input_local_fidelity(
            NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY[0], predict, predict, None,
            None, None, None, None)
    assert str(exin.value) == metric_type_error
    with pytest.raises(TypeError) as exin:
        futs._validate_input_local_fidelity(
            NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY[0], predict, predict,
            invalid_metric, None, None, None, None)
    assert str(exin.value) == metric_param_error

    with pytest.raises(TypeError) as exin:
        futs._validate_input_local_fidelity(
            NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY[0], predict_proba, predict,
            metric, '1', None, None, None)
    assert str(exin.value) == explained_class_type_error
    with pytest.raises(ValueError) as exin:
        futs._validate_input_local_fidelity(
            NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY[0], predict_proba, predict,
            metric, -1, None, None, None)
    assert str(exin.value) == explained_class_value_error
    with pytest.raises(ValueError) as exin:
        futs._validate_input_local_fidelity(
            NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY[0], predict_proba, predict,
            metric, 3, None, None, None)
    assert str(exin.value) == explained_class_value_error
    #
    with pytest.warns(UserWarning) as w:
        futs._validate_input_local_fidelity(NUMERICAL_NP_ARRAY,
                                            NUMERICAL_NP_ARRAY[0], predict,
                                            predict, metric, 3, None, 1, 1)
    assert len(w) == 1
    assert str(w[0].message) == explained_class_warning

    with pytest.raises(TypeError) as exin:
        futs._validate_input_local_fidelity(
            NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY[0], predict_proba, predict,
            metric, None, np.array([10, 11]), None, None)
    assert str(exin.value) == features_type_error
    with pytest.raises(IndexError) as exin:
        futs._validate_input_local_fidelity(
            NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY[0], predict, predict,
            metric, None, [10, 11], None, None)
    assert str(exin.value) == features_index_error.format(np.array([10, 11]))

    with pytest.raises(TypeError) as exin:
        futs._validate_input_local_fidelity(
            NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY[0], predict_proba, predict,
            metric, 1, [1, 2], 'a', None)
    assert str(exin.value) == fidelity_radius_type_error
    with pytest.raises(TypeError) as exin:
        futs._validate_input_local_fidelity(
            NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY[0], predict_proba, predict,
            metric, 1, [1, 2], None, None)
    assert str(exin.value) == fidelity_radius_type_error
    with pytest.raises(TypeError) as exin:
        futs._validate_input_local_fidelity(
            NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY[0], predict_proba, predict,
            metric, 1, [1, 2], 55.0, None)
    assert str(exin.value) == fidelity_radius_type_error
    #
    with pytest.raises(ValueError) as exin:
        futs._validate_input_local_fidelity(
            NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY[0], predict, predict,
            metric, None, [1, 2], 0, None)
    assert str(exin.value) == fidelity_radius_value_error
    with pytest.raises(ValueError) as exin:
        futs._validate_input_local_fidelity(
            NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY[0], predict, predict,
            metric, None, [1, 2], 101, None)
    assert str(exin.value) == fidelity_radius_value_error

    with pytest.raises(TypeError) as exin:
        futs._validate_input_local_fidelity(
            NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY[0], predict, predict,
            metric, None, None, 100, None)
    assert str(exin.value) == samples_number_type_error
    with pytest.raises(TypeError) as exin:
        futs._validate_input_local_fidelity(
            NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY[0], predict, predict,
            metric, None, None, 100, 55.0)
    assert str(exin.value) == samples_number_type_error
    #
    with pytest.raises(ValueError) as exin:
        futs._validate_input_local_fidelity(
            NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY[0], predict, predict,
            metric, None, None, 100, 0)
    assert str(exin.value) == samples_number_value_error
    with pytest.raises(ValueError) as exin:
        futs._validate_input_local_fidelity(
            NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY[0], predict, predict,
            metric, None, None, 100, -42)
    assert str(exin.value) == samples_number_value_error

    clf = fumm.KNN(k=3)
    clf.fit(NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY_TARGET)

    with pytest.raises(ValueError) as exin:
        futs._validate_input_local_fidelity(
            NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY[0], clf.predict_proba,
            predict, metric, 10, None, 10, 1)
    assert str(exin.value) == explained_class_value_error

    # All OK
    assert futs._validate_input_local_fidelity(
        NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY[0], clf.predict_proba, predict,
        metric, 1, [0, 1], 10, 1)


def test_local_fidelity_score():
    """
    Tests the ``local_fidelity_score`` function.

    This function tests the
    :func:`fatf.utils.transparency.surrogate_evaluation.local_fidelity_score`
    function.
    """
    accuracy_warning = ('Some of the given labels are not present in either '
                        'of the input arrays: {}.')
    fatf.setup_random_seed()

    def accuracy(global_predictions, local_predictions):
        global_predictions[global_predictions >= 0.5] = 1
        global_predictions[global_predictions < 0.5] = 0

        local_predictions[local_predictions >= 0.5] = 1
        local_predictions[local_predictions < 0.5] = 0

        confusion_matrix = fumt.get_confusion_matrix(
            global_predictions, local_predictions, labels=[0, 1])
        accuracy = fummet.accuracy(confusion_matrix)

        return accuracy

    def accuracy_prob(global_predictions,
                      local_predictions,
                      global_proba=True,
                      local_proba=True):
        if global_proba:
            global_predictions = np.argmax(global_predictions, axis=1)
        if local_proba:
            local_predictions = np.argmax(local_predictions, axis=1)

        confusion_matrix = fumt.get_confusion_matrix(
            global_predictions, local_predictions, labels=[0, 1, 2])
        accuracy = fummet.accuracy(confusion_matrix)

        return accuracy

    def accuracy_proba_np(global_predictions, local_predictions):
        return accuracy_prob(
            global_predictions,
            local_predictions,
            global_proba=False,
            local_proba=True)

    def accuracy_proba_nn(global_predictions, local_predictions):
        return accuracy_prob(
            global_predictions,
            local_predictions,
            global_proba=False,
            local_proba=False)

    def reg_dist(global_predictions, local_predictions):
        return (global_predictions - local_predictions).sum()

    predictor = fumm.KNN(k=3)
    predictor.fit(NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY_TARGET)

    regressor = fumm.KNN(k=3, mode='regressor')
    regressor.fit(NUMERICAL_NP_ARRAY_LOCAL, NUMERICAL_NP_ARRAY_LOCAL_TARGET)

    regressor_23 = fumm.KNN(k=3, mode='regressor')
    regressor_23.fit(NUMERICAL_NP_ARRAY_LOCAL[:, [2, 3]],
                     NUMERICAL_NP_ARRAY_LOCAL_TARGET)

    # Structured array
    predictor_struct = fumm.KNN(k=3)
    predictor_struct.fit(NUMERICAL_STRUCT_ARRAY, NUMERICAL_NP_ARRAY_TARGET)
    #
    regressor_struct_cd = fumm.KNN(k=3, mode='regressor')
    regressor_struct_cd.fit(NUMERICAL_STRUCT_ARRAY_LOCAL[['c', 'd']],
                            NUMERICAL_NP_ARRAY_LOCAL_TARGET)

    # Global: probabilistic...
    # ...local: regressor
    comparison = futs.local_fidelity_score(
        NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY[0], predictor.predict_proba,
        regressor.predict, accuracy, 2)
    assert np.isclose(comparison, 0.26)
    # ...local: classifier
    comparison = futs.local_fidelity_score(
        NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY[0], predictor.predict_proba,
        predictor.predict, accuracy, 2)
    assert np.isclose(comparison, 1.0)
    # ...local: probabilistic
    with pytest.warns(UserWarning) as w:
        comparison = futs.local_fidelity_score(
            NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY[0], predictor.predict_proba,
            predictor.predict_proba, accuracy_prob)
    assert len(w) == 1
    assert str(w[0].message) == accuracy_warning.format(set([1]))
    assert np.isclose(comparison, 1.0)

    # Global: classifier...
    # ...local: probabilistic
    with pytest.warns(UserWarning) as w:
        comparison = futs.local_fidelity_score(
            NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY[0], predictor.predict,
            predictor.predict_proba, accuracy_proba_np)
    assert len(w) == 1
    assert str(w[0].message) == accuracy_warning.format(set([1]))
    assert np.isclose(comparison, 1.0)
    # ...local: classifier
    with pytest.warns(UserWarning) as w:
        comparison = futs.local_fidelity_score(
            NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY[0], predictor.predict,
            predictor.predict, accuracy_proba_nn)
    assert len(w) == 1
    assert str(w[0].message) == accuracy_warning.format(set([1]))
    assert np.isclose(comparison, 1.0)

    # Global: regressor...
    # ...local: regressor
    comparison = futs.local_fidelity_score(
        NUMERICAL_NP_ARRAY,
        NUMERICAL_NP_ARRAY[0],
        regressor.predict,
        regressor_23.predict,
        reg_dist,
        explained_feature_indices=[2, 3])
    assert np.isclose(comparison, 0)

    # Structured array
    # Global: probabilistic...
    # ...local: regressor
    comparison = futs.local_fidelity_score(
        NUMERICAL_STRUCT_ARRAY,
        NUMERICAL_STRUCT_ARRAY[0],
        predictor_struct.predict_proba,
        regressor_struct_cd.predict,
        accuracy,
        0,
        explained_feature_indices=['c', 'd'])
    assert np.isclose(comparison, 0.94)
