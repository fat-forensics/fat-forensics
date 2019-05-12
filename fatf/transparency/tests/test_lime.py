"""
Tests the LIME wrapper.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
#         Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import importlib
import pytest
import sys

from typing import Dict, List, Tuple

import numpy as np

import fatf.transparency.lime as ftl
import fatf.utils.models as fum
import fatf.utils.testing.imports as futi

from fatf.exceptions import IncompatibleModelError, IncorrectShapeError

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
    dtype=[('a', int), ('b', int), ('c', float), ('d', float)])
LABELS = np.array([2, 0, 1, 1, 0, 2])
SAMPLE = np.array([0, 1, 0.08, 0.54])
SAMPLE_STRUCT = np.array(
    [(0, 1, 0.08, 0.54)],
    dtype=[('a', 'i'), ('b', 'i'), ('c', 'f'), ('d', 'f')])[0]
CLF = fum.KNN()
CLF.fit(NUMERICAL_NP_ARRAY, LABELS)
CLASS_NAMES = ['class0', 'class1', 'class2']
FEATURE_NAMES = ['feat0', 'feat1', 'feat2', 'feat3']

NUMERICAL_RESULTS = {
    'class0': [('feat0 <= 0.00', -0.415),
               ('0.50 < feat1 <= 1.00', -0.280),
               ('0.07 < feat2 <= 0.22', 0.037),
               ('0.34 < feat3 <= 0.58', -0.007)],
    'class1': [('0.50 < feat1 <= 1.00', 0.202),
               ('0.07 < feat2 <= 0.22', -0.076),
               ('feat0 <= 0.00', 0.019),
               ('0.34 < feat3 <= 0.58', -0.018)],
    'class2': [('feat0 <= 0.00', 0.395),
               ('0.50 < feat1 <= 1.00', 0.077),
               ('0.07 < feat2 <= 0.22', 0.039),
               ('0.34 < feat3 <= 0.58', 0.025)]
}
CATEGORICAL_RESULTS = {
    'class0': [('feat0=0', -0.413),
               ('feat1=1', -0.282),
               ('0.07 < feat2 <= 0.22', 0.0366),
               ('0.34 < feat3 <= 0.58', -0.00717)],
    'class1': [('feat1=1', 0.2048),
               ('0.07 < feat2 <= 0.22', -0.0767),
               ('feat0=0', 0.0179),
               ('0.34 < feat3 <= 0.58', -0.018)],
    'class2': [('feat0=0', 0.395),
               ('feat1=1', 0.077),
               ('0.07 < feat2 <= 0.22', 0.039),
               ('0.34 < feat3 <= 0.58', 0.025)]
}
REGRESSION_RESULTS = [
    ('feat0 <= 0.00', 1.332),
    ('0.50 < feat1 <= 1.00', 0.767),
    ('0.34 < feat3 <= 0.58', 0.149),
    ('0.07 < feat2 <= 0.22', -0.048)
]
# yapf: enable

USER_WARNING_MODEL_PRED = ('Since both, a model and a predictive function, '
                           'are provided only the latter will be used.')
LOG_WARNING = 'The model can only be used for LIME in a regressor mode.'


class InvalidModel(object):
    """
    An invalid model class -- it does not implement a ``predict_proba`` method.
    """

    def __init__(self):
        """
        Invalid initialisation.
        """

    def fit(self, X, y):
        """
        Invalid fit.
        """


class NonProbabilisticModel(InvalidModel):
    """
    A model that is not probabilistic -- no ``predict_proba`` function.
    """

    def __init__(self, prediction_function):
        """
        Non-probabilistic initialisation.
        """
        super().__init__()
        self.prediction_function = prediction_function

    def predict(self, X):
        """
        Non-probabilistic predict.
        """
        return self.prediction_function(X)


CLF_NON_PROBA = NonProbabilisticModel(CLF.predict)


def test_import_when_missing():
    """
    Tests importing :mod:`fatf.transparency.lime` module when LIME is missing.
    """
    assert 'fatf.transparency.lime' in sys.modules
    warning_msg = (
        'Lime package is not installed on your system. You must install it in '
        'order to use the fatf.transparency.lime module. One possibility is '
        'to install LIME alongside this package with: pip install fatf[lime].')
    with futi.module_import_tester('lime', when_missing=True):
        with pytest.warns(ImportWarning) as w:
            importlib.reload(ftl)
        assert len(w) == 1
        assert str(w[0].message) == warning_msg
    assert 'fatf.transparency.lime' in sys.modules


def _is_explanation_equal(dict1: Dict[str, List[Tuple[str, float]]],
                          dict2: Dict[str, List[Tuple[str, float]]]) -> bool:
    """
    Tests if the two dictionaries of a given structure are equal.

    The both of the input parameters must be a dictionary with string keys and
    list values. The latter is composed of 2-tuples of strings and floats.

    The keys in the dictionary and the tuples must match exactly, while the
    floats only need to be approximately equal. The ordering of the tuples in
    the list does not need to be the same.

    Parameters
    ----------
    dict1 : Dictionary[string, List[Tuple[string, float]]]
        The first dictionary to be compared.
    dict2 : Dictionary[string, List[Tuple[string, float]]]
        The second dictionary to be compared.

    Returns
    -------
    equal : boolean
        ``True`` if the dictionaries are the same, ``False`` otherwise.
    """
    if set(dict1.keys()) == set(dict2.keys()):
        equal = True
        for key in dict1:
            val1 = sorted(dict1[key])
            val2 = sorted(dict2[key])

            if len(val1) != len(val2):
                equal = False
                break

            for i in range(len(val1)):
                if val1[i][0] != val2[i][0]:
                    equal = False
                    break
                if abs(val1[i][1] - val2[i][1]) > 1e-1:
                    equal = False
                    break

            if not equal:
                break
    else:
        equal = False
    return equal


def test_is_explanation_equal():
    """
    Tests ``_is_explanation_equal`` function included above in this test suite.
    """
    # yapf: disable
    dict1 = {'a': [('a1', 0.1), ('a2', 0.9)],
             'b': [('b1', 0.55), ('b2', 0.2222)],
             'c': [('c', 7)]}
    dict2 = {'a': [('a2', 0.8), ('a1', 0.0)],
             'b': [('b1', 0.5), ('b2', 0.2)],
             'c': [('c', 6.9)]}
    dict3 = {'a': [('a3', 0.8), ('a1', 0.0)],
             'b': [('b1', 0.5), ('b2', 0.2)],
             'c': [('c', 6.9)]}
    dict4 = {'a': [('a1', 0.05), ('a2', 0.88)],
             'b': [('b1', 0.5), ('b2', 0.2), ('b3', 0)],
             'c': [('c', 6.95)]}
    dict5 = {'a': [('a1', 0.1), ('a2', 0.9)],
             'b': [('b1', 0.5), ('b2', 0.2)],
             'c': [('c', 6.95)],
             'd': [('d1', 6)]}
    dict6 = {'a': [('a1', 0.1), ('a2', 0.9)],
             'b': [('b1', 0.5), ('b2', 0.2)],
             'c': [('c', 6.899999999)]}
    # yapf: enable

    assert _is_explanation_equal(dict1, dict1)
    assert _is_explanation_equal(dict1, dict2)
    assert not _is_explanation_equal(dict1, dict3)
    assert not _is_explanation_equal(dict1, dict4)
    assert not _is_explanation_equal(dict1, dict5)
    assert not _is_explanation_equal(dict1, dict6)

    assert _is_explanation_equal(dict2, dict1)
    assert _is_explanation_equal(dict2, dict2)
    assert not _is_explanation_equal(dict2, dict3)
    assert not _is_explanation_equal(dict2, dict4)
    assert not _is_explanation_equal(dict2, dict5)
    assert _is_explanation_equal(dict2, dict6)

    assert not _is_explanation_equal(dict3, dict1)
    assert not _is_explanation_equal(dict3, dict2)
    assert _is_explanation_equal(dict3, dict3)
    assert not _is_explanation_equal(dict3, dict4)
    assert not _is_explanation_equal(dict3, dict5)
    assert not _is_explanation_equal(dict3, dict6)

    assert not _is_explanation_equal(dict4, dict1)
    assert not _is_explanation_equal(dict4, dict2)
    assert not _is_explanation_equal(dict4, dict3)
    assert _is_explanation_equal(dict4, dict4)
    assert not _is_explanation_equal(dict4, dict5)
    assert not _is_explanation_equal(dict4, dict6)

    assert not _is_explanation_equal(dict5, dict1)
    assert not _is_explanation_equal(dict5, dict2)
    assert not _is_explanation_equal(dict5, dict3)
    assert not _is_explanation_equal(dict5, dict4)
    assert _is_explanation_equal(dict5, dict5)
    assert not _is_explanation_equal(dict5, dict6)

    assert not _is_explanation_equal(dict6, dict1)
    assert _is_explanation_equal(dict6, dict2)
    assert not _is_explanation_equal(dict6, dict3)
    assert not _is_explanation_equal(dict6, dict4)
    assert not _is_explanation_equal(dict6, dict5)
    assert _is_explanation_equal(dict6, dict6)


def test_lime_init():
    """
    Tests :mod:`fatf.transparency.lime.Lime` object initialisation.

    This only looks into cases where the initialisation would fail.
    """
    attribute_error = 'The following named parameters are not valid: {}.'
    shape_error_data = ('The data parameter must be a 2-dimensional numpy '
                        'array.')
    value_error_cat = 'LIME does not support non-numerical data arrays.'
    value_error = ("The mode must be either 'classification' or 'regression'. "
                   "'{}' given.")
    incompatible_model_error = ('LIME requires a model object to have a fit '
                                'method and optionally a predict_proba '
                                'method.')
    type_error_predictor = ('The predict_fn parameter is not callable -- it '
                            'has to be a function.')
    type_error_struct_indices = ('The categorical_features parameter either '
                                 'has to be a list, a numpy array or None.')
    incorrect_shape_struct_indices = ('categorical_features array/list is not '
                                      '1-dimensional.')
    value_error_struct_indices = ('Since categorical_features is an array of '
                                  'indices for a structured array, all of its '
                                  'elements should be strings.')
    value_error_struct_incorrect_indices = (
        'Indices given in the categorical_features parameter are not valid '
        'for the input data array.')
    #
    attribute_error_explain = ('The following named parameters are not valid: '
                               '{}.')
    incorrect_shape_error_explain = ('The instance to be explained should be '
                                     '1-dimensional.')
    value_error_explain = ('The instance to be explained should be purely '
                           'numerical -- LIME does not support categorical '
                           'features.')

    # Wrong named parameter
    with pytest.raises(AttributeError) as exin:
        ftl.Lime(NUMERICAL_NP_ARRAY, model=CLF, lorem='ipsum')
    assert str(exin.value) == attribute_error.format("{'lorem'}")

    # Not a 2-dimensional array
    with pytest.raises(IncorrectShapeError) as exin:
        ftl.Lime(np.ones((6, 4, 4)))
    assert str(exin.value) == shape_error_data

    # Not a numerical array
    with pytest.raises(ValueError) as exin:
        lime = ftl.Lime(np.ones((6, 4), dtype='U1'))
    assert str(exin.value) == value_error_cat

    # A structured data array with weird categorical indices type
    with pytest.raises(TypeError) as exin:
        ftl.Lime(NUMERICAL_STRUCT_ARRAY, categorical_features='')
    assert str(exin.value) == type_error_struct_indices

    # A structured data array with weird categorical indices shape
    with pytest.raises(IncorrectShapeError) as exin:
        ftl.Lime(NUMERICAL_STRUCT_ARRAY, categorical_features=[['a']])
    assert str(exin.value) == incorrect_shape_struct_indices

    # A structured data array with non-textual categorical indices
    with pytest.raises(ValueError) as exin:
        ftl.Lime(NUMERICAL_STRUCT_ARRAY, categorical_features=np.array([3, 2]))
    assert str(exin.value) == value_error_struct_indices

    # A structured data array with incorrect categorical indices
    with pytest.raises(ValueError) as exin:
        ftl.Lime(NUMERICAL_STRUCT_ARRAY, categorical_features=['a', 'e', 'b'])
    assert str(exin.value) == value_error_struct_incorrect_indices

    # Wrong operation mode
    with pytest.raises(ValueError) as exin:
        ftl.Lime(NUMERICAL_NP_ARRAY, mode='c')
    assert str(exin.value) == value_error.format('c')

    # Invalid model
    invalid_model = InvalidModel()
    with pytest.raises(IncompatibleModelError) as exin:
        ftl.Lime(
            NUMERICAL_NP_ARRAY, model=invalid_model, mode='classification')
    assert str(exin.value) == incompatible_model_error
    with pytest.raises(IncompatibleModelError) as exin:
        ftl.Lime(NUMERICAL_NP_ARRAY, model='a', mode='classification')
    assert str(exin.value) == incompatible_model_error

    # Invalid predictive function
    with pytest.raises(TypeError) as exin:
        ftl.Lime(NUMERICAL_NP_ARRAY, predict_fn='a', mode='regression')
    assert str(exin.value) == type_error_predictor

    ###########################################################################
    # Test explain_instance for exceptions and errors
    lime = ftl.Lime(NUMERICAL_NP_ARRAY)

    # Incorrect parameter
    with pytest.raises(AttributeError) as exin:
        lime.explain_instance(SAMPLE, weird_named_argument='yes')
    assert str(exin.value) == attribute_error_explain.format(
        "{'weird_named_argument'}")

    # Incorrect shape
    with pytest.raises(IncorrectShapeError) as exin:
        lime.explain_instance(NUMERICAL_STRUCT_ARRAY)
    assert str(exin.value) == incorrect_shape_error_explain

    # Not numerical
    with pytest.raises(ValueError) as exin:
        lime.explain_instance(np.ones((5, ), dtype='U1'))
    assert str(exin.value) == value_error_explain


def test_explain_instance_classification(caplog):
    """
    Tests :mod:`fatf.transparency.lime.Lime.explain_instance` method.

    These tests are for a classification task.
    """
    runtime_error_no_predictor = 'A predictive function is not available.'
    runtime_error_non_prob = ('The predictive model is not probabilistic. '
                              'Please specify a predictive function instead.')

    # Check logging
    assert len(caplog.records) == 0

    # Non-probabilistic model -- function -- probabilistic function
    with pytest.warns(UserWarning) as warning:
        lime = ftl.Lime(
            NUMERICAL_NP_ARRAY,
            model=CLF_NON_PROBA,
            predict_fn=CLF.predict_proba,
            class_names=CLASS_NAMES,
            feature_names=FEATURE_NAMES)
    assert len(warning) == 1
    assert str(warning[0].message) == USER_WARNING_MODEL_PRED
    explained = lime.explain_instance(SAMPLE, predict_fn=CLF.predict_proba)
    assert _is_explanation_equal(explained, NUMERICAL_RESULTS)
    # Non-probabilistic model -- function -- no function
    with pytest.warns(UserWarning) as warning:
        lime = ftl.Lime(
            NUMERICAL_NP_ARRAY,
            model=CLF_NON_PROBA,
            predict_fn=CLF.predict_proba,
            class_names=CLASS_NAMES,
            feature_names=FEATURE_NAMES)
    assert len(warning) == 1
    assert str(warning[0].message) == USER_WARNING_MODEL_PRED
    explained = lime.explain_instance(SAMPLE)
    assert _is_explanation_equal(explained, NUMERICAL_RESULTS)
    # Non-probabilistic model -- no function -- probabilistic function
    lime = ftl.Lime(
        NUMERICAL_NP_ARRAY,
        model=CLF_NON_PROBA,
        class_names=CLASS_NAMES,
        feature_names=FEATURE_NAMES)
    explained = lime.explain_instance(SAMPLE, predict_fn=CLF.predict_proba)
    assert _is_explanation_equal(explained, NUMERICAL_RESULTS)
    # Non-probabilistic model -- no function -- no function
    lime = ftl.Lime(
        NUMERICAL_NP_ARRAY,
        model=CLF_NON_PROBA,
        class_names=CLASS_NAMES,
        feature_names=FEATURE_NAMES)
    with pytest.raises(RuntimeError) as exin:
        lime.explain_instance(SAMPLE_STRUCT)
    assert str(exin.value) == runtime_error_non_prob

    # Check logging
    assert len(caplog.records) == 4
    for i in range(4):
        assert caplog.records[i].levelname == 'WARNING'
        assert caplog.records[i].getMessage() == LOG_WARNING

    # No model -- function -- probabilistic function
    lime = ftl.Lime(
        NUMERICAL_STRUCT_ARRAY,
        predict_fn=CLF.predict_proba,
        class_names=CLASS_NAMES,
        feature_names=FEATURE_NAMES)
    explained = lime.explain_instance(SAMPLE, predict_fn=CLF.predict_proba)
    assert _is_explanation_equal(explained, NUMERICAL_RESULTS)
    # No model -- function -- no function
    lime = ftl.Lime(
        NUMERICAL_STRUCT_ARRAY,
        predict_fn=CLF.predict_proba,
        class_names=CLASS_NAMES,
        feature_names=FEATURE_NAMES)
    explained = lime.explain_instance(SAMPLE)
    assert _is_explanation_equal(explained, NUMERICAL_RESULTS)
    # No model -- no function -- probabilistic function
    lime = ftl.Lime(
        NUMERICAL_NP_ARRAY,
        class_names=CLASS_NAMES,
        feature_names=FEATURE_NAMES)
    explained = lime.explain_instance(SAMPLE, predict_fn=CLF.predict_proba)
    assert _is_explanation_equal(explained, NUMERICAL_RESULTS)
    # No model -- no function -- no function
    lime = ftl.Lime(
        NUMERICAL_NP_ARRAY,
        class_names=CLASS_NAMES,
        feature_names=FEATURE_NAMES)
    with pytest.raises(RuntimeError) as exin:
        lime.explain_instance(SAMPLE)
    assert str(exin.value) == runtime_error_no_predictor

    # Check logging
    assert len(caplog.records) == 4

    # Probabilistic model -- probabilistic function -- empty call
    with pytest.warns(UserWarning) as warning:
        lime = ftl.Lime(
            NUMERICAL_NP_ARRAY,
            model=CLF,
            predict_fn=CLF.predict_proba,
            class_names=CLASS_NAMES,
            feature_names=FEATURE_NAMES)
    assert len(warning) == 1
    assert str(warning[0].message) == USER_WARNING_MODEL_PRED
    explained = lime.explain_instance(SAMPLE_STRUCT)
    assert _is_explanation_equal(explained, NUMERICAL_RESULTS)
    #
    # Probabilistic model -- probabilistic function -- non-empty call
    with pytest.warns(UserWarning) as warning:
        lime = ftl.Lime(
            NUMERICAL_NP_ARRAY,
            model=CLF,
            predict_fn=CLF.predict_proba,
            class_names=CLASS_NAMES,
            feature_names=FEATURE_NAMES)
    assert len(warning) == 1
    assert str(warning[0].message) == USER_WARNING_MODEL_PRED
    explained = lime.explain_instance(SAMPLE, predict_fn=CLF.predict_proba)
    assert _is_explanation_equal(explained, NUMERICAL_RESULTS)
    #
    # Probabilistic model -- no function -- empty call
    lime = ftl.Lime(
        NUMERICAL_STRUCT_ARRAY,
        model=CLF,
        class_names=CLASS_NAMES,
        feature_names=FEATURE_NAMES)
    explained = lime.explain_instance(SAMPLE)
    assert _is_explanation_equal(explained, NUMERICAL_RESULTS)
    #
    # Probabilistic model -- no function -- non-empty call
    lime = ftl.Lime(
        NUMERICAL_STRUCT_ARRAY,
        model=CLF,
        class_names=CLASS_NAMES,
        feature_names=FEATURE_NAMES)
    explained = lime.explain_instance(
        SAMPLE_STRUCT, predict_fn=CLF.predict_proba)
    assert _is_explanation_equal(explained, NUMERICAL_RESULTS)

    # Check logging
    assert len(caplog.records) == 4

    ###########################################################################
    # Test with categorical features: feat0 and feat1

    cat_feat = [0, 1]
    lime = ftl.Lime(
        NUMERICAL_NP_ARRAY,
        model=CLF,
        class_names=CLASS_NAMES,
        feature_names=FEATURE_NAMES,
        categorical_features=cat_feat)
    explained = lime.explain_instance(SAMPLE_STRUCT)
    assert _is_explanation_equal(CATEGORICAL_RESULTS, explained)

    cat_feat = ['a', 'b']
    lime = ftl.Lime(
        NUMERICAL_STRUCT_ARRAY,
        model=CLF,
        class_names=CLASS_NAMES,
        feature_names=FEATURE_NAMES,
        categorical_features=cat_feat)
    explained = lime.explain_instance(SAMPLE)
    assert _is_explanation_equal(CATEGORICAL_RESULTS, explained)

    # Check logging
    assert len(caplog.records) == 4


def test_explain_instance_regression(caplog):
    """
    Tests :mod:`fatf.transparency.lime.Lime.explain_instance` method.

    These tests are for a regression task.
    """
    # Check logging
    assert len(caplog.records) == 0

    # Regression a non-probabilistic model
    lime = ftl.Lime(
        NUMERICAL_STRUCT_ARRAY,
        mode='regression',
        model=CLF_NON_PROBA,
        class_names=CLASS_NAMES,
        feature_names=FEATURE_NAMES)
    explained = lime.explain_instance(SAMPLE)
    assert _is_explanation_equal({'a': explained}, {'a': REGRESSION_RESULTS})

    # Check logging
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == 'WARNING'
    assert caplog.records[0].getMessage() == LOG_WARNING

    # Regression a probabilistic model
    lime = ftl.Lime(
        NUMERICAL_NP_ARRAY,
        mode='regression',
        model=CLF,
        class_names=CLASS_NAMES,
        feature_names=FEATURE_NAMES)
    explained = lime.explain_instance(SAMPLE_STRUCT)
    assert _is_explanation_equal({'a': explained}, {'a': REGRESSION_RESULTS})

    # Regression with a model and function
    with pytest.warns(UserWarning) as warning:
        lime = ftl.Lime(
            NUMERICAL_STRUCT_ARRAY,
            mode='regression',
            model=CLF,
            predict_fn=CLF_NON_PROBA.predict,
            class_names=CLASS_NAMES,
            feature_names=FEATURE_NAMES)
    assert len(warning) == 1
    assert str(warning[0].message) == USER_WARNING_MODEL_PRED
    explained = lime.explain_instance(SAMPLE_STRUCT)
    assert _is_explanation_equal({'a': explained}, {'a': REGRESSION_RESULTS})

    # Regression without a model
    lime = ftl.Lime(
        NUMERICAL_NP_ARRAY,
        mode='regression',
        class_names=CLASS_NAMES,
        feature_names=FEATURE_NAMES)
    explained = lime.explain_instance(SAMPLE, predict_fn=CLF_NON_PROBA.predict)
    assert _is_explanation_equal({'a': explained}, {'a': REGRESSION_RESULTS})

    # Check logging
    assert len(caplog.records) == 1
