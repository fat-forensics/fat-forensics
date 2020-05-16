"""
Tests the LIME wrapper.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
#         Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import pytest

try:
    import lime
except ImportError:  # pragma: no cover
    pytest.skip(
        'Skipping lime wrapper tests -- lime missing.',
        allow_module_level=True)
else:
    del lime

import importlib
import sys

import numpy as np

import fatf.transparency.lime as ftl
import fatf.utils.models as fum
import fatf.utils.testing.imports as futi
import fatf.utils.testing.transparency as futt

from fatf.exceptions import IncompatibleModelError, IncorrectShapeError

# yapf: disable
FUTURE_WARNING = (
    'The LIME wrapper will be deprecated in FAT Forensics version '
    '0.0.3. Please consider using the TabularBlimeyLime explainer '
    'class implemented in the fatf.transparency.predictions.'
    'surrogate_explainers module instead. Alternatively, you may '
    'consider building a custom surrogate explainer using the '
    'functionality implemented in FAT Forensics -- see the *Tabular '
    'Surrogates* how-to guide for more details.')

SAMPLE = np.array([0, 1, 0.08, 0.54])
SAMPLE_STRUCT = np.array(
    [(0, 1, 0.08, 0.54)],
    dtype=[('a', 'i'), ('b', 'i'), ('c', 'f'), ('d', 'f')])[0]
CLF = fum.KNN()
CLF.fit(futt.NUMERICAL_NP_ARRAY, futt.LABELS)
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

CLF_NON_PROBA = futt.NonProbabilisticModel(CLF.predict)


def test_import_when_missing():
    """
    Tests importing :mod:`fatf.transparency.lime` module when LIME is missing.
    """
    assert 'fatf.transparency.lime' in sys.modules
    exception_msg = (
        'Lime package is not installed on your system. You must install it in '
        'order to use the fatf.transparency.lime module. One possibility is '
        'to install LIME alongside this package with: pip install '
        'fat-forensics[lime].')
    with futi.module_import_tester('lime', when_missing=True):
        with pytest.raises(ImportError) as exin:
            importlib.reload(ftl)
        assert str(exin.value) == exception_msg
    assert 'fatf.transparency.lime' in sys.modules


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
    with pytest.warns(FutureWarning) as warning:
        with pytest.raises(AttributeError) as exin:
            ftl.Lime(futt.NUMERICAL_NP_ARRAY, model=CLF, lorem='ipsum')
        assert str(exin.value) == attribute_error.format("{'lorem'}")
    assert len(warning) == 1
    assert str(warning[0].message) == FUTURE_WARNING

    # Not a 2-dimensional array
    with pytest.warns(FutureWarning) as warning:
        with pytest.raises(IncorrectShapeError) as exin:
            ftl.Lime(np.ones((6, 4, 4)))
        assert str(exin.value) == shape_error_data
    assert len(warning) == 1
    assert str(warning[0].message) == FUTURE_WARNING

    # Not a numerical array
    with pytest.warns(FutureWarning) as warning:
        with pytest.raises(ValueError) as exin:
            lime = ftl.Lime(np.ones((6, 4), dtype='U1'))
        assert str(exin.value) == value_error_cat
    assert len(warning) == 1
    assert str(warning[0].message) == FUTURE_WARNING

    # A structured data array with weird categorical indices type
    with pytest.warns(FutureWarning) as warning:
        with pytest.raises(TypeError) as exin:
            ftl.Lime(futt.NUMERICAL_STRUCT_ARRAY, categorical_features='')
        assert str(exin.value) == type_error_struct_indices
    assert len(warning) == 1
    assert str(warning[0].message) == FUTURE_WARNING

    # A structured data array with weird categorical indices shape
    with pytest.warns(FutureWarning) as warning:
        with pytest.raises(IncorrectShapeError) as exin:
            ftl.Lime(futt.NUMERICAL_STRUCT_ARRAY, categorical_features=[['a']])
        assert str(exin.value) == incorrect_shape_struct_indices
    assert len(warning) == 1
    assert str(warning[0].message) == FUTURE_WARNING

    # A structured data array with non-textual categorical indices
    with pytest.warns(FutureWarning) as warning:
        with pytest.raises(ValueError) as exin:
            ftl.Lime(
                futt.NUMERICAL_STRUCT_ARRAY,
                categorical_features=np.array([3, 2]))
        assert str(exin.value) == value_error_struct_indices
    assert len(warning) == 1
    assert str(warning[0].message) == FUTURE_WARNING

    # A structured data array with incorrect categorical indices
    with pytest.warns(FutureWarning) as warning:
        with pytest.raises(ValueError) as exin:
            ftl.Lime(
                futt.NUMERICAL_STRUCT_ARRAY,
                categorical_features=['a', 'e', 'b'])
        assert str(exin.value) == value_error_struct_incorrect_indices
    assert len(warning) == 1
    assert str(warning[0].message) == FUTURE_WARNING

    # Wrong operation mode
    with pytest.warns(FutureWarning) as warning:
        with pytest.raises(ValueError) as exin:
            ftl.Lime(futt.NUMERICAL_NP_ARRAY, mode='c')
        assert str(exin.value) == value_error.format('c')
    assert len(warning) == 1
    assert str(warning[0].message) == FUTURE_WARNING

    # Invalid model
    invalid_model = futt.InvalidModel()
    with pytest.warns(FutureWarning) as warning:
        with pytest.raises(IncompatibleModelError) as exin:
            ftl.Lime(
                futt.NUMERICAL_NP_ARRAY,
                model=invalid_model,
                mode='classification')
        assert str(exin.value) == incompatible_model_error
    assert len(warning) == 1
    assert str(warning[0].message) == FUTURE_WARNING
    with pytest.warns(FutureWarning) as warning:
        with pytest.raises(IncompatibleModelError) as exin:
            ftl.Lime(futt.NUMERICAL_NP_ARRAY, model='a', mode='classification')
        assert str(exin.value) == incompatible_model_error
    assert len(warning) == 1
    assert str(warning[0].message) == FUTURE_WARNING

    # Invalid predictive function
    with pytest.warns(FutureWarning) as warning:
        with pytest.raises(TypeError) as exin:
            ftl.Lime(
                futt.NUMERICAL_NP_ARRAY, predict_fn='a', mode='regression')
        assert str(exin.value) == type_error_predictor
    assert len(warning) == 1
    assert str(warning[0].message) == FUTURE_WARNING

    ###########################################################################
    # Test explain_instance for exceptions and errors
    with pytest.warns(FutureWarning) as warning:
        lime = ftl.Lime(futt.NUMERICAL_NP_ARRAY)
    assert len(warning) == 1
    assert str(warning[0].message) == FUTURE_WARNING

    # Incorrect parameter
    with pytest.raises(AttributeError) as exin:
        lime.explain_instance(SAMPLE, weird_named_argument='yes')
    assert str(exin.value) == attribute_error_explain.format(
        "{'weird_named_argument'}")

    # Incorrect shape
    with pytest.raises(IncorrectShapeError) as exin:
        lime.explain_instance(futt.NUMERICAL_STRUCT_ARRAY)
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
    with pytest.warns(None) as warning:
        lime = ftl.Lime(
            futt.NUMERICAL_NP_ARRAY,
            model=CLF_NON_PROBA,
            predict_fn=CLF.predict_proba,
            class_names=CLASS_NAMES,
            feature_names=FEATURE_NAMES)
    assert len(warning) == 2
    assert str(warning[0].message) == FUTURE_WARNING
    assert str(warning[1].message) == USER_WARNING_MODEL_PRED
    explained = lime.explain_instance(SAMPLE, predict_fn=CLF.predict_proba)
    assert futt.is_explanation_equal_list(explained, NUMERICAL_RESULTS)
    # Non-probabilistic model -- function -- no function
    with pytest.warns(None) as warning:
        lime = ftl.Lime(
            futt.NUMERICAL_NP_ARRAY,
            model=CLF_NON_PROBA,
            predict_fn=CLF.predict_proba,
            class_names=CLASS_NAMES,
            feature_names=FEATURE_NAMES)
    assert len(warning) == 2
    assert str(warning[0].message) == FUTURE_WARNING
    assert str(warning[1].message) == USER_WARNING_MODEL_PRED
    explained = lime.explain_instance(SAMPLE)
    assert futt.is_explanation_equal_list(explained, NUMERICAL_RESULTS)
    # Non-probabilistic model -- no function -- probabilistic function
    with pytest.warns(FutureWarning) as warning:
        lime = ftl.Lime(
            futt.NUMERICAL_NP_ARRAY,
            model=CLF_NON_PROBA,
            class_names=CLASS_NAMES,
            feature_names=FEATURE_NAMES)
    assert len(warning) == 1
    assert str(warning[0].message) == FUTURE_WARNING
    explained = lime.explain_instance(SAMPLE, predict_fn=CLF.predict_proba)
    assert futt.is_explanation_equal_list(explained, NUMERICAL_RESULTS)
    # Non-probabilistic model -- no function -- no function
    with pytest.warns(FutureWarning) as warning:
        lime = ftl.Lime(
            futt.NUMERICAL_NP_ARRAY,
            model=CLF_NON_PROBA,
            class_names=CLASS_NAMES,
            feature_names=FEATURE_NAMES)
    assert len(warning) == 1
    assert str(warning[0].message) == FUTURE_WARNING
    with pytest.raises(RuntimeError) as exin:
        lime.explain_instance(SAMPLE_STRUCT)
    assert str(exin.value) == runtime_error_non_prob

    # Check logging
    assert len(caplog.records) == 4
    for i in range(4):
        assert caplog.records[i].levelname == 'WARNING'
        assert caplog.records[i].getMessage() == LOG_WARNING

    # No model -- function -- probabilistic function
    with pytest.warns(FutureWarning) as warning:
        lime = ftl.Lime(
            futt.NUMERICAL_STRUCT_ARRAY,
            predict_fn=CLF.predict_proba,
            class_names=CLASS_NAMES,
            feature_names=FEATURE_NAMES)
    assert len(warning) == 1
    assert str(warning[0].message) == FUTURE_WARNING
    explained = lime.explain_instance(SAMPLE, predict_fn=CLF.predict_proba)
    assert futt.is_explanation_equal_list(explained, NUMERICAL_RESULTS)
    # No model -- function -- no function
    with pytest.warns(FutureWarning) as warning:
        lime = ftl.Lime(
            futt.NUMERICAL_STRUCT_ARRAY,
            predict_fn=CLF.predict_proba,
            class_names=CLASS_NAMES,
            feature_names=FEATURE_NAMES)
    assert len(warning) == 1
    assert str(warning[0].message) == FUTURE_WARNING
    explained = lime.explain_instance(SAMPLE)
    assert futt.is_explanation_equal_list(explained, NUMERICAL_RESULTS)
    # No model -- no function -- probabilistic function
    with pytest.warns(FutureWarning) as warning:
        lime = ftl.Lime(
            futt.NUMERICAL_NP_ARRAY,
            class_names=CLASS_NAMES,
            feature_names=FEATURE_NAMES)
    assert len(warning) == 1
    assert str(warning[0].message) == FUTURE_WARNING
    explained = lime.explain_instance(SAMPLE, predict_fn=CLF.predict_proba)
    assert futt.is_explanation_equal_list(explained, NUMERICAL_RESULTS)
    # No model -- no function -- no function
    with pytest.warns(FutureWarning) as warning:
        lime = ftl.Lime(
            futt.NUMERICAL_NP_ARRAY,
            class_names=CLASS_NAMES,
            feature_names=FEATURE_NAMES)
    assert len(warning) == 1
    assert str(warning[0].message) == FUTURE_WARNING
    with pytest.raises(RuntimeError) as exin:
        lime.explain_instance(SAMPLE)
    assert str(exin.value) == runtime_error_no_predictor

    # Check logging
    assert len(caplog.records) == 4

    # Probabilistic model -- probabilistic function -- empty call
    with pytest.warns(None) as warning:
        lime = ftl.Lime(
            futt.NUMERICAL_NP_ARRAY,
            model=CLF,
            predict_fn=CLF.predict_proba,
            class_names=CLASS_NAMES,
            feature_names=FEATURE_NAMES)
    assert len(warning) == 2
    assert str(warning[0].message) == FUTURE_WARNING
    assert str(warning[1].message) == USER_WARNING_MODEL_PRED
    explained = lime.explain_instance(SAMPLE_STRUCT)
    assert futt.is_explanation_equal_list(explained, NUMERICAL_RESULTS)
    #
    # Probabilistic model -- probabilistic function -- non-empty call
    with pytest.warns(None) as warning:
        lime = ftl.Lime(
            futt.NUMERICAL_NP_ARRAY,
            model=CLF,
            predict_fn=CLF.predict_proba,
            class_names=CLASS_NAMES,
            feature_names=FEATURE_NAMES)
    assert len(warning) == 2
    assert str(warning[0].message) == FUTURE_WARNING
    assert str(warning[1].message) == USER_WARNING_MODEL_PRED
    explained = lime.explain_instance(SAMPLE, predict_fn=CLF.predict_proba)
    assert futt.is_explanation_equal_list(explained, NUMERICAL_RESULTS)
    #
    # Probabilistic model -- no function -- empty call
    with pytest.warns(FutureWarning) as warning:
        lime = ftl.Lime(
            futt.NUMERICAL_STRUCT_ARRAY,
            model=CLF,
            class_names=CLASS_NAMES,
            feature_names=FEATURE_NAMES)
    assert len(warning) == 1
    assert str(warning[0].message) == FUTURE_WARNING
    explained = lime.explain_instance(SAMPLE)
    assert futt.is_explanation_equal_list(explained, NUMERICAL_RESULTS)
    #
    # Probabilistic model -- no function -- non-empty call
    with pytest.warns(FutureWarning) as warning:
        lime = ftl.Lime(
            futt.NUMERICAL_STRUCT_ARRAY,
            model=CLF,
            class_names=CLASS_NAMES,
            feature_names=FEATURE_NAMES)
    assert len(warning) == 1
    assert str(warning[0].message) == FUTURE_WARNING
    explained = lime.explain_instance(
        SAMPLE_STRUCT, predict_fn=CLF.predict_proba)
    assert futt.is_explanation_equal_list(explained, NUMERICAL_RESULTS)

    # Check logging
    assert len(caplog.records) == 4

    ###########################################################################
    # Test with categorical features: feat0 and feat1

    cat_feat = [0, 1]
    with pytest.warns(FutureWarning) as warning:
        lime = ftl.Lime(
            futt.NUMERICAL_NP_ARRAY,
            model=CLF,
            class_names=CLASS_NAMES,
            feature_names=FEATURE_NAMES,
            categorical_features=cat_feat)
    assert len(warning) == 1
    assert str(warning[0].message) == FUTURE_WARNING
    explained = lime.explain_instance(SAMPLE_STRUCT)
    assert futt.is_explanation_equal_list(CATEGORICAL_RESULTS, explained)

    cat_feat = ['a', 'b']
    with pytest.warns(FutureWarning) as warning:
        lime = ftl.Lime(
            futt.NUMERICAL_STRUCT_ARRAY,
            model=CLF,
            class_names=CLASS_NAMES,
            feature_names=FEATURE_NAMES,
            categorical_features=cat_feat)
    assert len(warning) == 1
    assert str(warning[0].message) == FUTURE_WARNING
    explained = lime.explain_instance(SAMPLE)
    assert futt.is_explanation_equal_list(CATEGORICAL_RESULTS, explained)

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
    with pytest.warns(FutureWarning) as warning:
        lime = ftl.Lime(
            futt.NUMERICAL_STRUCT_ARRAY,
            mode='regression',
            model=CLF_NON_PROBA,
            class_names=CLASS_NAMES,
            feature_names=FEATURE_NAMES)
    assert len(warning) == 1
    assert str(warning[0].message) == FUTURE_WARNING
    explained = lime.explain_instance(SAMPLE)
    assert futt.is_explanation_equal_list({'a': explained},
                                          {'a': REGRESSION_RESULTS})

    # Check logging
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == 'WARNING'
    assert caplog.records[0].getMessage() == LOG_WARNING

    # Regression a probabilistic model
    with pytest.warns(FutureWarning) as warning:
        lime = ftl.Lime(
            futt.NUMERICAL_NP_ARRAY,
            mode='regression',
            model=CLF,
            class_names=CLASS_NAMES,
            feature_names=FEATURE_NAMES)
    assert len(warning) == 1
    assert str(warning[0].message) == FUTURE_WARNING
    explained = lime.explain_instance(SAMPLE_STRUCT)
    assert futt.is_explanation_equal_list({'a': explained},
                                          {'a': REGRESSION_RESULTS})

    # Regression with a model and function
    with pytest.warns(None) as warning:
        lime = ftl.Lime(
            futt.NUMERICAL_STRUCT_ARRAY,
            mode='regression',
            model=CLF,
            predict_fn=CLF_NON_PROBA.predict,
            class_names=CLASS_NAMES,
            feature_names=FEATURE_NAMES)
    assert len(warning) == 2
    assert str(warning[0].message) == FUTURE_WARNING
    assert str(warning[1].message) == USER_WARNING_MODEL_PRED
    explained = lime.explain_instance(SAMPLE_STRUCT)
    assert futt.is_explanation_equal_list({'a': explained},
                                          {'a': REGRESSION_RESULTS})

    # Regression without a model
    with pytest.warns(FutureWarning) as warning:
        lime = ftl.Lime(
            futt.NUMERICAL_NP_ARRAY,
            mode='regression',
            class_names=CLASS_NAMES,
            feature_names=FEATURE_NAMES)
    assert len(warning) == 1
    assert str(warning[0].message) == FUTURE_WARNING
    explained = lime.explain_instance(SAMPLE, predict_fn=CLF_NON_PROBA.predict)
    assert futt.is_explanation_equal_list({'a': explained},
                                          {'a': REGRESSION_RESULTS})

    # Check logging
    assert len(caplog.records) == 1
