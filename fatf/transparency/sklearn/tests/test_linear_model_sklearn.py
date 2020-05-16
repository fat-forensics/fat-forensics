"""
This module tests scikit-learn linear models explainers.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import pytest

try:
    import sklearn
except ImportError:  # pragma: no cover
    pytest.skip(
        'Skipping tests of scikit-learn linear models explainers '
        '-- scikit-learn is not installed.',
        allow_module_level=True)

import sklearn.cluster
import sklearn.discriminant_analysis
import sklearn.dummy
import sklearn.exceptions
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.neighbors
import sklearn.svm
import sklearn.tree

import numpy as np

import fatf

import fatf.transparency.sklearn.linear_model as ftsl
import fatf.utils.tools as fut

_SKLEARN_VERSION = [int(i) for i in sklearn.__version__.split('.')[:2]]
_SKLEARN_0_20 = fut.at_least_verion([0, 20], _SKLEARN_VERSION)
_SKLEARN_0_22 = fut.at_least_verion([0, 22], _SKLEARN_VERSION)
_SKLEARN_0_23 = fut.at_least_verion([0, 23], _SKLEARN_VERSION)

# yapf: disable
LINEAR_CLASSIFIERS = [
    sklearn.svm.LinearSVC,
    sklearn.linear_model.RidgeClassifier,
    sklearn.linear_model.RidgeClassifierCV,
    sklearn.discriminant_analysis.LinearDiscriminantAnalysis,
    #
    sklearn.linear_model.LogisticRegression,
    sklearn.linear_model.LogisticRegressionCV
]
# These three linear models get different results in different Python versions,
# hence will not be tested for parameters.
LINEAR_CLASSIFIERS_ = [
    sklearn.linear_model.SGDClassifier,
    sklearn.linear_model.PassiveAggressiveClassifier,
    sklearn.linear_model.Perceptron
]
LINEAR_REGRESSORS = [
    sklearn.linear_model.BayesianRidge,
    sklearn.linear_model.ElasticNet,
    sklearn.linear_model.ElasticNetCV,
    sklearn.linear_model.Lasso,
    sklearn.linear_model.LassoCV,
    sklearn.linear_model.Lars,
    sklearn.linear_model.LarsCV,
    sklearn.linear_model.LassoLars,
    sklearn.linear_model.LassoLarsCV,
    sklearn.linear_model.LassoLarsIC,
    sklearn.linear_model.ARDRegression,
    sklearn.linear_model.HuberRegressor,
    sklearn.linear_model.TheilSenRegressor,
    sklearn.linear_model.OrthogonalMatchingPursuit,
    sklearn.linear_model.OrthogonalMatchingPursuitCV,
    #
    sklearn.linear_model.Ridge,
    sklearn.linear_model.RidgeCV,
    #
    sklearn.linear_model.PassiveAggressiveRegressor,
    sklearn.linear_model.SGDRegressor
]
LINEAR_REGRESSORS_ = [
    sklearn.linear_model.LinearRegression,
    sklearn.svm.LinearSVR
]
LINEAR_MULTITASK_REGRESSORS = [
    sklearn.linear_model.MultiTaskLasso,
    sklearn.linear_model.MultiTaskLassoCV,
    sklearn.linear_model.MultiTaskElasticNet,
    sklearn.linear_model.MultiTaskElasticNetCV
]
NON_LINEAR_MODELS = [
    sklearn.cluster.KMeans,
    sklearn.dummy.DummyRegressor,
    sklearn.naive_bayes.BernoulliNB,
    sklearn.neighbors.KNeighborsClassifier,
    sklearn.neighbors.KNeighborsRegressor,
    sklearn.tree.DecisionTreeClassifier,
    sklearn.tree.DecisionTreeRegressor
]
NON_LINEAR_MODELS_ = [
    sklearn.dummy.DummyClassifier,
    sklearn.linear_model.RANSACRegressor
]

LINEAR_REG_COEF = [
    np.array([0.001, -0.003, 0.004, -0.008]),
    np.array([0.000, -0.002, 0.000, -0.009]),
    np.array([0.000, -0.002, 0.000, -0.009]),
    np.array([0., -0.001, 0., -0.009]),
    np.array([0., -0.002, 0., -0.009]),
    np.array([0.035, -0.004, 0.020, -0.005]),
    np.array([0.001, 0., 0., -0.007]),
    np.array([0., 0., 0., 0.]),
    np.array([0.001, 0., 0., -0.007]),
    np.array([0., 0., 0., -0.007]),
    np.array([0.104, 0.000, 0.028, 0.000]),
    np.array([0.029, -0.004, 0.018, -0.005]),
    np.array([-0.028, -0.039, 0.028, -0.007]),
    np.array([0., 0., 0., -0.009]),
    np.array([0., 0., 0., -0.009]),
    np.array([0.033, -0.003, 0.019, -0.005]),
    np.array([0.021, -0.003, 0.017, -0.006]),
    #
    np.array([0.017, -0.003, 0.040, -0.005]),
    np.array([1.219, 10.356, -0.982, -19.025])  # / 1e+10
]
if not _SKLEARN_0_20:  # pragma: nocover
    LINEAR_REG_COEF[17] = np.array([0.013, 0.001, 0.033, -0.008])
    LINEAR_REG_COEF[18] = np.array([-2.663, -1.089, -14.857, 23.487])
LINEAR_REG_COEF_ = [
    np.array([0.035, -0.004, 0.020, -0.005]),
    np.array([0.012, 0.007, 0.027, -0.016])
]
LINEAR_CLF_COEF_36 = [
    np.array([[-28.064, -84.191, -65.482, -299.345]]),
    np.array([[0.001, -0.007, 0.004, -0.042]]),
    np.array([[-3., -9., -7., -32.]]),
]
LINEAR_CLF_COEF_37 = [
    np.array([[84.191, 9.355, 130.964, -514.500]]),
    np.array([[0.003, -0.003, 0.008, -0.038]]),
    np.array([[9., 1., 14., -55.]])
]
LINEAR_CLF_COEF = [
    np.array([[0.004, -0.003, 0.013, -0.033]]),
    np.array([[0.065, -0.007, 0.039, -0.011]]),
    np.array([[0.042, -0.007, 0.033, -0.012]]),
    np.array([[2.220, -0.224, 1.261, -0.326]]),
    np.array([[0.021, -0.024, 0.063, -0.195]]),
    np.array([[0.001, -0.003, 0.001, -0.015]])
]
if _SKLEARN_0_23:  # pragma: nocover
    LINEAR_CLF_COEF[2] = np.array([0.069, -0.007, 0.039, -0.010])
LINEAR_MUL_REG_COEF = [
    np.array([[0., -0.001, 0., -0.009], [0., -0.001, 0., -0.009]]),
    np.array([[0., -0.002, 0., -0.009], [0., -0.002, 0., -0.009]]),
    np.array([[0., -0.002, 0., -0.009], [0., -0.002, 0., -0.009]]),
    np.array([[0., -0.002, 0., -0.009], [0., -0.002, 0., -0.009]])
]
# yapf: enable

DATA = [[0, 5, 1, 55], [1, 12, 12, 64], [2, 88, 12, 43], [3, 9, 7, 32],
        [5, 9, -2, 71], [6, 1, 19, -22], [7, 1, 14, -33], [8, 1, 14, -44],
        [9, 1, 14, -55], [10, 11, 14, -66]]
LABELS = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
LABELS_MULTITASK = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 1], [1, 1],
                    [1, 1], [1, 1], [1, 1]]


def get_kwargs(clf_name):
    """
    Prepares positional arguments for sklearn model initialisation.

    Parameters
    ----------
    clf_name : string
        The name of the scikit-learn classifer.

    Returns
    -------
    kwargs : Dictionary[string, Union[string, number]]
        A dictionary with named arguments appropriate for the selected model.
    """
    assert isinstance(clf_name, str), 'Classifier name must be a string.'

    if clf_name.endswith('CV'):
        kwargs = dict(cv=5)
    elif clf_name == 'LogisticRegression':
        kwargs = dict(solver='liblinear')
    elif (clf_name.startswith('PassiveAggressive')
          or clf_name in ('Perceptron', 'SGDClassifier', 'SGDRegressor')):
        kwargs = dict(max_iter=40, tol=1e-1, random_state=42)
    # This model is not currently tested.
    # elif clf_name == 'LinearSVR':
    #     kwargs = dict(max_iter=5, tol=1, random_state=42)
    else:
        kwargs = dict()
    return kwargs


def test_validate_classifier_list():
    """
    Validates the list of all the scikit-learn models used for testing.
    """
    for clf in LINEAR_CLASSIFIERS + LINEAR_CLASSIFIERS_:
        name = clf.__name__
        kwargs = get_kwargs(name)
        clf_instance = clf(**kwargs)
        clf_instance.fit(DATA, LABELS)

        assert hasattr(clf_instance, 'classes_')

    for clf in LINEAR_REGRESSORS:
        name = clf.__name__
        kwargs = get_kwargs(name)
        clf_instance = clf(**kwargs)
        clf_instance.fit(DATA, LABELS)

        assert not hasattr(clf_instance, 'classes_')

    for clf in LINEAR_MULTITASK_REGRESSORS:
        name = clf.__name__
        kwargs = get_kwargs(name)
        clf_instance = clf(**kwargs)
        clf_instance.fit(DATA, LABELS_MULTITASK)

        assert not hasattr(clf_instance, 'classes_')


def test_is_scikit_linear():
    """
    Tests :func:`fatf.transparency.sklearn.linear_model._is_scikit_linear`.
    """
    assertion_error_clf = 'Invalid sklearn predictor.'

    clf_instance = object()
    with pytest.raises(AssertionError) as excinfo:
        ftsl._is_scikit_linear(clf_instance)
    assert str(excinfo.value) == assertion_error_clf
    clf_instance = 'object'
    with pytest.raises(AssertionError) as excinfo:
        ftsl._is_scikit_linear(clf_instance)
    assert str(excinfo.value) == assertion_error_clf
    clf_instance = object
    with pytest.raises(AssertionError) as excinfo:
        ftsl._is_scikit_linear(clf_instance)
    assert str(excinfo.value) == assertion_error_clf

    for clf in NON_LINEAR_MODELS + NON_LINEAR_MODELS_:
        clf_instance = clf()
        assert ftsl._is_scikit_linear(clf_instance) is False

    mdl = (LINEAR_CLASSIFIERS + LINEAR_CLASSIFIERS_ + LINEAR_REGRESSORS
           + LINEAR_REGRESSORS_ + LINEAR_MULTITASK_REGRESSORS)  # yapf: disable
    for clf in mdl:
        clf_instance = clf()
        assert ftsl._is_scikit_linear(clf_instance) is True


def test_is_fitted_linear():
    """
    Tests :func:`fatf.transparency.sklearn.linear_model._is_fitted_linear`.
    """
    unfit_error = ("This {{}} instance is not fitted yet. Call 'fit' with "
                   'appropriate arguments before using this {}.'.format(
                       'estimator' if _SKLEARN_0_22 else 'method'))

    for clf in LINEAR_CLASSIFIERS + LINEAR_CLASSIFIERS_ + LINEAR_REGRESSORS:
        name = clf.__name__
        kwargs = get_kwargs(name)
        clf_instance = clf(**kwargs)

        with pytest.raises(sklearn.exceptions.NotFittedError) as excinfo:
            ftsl._is_fitted_linear(clf_instance)
        msg = unfit_error.format(clf_instance.__class__.__name__)
        assert str(excinfo.value) == msg

        clf_instance.fit(DATA, LABELS)
        assert ftsl._is_fitted_linear(clf_instance)

    for clf in LINEAR_MULTITASK_REGRESSORS:
        name = clf.__name__
        kwargs = get_kwargs(name)
        clf_instance = clf(**kwargs)

        with pytest.raises(sklearn.exceptions.NotFittedError) as excinfo:
            ftsl._is_fitted_linear(clf_instance)
        msg = unfit_error.format(clf_instance.__class__.__name__)
        assert str(excinfo.value) == msg

        clf_instance.fit(DATA, LABELS_MULTITASK)
        assert ftsl._is_fitted_linear(clf_instance)


def test_linear_classifier_coefficients():
    """
    Tests linear scikit-learn classifier coefficient extraction.

    Tests :func:`fatf.transparency.sklearn.linear_model.\
linear_classifier_coefficients` function.
    """
    fatf.setup_random_seed()

    type_error = ('This functionality is designated for linear-like '
                  'scikit-learn predictor instances only. Instead got: {}.')
    unfit_error = ("This {{}} instance is not fitted yet. Call 'fit' with "
                   'appropriate arguments before using this {}.'.format(
                       'estimator' if _SKLEARN_0_22 else 'method'))

    for clf in NON_LINEAR_MODELS:
        clf_instance = clf()
        clf_instance.fit(DATA, LABELS)

        with pytest.raises(TypeError) as excinfo:
            ftsl.linear_classifier_coefficients(clf_instance)
        name = str(clf).strip("<>' ")[7:]
        assert str(excinfo.value) == type_error.format(name)

    for i, clf in enumerate(LINEAR_REGRESSORS):
        name = clf.__name__
        kwargs = get_kwargs(name)
        clf_instance = clf(**kwargs)

        with pytest.raises(sklearn.exceptions.NotFittedError) as excinfo:
            ftsl.linear_classifier_coefficients(clf_instance)
        msg = unfit_error.format(clf_instance.__class__.__name__)
        assert str(excinfo.value) == msg

        clf_instance.fit(DATA, LABELS)

        coef = ftsl.linear_classifier_coefficients(clf_instance)
        if name == 'SGDRegressor':
            assert np.allclose(coef / 1e+10, LINEAR_REG_COEF[i], atol=1e-3)
        else:
            assert np.allclose(coef, LINEAR_REG_COEF[i], atol=1e-3)

    for i, clf in enumerate(LINEAR_CLASSIFIERS):
        name = clf.__name__
        kwargs = get_kwargs(name)
        clf_instance = clf(**kwargs)

        with pytest.raises(sklearn.exceptions.NotFittedError) as excinfo:
            ftsl.linear_classifier_coefficients(clf_instance)
        msg = unfit_error.format(clf_instance.__class__.__name__)
        assert str(excinfo.value) == msg

        clf_instance.fit(DATA, LABELS)

        coef = ftsl.linear_classifier_coefficients(clf_instance)
        assert np.allclose(coef, LINEAR_CLF_COEF[i], atol=1e-3)

    for i, clf in enumerate(LINEAR_MULTITASK_REGRESSORS):
        name = clf.__name__
        kwargs = get_kwargs(name)
        clf_instance = clf(**kwargs)

        with pytest.raises(sklearn.exceptions.NotFittedError) as excinfo:
            ftsl.linear_classifier_coefficients(clf_instance)
        msg = unfit_error.format(clf_instance.__class__.__name__)
        assert str(excinfo.value) == msg

        clf_instance.fit(DATA, LABELS_MULTITASK)

        coef = ftsl.linear_classifier_coefficients(clf_instance)
        assert np.allclose(coef, LINEAR_MUL_REG_COEF[i], atol=1e-3)


class TestSKLearnLinearModelExplainer(object):
    """
    Tests the ``SKLearnLinearModelExplainer`` class.

    Tests the
    :class:`fatf.transparency.sklearn.linear_model.SKLearnLinearModelExplainer`
    class.
    """

    feature_names = ['feature {}'.format(i) for i in range(4)]
    class_names = ['class {}'.format(i) for i in range(2)]

    def test_linear_classifiers(self):
        """
        Tests ``SKLearnLinearModelExplainer`` with linear classifiers.
        """
        fatf.setup_random_seed()

        for i, clf in enumerate(LINEAR_CLASSIFIERS):
            name = clf.__name__
            kwargs = get_kwargs(name)
            clf_instance = clf(**kwargs)
            clf_instance.fit(DATA, LABELS)

            ske = ftsl.SKLearnLinearModelExplainer(
                clf_instance, self.feature_names, self.class_names)
            #
            assert ske.clf == clf_instance
            assert ske.is_classifier is True
            assert ske.feature_names == self.feature_names
            assert ske.class_names == self.class_names
            assert ske.features_number == 4
            assert np.array_equal(ske.classes_array, [0, 1])

            coef = ske.feature_importance()
            assert np.allclose(coef, LINEAR_CLF_COEF[i], atol=1e-3)

    def test_linear_regressors(self):
        """
        Tests ``SKLearnLinearModelExplainer`` with linear regressors.
        """
        fatf.setup_random_seed()

        for i, clf in enumerate(LINEAR_REGRESSORS):
            name = clf.__name__
            kwargs = get_kwargs(name)
            clf_instance = clf(**kwargs)
            clf_instance.fit(DATA, LABELS)

            ske = ftsl.SKLearnLinearModelExplainer(
                clf_instance, feature_names=self.feature_names)
            #
            assert ske.clf == clf_instance
            assert ske.is_classifier is False
            assert ske.feature_names == self.feature_names
            assert ske.class_names is None
            assert ske.features_number == 4
            assert ske.classes_array is None

            coef = ske.feature_importance()
            if name == 'SGDRegressor':
                assert np.allclose(coef / 1e+10, LINEAR_REG_COEF[i], atol=1e-3)
            else:
                assert np.allclose(coef, LINEAR_REG_COEF[i], atol=1e-3)

    def test_linear_multitask_regressors(self):
        """
        Tests ``SKLearnLinearModelExplainer`` with linear multitask regressors.
        """
        fatf.setup_random_seed()

        for i, clf in enumerate(LINEAR_MULTITASK_REGRESSORS):
            name = clf.__name__
            kwargs = get_kwargs(name)
            clf_instance = clf(**kwargs)
            clf_instance.fit(DATA, LABELS_MULTITASK)

            ske = ftsl.SKLearnLinearModelExplainer(
                clf_instance, feature_names=self.feature_names)
            #
            assert ske.clf == clf_instance
            assert ske.is_classifier is False
            assert ske.feature_names == self.feature_names
            assert ske.class_names is None
            assert ske.features_number == 4
            assert ske.classes_array is None

            coef = ske.feature_importance()
            assert np.allclose(coef, LINEAR_MUL_REG_COEF[i], atol=1e-3)

    def test_non_linear_models(self):
        """
        Tests ``SKLearnLinearModelExplainer`` with non-linear models.
        """
        type_error = ('This functionality is designated for linear-like '
                      'scikit-learn predictor instances only. Instead got: '
                      '{}.')

        for clf in NON_LINEAR_MODELS:
            clf_instance = clf()
            clf_instance.fit(DATA, LABELS)

            with pytest.raises(TypeError) as excinfo:
                ftsl.SKLearnLinearModelExplainer(clf_instance)
            name = str(clf).strip("<>' ")[7:]
            assert str(excinfo.value) == type_error.format(name)
