"""
.. versionadded:: 0.0.2

The :mod:`fatf.transparency.sklearn.linear_model` module implements linear
scikit-learn model explainers.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

from typing import Union

import sklearn.linear_model
import sklearn.utils.validation

import numpy as np

import fatf.transparency.sklearn.tools as ftst
import fatf.utils.tools as fut

__all__ = ['linear_classifier_coefficients', 'SKLearnLinearModelExplainer']

_SKLEARN_VERSION = [int(i) for i in sklearn.__version__.split('.')[:2]]
_SKLEARN_0_22 = fut.at_least_verion([0, 22], _SKLEARN_VERSION)

if _SKLEARN_0_22:  # pragma: nocover
    # pylint: disable=invalid-name,protected-access,no-member
    _linear_base = sklearn.linear_model._base
    _linear_coordinate_descent = sklearn.linear_model._coordinate_descent
    _lienar_stochastic_gradient = sklearn.linear_model._stochastic_gradient
    _linear_bayes = sklearn.linear_model._bayes
    _linear_theil = sklearn.linear_model._theil_sen
    _linear_omp = sklearn.linear_model._omp
    _linear_ridge = sklearn.linear_model._ridge
    _linear_angles = sklearn.linear_model._least_angle
else:  # pragma: nocover
    _linear_base = sklearn.linear_model.base  # pylint: disable=invalid-name
    # pylint: disable=invalid-name
    _linear_coordinate_descent = sklearn.linear_model.coordinate_descent
    _lienar_stochastic_gradient = sklearn.linear_model.stochastic_gradient
    _linear_bayes = sklearn.linear_model.bayes
    _linear_theil = sklearn.linear_model.theil_sen
    _linear_omp = sklearn.linear_model.omp
    _linear_ridge = sklearn.linear_model.ridge
    _linear_angles = sklearn.linear_model.least_angle

_LINEAR = (_linear_base.LinearModel, _linear_coordinate_descent.LinearModelCV)
_LINEAR_CLASSIFIER = (_linear_base.LinearClassifierMixin, )
_LINEAR_REGRESSOR = (
    _linear_base.LinearRegression,
    _lienar_stochastic_gradient.BaseSGDRegressor,
    _linear_bayes.BayesianRidge,
    _linear_bayes.ARDRegression,
    _linear_coordinate_descent.ElasticNet,
    _linear_coordinate_descent.ElasticNetCV,
    _linear_coordinate_descent.LassoCV,
    _linear_theil.TheilSenRegressor,
    _linear_omp.OrthogonalMatchingPursuit,
    _linear_omp.OrthogonalMatchingPursuitCV,
    _linear_ridge.Ridge,
    _linear_ridge.RidgeCV,
    sklearn.linear_model.HuberRegressor,
    _linear_angles.Lars,
    sklearn.svm.LinearSVR,
    #
    _linear_coordinate_descent.MultiTaskLassoCV,
    _linear_coordinate_descent.MultiTaskElasticNetCV)


def _is_scikit_linear(clf: sklearn.base.BaseEstimator) -> bool:
    """
    Checks whether a scikit-learn model is a linear model.

    Children of the following classes are considered linear:

    - ``sklearn.linear_model.base.LinearModel``,
    - ``sklearn.linear_model.coordinate_descent.LinearModelCV``,
    - ``sklearn.linear_model.base.LinearRegression``, and
    - ``sklearn.linear_model.base.LinearClassifierMixin``.

    Parameters
    ----------
    clf : sklearn.base.BaseEstimator
        A scikit-learn predictor.

    Returns
    -------
    is_scikit_linear : boolean
        ``True`` if the predictor is any of the scikit-learn linear models,
        ``False`` otherwise.
    """

    assert ftst.is_sklearn_model_instance(clf), 'Invalid sklearn predictor.'
    is_scikit_linear = isinstance(clf, (_LINEAR_REGRESSOR, _LINEAR_CLASSIFIER))
    return is_scikit_linear


def _is_fitted_linear(clf: sklearn.base.BaseEstimator) -> bool:
    """
    Checks whether a scikit-learn linear model is fitted.

    The check succeeds if the ``clf`` classifier has a ``coef_`` attribute.

    Parameters
    ----------
    clf : sklearn.base.BaseEstimator
        A linear scikit-learn model.

    Raises
    ------
    sklearn.exceptions.NotFittedError
        The scikit-learn package will raise this exception if the model is not
        fitted.

    Returns
    -------
    is_fitted_linear : boolean
        ``True`` if the linear predictor is fitted, ``False`` otherwise.
    """
    assert _is_scikit_linear(clf), 'Has to be an linear scikit-learn model.'

    is_fitted_linear = False

    # (clf, ['coef_', 'intercept_'], all_or_any=any)
    if _SKLEARN_0_22:  # pragma: nocover
        # pylint: disable=no-value-for-parameter
        sklearn.utils.validation.check_is_fitted(clf)
    else:  # pragma: nocover
        sklearn.utils.validation.check_is_fitted(clf, 'coef_', all_or_any=all)

    is_fitted_linear = True
    return is_fitted_linear


def linear_classifier_coefficients(
        clf: sklearn.base.BaseEstimator) -> np.ndarray:
    """
    Extracts coefficients (feature importances) of a linear scikit-learn model.

    .. versionadded:: 0.0.2

    .. note::
       Please note that for the coefficients (feature importances) to be
       comparable the values of all features had to be normalised to the same
       range before training the model.

    Parameters
    ----------
    clf : sklearn.base.BaseEstimator
        A linear scikit-learn model.

    Raises
    ------
    sklearn.exceptions.NotFittedError
        The scikit-learn package (``sklearn.utils.validation.check_is_fitted``
        function) will raise this exception if the model is not fitted.
    TypeError
        The ``clf`` classifier is not a scikit-learn linear model.

    Returns
    -------
    coefficients : numpy.ndarray
        A numpy array that holds coefficients of the ``clf`` linear model.
        (The order of the coefficients corresponds to the order of the features
        in the training data array).
    """
    # Has to be a linear sklearn model
    if not _is_scikit_linear(clf):
        raise TypeError('This functionality is designated for linear-like '
                        'scikit-learn predictor instances only. Instead got: '
                        '{}.{}.'.format(clf.__module__,
                                        clf.__class__.__name__))
    assert _is_fitted_linear(clf), 'Has to be a fitted sklearn linear model.'

    assert hasattr(clf, 'coef_'), 'coef_ attribute missing.'
    coefficients = clf.coef_
    # assert hasattr(clf, 'intercept_'), 'intercept_ attribute missing.'
    # intercept = clf.intercept_

    return coefficients


class SKLearnLinearModelExplainer(ftst.SKLearnExplainer):
    """
    A scikit-learn linear model explainer class.

    .. versionadded:: 0.0.2

    This class implements a ``feature_importance`` method that returns
    coefficients of the linear ``clf`` model. This coefficients can be
    interpreted as features (positive or negative) importance.

    .. note::
       Please note that for the coefficients (feature importances) to be
       comparable the values of all features had to be normalised to the same
       range before training the model.

    For other functionality, parameters, attributes, logs, warnings and errors
    implemented by this class please see its parent class:
    :class:`fatf.transparency.sklearn.tools.SKLearnExplainer`.
    """

    # pylint: disable=abstract-method

    def feature_importance(self) -> np.ndarray:
        """
        Extracts features importance from the ``clf`` predictor.

        Returns
        -------
        feature_importance_array : numpy.ndarray
            A numpy array with coefficients of the ``clf`` linear model.
            (The order of the coefficients corresponds to the order of the
            features in the training data array.)
        """
        feature_importance_array = linear_classifier_coefficients(self.clf)
        return feature_importance_array

    def _is_classifier(self) -> bool:
        """
        Decides whether the linear ``clf`` model is a classifier or regressor.

        Returns
        -------
        is_classifier : boolean
            ``True`` if the linear ``clf`` model is a classifier and ``False``
            if it is a regressor.
        """
        if isinstance(self.clf, _LINEAR_CLASSIFIER):
            is_classifier = True
        elif isinstance(self.clf, _LINEAR_REGRESSOR):
            is_classifier = False
        else:
            assert False, 'Not a linear predictive model?'  # pragma: no cover
        return is_classifier

    def _validate_kind_fitted(self) -> bool:
        """
        Validates that the ``clf`` model is *linear* and *fitted*.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            The scikit-learn package
            (``sklearn.utils.validation.check_is_fitted`` function) will raise
            this exception if the model is not fitted.
        TypeError
            The ``clf`` classifier is not a scikit-learn linear model.

        Returns
        -------
        is_linear_fitted : boolean
            ``True`` if the ``clf`` model is linear and fitted. ``False`` if
            the model is either not fitted or is not linear.
        """
        is_linear_fitted = False

        if not _is_scikit_linear(self.clf):
            raise TypeError('This functionality is designated for linear-like '
                            'scikit-learn predictor instances only. Instead '
                            'got: {}.{}.'.format(self.clf.__module__,
                                                 self.clf.__class__.__name__))

        assert _is_fitted_linear(self.clf), 'Has to be a fitted linear model.'

        is_linear_fitted = True
        return is_linear_fitted

    def _get_features_number(self) -> int:
        """
        Extracts the number of features expected by the ``clf`` model.

        Returns
        -------
        features_number : integer
            The number of features that the ``clf`` model is expecting (was
            trained on).
        """
        if self.is_classifier:
            features_number = self.clf.coef_.shape[1]
        else:
            coef_shape_dim = len(self.clf.coef_.shape)
            if coef_shape_dim == 1:  # Single-task regression
                features_number = self.clf.coef_.shape[0]
            elif coef_shape_dim == 2:  # Multi-task regression
                features_number = self.clf.coef_.shape[1]
            else:
                assert False, 'Incompatible _coef shape.'  # pragma: nocover
        return features_number

    def _get_classes_array(self) -> Union[np.ndarray, None]:
        """
        Extracts the unique class id's that the ``clf`` model can output.

        Returns
        -------
        classes_array : Union[None, List[Union[string, integer]]]
            ``None`` if the ``clf`` is a regressor. A numpy array with the
            unique class id's (unique elements of the target, i.e. ground
            truth, array used to fit the ``clf`` model).
        """
        if self.is_classifier:
            classes_array = self.clf.classes_
        else:
            classes_array = None
        return classes_array
