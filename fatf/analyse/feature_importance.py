"""
Functions for calculating feature importance and 
Individual Conditional Expectation (ICE)
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
# License: new BSD

from typing import List, Dict, Union, Tuple
import warnings

import numpy as np

import fatf.utils.validation as fuv
from fatf.utils.validation import (is_2d_array, 
                                   check_model_functionality,
                                   check_indices,
                                   is_numerical_array,
                                   is_structured_array)
from fatf.exceptions import (MissingImplementationError, 
                             IncompatibleModelError, IncorrectShapeError)

__all__ = ['individual_conditional_expectation', 'partial_dependence']


def _check_input(
                X: np.ndarray, 
                model: object, 
                feature: Union[int, str],
                is_categorical: bool,
                steps: int,
                check_x: bool = True,
                check_model: bool = True,
                check_feature: bool = True,
                check_steps : bool = True) -> None:
    """
    Checks if input is compatible to compute partial depedence and invdividual
    conditional expectations.

    Parameters
    ----------
    X : np.ndarray
        Data matrix (can be structured or regular np.ndarray)
    model : object 
        Which is fitted model containing functions fit(X, Y), predict(X)
        and predict_proba(X)
    feature : Union[integer, string] 
        Corresponding to column in X for feature to compute ICE
    is_categorical : boolean
        If feature is categorical (do not numerically interpolate)
    steps : integer
        How many steps to sample with between feature min amd max.
    check_x : boolean
        Whether to check if X is valid or not.
    check_model : boolean
        Whether to check if model is valid or not.
    check_feature : boolean
        Whether to check if feature is valid or not.
    check_steps : boolean
        Whether to check if steps is compatible with is_categorical value.

    Warns
    -----
    UserWarning
        If is_categorical is True but steps is defined, warn the user that the
        number of steps will be the number of unique values of feature in the
        dataset.

    Raises
    ------
    IncorrectShapeError
        X must be a 2-dimensional array.
    IncompatibleModelError
        Model does not contain required method predict_proba()
    ValueError
        Feature given is an invalid index to np.ndarray X
    """
    if check_x:
        if not fuv.is_2d_array(X):
            raise IncorrectShapeError('X must be 2-dimensional array.')
    if check_model:
        if not fuv.check_model_functionality(model, require_probabilities=True):
            raise IncompatibleModelError(
                'Partial dependence and individal conditional expectiations '
                'requires model object to have method predict_proba().')
        #TODO: check if model is already trained or not
    if check_feature:
        f = np.array([feature], dtype=type(feature))
        invalid_indices = fuv.check_indices(X, f)
        if not np.array_equal(invalid_indices, np.array([], dtype=f.dtype)):
            raise ValueError('Invalid features %s given' %str(invalid_indices))
    if check_steps:
        if is_categorical and steps is not None:
            message = ('Feature is defined as categorical but number of steps '
                       'is defined. The numer of steps used will be the '
                       'number of unique values in the dataset for feature.')
            warnings.warn(message, category=UserWarning)



def _interpolate_array(
                      X: np.ndarray,
                      feature: int,
                      is_categorical: bool,
                      steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates array which has interpolated between maximum and minimum
    for feature for every datapoint taking a normal np.ndarray

    Parameters
    ----------
    X : numpy.ndarray
        Data matrix of all the same dtype and indexed with ints
    feature : integer
        Corresponding to column in X to interpolate for
    is_categorical : boolean
        If feature is categorical (do not numerically interpolate)
    steps : integer
        How many steps to sample with between feature min amd max. Defaults
        to 100.
    
    Warns
    -----
    UserWarning
        Feature is not numerical and is specified as non-categoriceal.

    Returns
    -------
    X_sampled : numpy.ndarray
        Numpy array with shape (n_samples, steps, n_features) with the dataset 
        repeated with different values of feature for each point
    values : numpy.ndarray
        Array of values that have been interpolated between the maximum and
        minimum for the feature specified.
    """
    is_struct = fuv.is_structured_array(X)
    if is_struct:
        feat = X[feature]
    else:
        feat = X[:, feature]
    if not fuv.is_numerical_array(feat) and not is_categorical:
        message = ('Feature %s is not numerical and not specified as '
                   'categorical. Samples will be generated by using values '
                   'contained in the dataset.')
        warnings.warn(message, category=UserWarning)
        is_categorical = True
    if is_categorical:
        values = np.unique(feat)
        steps = np.unique(feat).shape[0]
    else:
        values = np.linspace(min(feat), max(feat), steps)
    samples = []
    if is_struct:
        X_sampled = np.zeros((X.shape[0], steps), dtype=X.dtype)
        for i in range(0, X.shape[0]):
            X_sampled[i] = np.repeat(X[np.newaxis, i], steps, axis=0)
            X_sampled[i][feature] = values
    else:
        X_sampled = np.zeros((X.shape[0], steps, X.shape[1]), dtype=X.dtype)
        for i in range(0, X.shape[0]):
            X_sampled[i, :, :] = np.tile(X[i, :], (steps, 1))
            X_sampled[i, :, feature] = values
    return X_sampled, values


def individual_conditional_expectation(
        X: np.ndarray, 
        model: object, 
        feature: Union[int, str],
        is_categorical: bool = False,
        steps: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates Individual Conditional Expectation for all class for feature
    specified.

    Parameters
    ----------
    X : numpy.ndarray
        Data matrix (can be structured or regular numpy.ndarray)
    model: object 
        Which is fitted model containing functions fit(X, Y), predict(X)
        and predict_proba(X)
    feature : Union[integer, string] 
        Corresponding to column in X for feature to compute ICE
    is_categorical : boolean
        If feature is categorical (do not numerically interpolate)
    steps : integer
        How many steps to sample with between feature min amd max. If 
        is_categorical is True, then the number of unique values for feature in
        the dataset is used. Else, steps defaults to 100.

    Returns
    -------
    probs : numpy.ndarray
        Shape [n_samples, steps, n_classes] that contains 
    values : numpy.ndarray
        Shape [steps] specifying the interpolation values that have been tested
    """
    _check_input(X, model, feature, is_categorical, steps)
    n_classes = model.predict_proba(X[0:1]).shape[1]
    X_sampled, values = _interpolate_array(X, feature, is_categorical, steps)
    probs = np.zeros((X.shape[0], steps, n_classes), dtype=np.float)
    for i in range(0, X.shape[0]):
        X_pred = X_sampled[i]
        probas = model.predict_proba(X_pred)
        probs[i, :, :] = probas
    return probs, values

def partial_dependence(
        X: np.ndarray,
        model: object,
        feature: Union[int, str],
        is_categorical: bool = False,
        steps: int = 100
) -> Tuple[np.ndarray, np.array]:
    """
    Calculates partial dependence for all classes for feature. Takes the mean 
    of the output of individual_conditional_expectation function over all 
    training data points.

    Parameters
    ----------
    X : numpy.ndarray 
        Data matrix (can be structured or regular np.ndarray)
    model : object 
        Which is fitted model containing functions fit(X, Y), predict(X)
        and predict_proba(X)
    feature : Union[integer, string] 
        Corresponding to column in X for feature to compute ICE
    is_categorical : boolean
        If feature is categorical (do not numerically interpolate)
    steps : integer
        How many steps to sample with between feature min amd max. Defaults
        to 100.

    Returns
    -------
    probs : numpy.ndarray
        Shape [steps, n_classes] that contains 
    values : numpy.ndarray
        Shape [steps] specifying the interpolation values that have been tested
    """
    ice, values = individual_conditional_expectation(X, 
                                                     model, 
                                                     feature, 
                                                     is_categorical,
                                                     steps=steps)
    pd = np.mean(ice, axis=0)
    return pd, values
