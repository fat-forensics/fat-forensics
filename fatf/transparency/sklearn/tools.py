"""
.. versionadded:: 0.0.2

The :mod:`fatf.transparency.sklearn.tools` module implements a base
scikit-learn explainer.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import abc
import logging
import warnings

from typing import List, Optional, Union

import sklearn

import numpy as np

import fatf.utils.array.validation as fuav
import fatf.utils.transparency.explainers as fute

__all__ = ['is_sklearn_model', 'is_sklearn_model_instance', 'SKLearnExplainer']

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def is_sklearn_model(clf: Union[object, type]) -> bool:
    """
    Checks whether a class instance or a class is a scikit-learn predictor.

    .. versionadded:: 0.0.2

    This is achieved by checking inheritance from
    ``sklearn.base.BaseEstimator``.

    Parameters
    ----------
    clf : Union[object, type]
        A Python object (class instance) or a class to be checked.

    Returns
    -------
    is_valid_model : boolean
        ``True`` if ``clf`` inherits from ``sklearn.base.BaseEstimator``,
        ``False`` otherwise.
    """
    if isinstance(clf, type):  # Check and object
        is_valid_model = issubclass(clf, sklearn.base.BaseEstimator)
    else:  # Check an instance
        is_valid_model = isinstance(clf, sklearn.base.BaseEstimator)
    return is_valid_model


def is_sklearn_model_instance(clf: object) -> bool:
    """
    Checks whether a class instance (object) is a scikit-learn predictor.

    .. versionadded:: 0.0.2

    This function is similar to
    :func:`fatf.transparency.sklearn.tools.is_sklearn_model` but it enforces
    the ``clf`` to be an **instance** of an object in addition to checking
    whether it inherits from ``sklearn.base.BaseEstimator``.

    Parameters
    ----------
    clf : object
        A Python object (class instance) to be checked.

    Returns
    -------
    is_valid_model : boolean
        ``True`` if the object inherits from ``sklearn.base.BaseEstimator``,
        ``False`` otherwise.
    """
    is_valid_model_instance = (not isinstance(clf, type)
                               and is_sklearn_model(clf))
    return is_valid_model_instance


def _validate_input(clf: sklearn.base.BaseEstimator,
                    feature_names: Union[None, List[str]],
                    class_names: Union[None, List[str]]) -> bool:
    """
    Validates input of the ``SKLearnExplainer`` class initialiser.

    Parameters
    ----------
    clf : sklearn.base.BaseEstimator
        A scikit-learn model.
    feature_names : Union[None, List[string]]
        A list of strings representing feature names in order they appear in
        the numpy array used to train the ``clf`` classifier.
    class_names : Union[None, List[string]])
        A list of strings representing class names. The order of this list has
        to correspond to the lexicographical ordering of the unique values in
        the target (ground truth) array used to train the ``clf`` classifier.

    Raises
    ------
    TypeError
        The ``clf`` object is not a scikit-learn classifier -- it does not
        inherit form the ``sklearn.base.BaseEstimator``. ``feature_names``
        parameter is neither a Python list nor ``None``. One of the elements of
        the ``feature_names`` list is not a string. The ``class_names``
        parameter is neither a Python list nor ``None``. One of the elements of
        the ``class_names`` list is not a string.
    ValueError
        Either the ``feature_names`` or ``class_names`` list is empty.

    Returns
    -------
    is_valid : boolean
        ``True`` if the input is valid, ``False`` otherwise.
    """
    # pylint: disable=too-many-branches
    is_valid = False

    if not is_sklearn_model_instance(clf):
        raise TypeError('The model has to be a scikit-learn classifier, i.e., '
                        'it has to inherit from sklearn.base.BaseEstimator.')

    if feature_names is not None:
        if isinstance(feature_names, list):
            if not feature_names:
                raise ValueError('The feature names list cannot be empty.')
            for feature_name in feature_names:
                if not isinstance(feature_name, str):
                    raise TypeError('All elements of the feature_names list '
                                    'have to be strings.')
        else:
            raise TypeError('The feature_names parameter has to be either '
                            'None or a Python list.')

    if class_names is not None:
        if isinstance(class_names, list):
            if not class_names:
                raise ValueError('The class names list cannot be empty.')
            for class_name in class_names:
                if not isinstance(class_name, str):
                    raise TypeError('All elements of the class_names list '
                                    'have to be strings.')
        else:
            raise TypeError('The class_names parameter has to be either None '
                            'or a Python list.')

    is_valid = True
    return is_valid


class SKLearnExplainer(fute.Explainer):
    """
    Implements a base scikit-learn model explainer class.

    .. versionadded:: 0.0.2

    Every scikit-learn model explainer class should inherit from this class.
    It should also overwrite the following four private methods:

    .. currentmodule:: fatf.transparency.sklearn.tools

    - :func:`~SKLearnExplainer._validate_kind_fitted`,
    - :func:`~SKLearnExplainer._is_classifier`,
    - :func:`~SKLearnExplainer._get_features_number`, and
    - :func:`~SKLearnExplainer._get_classes_array`.

    For their expected functionality please see their respective documentation.

    The explainer should also implement one of the explanatory methods that are
    inherited from ``SKLearnExplainer``'s parent class
    (:class:`fatf.utils.transparency.explainers.Explainer`):

    .. currentmodule:: fatf.utils.transparency.explainers

    - :func:`~Explainer.feature_importance`,
    - :func:`~Explainer.explain_model`, and/or
    - :func:`~Explainer.explain_instance`.

    .. currentmodule:: fatf.transparency.sklearn.tools

    Alternatively, a new method that explains an aspect of the model or its
    predictions can be introduced.

    This class loggs an *information* if the feature names were not given and
    are inferred from the provided number of features using "feature %d"
    pattern. An *information* is also logged if the class names were not given
    and are inferred from the provided class id's (using the ``classes_array``
    attribute) using "class %s" pattern.

    Parameters
    ----------
    clf : sklearn.base.BaseEstimator
        A scikit-learn model.
    feature_names : Optional[List[string]]
        A list of strings representing feature names in order they appear in
        the numpy array used to train the ``clf`` predictive model.
    class_names : Optional[List[string]]
        A list of strings representing class names. The order of this list has
        to correspond to the lexicographical ordering of the unique values in
        the target (ground truth) array used to train the ``clf`` predictor.
        For example, if your target array has the following values
        ``['aa', 'a', '0', 'b']``, your class names should be given for the
        following ordering of the class id's: ``['0', 'a', 'aa', 'b']``.

    Warns
    -----
    UserWarning
        Features number is not given, therefore the length of the features name
        list cannot be validated. Classes array is not given, therefore the
        length of class names array cannot be validated.

    Raises
    ------
    TypeError
        The ``clf`` object is not a scikit-learn classifier -- it does not
        inherit form the ``sklearn.base.BaseEstimator``. ``feature_names``
        parameter is neither a Python list nor ``None``. One of the elements of
        the ``feature_names`` list is not a string. The ``class_names``
        parameter is neither a Python list nor ``None``. One of the elements of
        the ``class_names`` list is not a string.
    ValueError
        Either the ``feature_names`` or ``class_names`` list is empty. The
        length of the ``feature_names`` list is different than the features
        number extracted from the classifier. The length of the ``class_names``
        list is different than the length of the ``classes_array`` extracted
        from the classifier.

    Attributes
    ----------
    clf : sklearn.base.BaseEstimator
        A fitted scikit-learn model.
    feature_names : Union[None, List[string]]
        Either ``None`` or a list of feature names in the order they appear in
        the numpy array used to train the ``clf`` classifier.
    class_names : Union[None, List[string]]
        Either ``None`` or a list of class names in the order of
        lexicographically sorted unique values in the target (ground truth)
        array used to train the ``clf`` predictor (class id's).
    is_classifier : boolean
        If ``True``, the predictive model held under the ``clf`` attribute is a
        classifier. If ``False``, it is a regressor. (Set using the ``clf``
        attribute via the ``_is_classifier`` method.)
    features_number : Union[None, integer]
        Either ``None`` or the number of features in the ``clf`` model.
        (Extracted from the ``clf`` attribute with the ``_get_features_number``
        method.)
    classes_array : Union[None, numpy.ndarray]
        Either ``None`` or a 1-dimensional numpy array holding all the possible
        model predictions (only for classifiers). For regressors this should
        always be ``None``.
    """

    def __init__(self,
                 clf: sklearn.base.BaseEstimator,
                 feature_names: Optional[List[str]] = None,
                 class_names: Optional[List[str]] = None) -> None:
        """
        Initialises the ``SKLearnExplainer`` class.
        """
        # Validate the input
        assert _validate_input(clf, feature_names,
                               class_names), 'Invalid init parameters.'
        self.clf = clf
        self.feature_names = feature_names
        self.class_names = class_names

        # Check whether the model is of the right type and is fitted
        assert self._validate_kind_fitted(), 'Unfitted or wrong type model.'

        # Classifier or regressor
        self.is_classifier = self._is_classifier()
        assert isinstance(self.is_classifier, bool), 'Has to be boolean.'

        # The number of features (number of columns in a data array) expected
        # by the classifier
        self.features_number = self._get_features_number()
        if self.features_number is not None:
            assert isinstance(self.features_number, int), 'Wrong type.'

        # Get the list of classes that the predictive model can output
        self.classes_array = self._get_classes_array()
        if self.classes_array is not None:
            assert isinstance(self.classes_array, np.ndarray), 'Bad type.'
        if self.classes_array is not None:
            assert fuav.is_1d_array(self.classes_array), 'Must be 1-D array.'
            assert (fuav.is_numerical_array(self.classes_array)
                    or fuav.is_textual_array(self.classes_array)), 'Bad type.'

        # A regressor must not have class names
        if not self.is_classifier:
            assert self.classes_array is None and self.class_names is None, \
                "Regressor's class_names and classes_array must both be None."

        # Validate feature names length
        if self.feature_names is None:
            if self.features_number is not None:
                logger.info('Generating missing feature names from the number '
                            'of features using "feature %d" pattern.')
                self.feature_names = [
                    'feature {}'.format(i) for i in range(self.features_number)
                ]
        else:
            if self.features_number is None:
                warnings.warn(
                    'Cannot validate the length of feature names list since '
                    'the _get_features_number method '
                    'returned None.', UserWarning)
            else:
                if len(self.feature_names) != self.features_number:
                    raise ValueError('The length of the feature_names list '
                                     'is different than the number of '
                                     'features extracted from the classifier.')

        # Validate class names length
        if self.class_names is None:
            if self.classes_array is not None:
                logger.info('Generating missing class names from the array of '
                            'classes output by the classifier using '
                            '"class %s" pattern.')
                self.class_names = [
                    'class {}'.format(i) for i in self.classes_array
                ]
        else:
            if self.classes_array is None:
                warnings.warn(
                    'Cannot validate the length of class names list since the '
                    '_get_classes_array method returned None.', UserWarning)
            else:
                if self.classes_array.shape[0] != len(self.class_names):
                    raise ValueError('The length of the class_names list is '
                                     'different than the length of the '
                                     'classes array extracted from the '
                                     'classifier.')

    def map_class(self, clf_class: Union[int, str]) -> str:
        """
        Maps a class id output by the classifier to a class name.

        A mapping will only be provided if the class was initialised with class
        names or an array of possible predictions was extracted form the
        classifier.

        Parameters
        ----------
        clf_class : Union[integer, string]
            A class id output by the classifier.

        Raises
        ------
        RuntimeError
            The error is raised when trying to map a class for a regressor.
            It is also raised if the class was not sufficiently initialised,
            i.e., either ``classes_array`` or ``class_names`` attributes are
            missing.
        TypeError
            The ``clf_class`` parameter is neither integer nor string.
        ValueError
            Given ``clf_class`` is not one of the values that the classifier
            can output.

        Returns
        -------
        mapped_class : string
            A class name corresponding to the class id.
        """
        if not self.is_classifier:
            raise RuntimeError('This functionality is not available for '
                               'regressors.')

        if not isinstance(clf_class, (str, int)):
            raise TypeError('The clf_class parameter must either be integer '
                            'or string.')

        if self.class_names is None or self.classes_array is None:
            raise RuntimeError('class_names and/or classes_array have not '
                               'been initialised; mapping is not supported.')
        if clf_class not in self.classes_array:
            possible_classes = self.classes_array.tolist()
            raise ValueError('Given class value is invalid for this '
                             'classifier. The following values are possible: '
                             '{}.'.format(possible_classes))

        class_index = np.where(self.classes_array == clf_class)[0]
        assert len(class_index.shape) == 1, 'Has to be a 1-D array.'
        assert class_index.shape[0] == 1, 'Has to be a single index.'

        mapped_class = self.class_names[class_index[0]]
        return mapped_class

    @abc.abstractmethod
    def _validate_kind_fitted(self) -> bool:  # pragma: no cover
        """
        Implements a kind check and a fit check of a predictive model.

        This method is called upon initialising the class and checks whether
        the ``self.clf`` predictor is of the right kind. For example, when
        implementing an explainer for scikit-learn linear models this method
        should check whether the ``self.clf`` is a linear model and whether it
        has been fitted. If any of these conditions is not satisfied this
        method should raise an appropriate exception: for a wrong model type
        this should be a ``ValueError``; for an unfit model this should be
        sklearn's ``sklearn.exceptions.NotFittedError`` (consider using
        scikit's ``sklearn.utils.validation.check_is_fitted`` function to raise
        this exception).

        Raises
        ------
        NotImplementedError
            This error is always raised since the method is an abstract method.

        Returns
        -------
        is_valid : boolean
            ``True`` if the kind of the ``self.clf`` model is correct and the
            model is fitted, ``False`` otherwise.
        """
        raise NotImplementedError('This method has not been implemented.')
        # pylint: disable=unreachable
        is_valid = False
        return is_valid

    @abc.abstractmethod
    def _is_classifier(self) -> bool:  # pragma: no cover
        """
        Indicates whether the ``clf`` model is a *classifier* or a *regressor*.

        This method should return ``True`` if the model that this class
        explains is a classifier and ``False`` if it is a regressor.

        Raises
        ------
        NotImplementedError
            This error is always raised since the method is an abstract method.

        Returns
        -------
        is_classifier : boolean
            ``True`` if the ``self.clf`` model is a classifier or ``False``
            when it is a regressor.
        """
        raise NotImplementedError('This method has not been implemented.')
        # pylint: disable=unreachable
        is_classifier = False
        return is_classifier

    @abc.abstractmethod
    def _get_features_number(self) -> Union[int, None]:  # pragma: no cover
        """
        Returns the number of features that the model accepts or ``None``.

        If it is possible to extract the number of features (columns) expected
        by the ``self.clf`` predictor, this method should return this number.
        Otherwise, it must return ``None``.

        Raises
        ------
        NotImplementedError
            This error is always raised since the method is an abstract method.

        Returns
        -------
        features_number : Union[integer, None]
            The number of features accepted by the classifier or ``None``.
        """
        raise NotImplementedError('This method has not been implemented.')
        # pylint: disable=unreachable
        features_number = None
        return features_number

    @abc.abstractmethod
    def _get_classes_array(
            self) -> Union[np.ndarray, None]:  # pragma: no cover
        """
        Retrieves the array with classes that the predictive model can output.

        For regressors this method must return ``None``. For classifier it
        should return a 1-dimensional numpy array that holds all the possible
        classification results that the model can output if they are possible
        to extract form ``self.clf`` or ``None`` otherwise.

        Raises
        ------
        NotImplementedError
            This error is always raised since the method is an abstract method.

        Returns
        -------
        features_number : Union[numpy.ndarray, None]
            A 1-dimensional numpy array holding all the possible model
            predictions (only for classifiers) or ``None``.
        """
        raise NotImplementedError('This method has not been implemented.')
        # pylint: disable=unreachable
        classes_array = None
        return classes_array
