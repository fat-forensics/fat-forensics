"""
The :mod:`fatf.core.model` module includes all custom predictive models used
for FAT-Forensics testing and examples.
"""

# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: BSD 3 clause

import abc
import numpy as np

from fatf.exceptions import (
    CustomValueError,
    MissingImplementationException,
    PrefittedModelException,
    UnfittedModelException
)
from fatf.utils.distance import euclidean_vector_distance

class Model(abc.ABC):
    """An abstract Model class with required fit and predict methods and an
    optional predict_proba method.

    Raises
    ------
    MissingImplementationException
        Any of the required methods (__init__, fit, predict) are not
        implemented.
    """

    @abc.abstractmethod
    def __init__(self) -> None:
        """Inits the abstract Model class."""

    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fits this model.

        Args
        ----
        X : np.ndarray
            A 2-dimensional numpy array with data used to fit the model.

        y : np.ndarray
            A 1-dimensional numpy array with labels used to fit the model.
        """

    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> None:
        """Predicts new data points using this model.

        Args
        ----
        X : np.ndarray
            A 2-dimensional numpy array with data to predict labels for.
        """

    def predict_proba(self, X: np.ndarray) -> None:
        """Predicts probabilities of new data points using this model.

        Args
        ----
        X : np.ndarray
            A 2-dimensional numpy array with data to predict probabilities for.

        Raises
        ------
        MissingImplementationException
            Always, since this is an abstract method that is not implemented.
        """
        raise MissingImplementationException()


class KNN(Model):
    """A K-Nearest Neighbours model that uses Euclidean distance.

    Args
    ----
    k : int, optional
        The number of neighbours used to make a prediction. Defaults to 3.

    Raises
    ------
    CustomValueError
        Raised when the k parameter is not a positive integer.

    PrefittedModelException
        Raised when trying to fit a model that has already been fitted. Usually
        when calling the fit method for the second time. Try using the clear
        method to reset the model before fitting it again.

    UnfittedModelException
        Raised when trying to predict data when a model has not been fitted yet.
        Try using the fit method to fit the model first.

    Attributes
    ----------
    _k : int
        The number of neighbours used to make a prediction.

    _X : np.ndarray
        The KNN training data.

    _y : np.ndarray
        The KNN training labels.

    _X_n : int
        The number of data points in the training set.

    _unique_y : np.ndarray
        An array with unique labels in the training labels set.

    _is_fitted : bool
        A Boolean variable indicating whether the model is fitted.
    """

    def __init__(self, k: int = 3) -> None:
        """Initialises the KNN model with selected k parameter and sets the
        internal attributes."""
        if k < 0:
            raise CustomValueError('k has to be positive.')
        elif not isinstance(k, int):
            raise CustomValueError('k has to be an integer.')
        else:
            self._k = k

        self._X = np.ndarray((0,))
        self._X_n = int()
        self._y = np.ndarray((0,))
        self._unique_y = np.ndarray((0,))
        self._unique_y_counts = np.ndarray((0,))
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model.

        Args
        ----
        X : np.ndarray
            The KNN training data.

        y : np.ndarray
            The KNN training labels.

        Raises
        ------
        PrefittedModelException
            Raised when trying to fit the model that has already been fitted.
            Usually when calling the fit method for the second time. Try using
            the clear method to reset the model before fitting it again.
        """
        if self._is_fitted:
            raise PrefittedModelException('This model has already been fitted.')
        else:
            # TODO: Check for square and numerical (complex and simple) array
            # TODO: If empty array raise an error
            # TODO: Compare if the number of labels is the same as the number of
            # data points

            self._X = X
            self._y = y
            self._unique_y, self._unique_y_counts = np.unique(
                self._y,
                return_counts=True
            )
            self._X_n = self._X.shape[0]
            self._is_fitted = True

    def clear(self) -> None:
        """Clear (unfit) the model.

        Raises
        ------
        UnfittedModelException
            Raised when trying to clear a model that has not been fitted yet.
            Try using the fit method to fit the model first.
        """
        if not self._is_fitted:
            raise UnfittedModelException('This model has not been fitted yet.')
        else:
            self._X = np.ndarray((0,))
            self._X_n = int()
            self._y = np.ndarray((0,))
            self._unique_y = np.ndarray((0,))
            self._unique_y_counts = np.ndarray((0,))
            self._is_fitted = False

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict new instances with the fitted model.

        Args
        ----
        X : np.ndarray
            The data to be predicted.

        Raises
        ------
        UnfittedModelException
            Raised when trying to predict data when the model has not been
            fitted yet. Try using the fit method to fit the model first.

        Returns
        -------
        predictions : np.ndarray
            Predicted class labels for each data sample.
        """
        if not self._is_fitted:
            raise UnfittedModelException('This model has not been fitted yet.')

        # TODO: Check for square and numerical (complex and simple) array
        # TODO: Check for size of the array -- if no data points return an empty
        # array

        predictions = np.array((0,))

        if self._k < self._X_n:
            distances = euclidean_vector_distance(X, self._X)
            knn = np.argpartition(distances, self._k)
            predictions = []
            for row in knn:
                close_labels = self._y[row[:self._k]]
                values, counts = np.unique(close_labels, return_counts=True)
                # TODO: What if the counts for the number of neighbours of
                # different classes are the same, i.e. a tie. Fix: choose the
                # overall majority class.
                predictions.append(values[np.argmax(counts)])
            predictions = np.array(predictions)
        else:
            majority_label = self._unique_y[np.argmax(self._unique_y_counts)]
            predictions = np.array([majority_label] * X.shape[0])

        return predictions
