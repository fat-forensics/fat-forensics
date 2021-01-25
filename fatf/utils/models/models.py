"""
The :mod:`fatf.utils.models.models` module holds custom models.

The models implemented in this module are mainly used for used for
FAT Forensics package testing and the examples in the documentation.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import abc

from typing import Optional

import numpy as np

import fatf.utils.array.tools as fuat
import fatf.utils.array.validation as fuav
import fatf.utils.distances as fud

from fatf.exceptions import (IncorrectShapeError, PrefittedModelError,
                             UnfittedModelError)

__all__ = ['KNN']


class Model(abc.ABC):
    """
    An abstract class used to implement predictive models.

    This abstract class requires ``fit`` and ``predict`` methods and defines
    an optional ``predict_proba`` method.

    This is a scikit-learn-inspired model specification and it is being relied
    on through out this package.

    Raises
    ------
    NotImplementedError
        Any of the required methods -- ``fit`` or ``predict`` -- is not
        implemented.
    """
    # pylint: disable=invalid-name

    @abc.abstractmethod
    def __init__(self) -> None:
        """
        Initialises the abstract model class.
        """

    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits this predictive model.

        Parameters
        ----------
        X : numpy.ndarray
            A 2-dimensional numpy data array used to fit the model.
        y : numpy.ndarray
            A 1-dimensional numpy labels array used to fit the model.
        """

    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> None:
        """
        Predicts labels of new data points using this model.

        Parameters
        ----------
        X : numpy.ndarray
            A 2-dimensional numpy data array for which labels are predicted.
        """

    def predict_proba(self, X: np.ndarray) -> None:
        """
        Predicts probabilities of labels for new data points using this model.

        Parameters
        ----------
        X : numpy.ndarray
            A 2-dimensional numpy data array for which labels probabilities are
            predicted.

        Raises
        ------
        NotImplementedError
            By default this method is not required, hence it raises a
            ``NotImplementedError``.
        """
        raise NotImplementedError


class KNN(Model):
    """
    A K-Nearest Neighbours model based on Euclidean distance.

    When the ``k`` parameter is set to 0 the model works as a majority class
    classifier. In case the count of neighbours (within ``k``) results in a
    tie the overall majority class for the whole training data is returned.
    Finally, when the training data contains categorical (i.e. non-numerical,
    e.g. strings) columns the distance for these columns is 0 when the value
    matches and 1 otherwise.

    This model can operate in two modes: *classifier* or *regressor*. The first
    one works for categorical and numerical targets and provides two predictive
    methods: ``predict`` -- for predicting labels and ``predict_proba`` for
    predicting probabilities of labels. The regressor mode, on the other hand,
    requires the target to be numerical and it only supports the ``predict``
    method, which returns the average of the target value of the ``k``
    neighbours for the queried data point.

    Parameters
    ----------
    k : integer, optional (default=3)
        The number of neighbours used to make a prediction. Defaults to 3.
    mode : string, optional (default='classifier')
        The mode in which the model will operate. Either ``'classifier'``
        (``'c'``) or ``'regressor'`` (``'r'``). In the latter case
        ``predict_proba`` method is disabled.

    Raises
    ------
    PrefittedModelError
        Raised when trying to fit a model that has already been fitted. Usually
        raised when calling the ``fit`` method for the second time. Try using
        the ``clear`` method to reset the model before fitting it again.
    TypeError
        The ``k`` parameter is not an integer.
    UnfittedModelError
        Raised when trying to predict data with a model that has not been
        fitted yet. Try using the ``fit`` method to fit the model first.
    ValueError
        The ``k`` parameter is a negative number or the ``mode`` parameter does
        not have one of the allowed values: ``'c'``, ``'classifier'``, ``'r'``
        or ``'regressor'``.

    Attributes
    ----------
    _MODES : Set[string]
        Possible modes of the KNN model: ``'classifier'`` (``'c'``) or
        ``'regressor'`` (``'r'``).
    _k : integer
        The number of neighbours used to make a prediction.
    _is_classifier : boolean
        True when the model is initialised (and operates) as a classifier.
        False when it acts as a regressor.
    _is_fitted : boolean
        A Boolean variable indicating whether the model is fitted.
    _X : numpy.ndarray
        The KNN model training data.
    _y : numpy.ndarray
        The KNN model training labels.
    _X_n : integer
        The number of data points in the training set.
    _unique_y : numpy.ndarray
        An array with unique labels in the training labels set ordered
        lexicographically.
    _unique_y_counts : numpy.ndarray
        An array with counts of the unique labels in the training labels set.
    _unique_y_probabilities : numpy.ndarray
        Probabilities of labels calculated using their frequencies in the
        training data.
    _majority_label : Union[string, integer, float]
        The most common label in the training set.
    _is_structured : boolean
        A Boolean variable indicating whether the model has been fitted on a
        structured numpy array.
    _categorical_indices : numpy.ndarray
        An array with categorical indices in the training array.
    _numerical_indices : numpy.ndarray
        An array with numerical indices in the training array.
    """
    # pylint: disable=too-many-instance-attributes
    _MODES = set(['classifier', 'c', 'regressor', 'r'])

    def __init__(self, k: int = 3, mode: Optional[str] = None) -> None:
        """
        Initialises the KNN model with the selected ``k`` parameter.
        """
        super().__init__()
        if not isinstance(k, int):
            raise TypeError('The k parameter has to be an integer.')
        if k < 0:
            raise ValueError('The k parameter has to be a positive integer.')

        if mode is None:
            self._is_classifier = True
        else:
            if mode in self._MODES:
                self._is_classifier = mode[0] == 'c'
            else:
                raise ValueError(('The mode parameter has to have one of the '
                                  'following values {}.').format(self._MODES))

        self._k = k

        self._is_fitted = False
        self._X = np.ndarray((0, 0))  # pylint: disable=invalid-name
        self._y = np.ndarray((0, ))
        self._X_n = int()  # pylint: disable=invalid-name
        self._unique_y = np.ndarray((0, ))
        self._unique_y_counts = np.ndarray((0, ))
        self._unique_y_probabilities = np.ndarray((0, ))
        self._majority_label = None
        self._is_structured = False
        self._categorical_indices = np.ndarray((0, ))
        self._numerical_indices = np.ndarray((0, ))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the model.

        Parameters
        ----------
        X : numpy.ndarray
            The KNN training data.
        y : numpy.ndarray
            The KNN training labels.

        Raises
        ------
        IncorrectShapeError
            Either the ``X`` array is not 2-dimensional, the ``y`` array is not
            1-dimensional, the number of rows in ``X`` is not the same as the
            number of elements in ``y`` or the ``X`` array has 0 rows or 0
            columns.
        PrefittedModelError
            Trying to fit the model when it has already been fitted. Usually
            raised when calling the ``fit`` method for the second time without
            clearing the model first.
        TypeError
            Trying to fit a KNN predictor in a regressor mode with
            non-numerical target variable.
        """
        if self._is_fitted:
            raise PrefittedModelError('This model has already been fitted.')
        if not fuav.is_2d_array(X):
            raise IncorrectShapeError('The training data must be a 2-'
                                      'dimensional array.')
        if not fuav.is_1d_array(y):
            raise IncorrectShapeError('The training data labels must be a 1-'
                                      'dimensional array.')
        if X.shape[0] == 0:
            raise IncorrectShapeError('The data array has to have at least '
                                      'one data point.')
        # If the array is structured the fuav.is_2d_array function takes care
        # of checking whether there is at least one column
        if not fuav.is_structured_array(X) and X.shape[1] == 0:
            raise IncorrectShapeError('The data array has to have at least '
                                      'one feature.')
        if X.shape[0] != y.shape[0]:
            raise IncorrectShapeError('The number of samples in X must be the '
                                      'same as the number of labels in y.')
        if not self._is_classifier and not fuav.is_numerical_array(y):
            raise TypeError('Regressor can only be fitted for a numerical '
                            'target vector.')

        numerical_indices, categorical_indices = fuat.indices_by_type(X)
        self._numerical_indices = numerical_indices
        self._categorical_indices = categorical_indices

        self._is_structured = fuav.is_structured_array(X)
        self._X = X
        self._y = y

        if self._is_classifier:
            unique_y, unique_y_counts = np.unique(self._y, return_counts=True)
            # Order labels lexicographically.
            unique_y_sort_index = np.argsort(unique_y)
            self._unique_y = unique_y[unique_y_sort_index]
            self._unique_y_counts = unique_y_counts[unique_y_sort_index]

            # How many other labels have the same count.
            top_y_index = self._unique_y_counts == np.max(
                self._unique_y_counts)
            top_y_unique_sorted = np.sort(self._unique_y[top_y_index])
            self._majority_label = top_y_unique_sorted[0]

            self._unique_y_probabilities = (
                self._unique_y_counts / self._y.shape[0])
        else:
            self._majority_label = self._y.mean()
            self._unique_y = np.ndarray((0, ))
            self._unique_y_counts = np.ndarray((0, ))
            self._unique_y_probabilities = np.ndarray((0, ))

        self._X_n = self._X.shape[0]
        self._is_fitted = True

    def clear(self) -> None:
        """
        Clears (unfits) the model.

        Raises
        ------
        UnfittedModelError
            Raised when trying to clear a model that has not been fitted yet.
            Try using the fit method to ``fit`` the model first.
        """
        if not self._is_fitted:
            raise UnfittedModelError('This model has not been fitted yet.')

        self._is_fitted = False
        self._X = np.ndarray((0, 0))
        self._y = np.ndarray((0, ))
        self._X_n = int()
        self._unique_y = np.ndarray((0, ))
        self._unique_y_counts = np.ndarray((0, ))
        self._unique_y_probabilities = np.ndarray((0, ))
        self._majority_label = None
        self._is_structured = False
        self._categorical_indices = np.ndarray((0, ))
        self._numerical_indices = np.ndarray((0, ))

    def _get_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Gets distances for a mixture of numerical and categorical features.

        For numerical columns the distance is calculated as the Euclidean
        distance. For categorical columns (i.e. non-numerical, e.g. strings)
        the distance is 0 when the value matches and 1 otherwise.

        Parameters
        ----------
        X : numpy.ndarray
            A data array for which distances to the training data will be
            calculated.

        Raises
        ------
        AssertionError
            Raised when the model is not fitted, X is not a 2-dimensional
            array or X's dtype is different than training data's dtype. It is
            also raised when the distances matrix is not 2-dimensional.

        Returns
        -------
        distances : numpy.ndarray
            An array of distances between X and the training data.
        """
        # pylint: disable=invalid-name
        assert self._is_fitted, 'Cannot calculate distances on unfitted model.'
        assert fuav.is_2d_array(X), 'X must be a 2-dimensional array.'
        assert fuav.are_similar_dtype_arrays(X, self._X), \
            'X must have the same dtype as the training data.'

        distances_shape = (self._X.shape[0], X.shape[0])
        categorical_distances = np.zeros(distances_shape)
        numerical_distances = np.zeros(distances_shape)

        if self._is_structured:
            if self._categorical_indices.size:
                categorical_distances = fud.binary_array_distance(
                    self._X[self._categorical_indices],
                    X[self._categorical_indices])
            if self._numerical_indices.size:
                numerical_distances = fud.euclidean_array_distance(
                    self._X[self._numerical_indices],
                    X[self._numerical_indices])
        else:
            if self._categorical_indices.size:
                categorical_distances = fud.binary_array_distance(
                    self._X[:, self._categorical_indices],
                    X[:, self._categorical_indices])
            if self._numerical_indices.size:
                numerical_distances = fud.euclidean_array_distance(
                    self._X[:, self._numerical_indices],
                    X[:, self._numerical_indices])

        assert categorical_distances.shape == numerical_distances.shape, \
            'Different number of point-wise distances for these feature types.'
        distances = categorical_distances + numerical_distances
        assert fuav.is_2d_array(distances), 'Distances matrix must be 2D.'

        return distances

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts labels of new instances with the fitted model.

        Parameters
        ----------
        X : numpy.ndarray
            The data for which labels will be predicted.

        Raises
        ------
        IncorrectShapeError
            X is not a 2-dimensional array, it has 0 rows or it has a different
            number of columns than the training data.
        UnfittedModelError
            Raised when trying to predict data when the model has not been
            fitted yet. Try using the ``fit`` method to fit the model first.
        ValueError
            X has a different dtype than the data used to fit the model.

        Returns
        -------
        predictions : numpy.ndarray
            Predicted class labels for each data point.
        """
        # pylint: disable=too-many-locals,too-many-branches
        if not self._is_fitted:
            raise UnfittedModelError('This model has not been fitted yet.')
        if not fuav.is_2d_array(X):
            raise IncorrectShapeError('X must be a 2-dimensional array. If '
                                      'you want to predict a single data '
                                      'point please format it as a single row '
                                      'in a 2-dimensional array.')
        if not fuav.are_similar_dtype_arrays(X, self._X):
            raise ValueError('X must have the same dtype as the training '
                             'data.')
        if not X.shape[0]:
            raise IncorrectShapeError('X must have at least one row.')
        # No need to check for columns in a structured array -> this is handled
        # by the dtype checker.
        if not fuav.is_structured_array(X):
            if X.shape[1] != self._X.shape[1]:
                raise IncorrectShapeError(('X must have the same number of '
                                           'columns as the training data '
                                           '({}).').format(self._X.shape[1]))

        predictions = np.empty((X.shape[0], ))

        if self._k < self._X_n:
            distances = self._get_distances(X)
            # If there are 3 nearest neighbours within distances 1, 2 and 2 and
            # k is set to 2, then argpartition will always take the first
            # within distance 2.
            knn = np.argpartition(distances, self._k, axis=0)
            predictions = []
            for column in knn.T:
                close_labels = self._y[column[:self._k]]
                if self._is_classifier:
                    values, counts = np.unique(
                        close_labels, return_counts=True)
                    # If there is a tie in the counts take into consideration
                    # the overall label count in the training data to resolve
                    # it.
                    top_label_index = counts == counts.max()
                    top_label_unique_sorted = np.sort(values[top_label_index])
                    assert len(top_label_unique_sorted.shape) == 1, \
                        'This should be a flat array.'
                    if top_label_unique_sorted.shape[0] > 1:
                        # Resolve the tie.
                        # Get count of these label for the training data.
                        labels_filter = np.array(
                            self._unique_y.shape[0] * [False])
                        for top_prediction in top_label_unique_sorted:
                            unique_y_filter = self._unique_y == top_prediction
                            np.logical_or(
                                labels_filter,
                                unique_y_filter,
                                out=labels_filter)
                        g_top_label = self._unique_y[labels_filter]
                        g_top_label_counts = (
                            self._unique_y_counts[labels_filter])

                        # What if any of the global labels have the same count?
                        g_top_label_index = g_top_label_counts == np.max(
                            g_top_label_counts)
                        g_top_label_sorted = np.sort(
                            g_top_label[g_top_label_index])

                        prediction = g_top_label_sorted[0]
                    else:
                        prediction = top_label_unique_sorted[0]
                else:
                    prediction = close_labels.mean()

                predictions.append(prediction)
            predictions = np.array(predictions)
        else:
            predictions = np.array(X.shape[0] * [self._majority_label])

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates label probabilities for new instances with the fitted model.

        Parameters
        ----------
        X : numpy.ndarray
            The data for which labels probabilities will be predicted.

        Raises
        ------
        IncorrectShapeError
            X is not a 2-dimensional array, it has 0 rows or it has a different
            number of columns than the training data.
        UnfittedModelError
            Raised when trying to predict data when the model has not been
            fitted yet. Try using the ``fit`` method to fit the model first.
        RuntimeError
            Raised when trying to use this method when the predictor is
            initialised as a regressor.
        ValueError
            X has a different dtype than the data used to fit the model.

        Returns
        -------
        probabilities : numpy.ndarray
            Probabilities of each instance belonging to every class. The labels
            in the return array are ordered by lexicographic order.
        """
        if not self._is_classifier:
            raise RuntimeError('This functionality is not available for a '
                               'regressor.')

        if not self._is_fitted:
            raise UnfittedModelError('This model has not been fitted yet.')
        if not fuav.is_2d_array(X):
            raise IncorrectShapeError('X must be a 2-dimensional array. If '
                                      'you want to predict a single data '
                                      'point please format it as a single row '
                                      'in a 2-dimensional array.')
        if not fuav.are_similar_dtype_arrays(X, self._X):
            raise ValueError('X must have the same dtype as the training '
                             'data.')
        if not X.shape[0]:
            raise IncorrectShapeError('X must have at least one row.')
        # No need to check for columns in a structured array -> this is handled
        # by the dtype checker.
        if not fuav.is_structured_array(X):
            if X.shape[1] != self._X.shape[1]:
                raise IncorrectShapeError(('X must have the same number of '
                                           'columns as the training data '
                                           '({}).').format(self._X.shape[1]))

        probabilities = np.empty((X.shape[0], self._unique_y.shape[0]))

        if self._k < self._X_n:
            distances = self._get_distances(X)
            knn = np.argpartition(distances, self._k, axis=0)
            probabilities = []
            for column in knn.T:
                close_labels = self._y[column[:self._k]]
                values, counts = np.unique(close_labels, return_counts=True)
                total_counts = np.sum(counts)
                probs = np.zeros((self._unique_y.shape[0], ))
                for i in range(values.shape[0]):
                    ind = np.where(self._unique_y == values[i])[0]
                    probs[ind] = counts[i] / total_counts
                probabilities.append(probs)
            probabilities = np.array(probabilities)
        else:
            probabilities = np.tile(self._unique_y_probabilities,
                                    (X.shape[0], 1))
        return probabilities
