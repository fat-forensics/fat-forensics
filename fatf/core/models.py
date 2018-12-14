"""
The :mod:`fatf.core.models` has some basic classification models for structured and 
unstructured np.ndarrays.
"""

# Author: Alex Hepburn <ah13558@bristol.ac.uk>
# License: BSD Clause 3

from collections import Counter
from typing import List

import numpy as np
from scipy.spatial import distance

from fatf.exceptions import CustomValueError


class KNN(object):
    """K-nearest-neighbour classifier for non-structured np.ndarray

    Attributes
    ----
    k : int 
        How many nearest neighbours to use
    _X_train : np.ndarray 
        Containing training data
    _Y_train : np.array 
        Containing training labels
    """

    def __init__(self, k: int = 1) -> None:
        self.k = k
        self._X_train = None
        self._Y_train = None

    def fit(self, X: np.ndarray, labels: np.array):
        """Fits model to training data X

        Args
        ----
        X : np.ndarray
            Training data for algorithm to use.
        labels : np.array
            Labels corresponding to the training data

        Raises
        ----
        CustomValueError:
            Number of samples in X and 
        """
        if X.shape[0] != labels.shape[0]:
            raise CustomValueError('Number of samples in X and labels given'
                                   'must be equal')
        self._X_train = X
        self._Y_train = labels
        self.n_classes = np.unique(labels).shape[0]

    def predict(self, X_test: np.ndarray) -> np.array:
        """Returns predictions for X_test

        Args
        ----
        X_test : np.ndarray
            Data that predictions will be calculated for
        
        Returns
        ----
        predictions : np.array
            Array containing predictions for each sample in X_test
        """
        counters = self._get_counters(X_test)
        predictions = np.array([c.most_common(1)[0][0] for c in counters])
        return predictions

    def _get_counters(self, X_test: np.array) -> List[Counter]:
        """Gets list of counters for each test point

        Args
        ----
        X_test : np.ndarray
            Data that predictions will be calculated for
        
        Returns
        ----
        counters : List[Counter]
            List of counters with k points for each data point. Counts how
            many predictions for each class there is.
        """
        counters = []
        for i in range(0, X_test.shape[0]):
            sample = X_test[i, :].reshape(1, -1)
            dist = distance.cdist(sample, self._X_train, 'euclidean')[0]
            ind = np.argpartition(dist, self.k)[:self.k]
            counters.append(Counter(self._Y_train[ind]))
        return counters

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Returns probabilities for each test sample for each class

        Args
        ----
        X_test : np.ndarray
            Data that predictions will be calculated for
        
        Returns
        ----
        probas : np.ndarray
            Returns matrix of probabilities for each sample with the probability
            of that sample being predicted each class.
        """
        counters = self._get_counters(X_test)
        probas = np.ndarray((X_test.shape[0], self.n_classes))
        for i in range(0, len(counters)):
            for j in range(0, self.n_classes):
                probas[i][j] = counters[i][j] / self.k
        return probas


class KNN_structured(object):
    """K-nearest-neighbour classifier for structured arrays. Only works with
    purely numerical data currently.

    Attributes
    ----
    k : int 
        How many nearest neighbours to use
    _X_train : np.ndarray 
        Containing training data
    _Y_train : np.array 
        Containing training labels
    """

    def __init__(self, k: int = 4) -> None:
        self.k = k
        self._X_train = None
        self._Y_train = None

    def fit(self, X: np.ndarray, labels: np.array):
        """Fits model to training data X

        Args
        ----
        X : np.ndarray
            Training data for algorithm to use.
        labels : np.array
            Labels corresponding to the training data

        Raises
        ----
        CustomValueError:
            Number of samples in X and 
        """
        X = np.array(X.tolist())
        if X.shape[0] != labels.shape[0]:
            raise CustomValueError('Number of samples in X and labels given'
                                   'must be equal')
        self._X_train = X
        self._Y_train = labels
        self.n_classes = np.unique(labels).shape[0]

    def predict(self, X_test: np.ndarray) -> np.array:
        """Returns predictions for X_test

        Args
        ----
        X_test : np.ndarray
            Data that predictions will be calculated for
        
        Returns
        ----
        predictions : np.array
            Array containing predictions for each sample in X_test
        """
        X_test = np.ndarray(X_test.tolist())
        counters = self._get_counters(X_test)
        return np.array([c.most_common(1)[0][0] for c in counters])

    def _get_counters(self, X_test: np.array) -> List[Counter]:
        """Gets list of counters for each test point

        Args
        ----
        X_test : np.ndarray
            Data that predictions will be calculated for
        
        Returns
        ----
        counters : List[Counter]
            List of counters with k points for each data point. Counts how
            many predictions for each class there is.
        """
        counters = []
        for i in range(0, X_test.shape[0]):
            sample = X_test[i, :].reshape(1, -1)
            dist = distance.cdist(sample, self._X_train, 'euclidean')[0]
            ind = np.argpartition(dist, self.k)[:self.k]
            counters.append(Counter(self._Y_train[ind]))
        return counters

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Returns probabilities for each test sample for each class

        Args
        ----
        X_test : np.ndarray
            Data that predictions will be calculated for
        
        Returns
        ----
        probas : np.ndarray
            Returns matrix of probabilities for each sample with the probability
            of that sample being predicted each class.
        """
        X_test = np.array(X_test.tolist())
        counters = self._get_counters(X_test)
        probas = np.ndarray((X_test.shape[0], self.n_classes))
        for i in range(0, len(counters)):
            for j in range(0, self.n_classes):
                probas[i][j] = counters[i][j] / self.k
        return probas
