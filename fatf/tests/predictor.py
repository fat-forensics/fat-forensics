"""
K-NN Classifier to use in training
Author: Alex Hepburn <ah13558@bristol.ac.uk>
License: new BSD
"""

from collections import Counter
from typing import List

import numpy as np
from scipy.spatial import distance


class KNN(object):
    '''K-nearest-neighbour classifier for testing purposes only

    Attributes:
        k: int of how many nearest neighbours to use
        _X_train: np.ndarray containing training data
        _Y_train: np.array containing training labels
    '''

    def __init__(self, k: int = 1) -> None:
        self.k = k
        self._X_train = None
        self._Y_train = None

    def fit(self, X: np.ndarray, labels: np.array):
        '''Fits model to training data X
        '''
        if X.shape[0] != labels.shape[0]:
            raise ValueError('Number of samples in X_train and labels in Y_train must be equal')
        self._X_train = X
        self._Y_train = labels
        self.n_classes = np.unique(labels).shape[0]

    def predict(self, X_test: np.ndarray) -> np.array:
        '''Returns predictions for X_test
        '''
        counters = self._get_counters(X_test)
        return np.array([c.most_common(1)[0][0] for c in counters])

    def _get_counters(self, X_test: np.array) -> List[Counter]:
        '''Gets list of counters for each test point
        '''
        counters = []
        for i in range(0, X_test.shape[0]):
            sample = X_test[i, :].reshape(1, -1)
            dist = distance.cdist(sample, self._X_train, 'euclidean')[0]
            ind = np.argpartition(dist, self.k)[:self.k]
            counters.append(Counter(self._Y_train[ind]))
        return counters

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        '''Returns probabilities for each test sample for each class
        '''
        counters = self._get_counters(X_test)
        probas = np.ndarray((X_test.shape[0], self.n_classes))
        for i in range(0, len(counters)):
            for j in range(0, self.n_classes):
                probas[i][j] = counters[i][j] / self.k
        return probas
