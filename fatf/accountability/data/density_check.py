"""
The :mod:`fatf.accountability.data.density_check` module holds the object and
functions relevant to performing density checks.
"""

# Author: Rafael Poyiadzi <rp13102@bristol.ac.uk>
# License: BSD 3 clause

import numpy as np
from typing import Optional, Union
from fatf.utils.validation import check_array_type

class DensityCheck(object):
    """Class for doing density check in the provided dataset.

    Performs density checks on the sample provided based on how far from each
    point lie k points.

    """
    def __init__(self,
                 dataset: np.ndarray,
                 alpha: Optional[Union[int, float]] = None,
                 neighbours: Optional[int] = None):
        
        self.numerical_features, self.categorical_features = check_array_type(dataset)
        self.dataset = dataset.copy(order = 'K')

        if len(self.dataset.dtype) == 0:
            self.structure_bool = False
            self.n_samples, self.n_features = self.dataset.shape
        else:
            self.structure_bool = True
            self.n_samples = self.dataset.shape[0]
        self.__check_alpha(alpha)
        self.__check_neighbours(neighbours)
        self.scores = None
        
    @property
    def alpha(self):
        return self._alpha
    
    @alpha.setter
    def alpha(self, alpha: Union[int, float, None]):
        self.__check_alpha(alpha)
    
    @property
    def neighbours(self):
        return self._neighbours
    
    @neighbours.setter
    def neighbours(self, neighbours: Union[int, None]):
        self.check_neighbours(neighbours)
    
    def __check_neighbours(self, neighbours: Union[int, None]):
        if not neighbours:
            self._neighbours = 7
        elif type(neighbours) not in [int]:
            raise TypeError('number of neighbours needs to be integer')
        elif (neighbours < 0 or neighbours > self.n_samples):
            raise ValueError('neighbours parameter needs to be between 1 and number of samples')
        else:
            self._neighbours = neighbours
            
    def __check_alpha(self, alpha):
        if not alpha:
            self._alpha = 0.90
        elif type(alpha) not in [float, int]:
            raise TypeError('alpha parameter needs to be numeric')
        elif (alpha < 0 or alpha > 1):
            raise ValueError('alpha parameter needs to be between 0 and 1')
        else:
            self._alpha = alpha
    
    def __get_distance(self, v0: np.ndarray, v1: np.ndarray):
        dist = 0
        for ftr in self.numerical_features:
            dist += (v0[ftr] - v1[ftr])**2
        return dist
    
    def __get_distance_matrix(self):
        D = np.zeros((self.n_samples, self.n_samples), dtype=float)
        for i in range(self.n_samples):
            v0 = self.dataset[i]
            for j in range(i):
                v1 = self.dataset[j]
                D[i, j] = self.__get_distance(v0, v1)
                D[j, i] = D[i, j]
        self.distance_matrix = D
        
    def get_scores(self, normalise: Optional[bool] = False):
        self.__get_distance_matrix()
        scores = np.zeros(self.n_samples)
        for i in range(self.n_samples):
            row = self.distance_matrix[i, :]
            boundary_neighbour = np.argsort(row)[self.neighbours + 1]
            scores[i] = row[boundary_neighbour]
        self.scores = scores
        if normalise:
            self.scores = self.scores - np.min(self.scores)
            self.scores = self.scores / np.max(self.scores)       
        return self.scores
    
    def filter_dataset(self, alpha: Optional[int] = None):
        if (not alpha and not self.alpha):
            raise ValueError('No alpha parameter provided')
        elif alpha:
            self.alpha = alpha
            
        if not self.scores:
            self.scores = self.get_scores()
        n_tokeep = int(self.n_samples * (1 - self.alpha))
        samples_tokeep = np.argsort(self.scores)[:n_tokeep]
        return self.dataset[samples_tokeep]