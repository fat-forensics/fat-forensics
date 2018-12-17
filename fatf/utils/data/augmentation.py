"""
The :mod:`fatf.utils.data.augmentation` module holds the object and
functions relevant to performing Mixup data generation.
"""

# Author: Rafael Poyiadzi <rp13102@bristol.ac.uk>
# License: BSD 3 clause

import numpy as np
import mypy
from fatf.utils.validation import check_array_type
from fatf.exceptions import CustomValueError
from typing import Optional, List, Union

class Mixup(object):
    """Object to perform data generation following the Mixup method.

    For a specific point, select points frm the dataset at random, then draw
    samples frm Beta distribution, and form new points according to the convex
    combinations of the points.
    """
    def __init__(self,
                 dataset: np.ndarray,
                 y: np.array,
                 beta_parameters: Optional[List[Union[int, float]]] = None,
                 balanced: Optional[bool] = False):
        
        # TODO: think about this parameter
        self.threshold = 0.50
        
        self.numerical_indices, self.categorical_indices = check_array_type(dataset)
        self.dataset = dataset.copy(order = 'K')
        if len(self.dataset.dtype) == 0:
            self.structure_bool = False
        else:
            self.structure_bool = True
            
        self.n_samples = self.dataset.shape[0]
        self.y = y.copy(order = 'K')
        self.balanced = balanced

        if self.dataset.shape[0] != self.y.shape[0]:
            raise ValueError('Input structures are not of equal length')
        self.check_beta_parameters(beta_parameters)
        
    @property
    def beta_parameters(self) -> List[Union[int, float]]:
        return self._beta_parameters
    
    @beta_parameters.setter
    def beta_parameters(self, 
                        beta_parameters: List[Union[int, float]]):
        self.check_beta_parameters(beta_parameters)
        
    def check_beta_parameters(self, 
                              beta_parameters: List[Union[int, float]]):
        if not beta_parameters:
            self._beta_parameters: List[Union[int, float]] = [2, 5]
        elif len(beta_parameters) != 2:
            raise ValueError('Need two parameters for beta distribution')
        elif not np.all([type(item) in [int, float] for item in beta_parameters]):
            raise TypeError('Beta parameters need to be int or float')
        elif not np.all([item > 0 for item in beta_parameters]):
            raise ValueError('Beta parameters need to be positive')
        else:
            self._beta_parameters = beta_parameters
            
    def sample(self,
               subject_instance: np.ndarray,
               subject_y: np.array,
               n_draws: Optional[int] = None,
               replacement: Optional[bool] = True):
        if not n_draws:
            self.n_draws = 1
        else:
            self.n_draws = n_draws
        
        if not subject_instance.dtype == self.dataset.dtype:
            raise CustomValueError('The input instance should have the same dtype as the input dataset')
        
        self.subject_instance = subject_instance
        self.subject_y = subject_y
        if not self.balanced:
            random_indices = np.random.choice(self.n_samples, 
                                              self.n_draws, 
                                              replace=replacement)
        elif self.n_draws > 10:
            unique, counts = np.unique(self.y, return_counts = True)
            total = sum(counts)
            random_indices = []
            for clss, count in zip(unique, counts):
                pos = np.where(self.y == clss)[0]
                n_tosample = int((count / total) * self.n_draws)
                random_indices_clss = np.random.choice(pos,
                                                       n_tosample,
                                                       replace=replacement)
                random_indices.extend(random_indices_clss)
            random_indices = np.array(random_indices)
        else:
            raise ValueError('Please consider a higher value for N')
        
        random_draws_lambda = np.random.beta(self.beta_parameters[0],
                                             self.beta_parameters[1],
                                             self.n_draws)
        if self.structure_bool:
            sampled_x = np.empty((self.n_draws, ), dtype=self.dataset.dtype)
        else:
            sampled_x = np.zeros((self.n_draws, self.dataset.shape[1]), dtype=self.dataset.dtype)
        
        if len(self.y.shape) == 1:
            sampled_y = np.zeros(self.n_draws).reshape(-1, 1)
        else:
            n_classes = self.y.shape[1]
            sampled_y = np.zeros((self.n_draws, n_classes)).reshape(-1, n_classes)
        for i in range(self.n_draws):
            l = random_draws_lambda[i]
            random_draw_x = self.dataset[random_indices[i]]
            for ftr in self.numerical_indices:
                sampled_x[i][ftr] = (1 - l) * random_draw_x[ftr] \
                                    + l * self.subject_instance[ftr]
            for ftr in self.categorical_indices:
                if l <= self.threshold:
                    sampled_x[i][ftr] = self.subject_instance[ftr]
                else:
                    sampled_x[i][ftr] = random_draw_x[ftr]
            sampled_y[i] = (1 - l) * self.y[random_indices[i]] \
                                + l * self.subject_y
        return sampled_x, sampled_y
  