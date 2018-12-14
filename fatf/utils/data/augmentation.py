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
        self.numerical_indices, self.categorical_indices = check_array_type(dataset)
        self.dataset = dataset.copy(order = 'K')
        if len(self.dataset.dtype) == 0:
            self.structure_bool = False
        else:
            self.structure_bool = True
            
        self.n_samples = self.dataset.shape[0]
        self.y = y.copy(order = 'K')
        if balanced:
            self.balanced = balanced
            self.n_positives = sum(y == 1)
            self.n_negatives = sum(y == 0)
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
        random_indices = np.random.choice(self.n_samples, 
                                          self.n_draws, 
                                          replace=replacement)
        
        random_draws_lambda = np.random.beta(self.beta_parameters[0],
                                             self.beta_parameters[1],
                                             self.n_draws)
        if self.structure_bool:
            sampled_x = np.empty((self.n_draws, ), dtype=self.dataset.dtype)
        else:
            sampled_x = np.zeros((self.n_draws, self.dataset.shape[1]), dtype=self.dataset.dtype)
        sampled_y = np.array(np.zeros(self.n_draws)).reshape(-1, 1)
        for i in range(self.n_draws):
            l = random_draws_lambda[i]
            random_draw_x = self.dataset[random_indices[i]]
            for ftr in self.numerical_indices:
                sampled_x[i][ftr] = (1 - l) * random_draw_x[ftr] \
                                    + l * self.subject_instance[ftr]
            sampled_y[i] = (1 - l) * self.y[random_indices[i]] \
                                + l * self.subject_y
        return sampled_x, sampled_y
  