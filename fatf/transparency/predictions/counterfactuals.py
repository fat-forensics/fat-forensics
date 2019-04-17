"""
The :mod:`fatf.path.to.the.file.in.the.module` module implements a counterfactual explainer.
"""

# Author: Rafael Poyiadzi <rp13102@bristol.ac.uk>
# License: BSD 3 clause
import numpy as np
import itertools
from typing import Optional, List, Any, Union, Dict, Tuple
from numbers import Number

import inspect
import warnings

import fatf.utils.models.validation as fumv
import fatf.utils.array.validation as fuav

from fatf.exceptions import IncorrectShapeError

Index = Union[str, int]

def combine_arrays(array1: np.ndarray,
                   array2: np.ndarray) -> List[int]:
    """Will combine two numpy arrays in an incremental manner.

    Form a list out of the elements of two numpy arrays. For example:
    array1 = np.array([1,2,3]) and array2 = np.array([4,5,6,7,8]), the output
    will be [1,4,2,5,3,6,7,8].

    Parameters
    ----------
    array1 : numpy array.
    array2 : numpy array

    Raises
    ------
    NA

    Returns
    -------
    List with ordered merge of the two arrays.
    """
    arrays = [array1, array2]
    array_lengths = [len(array1), len(array2)]
    sorted_lengths = np.argsort(np.array(array_lengths))
    min_val = array_lengths[sorted_lengths[0]]
    longest_list = arrays[sorted_lengths[1]]
    newlist = []
    for i in range(min_val):
        newlist.append(arrays[0][i])
        newlist.append(arrays[1][i])

    return newlist + longest_list[min_val:].tolist()


def textualise_counterfactuals():
    pass


class CounterfactualExplainer(object):


    def _validate_input(self, model, predictive_function, dataset,
                        categorical_indices, numerical_indices):
        input_is_valid = False
        # Validate model/predictive function
        if model is not None:
            if not fumv.check_model_functionality(model):
                raise RuntimeError('The model object requires a predict '
                                   'method to be used with this explainer.')
        if predictive_function is not None:
            if not isinstance(predictive_function, callable):
                raise TypeError('The predictive_function parameter should be '
                                'a python function.')
            # The predictive function is to have one non-optional parameter
            required_param_n = 0
            params = inspect.signature(predictive_function).parameters
            for param in params:
                if params[param].default is params[param].empty:
                    required_param_n += 1
            if required_param_n != 1:
                raise AttributeError('The predictive function requires '
                                     'exactly 1 non-optional parameter: a '
                                     'data array to be predicted.')

        if model is None and predictive_function is None:
            raise RuntimeError('You either need to specify a model or a '
                               'predictive_function parameter to initialise '
                               'a counterfactual explainer.')
        elif model is not None and predictive_function is not None:
            warnings.warn(
                'Both a model and a predictive_function parameters '
                'were supplied. A predictive functions takes the '
                'precedence during the execution.',
                UserWarning)

        # Validate data
        if dataset is not None:
            if not fuav.is_base_array(dataset):
                raise ValueError('The dataset has to be of a base type '
                                 '(strings and numbers).')
            if not fuav.is_2d_array(dataset):
                raise IncorrectShapeError('The data array has to be '
                                          '2-dimensional.')
            structured_dataset = fuav.is_structured_array(dataset)
        else:
            structured_dataset = None

        # TODO
        # Validate categorical and numerical indices
        if categorical_indices is not None:
            if not isinstance(categorical_indices, list):
                raise TypeError('categorical_indices parameter either has to '
                                'be a list or None.')
        if numerical_indices is not None:
            if not isinstance(numerical_indices, list):
                raise TypeError('numerical_indices parameter either has to '
                                'be a list or None.')
        # All have to be either ints (non-structured) or strings (structured)
        # they have to be valid for the array if given
        # They have to be disjoin

        input_is_valid = True
        return input_is_valid



    def __init__(
            self,
            model: Optional[object] = None,
            predictive_function: Optional[callable] = None,
            dataset: Optional[np.ndarray] = None,
            categorical_indices: Optional[List[Index]] = None,
            numerical_indices: Optional[List[Index]] = None,
            # TODO
            feature_ranges: Optional[Dict[Index, Any]] = None,
            stepsizes: Optional[Dict[Index, Union[int, np.ndarray]]] = None,
            distance_functions: Optional[Dict[Index, callable]] = None,
            max_comb: Optional[int] = 2):

        # Validate input
        assert self._validate_input(model, predictive_function, dataset,
                                    categorical_indices, numerical_indices
                                    ), 'The input must be valid.'

        # Select a predictive function
        if predictive_function is None:
            self.predict = model.predict
        else:
            self.predict = predictive_function

        # TODO
        # Choose categorical and numerical indices
        if dataset is not None:
            if fuav.is_structured_array(dataset):
                pass
            else:
                pass
        if categorical_indices is None:
            pass
        else:
            self.categorical_indices = set(categorical_indices)





        if feature_ranges is None:
            self._get_feature_ranges(dataset)
        else:
            # all_feature_indices = self.categorical_indices.union(self.numerical_indices)

            set0 = set(feature_ranges.keys())
            set1 = set(dataset.dtype.names)
            diff = list(set1.difference(set0))
            for ftr in set1:
                if ftr not in self.categorical_indices and ftr not in diff:
                    if len(feature_ranges[ftr].keys()) != 2:
                        if ftr not in diff:
                            diff.append(ftr)
            if not (set0 == set1):
                self.feature_ranges = feature_ranges
                self._get_feature_ranges(dataset, column_indices=diff)





        default_stepsize = 1
        self.max_comb = max_comb
        if not stepsizes:
            self.stepsizes = {}
            for ftr in dataset.dtype.names:
                if ftr not in self.categorical_indices:
                    self.stepsizes[ftr] = default_stepsize
        set0 = set(stepsizes.keys())
        set1 = set(dataset.dtype.names)
        if not np.all(set0 == set1):
            self.stepsizes = stepsizes
            for ftr in list(set1.difference(set0)):
                if ftr not in self.categorical_indices:
                    self.stepsizes[ftr] = default_stepsize
        else:
            self.stepsizes = stepsizes

        # Validate distance_finctions
        # either None, or a non-empty dictionary
        # all of the keys must be valid indices if dataset is given
        # Prepare distance functions
        if distance_functions is None:
            self.distance_functions = {}
            for ftr in dataset.dtype.names:
                if ftr not in self.categorical_indices:
                    self.distance_functions[ftr] = lambda x, y: np.abs(x - y)
                else:
                    self.distance_functions[ftr] = lambda x, y: int(x != y)
        else:
            self.distance_functions = distance_functions

    def _get_feature_ranges(
            self,
            dataset: np.ndarray,
            column_indices: Optional[List[Index]] = None
    ) -> List[Tuple[int, int]]:
        """

        """
        if not column_indices:
            features_to_complete = dataset.dtype.names
            self.feature_ranges = {}
        else:
            features_to_complete = column_indices

        for field_name in features_to_complete:
            if field_name not in self.categorical_indices:
                try:
                    min_val = self.feature_ranges[field_name]['min']
                except:
                    min_val = min(dataset[field_name])

                try:
                    max_val = self.feature_ranges[field_name]['max']
                except:
                    max_val = max(dataset[field_name])

                self.feature_ranges[field_name] = {'min':min_val,
                                                   'max': max_val}
            else:
                self.feature_ranges[field_name] = np.unique(dataset[field_name])

    def _get_distance(self,
                      instance_one: np.ndarray,
                      instance_two: np.ndarray) -> Number:
        dist = 0.0
        features = instance_one.dtype.names
        for feature in features:
            dist += self.distance_functions[feature](instance_one[feature],
                                                     instance_two[feature])
        return dist

    def _modify_instance(self,
                        x: np.ndarray,
                        ftrs: List[Index],
                        vals: Tuple[Any]):
        for idx, ftr in enumerate(ftrs):
            x[ftr] = vals[idx]

    def _get_value_combinations(self,
                               ftr_combination: List[Union[str, int]],
                                original_instance) -> List[tuple]:
        list_of_values = []
        for ftr in ftr_combination:
            if ftr not in self.categorical_indices:

                s0 = np.arange(self.feature_ranges[ftr]['min'],
                               original_instance[ftr],
                               self.stepsizes[ftr])

                s1 = np.arange(original_instance[ftr] + self.stepsizes[ftr],
                               self.feature_ranges[ftr]['max'],
                               self.stepsizes[ftr])

                combined = combine_arrays(s0[::-1], s1)

                list_of_values.append(combined)
            else:
                t = self.feature_ranges[ftr].tolist()
                t.pop(t.index(original_instance[ftr]))
                list_of_values.append(t)
        return itertools.product(*list_of_values)

    def explain_instance(self,
                         instance: np.ndarray,
                         counterfactual_class: Optional[int] = None) -> list:
        """

        """
        if counterfactual_class is None:
            # Predict the class and make sure that it is different
            pass

        ftrs = instance.dtype.names
        ftrs_indices = list(range(len(ftrs)))

        counterfactuals = []

        for n in range(1, self.max_comb+1):
            for ftr_combination_indices in itertools.combinations(ftrs_indices, n):
                scores = []
                ftr_combination = [ftrs[item] for item in ftr_combination_indices]
                for value_combination in self._get_value_combinations(ftr_combination, instance):
                    test_instance = instance.copy()
                    self._modify_instance(test_instance,
                                         ftr_combination,
                                         value_combination)
                    pred = self.predict(np.array(test_instance.tolist(), dtype=float).reshape(1, -1))
                    if pred[0] == counterfactual_class:
                        distance = self._get_distance(instance, test_instance)
                        scores.append((test_instance, distance))
                try:
                    best = np.argmin([item[1] for item in scores])
                    counterfactuals.append(scores[best])
                except:
                    pass
        return counterfactuals
