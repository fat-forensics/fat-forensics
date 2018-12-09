"""
The :mod:`fatf.path.to.the.file.in.the.module` module implements a counterfactual explainer.
"""

# Author: Rafael Poyiadzi <rp13102@bristol.ac.uk>
# License: BSD 3 clause
import numpy as np
import itertools

def combine_arrays(array1: np.array, array2: np.array) -> list:
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

class Explainer(object):
    def __init__(self, 
                 model,
                 categorical_features: list,
                 dataset: np.ndarray,
                 monotone = False,
                 stepsizes = None,
                 max_comb = 2,
                 feature_ranges=None,
                 dist_funcs=None):
        
        self.model = model
        self.monotone = monotone
        self.dataset = dataset
        self.categorical_features = categorical_features

        default_stepsize = 1
        if not feature_ranges:
            self.get_feature_ranges()
        else:
            set0 = set(feature_ranges.keys())
            set1 = set(self.dataset.dtype.names)
            diff = list(set1.difference(set0))
            for ftr in set1:
                if (ftr not in self.categorical_features and
                    ftr not in diff ):
                    if len(feature_ranges[ftr].keys()) != 2:
                        if ftr not in diff:
                            diff.append(ftr)
            if not np.all(set0 == set1):
                self.feature_ranges = feature_ranges
                self.get_feature_ranges(tocomplete = diff)
            
        self.max_comb = max_comb
        if not stepsizes:
            self.stepsizes = {}
            for ftr in self.dataset.dtype.names:
                if ftr not in self.categorical_features:
                    self.stepsizes[ftr] = default_stepsize
        set0 = set(stepsizes.keys())
        set1 = set(self.dataset.dtype.names)
        if not np.all(set0 == set1):
            self.stepsizes = stepsizes
            for ftr in list(set1.difference(set0)):
                if ftr not in self.categorical_features:
                    self.stepsizes[ftr] = default_stepsize
        else:      
            self.stepsizes = stepsizes

        if not dist_funcs:
            self.dist_funcs = {}
            for ftr in self.dataset.dtype.names:
                if ftr not in self.categorical_features:
                    self.dist_funcs[ftr] = lambda x, y: np.abs(x - y)
                else:
                    self.dist_funcs[ftr] = lambda x, y: int(x != y)
        else:
            self.dist_funcs = dist_funcs
    
    def dist(self, v0, v1):
        dist = 0
        features = v0.dtype.names
        for feature in features:
            dist += self.dist_funcs[feature](v0[feature], v1[feature])
        return dist  
      
    def get_feature_ranges(self, tocomplete = 'all'):
        if tocomplete == 'all':
            features_to_complete = self.dataset.dtype.names
            self.feature_ranges = {}
        else:
            features_to_complete = tocomplete
        for field_name in features_to_complete:
            if field_name not in self.categorical_features:
                try:
                    min_val = self.feature_ranges[field_name]['min']
                except:
                    min_val = min(self.dataset[field_name])
                
                try:
                    max_val = self.feature_ranges[field_name]['max']
                except:
                    max_val = max(self.dataset[field_name])
                    
                self.feature_ranges[field_name] = {'min':min_val, 
                                                   'max': max_val}
            else:
                self.feature_ranges[field_name] = np.unique(self.dataset[field_name])
    
    def modify_instance(self, x, ftrs, vals):
        for idx, ftr in enumerate(ftrs):
            x[ftr] = vals[idx]
    
    def get_value_combinations(self, ftr_combination):
        list_of_values = []
        for ftr in ftr_combination:
            if ftr not in self.categorical_features:
                s0 = np.arange(self.feature_ranges[ftr]['min'], 
                               self.instance[ftr], 
                               self.stepsizes[ftr])
                s1 = np.arange(self.instance[ftr] + self.stepsizes[ftr], 
                               self.feature_ranges[ftr]['max'], 
                               self.stepsizes[ftr])
                combined = combine_arrays(s0[::-1], s1)

                list_of_values.append(combined)
            else:
                list_of_values.append(self.feature_ranges[ftr])
        return itertools.product(*list_of_values)
    
    def pretty_print(self, all_scores):
        order = np.argsort([item[1] for item in all_scores])
        for idx in order:
            instance = all_scores[idx][0]
            sim = []
            for ftr in self.ftrs:
                if instance[ftr] == self.instance[ftr]:
                    sim.append('_')
                else:
                    sim.append('*')
            print('              ', sim)    
            print('New instance: ', instance, '// Distance: ', all_scores[idx][1])
            print('\n')
            
    def generate_counterfactual(self, 
                                instance: np.void,
                                target_class: int
                                ):
        self.target_class = target_class
        self.instance = instance.copy(order='K')
        self.ftrs = instance.dtype.names
        ftrs_indices = list(range(len(self.ftrs)))
        all_scores = []
        for n in range(1, self.max_comb+1):            
            ftr_combinations = list(itertools.combinations(ftrs_indices, n))
            
            for ftr_combination_indices in ftr_combinations:
                scores = []
                ftr_combination = [self.ftrs[item] for item in ftr_combination_indices]
                value_combinations = self.get_value_combinations(ftr_combination)
                for value_combination in value_combinations:
                    test_instance = self.instance.copy(order='K')
                    self.modify_instance(test_instance, 
                                         ftr_combination, 
                                         value_combination)
                    pred = self.model.predict(np.array(test_instance.tolist(), dtype=float).reshape(1, -1))
                    if pred[0] == self.target_class:
                        scores.append((test_instance, self.dist(self.instance, test_instance)))
                        #TODO: work on this
                        #only works for checking 1 variable atm
                        if self.monotone:
                            break
                try:
                    best = np.argmin([item[1] for item in scores])
                    all_scores.append(scores[best])
                except:
                    pass
        return self.pretty_print(all_scores)