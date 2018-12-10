"""
The :mod:`fatf.fairness.metrics` module holds the object to be used for 
performing fairness checks.
"""

# Author: Rafael Poyiadzis <rp13102@bristol.ac.uk>
# License: BSD 3 clause

from __future__ import division
import numpy as np
import itertools
import math
from fatf.utils.validation import check_array_type
from types import FunctionType

def euc_dist(v0: np.ndarray, v1: np.ndarray) -> float:
    return np.linalg.norm(v0 - v1)**2


def get_distance_mat(mat: np.ndarray, func: FunctionType) -> np.matrix:
    n = mat.shape[0]
    D = np.matrix(np.zeros(n**2).reshape(n, n))
    for i in range(n):
        v0 = np.squeeze(mat[i, :]).reshape(-1, 1)
        for j in range(i):
            v1 = np.squeeze(mat[j, :]).reshape(-1, 1)
            D[i, j] = func(v0, v1)
            D[j, i] = D[i, j]
    return D



# =============================================================================
# def remove_field(dataset, field):
#     """ Removes a field from a Structured Numpy Array.
# 
#     Description: Removes a field from a Structured Numpy Array.
# 
#     Args:
#         dataset: Structured Numpy Array containing the features.
#         field: String contatining the name of the field to be removed.
# 
#     Returns: Structured Numpy Array without the specified field.
# 
#     Raises:
#         ValueError: If field not in dataset.dtypes
#         """
#     field_names = list(dataset.dtype.names)
#     if field in field_names:
#         field_names.remove(field)
#     else:
#         raise ValueError('Field not found')
#     return dataset[field_names]
# =============================================================================


def get_bins_inf(boundaries: list) -> list:
    """
	Produces bins, given a set of boundaries.
    Example: boundaries = [20, 40]
            bins = [(-inf, 20), (20, 40), (40, inf)]
    """
    bins = []
    INF = math.inf
    n_boundaries = len(boundaries)
    bins.append((-INF, boundaries[0]))
    for i in range(n_boundaries-1):
        bins.append((boundaries[i], boundaries[i+1]))
    bins.append((boundaries[-1], INF))
    return bins



     
def tpr(cm):
    den = cm[0, 0] + cm[0, 1]
    num = cm[0, 0]
    if den == 0:
        return 0
    else:
        return num / den

def tnr(cm):
    den = cm[1, 1] + cm[1, 0]
    num = cm[1, 1]
    if den == 0:
        return 0
    else:
        return num / den
        
def fpr(cm):
    den = cm[0, 0] + cm[0, 1]
    num = cm[0, 1]
    if den == 0:
        return 0
    else:
        return num / den
    
def fnr(cm):
    den = cm[1, 1] + cm[1, 0]
    num = cm[1, 0]
    if den == 0:
        return 0
    else:
        return num / den
 
def tpr_mc(cm, idx):
    den = np.sum(cm[idx, :])
    num = cm[idx, idx]
    if den == 0:
        return 0
    else:
        return num / den

def fpr_mc(cm, idx):
    den = np.sum(cm[idx, :])
    num = den - cm[idx, idx]
    if den == 0:
        return 0
    else:
        return num / den
        
def tnr_mc(cm, idx):
    den = np.sum(np.diag(cm))
    num = den - cm[idx, idx]
    if den == 0:
        return 0
    else:
        return num / den

def fnr_mc(cm, idx):
    den = np.sum(cm[:, idx])
    num = den - cm[idx, idx]
    if den == 0:
        return 0
    else:
        return num / den
 
def treatment_mc(cm, idx):
    den = 0
    num = 0
    for row_number, row in enumerate(cm):

        fp_sum = np.sum(row) - np.squeeze(np.asarray(row))[row_number]
        den += fp_sum
        if row_number == idx:
            num = fp_sum
    return num / den

class FairnessChecks(object):
    def __init__(self, 
                 dataset: np.ndarray, 
                 targets: np.array,
                 distance_funcs: dict,
                 protected: str,
                 toignore: list = None,
                 ):
        self.dataset = dataset.copy(order='K')
        self.structured_bool = True
        if len(self.dataset.dtype) == 0:
            self.structured_bool = False

            
        self.protected_field = protected
        numerical_features, categorical_features = check_array_type(self.dataset)
        self.numerical_features = numerical_features.tolist()
        self.categorical_features = categorical_features.tolist()
        self.features = self.numerical_features + self.categorical_features
        if toignore:
            for item in toignore:
                if item in self.numerical_features:
                    self.numerical_features.pop(item)
                elif item in self.categorical_features:
                    self.categorical_features.pop(item)
        self.targets = targets
        self.distance_funcs = distance_funcs
        self.check_funcs()
        self.checks = {'accuracy': lambda x: sum(np.diag(x)) / np.sum(x),
                      'true_positives': lambda x: x[0, 0],
                      'true_negatives': lambda x: x[1, 1],
                      'false_positives': lambda x: x[0, 1],
                      'false_negatives': lambda x: x[1, 0],
                      'true_positive_rate': tpr,
                      'true_negative_rate': tnr,
                      'false_positive_rate': fpr,
                      'false_negative_rate': fnr,
                      'treatment': lambda x: x[0, 1] / x[1, 0]
                      }
        
        self.checks_multiclass = {'accuracy': lambda x, i: sum(np.diag(x)) / np.sum(x),
                                  'true_positives': lambda x, i: x[i, i],
                                  'true_negatives': lambda x, i: np.sum(np.diag(x)) - x[i, i],
                                  'false_positives': lambda x, i: np.sum(x[i, :]) - x[i, i],
                                  'false_negatives': lambda x, i: np.sum(x[:, i]) - x[i, i],
                                  'true_positive_rate': tpr_mc,
                                  'true_negative_rate': tnr_mc,
                                  'false_positive_rate': fpr_mc,
                                  'false_negative_rate': fnr_mc,
                                  'treatment': treatment_mc
                                  }

        
    @property
    def features_to_check(self):
        return self.__features_to_check
    
    @features_to_check.setter
    def features_to_check(self, features_to_check):
        self.__features_to_check = features_to_check
        
    @property
    def distance_funcs(self):
        return self.__distance_funcs
    
    @distance_funcs.setter
    def distance_funcs(self, distance_funcs):
        self.__distance_funcs = distance_funcs
        
    @property
    def targets(self):
        return self.__targets
    
    @targets.setter
    def targets(self, targets):
        self.__targets = targets
    
    def check_funcs(self):
        distance_funcs_keys = set(self.distance_funcs.keys())
        for field_name in self.numerical_features:
            if (field_name not in distance_funcs_keys or
                    self.distance_funcs[field_name] is None):
                self.distance_funcs[field_name] = lambda x, y: np.abs(x-y)
        
            if field_name in self.categorical_features:
                if (field_name not in distance_funcs_keys or 
                        self.distance_funcs[field_name] is None):
                    raise ValueError('missing distance function for %s: ', field_name)

    def check_systemic_bias(self, threshold: int = 0.1) -> list:
        """ Checks for systemic bias in the dataset.
    
        Description: Will check if similar instances, that differ only on the
                    protected attribute have been treated differently. Treated
                    refers to the 'target' of the instance.
    
        Args:
            dataset: Structured Numpy Array containing the features.
            treatments: Dictionary of Lists. Keys are: 'Protected', 'Feature',
                        'Target', 'ToIgnore'.
            distance_funcs: Dictionary of functions. One for each field that is
                            categorized as 'Feature'.
            threshold: Float value for what counts as similar. Default to 0.
    
        Returns: List of pairs of instances that are similar but have been treated
                differently.
    
        Raises:
            NA
        """
        n_samples = self.dataset.shape[0]
        if self.structured_bool:
            protected = self.dataset[self.protected_field]
        else:
            protected = self.dataset[:, self.protected_field].astype(int)
        distance_list = []
        for i in range(n_samples):
            v0 = self.dataset[i]
            protected0 = protected[i]
            target0 = self.targets[i]
            for j in range(i):
                v1 = self.dataset[j]
                protected1 = protected[j]
                target1 = self.targets[j]
                dist = self._apply_distance_funcs(v0, v1, toignore=[self.protected_field])
    
                same_protected = protected0 == protected1
                same_target = target0 == target1
                if (dist <= threshold and
                    same_protected == False and
                    same_target == False):
                    distance_list.append((dist, (i,j)))
        return distance_list

    def _apply_distance_funcs(self, v0: np.ndarray, v1: np.ndarray, toignore: list = None) -> int:
        """
        Computes the distance between two instances, based on the distance functions
        provided by the user.
        """
        dist = 0
        for feature in self.features:
            if feature not in toignore:
                dist += self.distance_funcs[feature](v0[feature], v1[feature])
        return dist

    def check_sampling_bias(self, 
                            features_to_check: list = None, 
                            return_weights=False, 
                            boundaries_for_numerical: dict = None) -> (dict, np.ndarray):
        """ Checks for sampling bias in the dataset.
    
        Description: Will check if the different sub-populations defined by the
                    features_to_check have similar representation (sample size).
                    *Only for Categorical data*.
    
        Args:
            dataset: Structured Numpy Array containing the features.
            treatments: Dictionary of Lists. Keys are: 'Protected', 'Feature',
                        'Target', 'ToIgnore'.
            features_to_check: List of Strings, for which features to consider
                                the sub-populations.
            return_weights: Boolean, on whether to return weights to be used for
                            cost-sensitive learning.
    
        Returns: Counts of data for each sub-population defined by the cross-product
                of the provided features.
    
        Raises:
            NA
            """
        if not boundaries_for_numerical:
            boundaries_for_numerical = {}
            
        if len(features_to_check) == 0:
            if self.features_to_check is None:
                raise ValueError('no features to check provided')
        else:
            self.features_to_check = features_to_check
        counts = {}
        cross_product = self._get_cross_product(boundaries_for_numerical)
        counts = self._get_counts(cross_product, boundaries_for_numerical)
        if not return_weights:
            return counts
        else:
            weights = self._get_weights_costsensitivelearning(counts, boundaries_for_numerical)
            return counts, weights

    def _get_weights_costsensitivelearning(self, counts: dict, boundaries_for_numerical: dict) -> np.ndarray:
        """ Computed weights to be used for cost-sensitive learning.
    
        Description: Computes weights for each instance to be used for cost-sensitive
                    learning. The weights correspond are inversely proportional
                    to the number of occurences of the given combination.
    
        Args:
            dataset: Structured Numpy Array containing the features.
            features_to_check: List of Strings, for which features to consider
                                the sub-populations.
            counts: Dictionary of the combinations of the features_to_check
                    (cross-product) with their number of occurences.
    
        Returns: Numpy Array of weights, one for each instance.
    
        Raises:
            NA
            """

        n_samples = self.dataset.shape[0]
        cumulative = sum([sum(item.values()) for item in counts.values()])
        comb_weights = {}
        for key, val in counts.items():
            comb_weights[key] = cumulative / sum(val.values())
        indices = list(range(n_samples))
        weights = np.array(np.zeros(len(indices))).reshape(-1, 1)
        for comb, vals in comb_weights.items():
            mask = self._get_mask(self.dataset, self.features_to_check, comb, boundaries_for_numerical)
            weights[mask] = vals
        min_weight = np.min(weights)
        weights /= min_weight
        return weights
    
    def _get_counts(self, cross_product: list, boundaries_for_numerical: dict) -> dict:
        """
        Applies the mask on the target_field of the dataset to get the counts.
        """
        counts_dict = {}
        for combination in cross_product:
            mask = self._get_mask(self.dataset, self.features_to_check, combination, boundaries_for_numerical)
            unique, counts = np.unique(self.targets[mask], return_counts = True)
            hist = dict(zip(unique, counts))
            if len(hist) != 0:
                counts_dict[combination] =  hist
        return counts_dict
    
    def _get_cross_product(self, boundaries_for_numerical: dict = None) -> list:
        """
        Cross-product of features.
        """
        if not boundaries_for_numerical:
            boundaries_for_numerical = {}
            
        features_dict = {}
        for feature in self.features_to_check:
            if feature in boundaries_for_numerical.keys():
               try:
                   features_dict[feature] = self._get_bins(boundaries_for_numerical[feature])
               except:
                   raise ValueError("No bins provided for numerical field")
            else:
                if self.structured_bool:
                    features_dict[feature] = set(self.dataset[feature])
                else:
                    features_dict[feature] = set(self.dataset[:, feature])
        cross_product = list(itertools.product(*features_dict.values()))
        return cross_product
    
    def _get_mask(self, dataset: np.ndarray, features_to_check: list, 
                  combination: tuple, boundaries_for_numerical: dict = None) -> np.ndarray:
        """ Gets a filtering mask for the combination of features provided.
    
        Description: Will return a filtering a mask for the dataset based
                    on the features and specific values provided.
    
        Args:
            dataset: Structured Numpy Array containing the features.
            features_to_check: List of Strings, for which features to consider
                                the sub-populations.
            combination: Tuple of feature values to be used to filted the dataset.
    
        Returns: Numpy Array, mask corresponding to the locations that match the
                combination of feature values provided.
    
        Raises:
            NA
            """
        if not boundaries_for_numerical:
            boundaries_for_numerical = {}
        n_samples = dataset.shape[0]
        mask = np.array(np.zeros(n_samples), dtype=bool)
        list_of_sets = []
        for idx, feature in enumerate(features_to_check):
            if self.structured_bool:
                field = dataset[feature]
            else:
                field = dataset[:, feature]
            if feature in boundaries_for_numerical.keys():
                pos = np.intersect1d(np.where(field >= combination[idx][0])[0],
                                     np.where(field < combination[idx][1])[0])
            else:
                pos = np.where(field == combination[idx])[0]
            list_of_sets.append(set(pos))
        intersection_indices = list(list_of_sets[0].intersection(*list_of_sets))
        mask[intersection_indices] = True
        return mask
    
    def _get_bins(self, boundaries: list) -> list:
        """
    	Produces bins, given a set of boundaries.
        Example: boundaries = [20, 40, 60]
                bins = [(20, 40), (40, 60)]
        """
        bins = []
        n_boundaries = len(boundaries)
        for i in range(n_boundaries-1):
            bins.append((boundaries[i], boundaries[i+1]))
        return bins
    
    def check_systematic_error(self, 
                               predictions: np.ndarray,
                               requested_checks: list = None,
                               features_to_check: list = None, 
                               boundaries_for_numerical: dict = None) -> dict:
        """ Checks for systematic error in the dataset.
    
        Description: Will check if the different sub-populations defined by the
                    features_to_check have similar behaviour under the model.
    
        Args:
            dataset: Structured Numpy Array containing the features.
            treatments: Dictionary of Lists. Keys are: 'Protected', 'Feature',
                        'Target', 'ToIgnore'.
            features_to_check: List of Strings, for which features to consider
                                the sub-populations.
            checks: Dictionary of Functions, to be applied on the confusion-matrix.
                    Default to None. If None, returns the confusion matrix.
    
        Returns: Dictionary of confusion matrices for each sub-population, if checks==None,
                else, Dictionary of Dictionaries, one for each sub-population.
    
        Raises:
            NA
            """
        if not boundaries_for_numerical:
            boundaries_for_numerical = {}
        self.predictions = predictions
        multiclass = False
        classes_list = list(set(self.targets))
        if len(classes_list) > 2:
            multiclass = True
            
        if not requested_checks:
            if multiclass:
                requested_checks = self.checks_multiclass.keys()
            else:
                requested_checks = self.checks.keys()
    
        if len(features_to_check) == 0:
            if self.features_to_check is None:
                raise ValueError('no features to check provided')
        else:
            self.features_to_check = features_to_check
        cross_product = self._get_cross_product(boundaries_for_numerical)
        summary = {}
        for combination in cross_product:
            filtered_predictions, filtered_targets = \
                self._apply_combination_filter(combination, boundaries_for_numerical)
            conf_mat = self._get_confusion_matrix(filtered_targets, 
                                                  filtered_predictions, 
                                                  classes_list)
            if not requested_checks:
                summary[combination] = conf_mat
            else:
                summary[combination] = {}
                if multiclass:
                    for idx, target_class in enumerate(classes_list):
                        summary[combination][target_class] = {}
                        for item in requested_checks:
                            summary[combination][target_class][item] = \
                                self.checks_multiclass[item](conf_mat, idx)
                else:
                    for item in requested_checks:
                        summary[combination][item] = self.checks[item](conf_mat)
        return summary
    
    def _apply_combination_filter(self, 
                                  combination: tuple, 
                                  boundaries_for_numerical: dict) -> (np.ndarray, np.ndarray):
        mask = self._get_mask(self.dataset, self.features_to_check, combination, boundaries_for_numerical)
        filtered_predictions = self.predictions[mask]
        filtered_targets = self.targets[mask]
        return filtered_predictions, filtered_targets

    def _get_confusion_matrix(self, 
                              target: np.ndarray, 
                              prediction: np.ndarray, 
                              labels: list) -> np.matrix:
        """ Confusion matrix.
    
        Description: Confusion matrix.
    
        Args:
            targets: Numpy Array containing the (true) targets.
            predictions: Numpy Array containing the predictions.
            labels: List of the labels. Important for the order of the
                    confusion matrix.
    
        Returns: Confusion matrix.
    
        Raises:
            NA
            """
        n = len(labels)
        cm = np.matrix(np.zeros(n**2), dtype=int).reshape(n, n)
        for idx, label in enumerate(labels):
            for idx2, label2 in enumerate(labels):
                for idx3, val in enumerate(target):
                    if (val == label and prediction[idx3] == label2):
                        cm[idx, idx2] += 1
        return cm #, normalize(cm, axis=1, norm='l1')

    def perform_checks_on_split(self, 
                                requested_checks: list = None,
                                get_summary: bool = False,
                                conditioned_field: str = None, 
                                condition=None):
        """ Performs a series of checks on the desired splits of the dataset
    
        Description: A function that will split the dataset according to the
                    protected field, and then perform a series of checks.
    
        Args:
            X: Structured Numpy Array containing the features.
            targets: Numpy Array containing the (true) targets.
            predictions: Numpy Array containing the predictions.
            protected: String for the name of the protected feature.
            checks: Dictionary with keys the names of the checks, and values
                    functions to be applied on the confusion matrix.
            conditioned_field: String for the name of the feature to
                            condition on. Default = None --> we consider
                            the whole dataset
            condition: String for the name of the value to
                            condition on. Default = None --> we consider
                            the whole dataset
    
        Returns: The checks applied to the confusion matrices of the sub-populations.
    
        Raises:
            NA
            """
        
        multiclass = False
        classes_list = list(set(self.targets))
        if len(classes_list) > 2:
            multiclass = True
            
        if not requested_checks:
            if multiclass:
                requested_checks = self.checks_multiclass.keys()
            else:
                requested_checks = self.checks.keys()
        
        dataset = self.dataset.copy(order='K')
        targets = self.targets.copy(order='K')
        predictions = self.predictions.copy(order='K')
        
        if conditioned_field is not None:
            dataset, targets, predictions = \
                    self._filter_dataset(conditioned_field, condition)
        
        split_datasets = self._split_dataset(self.protected_field, [0, 1])
# =============================================================================
#         print('\n')
#         print(split_datasets[0])
#         print('\n')
#         print(split_datasets[1])
# =============================================================================
        aggregated_checks = dict()
        for item in split_datasets:
            field_val = item[0]
            X = item[1][0]; targets = item[1][1]; predictions = item[1][2]
            conf_mat = self._get_confusion_matrix(targets, predictions, classes_list)
# =============================================================================
#             print('\n')
#             print(targets)
#             print(predictions)
#             print(classes_list)
#             print(conf_mat)
# =============================================================================
            checks_dict = {}
  
            if multiclass:
                for idx, target_class in enumerate(classes_list):
                    checks_dict[target_class] = {}
                    for item in requested_checks:
                        checks_dict[target_class][item] = \
                            self.checks_multiclass[item](conf_mat, idx)
            else:
                for ch in requested_checks:
                    checks_dict[ch] = self.checks[ch](conf_mat)
            aggregated_checks[field_val] = checks_dict
        self.aggregated_checks = aggregated_checks
        if (get_summary and not multiclass):
            summary = self._get_summary()
            return self.aggregated_checks, summary
        else:
            return self.aggregated_checks

    def _filter_dataset(self, feature: str, feature_value) -> (np.ndarray, np.ndarray, np.ndarray):
        """ Filters the data according to the feature provided.
    
        Description: Will filter the data.
    
        Args:
            X: Structured Numpy Array containing the features.
            targets: Numpy Array containing the (true) targets.
            predictions: Numpy Array containing the predictions.
            feature: String for the name of the protected feature to filter on.
            feature_value: The value of the protected feature to filter on.
    
        Returns: Filtered data, according to the protected feature.
    
        Raises:
            NA
            """
        n = self.dataset.shape[0]
        mask = np.zeros(n, dtype=bool)
        if self.structured_bool:
            pos = np.where(self.dataset[feature] == feature_value)[0]
        else:
            pos = np.where(self.dataset[:, feature] == feature_value)[0]
        mask[pos] = True
        filtered_dataset = self.dataset[mask]
        filtered_targets = self.targets[mask]
        filtered_predictions = self.predictions[mask]
        return filtered_dataset, filtered_targets, filtered_predictions

    def _remove_field(self, dataset: np.ndarray, field: str) -> np.ndarray:
        field_names = list(dataset.dtype.names)
        if field in field_names:
            field_names.remove(field)
        return dataset[field_names]
    
    def _split_dataset(self, feature: str, labels: list) -> list:
        """ Splits the data according to the protected feature provided.
    
        Description: Will split the data.
    
        Args:
            X: Structured Numpy Array containing the features.
            targets: Numpy Array containing the (true) targets.
            predictions: Numpy Array containing the predictions.
            feature: String for the name of the protected feature to split on.
    
        Returns: List of spit data, according to the protected feature.
    
        Raises:
            NA
            """
        splits = []
        for label in labels:
            splits.append((
                            label,
                           self._filter_dataset(feature, label)
                           ))
        return splits
    
    def _get_summary(self) -> dict:
        """ Compares the checks on the subpopulations.
    
        Description: A function that will compare the checks on the subpopulations,
                    and produce new measures.
    
        Args:
            aggregated_checks: Dictionary of dictionaries, one for each subpopulation.
                            Each sub-dictionary should have as keys the names of the
                            checks.
    
        Returns: Comparison of the checks.
    
        Raises:
            NA
            """

        labels = list(self.aggregated_checks.keys())
        positives_checks = self.aggregated_checks[labels[0]]
        negatives_checks = self.aggregated_checks[labels[1]]
        summary = {}
        for key in positives_checks.keys():
            if positives_checks[key] == negatives_checks[key]:
                summary[key] = 'Equal'
            else:
                summary[key] = 'Not Equal'
        try:           
            n_tpr = negatives_checks['true_positive_rate']
            p_tpr = positives_checks['true_positive_rate']
            if n_tpr == 0:
                summary['true_positive_rate_ratio'] = p_tpr
            else:
                summary['true_positive_rate_ratio'] = p_tpr / n_tpr
        except:
            pass
    
        return summary
    
    def counterfactual_fairness(self, 
                                test_model: object, 
                                protected: str, 
                                xtest_: np.ndarray, 
                                unique_targets: list) -> np.matrix:
        """ Checks counterfactual fairness of the model.
    
        Description: Will flip the protected attribute and generate new predictions,
                    to check the model's dependence on the protected feature.
    
        Args:
            model: The model trained. Should have .fit(X, y) and .predict(X) (that
                    returns the actual predictions, i.e. {0, 1})
                    functionality.
            X: Structured Numpy Array containing the features.
            targets: Numpy Array containing the (true) targets.
            protected: String for the name of the protected feature.
    
        Returns: Confusion matrix between the predictions before and after the
                flipping the protected feature
    
        Raises:
            NA
            """

        xtest = xtest_.copy(order='K')
        original_predictions = test_model.predict(np.array(xtest.tolist()))
        xtest[protected] = [int(not item) for item in xtest[protected]]
        modified_X = np.array(xtest.tolist())
        counterfactual_predicitons = test_model.predict(modified_X)
        conf_mat = self._get_confusion_matrix(original_predictions, 
                                              counterfactual_predicitons,
                                              unique_targets)
        return conf_mat

    def individual_fairness(self, 
                            model: object, 
                            X: np.ndarray, 
                            X_distance_func: FunctionType = euc_dist, 
                            predictions_distance_func: FunctionType = euc_dist) -> bool:
        """ Checks individual fairness -- 'Fairness through awareness'.
    
        Description: Will check whether similar instances get similar predictions.
    
        Args:
            X: Structured Numpy Array containing the features.
            predictions_proba: Numpy Array containing the prediction probabilities,
                                for both classes.
            X_distance_func: Function to be used to compute distance between
                            instances.
            predictions_distance_func: Function to be used to compute distance between
                            predictions.
    
        Returns: Will check whether 'fairness through awareness holds'
                d_X(x_i, x_j) <= d_f(f(x_i), f(x_j))
    
        Raises:
            NA
            """
        n = X.shape[0]
        X_distance_mat = get_distance_mat(X, X_distance_func)
        predictions_proba = model.predict_proba(X)
        y_distance_mat = get_distance_mat(predictions_proba, predictions_distance_func)
        for i in range(n):
            for j in range(i):
                if y_distance_mat[i, j] > X_distance_mat[i, j]:
                    return False
        return True