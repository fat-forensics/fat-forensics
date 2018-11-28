# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 12:45:58 2018

@author: rp13102
"""
import numpy as np
import itertools
from copy import deepcopy
from collections import Counter

def get_confusion_matrix(target, prediction, labels):
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

def filter_dataset(X, targets, predictions, feature, feature_value):
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
    n = X.shape[0]
    mask = np.zeros(n, dtype=bool)
    pos = np.where(X[feature] == feature_value)[0]
    mask[pos] = True
    filtered_dataset = X[mask]
    filtered_targets = targets[mask]
    filtered_predictions = predictions[mask]
    return filtered_dataset, filtered_targets, filtered_predictions

def split_dataset(X, targets, predictions, feature, labels):
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
                       filter_dataset(X, targets, predictions, feature, label)
                       ))
    return splits

def perform_checks_on_split(X, targets, predictions, protected, checks, conditioned_field=None, condition=None):
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
    if not conditioned_field:
        X, targets, predictions = filter_dataset(X, targets, predictions, conditioned_field, condition)
    split_datasets = split_dataset(X, targets, predictions, protected, [0, 1])
    aggregated_checks = dict()
    for item in split_datasets:
        field_val = item[0]
        X = item[1][0]; targets = item[1][1]; predictions = item[1][2]
        
        conf_matrix = get_confusion_matrix(targets, predictions, [1, 0])
        checks_dict = {}
        for key, func in checks.items():
            checks_dict[key] = func(conf_matrix)
        aggregated_checks[field_val] = checks_dict
        
    return aggregated_checks

def get_summary(aggregated_checks):
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
    labels = list(aggregated_checks.keys())
    positives_checks = aggregated_checks[labels[0]]
    negatives_checks = aggregated_checks[labels[1]]
    summary = {}
    for key in positives_checks.keys():
        if positives_checks[key] == negatives_checks[key]:
            summary[key] = 'Equal'
        else:
            summary[key] = 'Not Equal'
    n_tpr = negatives_checks['true_positive_rate']
    p_tpr = positives_checks['true_positive_rate']
    if n_tpr == 0:
        summary['true_positive_rate_ratio'] = p_tpr
    else:
        summary['true_positive_rate_ratio'] = p_tpr / n_tpr

    return summary  
     
def counterfactual_fairness(model, X, targets, protected):
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
    newx = np.array(X.tolist())
    model.fit(newx, targets)
    original_predictions = model.predict(newx)
    newx = X.copy()
    newx[protected] = [int(not item) for item in newx[protected]]
    modified_X = np.array(newx.tolist())
    model.fit(modified_X, targets)
    counterfactual_predicitons = model.predict(modified_X)
    conf_mat = get_confusion_matrix(original_predictions, counterfactual_predicitons, [1, 0])
    return conf_mat

def get_distance_mat(mat, func):
    n = mat.shape[0]
    D = np.matrix(np.zeros(n**2).reshape(n, n))
    for i in range(n):
        v0 = np.squeeze(mat[i, :]).reshape(-1, 1)
        for j in range(i):
            v1 = np.squeeze(mat[j, :]).reshape(-1, 1)
            D[i, j] = func(v0, v1)
            D[j, i] = D[i, j]
    return D

def individual_fairness(X, predictions_proba, X_distance_func, predictions_distance_func):
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
    y_distance_mat = get_distance_mat(predictions_proba, predictions_distance_func)
    for i in range(n):
        for j in range(i):
            if y_distance_mat[i, j] > X_distance_mat[i, j]:
                return False
    return True

def remove_field(dataset, field):
    """ Removes a field from a Structured Numpy Array.
    
    Description: Removes a field from a Structured Numpy Array.
    
    Args: 
        dataset: Structured Numpy Array containing the features.
        field: String contatining the name of the field to be removed.
        
    Returns: Structured Numpy Array without the specified field.
    
    Raises:
        ValueError: If field not in dataset.dtypes 
        """
    field_names = list(dataset.dtype.names)
    if field in field_names:
        field_names.remove(field)
    else:
        raise ValueError('Field not found')
    return dataset[field_names]

def apply_distance_funcs(v0, v1, treatments, distance_funcs):
    """
    Computes the distance between two instances, based on the distance functions
    provided by the user.
    """
    dist = 0
    for feature in treatments['Feature']:
        dist += distance_funcs[feature](v0[feature], v1[feature])
    return dist

def check_systemic_bias(dataset, treatments, distance_funcs, threshold = 0):
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
    n_samples = dataset.shape[0]
    protected = dataset[treatments['Protected'][0]]
    target = dataset[treatments['Target'][0]]
    distance_list = []
    for i in range(n_samples):
        v0 = dataset[i]
        protected0 = protected[i]
        target0 = target[i]
        for j in range(i):
            v1 = dataset[j]
            protected1 = protected[j]
            target1 = target[j]
            dist = apply_distance_funcs(v0, v1, treatments, distance_funcs)
            
            same_protected = protected0 == protected1
            same_target = target0 == target1
            
            if (dist <= threshold and
                same_protected == False and
                same_target == True):
                distance_list.append((dist, same_protected, same_target, (i,j))) 
    return distance_list

def get_cross_product(dataset, features_to_check):
    """
    Cross-product of features.
    """
    features_dict = {}
    for feature in features_to_check:
        if dataset.dtype.fields[feature] == 'int32':
            raise TypeError('Can only handle categorical features')
        features_dict[feature] = set(dataset[feature])
    cross_product = list(itertools.product(*features_dict.values()))   
    return cross_product

def get_mask(dataset, features_to_check, combination):
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
    n_samples = dataset.shape[0]
    mask = np.array(np.zeros(n_samples), dtype=bool)
    list_of_sets = []
    for idx, feature in enumerate(features_to_check):
        pos = np.where(dataset[feature] == combination[idx])[0]
        list_of_sets.append(set(pos))
    intersection_indices = list(list_of_sets[0].intersection(*list_of_sets))
    mask[intersection_indices] = True
    return mask

def get_counts(dataset, target_field, features_to_check, cross_product): 
    """
    Applies the mask on the target_field of the dataset to get the counts.
    """    
    counts = {}
    for combination in cross_product:
        mask = get_mask(dataset, features_to_check, combination)
        hist = Counter(dataset[mask][target_field])
        if len(hist) != 0:
            counts[combination] =  hist  
    return counts

def get_weights_costsensitivelearning(dataset, features_to_check, counts):
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
    n_samples = dataset.shape[0]
    cumulative = sum([sum(item.values()) for item in counts.values()])
    comb_weights = {}
    for key, val in counts.items():
        comb_weights[key] = cumulative / sum(val.values())
    indices = list(range(n_samples))
    weights = np.array(np.zeros(len(indices))).reshape(-1, 1)
    for i in range(n_samples):
        v0  = tuple(dataset[i][features_to_check])
        weights[i] = comb_weights[v0]
    
    min_weight = np.min(weights)
    weights /= min_weight
    return weights

def check_sampling_bias(dataset, treatments, features_to_check, return_weights=False):
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
    counts = {}
    cross_product = get_cross_product(dataset, features_to_check)
    counts = get_counts(dataset, treatments['Target'][0], features_to_check, cross_product)
         
    if not return_weights:
        return counts
    else:
        weights = get_weights_costsensitivelearning(dataset, features_to_check, counts)
        return counts, weights

def apply_combination_filter(dataset, prediction_field, target_field, features_to_check, combination):
    predictions = dataset[prediction_field]
    targets = dataset[target_field]
    mask = get_mask(dataset, features_to_check, combination)
    filtered_predictions = predictions[mask]
    filtered_targets = targets[mask]
    return filtered_predictions, filtered_targets

def check_systematic_error(dataset, treatments, features_to_check, checks=None):
    """ Checks for systematic error in the dataset.
    
    Description: Will check if the different sub-populations defined by the
                features_to_check have similar behaviour under the model.
                *Only for Categorical data*.
    
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
    cross_product = get_cross_product(dataset, features_to_check)
    summary = {}
    for combination in cross_product:
        filtered_predictions, filtered_targets = \
            apply_combination_filter(dataset, 'Prediction', treatments['Target'][0], 
                                     features_to_check, combination)
        conf_mat = get_confusion_matrix(filtered_targets, filtered_predictions, [0, 1])
        if not checks:
            summary[combination] = conf_mat
        else:
            summary[combination] = {}
            for key, func in checks.items():
                summary[combination][key] = func(conf_mat)
    return summary
        