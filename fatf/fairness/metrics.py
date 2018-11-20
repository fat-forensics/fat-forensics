# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 12:45:58 2018

@author: rp13102
"""
import numpy as np


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

def split_dataset(X, targets, predictions, feature):
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
    for label in set(X[feature]):
        splits.append(
                        label, 
                       filter_dataset(X, targets, predictions, feature, label)
                       )
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
    split_datasets = split_dataset(X, targets, predictions, protected)
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
    
