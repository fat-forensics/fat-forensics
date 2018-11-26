"""
Functions for calculating feature importance and 
Individual Conditional Expectation (ICE)
Author: Alex Hepburn <ah13558@bristol.ac.uk>
License: new BSD
"""

from typing import List, Dict

import numpy as np
import scipy

from fatf.tests.predictor import KNN


def individual_conditional_expectation(
        X_train: np.ndarray, 
        model: object, 
        feature: int,
        steps: int = 100
) -> np.ndarray:
    '''Calculate Individual Conditional Expectation for all classs for feature

    Args:
        X_train: np.ndarray containing training data of shape [n_samples, n_features]
        model: object which is fitted model containing functions fit(X, Y), predict(X)
            and predict_proba(X)
        feature: int corresponding to column in X_train for feature to compute ICE
        steps: int how many steps to sample with between feature min and max. 
            Default: 100

    Return:
        np.ndarray of shape [n_samples, steps, n_classes]
        np.array of shape [steps] specifying the interpolation done

    Example:
        >>>
    '''
    # Find n_classes without using attribute
    n_classes = model.predict_proba(X[0:1, :]).shape[1]
    ret = np.ndarray((X_train.shape[0], steps, n_classes))
    feat = X_train[:, feature]
    values = np.linspace(min(feat), max(feat), num=100)
    for i in range(0, X_train.shape[0]):
        X_pred = np.tile(X_train[i, :], (steps, 1))
        X_pred[:, feature] = values
        probas = model.predict_proba(X_pred)
        # Could possible have so it only returns for certain classes?
        ret[i, :, :] = probas
    return ret, values

def partial_depedence(
        X_train: np.ndarray,
        model: object,
        feature: int,
        category: List[int] = None,
        steps: int = 100
) -> np.ndarray:
    '''Calculate partial dependence for all classes for feature. Takes the mean of
        the output of individual_conditional_expectation function over all training
        data points.

    Args:
        X_train: np.ndarray containing training data of shape [n_samples, n_features]
        model: object which is fitted model containing functions fit(X, Y), predict(X)
            and predict_proba(X)
        feature: int corresponding to column in X_train for feature to compute ICE
        category: List[int] which classes to compute partial_dependence for. If None
            then computed for all classes. Default: None
        steps: int how many steps to sample with between feature min and max. 
            Default: 100

    Return:
        np.ndarray of shape [steps, n_classes]
        np.array of shape [steps] specifying the interpolation done

    Example:
        >>>
    '''
    if not category:
        predict = model.predict_proba(X_train[0:1, :])
        category = list(range(0, predict.shape[0]+1))
    ice, values = individual_conditional_expectation(X_train, model, 
                                                     feature, steps=steps)
    pd = np.mean(ice[:, :, category], axis=0)
    return pd, values

def plot_ICE(
        ice: np.ndarray,
        feature_name: str,
        values: np.array,
        category: int,
        category_name: str = None
) -> None:
# return a figure without import matplotlib.pyplot outside function
    '''Plot individual conditional expectations for class

    Args:
        ice: np.ndarray of shape [n_samples, steps, n_classes] containing probabilities
            outputted from ICE
        feature_name: str specificy which feature was used to calculate the ICE
        values: np.array containing values of feature tested for ICE
        category: int which class to plot probabilities for
        category_name: str name of class chosen. If None then the category_name will be 
            the category integer converted to str. Default: None
    '''
    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
    except ImportError as e:
        raise ImportError('plot_ICE function requires matplotlib package. This can' 
                          'be installed with "pip install matplotlib"')
    if not category_name:
        category_name = str(category)
    ax = plt.subplot(111)
    lines = np.zeros((ice.shape[0], ice.shape[1], 2))
    lines[:, :, 1] = ice[:, :, category]
    lines[:, :, 0] = np.tile(values, (ice.shape[0], 1))
    collect = LineCollection(lines, label='Individual Points', color='black')
    ax.add_collection(collect)
    mean = np.mean(ice[:, :, category], axis=0)
    ax.plot(values, mean, color='yellow', linewidth=10, alpha=0.6, label='Mean')
    ax.legend()
    ax.set_ylabel('Probability of belonging to class %s'%category_name)
    ax.set_xlabel(feature_name)
    ax.set_title('Individual Conditional Expecatation')
    plt.show()

def feature_importance(
    X_train: np.ndarray,
    model: object,
    category: List[int] = None,
    n_samples: int = 5000
) -> Dict[int:int]:
    '''Feature importance

    Args:
        X_train: np.ndarray containing training data of shape [n_samples, n_features]
        model: object which is fitted model containing functions fit(X, Y), predict(X)
            and predict_proba(X)
        category: List[int] list of classes to compute for. If None then computed for all
            classes. Default: None
        n_samples: int specifying how many samples to generate when computing feature
            importance

    Return:
        np.ndarray of shape [steps, n_classes]

    Example:
        >>>
    '''
    original_predictions = model.predict_proba(X_train)
    if not category:
        category = list(range(0, original_predictions.shape[1]))
    for i in range(0, X_train.shape[1]):
        X_temp = X_train
        X_temp[:, i] = _generate_samples()
        new_predictions = model.predict(X_temp)
        imp = _compute_importance(original_predictions, new_predictions)
    return np.zeros((1, 1))

def _generate_samples():
    return 0
def _compute_importance():
    return 0

if __name__ == '__main__':
    knn = KNN(k=2)
    X = np.array([
        [1.2, 2, 3],
        [2.3, 3, 4],
        [10.3, 2, 4],
        [4.0, 3, 5]
    ], dtype=np.float32)
    Y = np.array([0, 0, 1, 1])
    #X = np.random.rand(100, 10)
    #Y = np.random.randint(0,2,100)
    knn.fit(X, Y)
    ret, values = individual_conditional_expectation(X, knn, 0)
    pd = partial_depedence(X, knn, 0, 0)
    #plot_ICE(ret, 'feature 0', values, 0, 'Good')
    imp = feature_importance(X, knn)

