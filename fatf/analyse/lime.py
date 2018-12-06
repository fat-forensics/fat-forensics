"""
A wrapper for the LIME package to work with tabular data 
"""

# Author: Alex Hepburn <ah13558@bristol.ac.uk>
# License: BSD 3 Clause

from typing import Dict, List, Tuple
import sys
import logging

import numpy as np 

from fatf.utils import validation as val
from fatf.exceptions import *

try:
    import lime
    from lime.lime_tabular import LimeTabularExplainer
except ImportError as e:
    raise ImportError(
        'Lime class requires LIME package to be installed. This can be installed by: ' 
        'pip install lime')
try:
    import matplotlib.pyplot as plt
except ImportError as e:
    plt = None
    logging.warning(
        'Matplotlib is not installed. You will not be able to use plot_lime function. To use' 
        'please install matplotlib by: pip install matplotlib')


class Lime(object):
    """Wrapper for Lime tabular explainer
    
    Class that wraps LimeTabularExplainer from package implemented 
    in https://github.com/marcotcr/lime

    Args
    ----
    X : np.array
        The data to be used
    model: object 
        Object that contains method predict(x) that outputs predictions
        and predict_proba(x) that outputs probability vectors corresponding to
        the probability of an instance belonging to each class
    categorical_indices: np.array
        Indices that user would like to specify as categorical. Defaults to np.array([])
    class_names : list of strings
        List of class names in the order the model is using. Defaults to None so names
        will be '0', '1', ....
    feature_names : list of strings specifying feature names in the order the model is using.
        Defaults to None so names will be 'feature0', 'feature1', ...
    num_samples : int
        Number of samples to generate around x in LIME algorithm. Defaults to 5000 
    num_features : int
        Max number of features to use in LIME algorithm (takes top-n features)
    distance_metric : string. Defaults to None to use all features
        Defining distance to use in LIME algorithm, has to be valid metric for use 
        in scipy.spatial.distance.pdist. Defaults to 'euclidean'
        check for pdist function if can't take any function

    Attributes
    ----------
    tabular_explainer: LimeTabularExplainer
        tabular explainer object whose methods will be called inside the class.

    Example
    ----

        
    """

    def __init__(self,
                X: np.array, 
                model: object, 
                categorical_indices: np.array = np.array([]),
                class_names: List[str] = None,
                feature_names: List[str] = None,
                num_samples: int = 5000,
                num_features: int = 0, 
                distance_metric: str = 'euclidean') -> None:
        self._check_input(X, model, categorical_indices, class_names, feature_names)
        self._X = X
        self.model = model
        self._num_classes = model.predict_proba(X[0:1, :]).shape[1]
        self.num_features = num_features
        self.num_samples = num_samples
        self.distance_metric = distance_metric
        self.class_names = class_names
        self._categorical = categorical_indices
        self.feature_names = feature_names
        self._model = model
        if not self.num_features:
            self.num_features = self._X.shape[1]
        self.tabular_explainer = LimeTabularExplainer(
            self._X, 
            feature_names=self.feature_names, 
            categorical_features=self._categorical, 
            class_names=self.class_names, 
            discretize_continuous=True,
            sample_around_instance=True)

    def __str__(self) -> str:
        return str(self.as_dict)

    def _check_input(self, X: np.array, 
                           model: object, 
                           categorical_indices: np.array,
                           class_names: List[str],
                           feature_names: List[str],
                           check_x: bool = True,
                           check_model: bool = True,
                           check_indices: bool = True,
                           check_feature_names: bool = True,
                           check_class_names: bool = True):
        """Check input data and model is valid for LIME algorithm

        Args
        ----
        X : np.array
            The data to be used
        model : object
            Object that contains method predict(x) that outputs predictions
            and predict_proba(x) that outputs probability vectors corresponding to
            the probability of an instance belonging to each class.
        categorical_indices : np.array
            Indices that user would like to specify as categorical.
        class_names : List[str]
            List of class names in the order the model is using. Defaults to None so names
            will be '0', '1', ....
        feature_names : List[str]
            specifying feature names in the order the model is using. Defaults to None so
            names will be 'feature0', 'feature1', ...
        check_x : boolean
            If true then X will be checked. Defaults to True
        check_model : boolean
            If true then model will be checked. Defaults to True
        check_indices : boolean
            If true then indices will be checked. Defaults to True
        check_feature_names : boolean
            If true then feature names will be checked. Defaults to True
        check_class_names : boolean
            If true then class names will be checked. Defaults to True

        Raises
        ----
        MissingImplementationException:
            The input X is a numpy structured array and not ndarray with same data type
        IncorrectShapeException:
            The input X is not a 2-D array
        IncompatibleModelException:
            The model parameter does not contain a peridct_proba() method that is needed
            for LIME algorithm
        CustomValueError:
            Index given in categorical_indices parameter is out of range for features
            given in X
        CustomValueError:
            Number of feature names given does not equal to number of features in X
        CustomValueError:
            Number of class names given does not equal number of classes that model
            has been trained with
        """
        if check_x:
            numerical_ind, categorical_ind = val.check_array_type(X)
            if not np.array_equal(categorical_ind, np.array([])):
                raise MissingImplementationException(
                    'LIME not implemented for structured numpy arrays. X must be np.ndarray '
                    'of purely numerical values.')
            if not val.is_2d_array(X):
                raise IncorrectShapeException('X must be 2-D array.')
        if check_model:
            if not val.check_model_functionality(model, True):
                raise IncompatibleModelException(
                    'LIME requires model object to have method predict_proba() in order to '
                    'work')
        if check_indices:
            if not np.array_equal(categorical_indices, np.array([])):
                if not val.check_indices(X, categorical_indices):
                    raise CustomValueError(
                        'Indices given in categorical_indices not valid for input array X')
        if check_feature_names:
            if feature_names is not None:
                if len(feature_names) != X.shape[1]:
                    raise CustomValueError(
                        'Number of feature names given does not correspond to input array')
        if check_class_names:
            if class_names is not None:
                if len(class_names) != model.predict_proba(X[0:1, :]).shape[1]:
                    raise CustomValueError(
                        'Number of class names given does not correspond to model')

    @property
    def categorical_indices(self) -> List[int]:
        return self._categorical

    @categorical_indices.setter
    def categorical_indices(self, categorical: np.array) -> None:
        """
        Setter for categorical variable so we can reinitialise self.tabular_explainer  
        """
        self._check_input(self._X, self.model, categorical, None, None,
                          check_x=False, check_model=False, check_feature_names=False,
                          check_class_names=False)
        self._categorical = categorical
        self.tabular_explainer = LimeTabularExplainer(
            self._X, 
            feature_names=self.feature_names, 
            categorical_features=self._categorical, 
            class_names=self.class_names, 
            discretize_continuous=True,
            sample_around_instance=True)

    def explain_instance(self, x: np.ndarray, labels: np.array = np.array([])) -> Dict[str, Tuple[str, float]]:
        """Uses LIME tabular_explainer to explain instance

        Args
        ----
        x: np.array
            Instance to explain
        labels: np.array 
            of int labels to explain decisions for. If empty, then all labels
            are used. Defaults to np.array([])

        Raises
        ----
        CustomValueError:
            Entry of labels not found in dataset (i.e. two classes but label 2 is given)
        
        Returns
        ----
        explained: dict
            Dictionary where key is class_name corresponding to labels and values are
            (string, float) tuples that are the feature and importance.
        """
        if not np.array_equal(labels, np.array([])):
            if np.any(labels > self._num_classes):
                raise CustomValueError('Class %d not in dataset specified'%l)
        else:
            labels = list(range(0, self._num_classes))
        exp = self.tabular_explainer.explain_instance(
               x, 
               self.model.predict_proba,
               labels=labels, 
               num_features=self.num_features,
               num_samples=self.num_samples, 
               distance_metric=self.distance_metric)
        explained = {}
        for l in labels:
            explained[exp.class_names[l]] = exp.as_list(label=l)
        return explained

    def show_notebook(self, explained: Dict[str, Tuple[str, float]]) -> None:
        # TODO: implement show_notebook that takes explain_instance return
        raise MissingImplementationException(
            'show_notebook function not yet implemented')
        #self._exp.show_in_notebook(predict_proba=True)

# TODO: how to do type check with plt = None when matplotlib is not installed
def plot_lime(lime_explained: Dict[str, List[tuple]]) -> plt.Figure:
    '''
    Figures to display explainer

    Args:
        lime_explained: Dictionary returned from Lime.explain_instance. 

    Returns: Figure from matplotlib where it is split into as many subplots as there are 
        possible labels in Dataset.
    '''
    if not plt:
        raise ImportError('Matplotlib is not installed. You will not be able to use plot_lime ' 
                          'function. To use please install matplotlib by: pip install ' 
                          'matplotlib')
    sharey = False
    names = None
    # check if all features are used, all subplots can share y axis
    sets = []
    labels = []
    for k, v in lime_explained.items():
        sets.append(set([l[0] for l in v]))
        labels.append(k)
    if all(s==sets[0] for s in sets):
        sharey = True
        f, axs = plt.subplots(1, len(sets), sharey=sharey, sharex=True)
    else:
        f, axs = plt.subplots(len(sets), 1, sharex=True)
    f.suptitle('Local Explanations for classes')
    if sharey: # Make sure all barplots are in the same order if sharing
        names = list(sets[0])
    # Do the plotting
    for ax, label in zip(axs, labels):
        exp = lime_explained[label]
        vals = [x[1] for x in exp]
        unordered_names = [x[0] for x in exp]
        if sharey:
            # get bars in correct order for sharing y-axis
            ind = [unordered_names.index(item) for item in names]
            vals = [vals[i] for i in ind]
        else:
            names = unordered_names
        vals.reverse()
        l = names[::-1]
        colors = ['green' if x > 0 else 'red' for x in vals]
        pos = np.arange(len(exp)) + .5
        ax.barh(pos, vals, align='center', color=colors)
        ax.set_yticks(pos)
        ax.set_yticklabels(l)
        title = str(label)
        ax.set_title(title)
    return f
