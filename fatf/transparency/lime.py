"""
A wrapper for the LIME package to work with tabular data 
"""

# Author: Alex Hepburn <ah13558@bristol.ac.uk>
# License: BSD 3 Clause

from typing import Dict, List, Tuple
import sys
import logging

import numpy as np 

import fatf.utils.array.validation as fuav
import fatf.utils.models.validation as fumv
import fatf.utils.array.tools as fuat 

from fatf.exceptions import IncompatibleModelError, IncorrectShapeError
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

_NUMPY_NUMERICAL_KINDS_LIST = ['b', 'u', 'i', 'f', 'c']

#TODO: purely numerical for stuctured arrrays (log warning cast into most common type)
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
                distance_metric: str = 'euclidean',
                random_state: int = None) -> None:
        self._check_input(X, model, categorical_indices, class_names, feature_names)
        self._X, self._categorical = self._process_X_indices(X, categorical_indices)
        self._common_type = self._X.dtype
        self.model = model
        self._num_classes = model.predict_proba(self._X[0:1, :]).shape[1]
        self.num_features = num_features
        self.num_samples = num_samples
        self.distance_metric = distance_metric
        self.class_names = class_names
        self.feature_names = feature_names
        self._model = model
        self.random_state = random_state
        if not self.num_features:
            self.num_features = self._X.shape[1]
        self.tabular_explainer = LimeTabularExplainer(
            self._X, 
            feature_names=self.feature_names, 
            categorical_features=self._categorical, 
            class_names=self.class_names, 
            discretize_continuous=True,
            sample_around_instance=True,
            random_state = random_state)

    def __str__(self) -> str:
        return str(self.as_dict)

    def _check_input(self, X: np.array, 
                           model: object, 
                           categorical_indices: np.array,
                           class_names: List[str],
                           feature_names: List[str],
                           check_x: bool = True,
                           check_model: bool = True,
                           check_categorical_indices: bool = True,
                           check_feature_names: bool = True,
                           check_class_names: bool = True):
        """Check input data and model is valid for LIME algorithm

        Parameters
        ----------
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
        check_categorical_indices : boolean
            If true then categorical_indices will be checked. Defaults to True
        check_feature_names : boolean
            If true then feature names will be checked. Defaults to True
        check_class_names : boolean
            If true then class names will be checked. Defaults to True

        Raises
        ------
        NotImplementedError:
            The input X is a numpy structured array and not ndarray with same data type
        IncorrectShapeError:
            The input X is not a 2-D array
        IncompatibleModelError:
            The model parameter does not contain a peridct_proba() method that is needed
            for LIME algorithm
        ValueError:
            Index given in categorical_indices parameter is out of range for features
            given in X
        ValueError:
            Number of feature names given does not equal to number of features in X
        ValueError:
            Number of class names given does not equal number of classes that model
            has been trained with
        """
        if check_x:
            numerical_ind, categorical_ind = fuat.indices_by_type(X)
            if not np.array_equal(categorical_ind, np.array([])):
                raise NotImplementedError(
                    'LIME not implemented for non-numerical arrays.')
            if not fuav.is_2d_array(X):
                raise IncorrectShapeError('X must be 2-D array.')
        if check_model:
            if not fumv.check_model_functionality(model, True, 
                                                  suppress_warning=True):
                raise IncompatibleModelError(
                    'LIME requires model object to have method predict_proba() in order to '
                    'work')
        if check_categorical_indices:
            if not np.array_equal(categorical_indices, np.array([])):
                if not fuat.are_indices_valid(X, categorical_indices):
                    raise ValueError(
                        'Indices given in categorical_indices not valid for input array X')
        # need numerical version of X for use in model.predict_proba
        X, _ = self._process_X_indices(X, categorical_indices)
        if check_feature_names:
            if feature_names is not None:
                if len(feature_names) != numerical_ind.shape[0]:
                    raise ValueError(
                        'Number of feature names given does not correspond to input array')
        if check_class_names:
            if class_names is not None:
                if len(class_names) != model.predict_proba(X[0:1, :]).shape[1]:
                    raise ValueError(
                        'Number of class names given does not correspond to model')

    def _process_X_indices(self, X:np.array, categorical_indices: np.array) -> (np.ndarray, np.array):
        """Function that processes input array X into ndarray if user gives
        a structured array and categorical_indices into an array of integers

        Args
        ----
        X: np.array
            The data to be used

        categorical_indices: np.array
            Indices specified by user to be categorical variables

        Returns
        ----
        X_ndarray: np.ndarray
            X parsed as ndarray to all one numerical value
        categorical_indicies_numerical: np.ndarray
            Indices specified by user to be categorical converted into numerical
            indices
        """

        if len(X.dtype) != 0:
            common_type = bool
            type_index = 0
            for name in X.dtype.names:
                i = _NUMPY_NUMERICAL_KINDS_LIST.index(X.dtype[name].kind)
                if i > type_index:
                    type_index = i
                    common_type = X.dtype[name]
            new_dtypes = [(name, common_type) for name in X.dtype.names]
            X_ndarray = X.copy().astype(new_dtypes).view(common_type).reshape(X.shape + (-1,))
            logging.warning('Structured array was converted to ndarray for use in LIME algorithm. '
                            'The values were convered to type %s.' %common_type)
            categorical_indices_numerical = np.array([X.dtype.names.index(a) 
                                                      for a in list(categorical_indices)])
        else:
            X_ndarray = X
            categorical_indices_numerical = categorical_indices
        return X_ndarray, categorical_indices_numerical

    def explain_instance(self, instance: np.array, labels: np.array = np.array([])) -> Dict[str, Tuple[str, float]]:
        """Uses LIME tabular_explainer to explain instance

        Args
        ----
        instance: np.array
            Instance to explain
        labels: np.array 
            of int labels to explain decisions for. If empty, then all labels
            are used. Defaults to np.array([])

        Raises
        ----
        ValueError:
            Entry of labels not found in dataset (i.e. two classes but label 2 is given)
        
        Returns
        ----
        explained: dict
            Dictionary where key is class_name corresponding to labels and values are
            (string, float) tuples that are the feature and importance.
        """
        if len(instance.dtype) != 0:
            new_dtypes = [(name, self._common_type) for name in instance.dtype.names]
            x = instance.copy().astype(new_dtypes).view(self._common_type)
        else:
            x = instance
        if not np.array_equal(labels, np.array([])):
            if np.any(labels > self._num_classes):
                raise ValueError('Class %d not in dataset specified'%l)
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
        raise NotImplementedError(
            'show_notebook function not yet implemented')
        #self._exp.show_in_notebook(predict_proba=True)
