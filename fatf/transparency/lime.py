"""
A wrapper for the LIME package to work with tabular data 
"""

# Author: Alex Hepburn <ah13558@bristol.ac.uk>
# License: BSD 3 Clause

from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np 

import fatf.utils.array.validation as fuav
import fatf.utils.models.validation as fumv
import fatf.utils.array.tools as fuat 
from fatf.exceptions import IncompatibleModelError, IncorrectShapeError

try:
    import lime
    from lime.lime_tabular import LimeTabularExplainer
except ImportError as e:
    _warning_msg = ( # pylint: disable=invalid-name
        'Lime package is not installed on your system. You must install it in '
        'order to use the fatf.transparency.Lime class. One possibility is to '
        'install Lime alongside this package with: pip install fatf[vis].')
    warnings.warn(_warning_msg, ImportWarning)


class Lime(object):
    """
    Wrapper for LimeTabularExplainer class from package implemented in 
    https://github.com/marcotcr/lime

    Parameters
    ----------
    X : numpy.array
        The data to be used
    model : object 
        Object that contains method predict(x) that outputs predictions
        and predict_proba(x) that outputs probability vectors corresponding to
        the probability of an instance belonging to each class
    categorical_indices : numpy.array
        Indices that user would like to specify as categorical. Defaults to 
        np.array([])
    class_names : List[string]
        List of class names in the order the model is using. Defaults to None 
        so names will be '0', '1', ....
    feature_names : List[string]
        specifying feature names in the order the model is using. Defaults to 
        None so names will be 'feature0', 'feature1', ...
    num_samples : integer
        Number of samples to generate around x in LIME algorithm. 
        Defaults to 5000.
    num_features : integer
        Max number of features to use in LIME algorithm (takes top-n features)
        Defaults to None to use all features.
    distance_metric : string
        Defining distance to use in LIME algorithm, has to be valid metric for 
        use in scipy.spatial.distance.pdist. Defaults to 'euclidean' check for 
        pdist function if can't take any function

    Attributes
    ----------
    tabular_explainer: LimeTabularExplainer
        tabular explainer object whose methods will be called inside the class.        
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
        self._check_input(X, model, categorical_indices, class_names, 
                          feature_names, check_class_names=False)
        if fuav.is_structured_array(X):
            self._X = fuat.as_unstructured(X)
            self._categorical = np.array([X.dtype.names.index(y) for y in 
                                          categorical_indices])
        else:
            self._X = X
            self._categorical = categorical_indices
        self._check_input(self._X, model, self._categorical, class_names, 
                          feature_names, False, False, False, False, True)
        self._common_type = self._X.dtype
        self.model = model
        self._num_classes = model.predict_proba(self._X[[0], :]).shape[1]
        self.num_features = num_features
        self.num_samples = num_samples
        self.distance_metric = distance_metric
        self.class_names = class_names
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
            sample_around_instance=True,
            random_state = random_state)


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
        """
        Checks input data and model is valid for LIME algorithm

        Parameters
        ----------
        X : numpy.array
            The data to be used
        model : object
            Object that contains method predict(x) that outputs predictions
            and predict_proba(x) that outputs probability vectors corresponding 
            to the probability of an instance belonging to each class.
        categorical_indices : numpy.array
            Indices that user would like to specify as categorical.
        class_names : List[str]
            List of class names in the order the model is using. Defaults to 
            None so names will be '0', '1', ....
        feature_names : List[str]
            specifying feature names in the order the model is using. Defaults 
            to None so names will be 'feature0', 'feature1', ...
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
            The input X is a numpy structured array and not ndarray with same 
            data type
        IncorrectShapeError:
            The input X is not a 2-D array
        IncompatibleModelError:
            The model parameter does not contain a peridct_proba() method that 
            is needed for LIME algorithm
        ValueError:
            Index given in categorical_indices parameter is out of range for 
            features given in X or number of feature names given does not equal 
            to number of features in X or number of class names given does not 
            equal number of classes that model has been trained with
        """
        if check_x:
            if not fuav.is_2d_array(X):
                raise IncorrectShapeError('X must be 2-D array.')
            numerical_ind, categorical_ind = fuat.indices_by_type(X)
            if not np.array_equal(categorical_ind, np.array([])):
                raise NotImplementedError(
                    'LIME not implemented for non-numerical arrays.')
        if check_model:
            if not fumv.check_model_functionality(model, True, 
                                                  suppress_warning=True):
                raise IncompatibleModelError(
                    'LIME requires model object to have method '
                    'predict_proba().')
        if check_categorical_indices:
            if not np.array_equal(categorical_indices, np.array([])):
                if not fuat.are_indices_valid(X, categorical_indices):
                    raise ValueError(
                        'Indices given in categorical_indices not valid for '
                        'input array X')
        # need numerical version of X for use in model.predict_proba
        if check_feature_names:
            if feature_names is not None:
                if len(feature_names) != numerical_ind.shape[0]:
                    raise ValueError(
                        'Number of feature names given does not correspond to '
                        'input array')
        if check_class_names:
            if class_names is not None:
                if len(class_names) != model.predict_proba(X[[0], :]).shape[1]:
                    raise ValueError(
                        'Number of class names given does not correspond to '
                        'model')


    def explain_instance(self, 
                         instance: np.ndarray,
                         labels: Optional[np.ndarray] = None
                         ) -> Dict[str, Tuple[str, float]]:
        """
        Uses LIME tabular_explainer to explain instance

        Parameters
        ----------
        instance : numpy.array
            Instance to explain
        labels : numpy.array 
            Array of int labels to explain decisions for. If empty, then all 
            labels are used. Defaults to None

        Raises
        ------
        ValueError:
            Entry of labels not found in dataset (i.e. two classes but label 
            2 is given)
        
        Returns
        -------
        explained: Dictionary[string, Tuple[string, float]]
            Dictionary where key is class_name corresponding to labels and 
            values are (string, float) tuples that are the feature and 
            importance.
        """
        if fuav.is_structured_array(instance):
            x = fuat.as_unstructured(instance)[0]
        else:
            x = instance
        if labels is  None:
            labels = list(range(self._num_classes))
        else:
            if np.any(labels > self._num_classes):
                inc_labels = labels[labels>self._num_classes]
                raise ValueError('Class {} not in dataset '
                                 'specified'.format(inc_labels))
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
