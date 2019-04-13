"""
Wraps the LIME_ tabular data explainer.

.. _LIME: https://github.com/marcotcr/lime
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
#         Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

from typing import Dict, List, Optional, Tuple

import numpy as np

import fatf.utils.array.tools as fuat
import fatf.utils.array.validation as fuav
import fatf.utils.models.validation as fumv

from fatf.exceptions import IncompatibleModelError, IncorrectShapeError

try:
    import lime.lime_tabular
except ImportError:
    import warnings
    _warning_msg = (  # pylint: disable=invalid-name
        'Lime package is not installed on your system. You must install it in '
        'order to use the fatf.transparency.lime module. One possibility is '
        'to install LIME alongside this package with: pip install fatf[lime].')
    warnings.warn(_warning_msg, ImportWarning)
    del warnings


class Lime(object):
    """
    Wraps `LIME package's`__ ``lime.lime_tabular.LimeTabularExplainer`` class.

    .. warning:: contracly to LIME this atually does lcoal explanatinos TODO

    __ https://github.com/marcotcr/lime

    Parameters
    ----------
    data: numpy.ndarray
        A 2-dimensional numpy array with a dataset to be used.
    model : object
        An object that contains ``predict`` -- outputs predictions -- and
        ``predict_proba`` -- outputs probability vectors corresponding to the
        probability of an instance belonging to each class -- methods.
    categorical_indices : numpy.ndarray
        Column indices that should be treated as categorical features.
        Defaults to np.array([]).
    class_names : List[string], optional (default=None)
        List of class names in the same order as the one in the model's output.
        By default (``class_names=None``) the names will be assigned as
        consecutive numbers starting with 0.
    feature_names : List[string], optional (default=None)
        specifying feature names in the order they appear in the ``data``
        parameter. By defaults (``feature_names=None``) the names will be
        assigned names conforming to the following pattern: 'feature_0',
        'feature_1', etc.
    num_samples : integer, optional (default=5000)
        Number of samples that LIME will generate to probe selected region.
        Defaults to 5000.
    num_features : integer, optional (default=None)
        The maximum number of features (takes top-n features) that LIME can use
        for an explanation. By default (``num_features=None``) it will use all
        of the features.


    distance_metric : string, optional (default='euclidean')
        Specifies
        Defining distance to use in LIME algorithm, has to be valid metric for
        use in scipy.spatial.distance.pdist. Defaults to 'euclidean' check for
        pdist function if can't take any function.
    random_state : int
        Random state to be used in LIME algorithm when sampling data.

    Raises
    ------
    NotImplementedError:
        The inputdatais a numpy structured array and not ndarray with same
        data type.
    IncorrectShapeError:
        The input data is not a 2-D array.
    IncompatibleModelError:
        The model parameter does not contain a peridct_proba() method that
        is needed for LIME algorithm.
    ValueError:
        Index given in categorical_indices parameter is out of range for
        features given indataor number of feature names given does not
        equal to number of features indataor number of class names given
        does not equal number of classes that model has been trained with.

    Attributes
    ----------
    tabular_explainer: lime.lime_tabular.LimeTabularExplainer
        Tabular explainer object whose methods will be called inside the class.
    """

    INIT_PARAMS = []
    EXPLAIN_INSTANCE_PARAMS = []

    def __init__(self,
                 data: np.ndarray,


                 model: object,

                 local_explanation=True,


                 # Initialiser
                 categorical_indices: np.ndarray = None,

                 # This is the explainer
                 num_samples: int = 5000,
                 num_features: Optional[int] = None,
                 distance_metric: str = 'euclidean',

                 **kwargs) -> None:
        """
        Initialises a tabular LIME wrapper.
        """
        # Check data
        if not fuav.is_2d_array(data):
            raise IncorrectShapeError('data must be 2-D array.')
        if not fuav.is_numerical_array(data):
            raise ValueError('LIME does not support non-numerical arrays.')

        # Check categorical indices
        if categorical_indices is not None:
            if not fuat.are_indices_valid(data, categorical_indices):
                raise ValueError(
                    'Indices given in categorical_indices not valid for '
                    'input array data')

        # Honour native local explanation keyword
        local_explanation_keyword = 'sample_around_instance'
        if local_explanation_keyword not in kwargs:
            kwargs[local_explanation_keyword] = local_explanation

        if fuav.is_structured_array(data):
            if categorical_indices is not None:
                categorical_indices = np.array([data.dtype.names.index(y) for y in
                                              categorical_indices])
            data = fuat.as_unstructured(data)

        self.tabular_explainer = lime.lime_tabular.LimeTabularExplainer(
            data, categorical_features=categorical_indices, **kwargs)





        # Check model
        if not fumv.check_model_functionality(model, True,
                                              suppress_warning=True):
            raise IncompatibleModelError(
                'LIME requires model object to have method '
                'predict_proba().')

        self._num_classes = model.predict_proba(data[[0], :]).shape[1]
        # TODO: if feature_names are none assign feature_0, feature_1, etc
        # TODO: num_features > 0
        self.num_features = num_features
        self.num_samples = num_samples
        self.distance_metric = distance_metric
        self.model = model
        if self.num_features is None:
            self.num_features = data.shape[1]

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
            labels are used. Defaults to None.

        Raises
        ------
        ValueError:
            Entry of labels not found in dataset (i.e. two classes but label
            2 is given).

        Returns
        -------
        explained: Dictionary[string, Tuple[string, float]]
            Dictionary where key is class_name corresponding to labels and
            values are (string, float) tuples that are the feature and
            importance.
        """
        if fuav.is_structured_array(instance):
            instance = fuat.as_unstructured(instance)[0]
        if labels is  None:
            labels = list(range(self._num_classes))
        else:
            if np.any(labels > self._num_classes):
                inc_labels = labels[labels > self._num_classes]
                raise ValueError('Class {} not in dataset '
                                 'specified'.format(inc_labels))
        exp = self.tabular_explainer.explain_instance(
            instance,
            self.model.predict_proba,
            labels=labels,
            num_features=self.num_features,
            num_samples=self.num_samples,
            distance_metric=self.distance_metric)
        explained = {}
        for lab in labels:
            explained[exp.class_names[lab]] = exp.as_list(label=lab)
        return explained


def _validate_input(data: np.array,
                    model: object,
                    categorical_indices: np.array) -> bool:
    """
    Checks input data and model is valid for LIME algorithm

    Parameters
    ----------
   data : numpy.array
        The 2-D data matrix to be used.
    model : object
        Object that contains method predict(x) that outputs predictions
        and predict_proba(x) that outputs probability vectors corresponding
        to the probability of an instance belonging to each class.
    categorical_indices : numpy.array
        Indices that user would like to specify as categorical.
    feature_names : List[str]
        specifying feature names in the order the model is using. Defaults
        to None so names will be 'feature0', 'feature1', ...

    Raises
    ------
    NotImplementedError:
        The inputdatais a numpy structured array and not ndarray with same
        data type.
    IncorrectShapeError:
        The input data is not a 2-D array.
    IncompatibleModelError:
        The model parameter does not contain a peridct_proba() method that
        is needed for LIME algorithm.
    ValueError:
        Index given in categorical_indices parameter is out of range for
        features given indataor number of feature names given does not
        equal to number of features indataor number of class names given
        does not equal number of classes that model has been trained with.

    Returns
    -------
    input_is_valid : boolean
        ``True`` if the input is valid, ``False`` otherwise.
    """
    input_is_valid = True
    return input_is_valid
