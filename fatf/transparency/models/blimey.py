"""
Blimey implementation.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
# License: new BSD

from typing import Dict, Tuple, Union, Optional, List
import numpy as np

import fatf.utils.data.augmentation as fuda
import fatf.utils.data.discretization as fudd

import fatf.utils.validation as fuv
import fatf.utils.models.validation as fumv
import fatf.utils.array.tools as fuat
import fatf.utils.data.similarity_binary_mask as fuds
import fatf.utils.array.validation as fuav

from fatf.exceptions import IncompatibleModelError, IncorrectShapeError

__all__ = ['blimey']

Index = Union[int, str]


def _input_is_valid(dataset: np.ndarray,
                    augmentor: fuda.Augmentation,
                    discretizer: fudd.Discretization,
                    explainer: object,
                    global_model: object,
                    local_model: object,
                    class_names: Optional[List[str]] = None,
                    feature_names: Optional[List[str]] = None) -> bool:
    """
    Validates the input parameters of blimey class.

    Parameters
    ----------
    dataset : numpy.ndarray
        A 2-dimensional numpy array with a dataset to be discretized.
    augmentor : fatf.utils.data.augmentation.Augmentation,
        An object refence which is a subclass of :class:`fatf.utils.data.
        augmentation.Augmentation` which the data will be augmented using.
    discretizer : fatf.utils.data.discretization.Discretization
        An object reference which is a subclass of :class:`fatf.utils.data.
        discretization.Discretization` which will discretize the data for use
        in the local model.
    explainer : object
        An object reference which will explain an instance by training a 
        `local_model` on locally augmentated data.
    global_model : object
        A pretrained global model. This must contain method ``predict_proba``
        that will return a numpy array for probabilities of instances belonging
        to each of the classses.
    local_model : object
        An object reference used for prediction in the `explainer` object. 
        Must be compatible with the chosen explainer.
    class_names : List[str]
        A list of strings defining the names of classes.
    feature_names : List[str], optional (default=None)
        A list of strings defining the names of the features.

    Raises
    ------

    Returns
    -------
    is_input_ok : boolean
        ``True`` if input is valid, ``False`` otherwise.
    """
    is_input_ok = False

    if not fuav.is_2d_array(dataset):
        raise IncorrectShapeError('The input dataset must be a 2-dimensional '
                                  'array.')

    if not fuav.is_base_array(dataset):
        raise ValueError('The input dataset must only contain base types '
                         '(textual and numerical).')

    if not fumv.check_model_functionality(global_model, 
            require_probabilities=True):
        raise IncompatibleModelError(
            'This functionality requires the global model to be capable of '
            'outputting probabilities via predict_proba method.')

    #if not fuv.check_explainer_functionality(explainer):
    #    raise IncompatibleModelError(
    #        'This functionality requires the explainer object to be capable '
    #        'of outputting explanations via explain_instance method.')

    if not issubclass(augmentor, fuda.Augmentation):
        raise TypeError('The augmentor object must inherit from abstract '
                        'class fatf.utils.augmentation.Augmentation.')
    
    if not issubclass(discretizer, fudd.Discretization):
        raise TypeError('The discretizer object must inherit from abstract '
                        'class fatf.utils.discretization.Discretization.')

    if class_names is not None:
        if not isinstance(class_names, list):
            raise TypeError('The class_names parameter must be None or a '
                            'list.')
        else:
            for class_name in class_names:
                if (class_name is not None and not isinstance(class_name,
                        str)):
                    raise TypeError('The class_name has to be either None or a '
                                    'string or a list of strings.')

    if fuav.is_structured_array(dataset):
        features_number = len(dataset.dtype.names)
    else:
        features_number = dataset.shape[1]

    if feature_names is not None:
        if not isinstance(feature_names, list):
            raise TypeError('The feature_names parameter must be None or a '
                            'list.')
        else:
            if len(feature_names) != features_number:
                raise ValueError('The length of feature_names must be equal '
                                 'to the number of features in the dataset.')
            for feature in feature_names:
                if (feature is not None and not isinstance(feature, str)):
                    raise TypeError('The feature name has to be either None or '
                                    'a string or a list of strings.')
    
    is_input_ok = True
    return is_input_ok


class blimey(object):
    """
    Blimey class

    Parameters
    ----------
    dataset : numpy.ndarray
        A 2-dimensional numpy array with a dataset to be discretized.
    augmentor : fatf.utils.data.augmentation.Augmentation,
        An object refence which is a subclass of :class:`fatf.utils.data.
        augmentation.Augmentation` which the data will be augmented using.
    discretizer : fatf.utils.data.discretization.Discretization
        An object reference which is a subclass of :class:`fatf.utils.data.
        discretization.Discretization` which will discretize the data for use
        in the local model.
    explainer : object
        An object reference which will explain an instance by training a 
        `local_model` on locally augmentated data.
    global_model : object
        A pretrained global model. This must contain method ``predict_proba``
        that will return a numpy array for probabilities of instances belonging
        to each of the classses.
    local_model : object
        An object reference used for prediction in the `explainer` object. 
        Must be compatible with the chosen explainer.
    class_names : List[str]
        A list of strings defining the names of classes.
    feature_names : List[str], optional (default=None)
        A list of strings defining the names of the features.

    Raises
    ------

    Attributes
    ----------
    """
    def __init__(self,
                 dataset: np.ndarray,
                 augmentor: fuda.Augmentation,
                 discretizer: fudd.Discretization,
                 explainer: object,
                 global_model : object,
                 local_model: object,
                 class_names: Optional[List[str]] = None,
                 feature_names: Optional[List[str]] = None,
                 **kwargs):
        """
        Constructs an ``Blimey`` class.
        """
        assert _input_is_valid(dataset, augmentor, discretizer, explainer,
                               global_model, local_model, class_names,
                               feature_names)

        self.kwargs = kwargs
        self.dataset = dataset
        self.global_model = global_model
        self.augmentor = augmentor(dataset, **kwargs)
        self.discretizer = discretizer(dataset, feature_names=feature_names,
                                       **kwargs)
        self.discretized_dataset = self.discretizer.discretize(self.dataset)
        self.explainer_class = explainer
        self.prediction_probabilities = global_model.predict_proba(dataset)
        self.n_classes = self.prediction_probabilities.shape[1]
        self.local_model = local_model

        if fuav.is_structured_array(self.dataset):
            self.indices = self.dataset.dtype.names
        else:
            self.indices = np.arange(0, self.dataset.shape[1], 1)

        self.class_names = class_names
        self.feature_names = feature_names
        # TODO: categorical indices
        # pre-process class_names and feature_names
        # TODO: preprocess

    def explain_instance(self,
                         data_row: np.ndarray,
                         samples_number: Optional[int] = 100
                         ) -> Dict[str, Dict[Index, np.float64]]:
        """
        Generates explanations for data_row.
        """
        # TODO: validate input
        discretized_data_row = self.discretizer.discretize(data_row)
        # TODO: change feature names for discreitzied_data_row (only for this 
        # instance as different instances will find different feature names)

        sampled_data = self.augmentor.sample(
            data_row, samples_number=samples_number, **self.kwargs)

        binary_data = fuds.similarity_binary_mask(
            self.discretized_dataset, discretized_data_row)
        
        discretized_value_names = self.discretizer.feature_value_names
        
        discretized_feature_names = []
        for i, index in enumerate(self.indices):
            if index in discretized_value_names.keys():
                discretized_feature_names.append(discretized_value_names[index][int(discretized_data_row[index])])
            else:
                discretized_feature_names.append(self.feature_names[i])

        blimey_explanation = {}
        for i in range(self.n_classes):
            local_model = self.local_model(**self.kwargs)
            local_model.fit(binary_data, self.prediction_probabilities[:, i])
            explainer = self.explainer_class(self.dataset, local_model=local_model, feature_names=discretized_feature_names, **self.kwargs)
            explanation = explainer.explain_instance(discretized_data_row, **self.kwargs)
            blimey_explanation[self.class_names[i]] = explanation

        return blimey_explanation
        