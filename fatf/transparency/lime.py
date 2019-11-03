"""
.. deprecated:: 0.0.3
   This module will be deprecated in FAT Forensics version 0.0.3.
   Instead of wrapping the lime package a full (modular) version of the
   LIME surrogate explainer has been implemented -- see the :class:`fatf.\
transparency.predictions.surrogate_explainers.TabularBlimeyLime` class and the
   :ref:`how_to_tabular_surrogates` how-to guide for more details.

The :mod:`fatf.transparency.lime` module wraps the LIME_ explainer.

This module implements a generic tabular data LIME explainer that can be
customised to either explain models (:mod:`fatf.transparency.models.lime`
module) or predictions (:mod:`fatf.transparency.predictions.lime` module).

**This module requires the lime package to be installed.**

(At the moment, only the tabular data explainer is wrapped. Since LIME is
slowly being reimplemented withing this package, wrapping anything else
would be a waste of time.)

.. _LIME: https://github.com/marcotcr/lime
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
#         Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import logging
import warnings

from typing import Any, Dict, List, Tuple, Union

import numpy as np

import fatf.utils.array.tools as fuat
import fatf.utils.array.validation as fuav
import fatf.utils.models.validation as fumv

from fatf.exceptions import IncompatibleModelError, IncorrectShapeError

try:
    import lime.lime_tabular
except ImportError:
    _warning_msg = (  # pylint: disable=invalid-name
        'Lime package is not installed on your system. You must install it in '
        'order to use the fatf.transparency.lime module. One possibility is '
        'to install LIME alongside this package with: pip install fatf[lime].')
    warnings.warn(_warning_msg, ImportWarning)

__all__ = ['Lime']

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Lime(object):
    """
    Wraps LIME package's ``lime.lime_tabular.LimeTabularExplainer`` class.

    .. warning::
        Contrarily to the `LIME tabular explainer`__ this wrapper sets the
        ``sample_around_instance`` parameter to ``True`` meaning that by
        default it provides a local rather than a global explanation. This can
        be changed by either providing a ``sample_around_instance`` or a
        ``local_explanation`` parameter with the first one taking precedence.

    This LIME_ wrapper can be initialised with any of the
    ``lime.lime_tabular.LimeTabularExplainer`` named parameters. Additionally,
    one can also pass in any of the
    ``lime.lime_tabular.LimeTabularExplainer.explain_instance`` method named
    parameters, which will be saved within the object and used when the local
    ``explain_instance`` method is called. In case the same parameters are
    provided both when initialising the object and when explaining an instance
    the values passed to the latter take precedence.

    In addition to all the named parameters one may decide to specify a model
    to be used with the explainer -- the model requires ``predict`` method for
    the regressor mode and ``predict_proba`` method for the classification mode
    -- or a predictive function (``predict_fn``), which is accessed directly
    via the explainer. If both are given, the latter takes the precedence.

    For all of the available parameters please consult the
    `LIME API documentation`__.

    .. warning::
        Since LIME does not support structured arrays by default the predictive
        function and the model have to operate on unstructured types. If a
        structured data or structured data point are passed in they are
        converted to unstructured types and these are used inside the lime.
        If your data array is structured, consider using
        :func:`fatf.utils.array.tools.as_unstructured` function first to
        convert it to an unstructured array before training a model.

    This function loggs a warning if the model does not have a
    ``predict_proba`` method.

    .. _LIME: https://github.com/marcotcr/lime
    __ https://lime-ml.readthedocs.io/en/latest/lime.html
       #lime.lime_tabular.LimeTabularExplainer
    __ https://lime-ml.readthedocs.io/en/latest/lime.html

    Parameters
    ----------
    data : numpy.ndarray
        A 2-dimensional numerical numpy array with a dataset to be used.
    local_explanation : boolean, optional (default=True)
        If ``True`` the LIME explainer will sample data from the neighbourhood
        of the selected instance (a local explanation), otherwise the data will
        be sampled from the whole data distribution (a global explanation).
        This parameter controls ``sample_around_instance`` LIMES parameter.
        If both, ``local_explanation`` and ``sample_around_instance``, are
        provided, the latter takes the precedence.
    model : object, optional (default=None)
        An object that contains ``predict`` -- outputs predictions -- and/or
        ``predict_proba`` -- outputs probability vectors corresponding to the
        probability of an instance belonging to each class -- methods. The
        first method is used when LIME operates in a regressor mode while the
        latter is used with a LIME classification mode.
    predict_fn : function, optional (default=None)
        Alternatively to a whole model LIME can use a python function that
        either outputs regression results or classification probabilities. In
        case both ``model`` and ``predict_fn`` are provided, the latter takes
        the precedence.
    **kwargs : lime.lime_tabular.LimeTabularExplainer
        LIME optional parameters.

    Warns
    -----
    FutureWarning
        This class will be deprecated in FAT Forensics version 0.0.3.
    UserWarning
        The user is warned when both a ``model`` and a ``predict_fn`` are
        provided. In such a case the ``predict_fn`` takes the precedence.

    Raises
    ------
    AttributeError
        One of the named parameters is invalid for the LIME tabular explainer.
    IncorrectShapeError
        The input data is not a 2-dimensional array. The categorical indices
        list/array (``categorical_features``) is not 1-dimensional.
    IncompatibleModelError
        The model does not have ``fit`` and ``predict`` methods.
    TypeError
        Categorical features index parameter (``categorical_features``) is
        neither of the following: a list, a numpy array or a None. The
        ``pred_fn`` parameter is not a callable object, i.e. a function.
    ValueError
        The input data is not purely numerical. For a structured data array
        some of the categorical features indices (``categorical_features``) are
        not strings or they are not valid indices. The mode parameter is
        neither of 'classification' nor 'regression'.

    Attributes
    ----------
    _INIT_PARAMS : Set[string]
        A list of names of the ``LimeTabularExplainer`` named parameter.
    _EXPLAIN_INSTANCE_PARAMS : Set[string]
        A list of names of the ``LimeTabularExplainer.explain_instance``
        function named parameter.
    tabular_explainer : lime.lime_tabular.LimeTabularExplainer
        An initialised ``LimeTabularExplainer`` object.
    mode : string
        LIME mode of operation; ``'classification'`` or ``'regression'``.
    model : object
        A model to be used for LIME explanations.
    model_is_probabilistic : boolean
        An indicator whether the model is a probabilistic model, i.e. has a
        ``predict_proba`` method.
    explain_instance_params : Dictionary[string, Any]
        A dictionary that holds named parameters for the future calls of the
        local ``explain_instance`` method.
    """

    # pylint: disable=useless-object-inheritance,too-few-public-methods

    _INIT_PARAMS = set([
        'mode', 'training_labels', 'feature_names', 'categorical_features',
        'categorical_names', 'kernel_width', 'kernel', 'verbose',
        'class_names', 'feature_selection', 'discretize_continuous',
        'discretizer', 'random_state', 'sample_around_instance',
        'training_data_stats'
    ])
    _EXPLAIN_INSTANCE_PARAMS = set([
        'predict_fn', 'top_labels', 'num_features', 'num_samples', 'labels',
        'distance_metric', 'model_regressor'
    ])

    def __init__(self,
                 data: np.ndarray,
                 local_explanation: bool = True,
                 model: object = None,
                 **kwargs: Any) -> None:
        """
        Initialises a tabular LIME wrapper.
        """
        # pylint: disable=too-many-branches,too-many-statements

        warnings.warn(
            'The LIME wrapper will be deprecated in FAT Forensics version '
            '0.0.3. Please consider using the TabularBlimeyLime explainer '
            'class implemented in the fatf.transparency.predictions.'
            'surrogate_explainers module instead. Alternatively, you may '
            'consider building a custom surrogate explainer using the '
            'functionality implemented in FAT Forensics -- see the *Tabular '
            'Surrogates* how-to guide for more details.', FutureWarning)

        valid_params = self._INIT_PARAMS.union(self._EXPLAIN_INSTANCE_PARAMS)
        invalid_params = set(kwargs.keys()).difference(valid_params)
        if invalid_params:
            raise AttributeError('The following named parameters are not '
                                 'valid: {}.'.format(invalid_params))

        # Split parameters
        init_params = {
            key: kwargs[key]
            for key in kwargs if key in self._INIT_PARAMS
        }
        explain_params = {
            key: kwargs[key]
            for key in kwargs if key in self._EXPLAIN_INSTANCE_PARAMS
        }

        # Check data
        if not fuav.is_2d_array(data):
            raise IncorrectShapeError('The data parameter must be a '
                                      '2-dimensional numpy array.')
        if not fuav.is_numerical_array(data):
            raise ValueError('LIME does not support non-numerical data '
                             'arrays.')

        # Honour native local explanation keyword
        local_explanation_keyword = 'sample_around_instance'
        if local_explanation_keyword not in init_params:
            init_params[local_explanation_keyword] = local_explanation

        # Sort out a structured data array
        if fuav.is_structured_array(data):
            categorical_indices_keyword = 'categorical_features'
            categorical_indices = init_params.get(categorical_indices_keyword,
                                                  None)

            if categorical_indices is not None:
                if isinstance(categorical_indices, list):
                    categorical_indices = np.array(categorical_indices)
                elif isinstance(categorical_indices, np.ndarray):
                    pass
                else:
                    raise TypeError('The {} parameter either has to be a '
                                    'list, a numpy array or None.'.format(
                                        categorical_indices_keyword))

                if not fuav.is_1d_array(categorical_indices):
                    raise IncorrectShapeError(
                        '{} array/list is not '
                        '1-dimensional.'.format(categorical_indices_keyword))
                if not fuav.is_textual_array(categorical_indices):
                    raise ValueError('Since {} is an array of indices for '
                                     'a structured array, all of its elements '
                                     'should be strings.'.format(
                                         categorical_indices_keyword))

                # Check categorical indices
                if not fuat.are_indices_valid(data, categorical_indices):
                    raise ValueError(
                        'Indices given in the {} parameter '
                        'are not valid for the input data '
                        'array.'.format(categorical_indices_keyword))
                init_params[categorical_indices_keyword] = np.array(
                    [data.dtype.names.index(y) for y in categorical_indices])

            data = fuat.as_unstructured(data)

        # Get a LIME tabular explainer
        self.mode = init_params.get('mode', 'classification')
        if self.mode not in ['classification', 'regression']:
            raise ValueError("The mode must be either 'classification' or "
                             "'regression'. '{}' given.".format(self.mode))

        self.tabular_explainer = lime.lime_tabular.LimeTabularExplainer(
            data, **init_params)

        # Check the model
        self.model = model
        self.model_is_probabilistic = False
        if model is not None:
            if fumv.check_model_functionality(
                    model, require_probabilities=True, suppress_warning=True):
                self.model_is_probabilistic = True
            elif fumv.check_model_functionality(
                    model, require_probabilities=False, suppress_warning=True):
                self.model_is_probabilistic = False
                logger.warning('The model can only be used for LIME in a '
                               'regressor mode.')
            else:
                raise IncompatibleModelError('LIME requires a model object to '
                                             'have a fit method and '
                                             'optionally a predict_proba '
                                             'method.')

        # Check the predictive function and memorise parameters that may be
        # useful for explaining an instance
        pred_fn_name = 'predict_fn'
        if pred_fn_name in explain_params:
            prediction_function = explain_params[pred_fn_name]
            # Make sure that its a function
            if not callable(prediction_function):
                raise TypeError('The {} parameter is not callable -- it has '
                                'to be a function.'.format(pred_fn_name))

            # Warn the user if both a model and a function are provided
            if self.model is not None:
                warnings.warn(
                    'Since both, a model and a predictive function, are '
                    'provided only the latter will be used.', UserWarning)

        self.explain_instance_params = explain_params

    def explain_instance(
            self, instance: np.ndarray, **kwargs: Any
    ) -> Union[Dict[str, Tuple[str, float]], List[Tuple[str, float]]]:
        """
        Explains an instance with the LIME tabular explainer.

        This method wraps around ``explain_instance`` method_ in the LIME
        tabular explainer object.

        .. warning::
            Contrarily to the LIME tabular explainer this wrapper produces
            explanations for all of the classes for a classification task by
            default.

        If any of the named parameters for this function were specified when
        initialising this object they will be used unless they are also defined
        when calling this method, in which case the latter take the precedence.

        If all: a class-wide model, a class-wide prediction function and a
        local prediction function (via named parameter to this function) are
        specified, they are used in the following order:

        - local prediction function,

        - global prediction function, and finally

        - the model.

        Based on whether the task at hand is classification or regression
        either ``predict`` (regression) or ``predict_proba`` (classification)
        method of the model is used.

        .. _method: https://lime-ml.readthedocs.io/en/latest/lime.html
           #lime.lime_tabular.LimeTabularExplainer.explain_instance

        Parameters
        ----------
        instance : numpy.ndarray
            A 1-dimensional data point (numpy array) to be explained.
        **kwargs : lime.lime_tabular.LimeTabularExplainer.explain_instance
            LIME tabular explainer's ``explain_instance`` optional parameters.

        Raises
        ------
        AttributeError
            One of the named parameters is invalid for the ``explain_instance``
            method of the LIME tabular explainer.
        IncorrectShapeError
            The input ``instance`` is not a 1-dimensional numpy array.
        RuntimeError
            A predictive function is not available (neither as a ``model``
            attribute of this class, nor as a ``predict_fn`` parameter).
        ValueError
            The input ``instance`` is not purely numerical.

        Returns
        -------
        explanation : Dictionary[string, Tuple[string, float]] or \
List[Tuple[string, float]]
            For classification a dictionary where the keys correspond to class
            names and the values are tuples (string and float), which represent
            an explanation in terms of one of the features and the importance
            of this explanation. For regression a list of tuples (string and
            float) with the same meaning.
        """
        # pylint: disable=too-many-locals,too-many-branches
        invalid_params = set(kwargs.keys()).difference(
            self._EXPLAIN_INSTANCE_PARAMS)
        if invalid_params:
            raise AttributeError('The following named parameters are not '
                                 'valid: {}.'.format(invalid_params))

        if not fuav.is_1d_like(instance):
            raise IncorrectShapeError('The instance to be explained should be '
                                      '1-dimensional.')
        instance = fuat.as_unstructured(instance)
        if not fuav.is_numerical_array(instance):
            raise ValueError('The instance to be explained should be purely '
                             'numerical -- LIME does not support categorical '
                             'features.')

        # Merge local kwargs and object's kwargs
        named_arguments = dict(self.explain_instance_params)
        for kwarg in self._EXPLAIN_INSTANCE_PARAMS:
            if kwarg in kwargs:
                named_arguments[kwarg] = kwargs[kwarg]

        # If both a model and a predictor function is supplied
        pred_fn_name = 'predict_fn'
        if pred_fn_name in named_arguments:
            pred_fn = named_arguments[pred_fn_name]
            del named_arguments[pred_fn_name]
        elif self.model is not None:
            if self.mode == 'classification':
                if self.model_is_probabilistic:
                    pred_fn = self.model.predict_proba  # type: ignore
                else:
                    raise RuntimeError('The predictive model is not '
                                       'probabilistic. Please specify a '
                                       'predictive function instead.')
            else:
                pred_fn = self.model.predict  # type: ignore
        else:
            raise RuntimeError('A predictive function is not available.')

        # If unspecified, get explanations for all classes for classification
        lbls_name = 'labels'
        if lbls_name not in named_arguments and self.mode == 'classification':
            # Since we cannot get all of the class names/indices/quantity,
            # we need to resort to this dirty trick
            n_classes = pred_fn(np.array([instance])).shape[1]
            named_arguments[lbls_name] = range(n_classes)

        exp = self.tabular_explainer.explain_instance(instance, pred_fn,
                                                      **named_arguments)

        if self.mode == 'classification':
            explanation = {}
            for label in exp.available_labels():
                class_name = exp.class_names[label]
                class_explanation = exp.as_list(label=label)

                explanation[class_name] = class_explanation
        else:
            explanation = exp.as_list()

        return explanation
