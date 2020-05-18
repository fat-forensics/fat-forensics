"""
.. versionadded:: 0.0.2

The :mod:`fatf.transparency.predictions.surrogate_explainers` module implements
example surrogate explainers.

Guidelines and tips for building custom surrogate explainers for various
types of data (tabular, image and text) can be found in the :ref:`how_to_guide`
part of the documentation:

* :ref:`how_to_tabular_surrogates`.

The ``scikit-learn`` package is required for the surrogate tree and linear
model (LIME) explainers to work. When importing this module with
``scikit-learn`` missing, the user will only be warned about disabling this
functionality not being available and the two explainers will
be disabled to allow using the abstract SurrogateTabularExplainer without
the ``scikit-learn`` package installed.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
#         Kacper Sokol <K.Sokol@bristol.ac.uk>
# License: new BSD

# pylint: disable=too-many-lines

from numbers import Number
from typing import Any, Dict, List, Optional, Tuple, Union

import abc
import logging
import warnings

import scipy.stats

import numpy as np

from fatf.exceptions import IncompatibleModelError, IncorrectShapeError

import fatf.utils.array.tools as fuat
import fatf.utils.array.validation as fuav
import fatf.utils.data.augmentation as fuda
import fatf.utils.data.discretisation as fudd
import fatf.utils.data.transformation as fudt
import fatf.utils.distances as fud
import fatf.utils.kernels as fatf_kernels
import fatf.utils.models.validation as fumv
import fatf.utils.models.models as fumm
import fatf.utils.tools as fut

__all__ = ['SurrogateTabularExplainer',
           'TabularBlimeyLime',
           'TabularBlimeyTree']  # yapf: disable

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

try:
    # pylint: disable=ungrouped-imports
    import sklearn
    import sklearn.linear_model
    import sklearn.tree

    import fatf.transparency.sklearn.linear_model as ftslm
    import fatf.utils.data.feature_selection.sklearn as fudfs

    _SKLEARN_VERSION = [int(i) for i in sklearn.__version__.split('.')[:2]]
    _SKLEARN_0_22 = fut.at_least_verion([0, 22], _SKLEARN_VERSION)
except ImportError as exin:
    warnings.warn(
        'The TabularBlimeyLime and TabularBlimeyTree surrogate explainers '
        'require scikit-learn to be installed. Since scikit-learn is missing, '
        'this functionality will be disabled.', UserWarning)
    logger.error(str(exin))

    ReturnTree = None  # pylint: disable=invalid-name
    SKLEARN_MISSING = True
else:
    if _SKLEARN_0_22:  # pragma: nocover
        # pylint: disable=invalid-name,protected-access,no-member
        ReturnTree = sklearn.tree._classes.BaseDecisionTree
    else:  # pragma: nocover
        ReturnTree = (  # pylint: disable=invalid-name
            sklearn.tree.tree.BaseDecisionTree)
    SKLEARN_MISSING = False

BinSamplingValues = Dict[Union[str, int],
                         Dict[int, Tuple[float, float, float, float]]]
Explanation = Dict[str, Dict[str, float]]
ExplanationSurrogate = Union[Dict[str, fumm.Model], fumm.Model]
ExplanationTuple = Union[Explanation,  # yapf: disable
                         Tuple[Explanation, ExplanationSurrogate]]
Index = Union[int, str]


def _input_is_valid(
        dataset: np.ndarray, predictive_model: object, as_probabilistic: bool,
        as_regressor: bool, categorical_indices: Union[None, List[Index]],
        class_names: Union[None, List[str]], classes_number: Union[None, int],
        feature_names: Union[None, List[str]],
        unique_predictions: Union[None, List[Union[str, int]]]) -> bool:
    """
    Validates the input parameters of the ``SurrogateTabularExplainer`` class.

    For the input parameters description, warnings and exceptions please see
    the documentation of the :class:`fatf.transparency.predictions.\
surrogate_explainers.SurrogateTabularExplainer` class.

    Returns
    -------
    is_input_ok : boolean
        ``True`` if the input is valid, ``False`` otherwise.
    """
    # pylint: disable=too-many-arguments,too-many-statements,too-many-locals
    # pylint: disable=too-many-branches
    is_input_ok = False

    if not fuav.is_2d_array(dataset):
        raise IncorrectShapeError('The input dataset must be a 2-dimensional '
                                  'numpy array.')
    if not fuav.is_base_array(dataset):
        raise TypeError('The input dataset must only contain base types '
                        '(textual and/or numerical).')

    if not isinstance(as_probabilistic, bool):
        raise TypeError('The as_probabilistic parameter has to be a boolean.')
    if not isinstance(as_regressor, bool):
        raise TypeError('The as_regressor parameter has to be a boolean.')

    if as_regressor:
        is_functional = fumv.check_model_functionality(predictive_model, False,
                                                       True)
        if not is_functional:
            raise IncompatibleModelError(
                'With as_regressor set to True the predictive model '
                'needs to be capable of outputting numerical predictions '
                'via a *predict* method, which takes exactly one required '
                'parameter -- data to be predicted -- and outputs a '
                '1-dimensional array with numerical predictions.')
    else:
        if as_probabilistic:
            is_functional = fumv.check_model_functionality(
                predictive_model, True, True)
            if not is_functional:
                raise IncompatibleModelError(
                    'With as_predictive set to True the predictive model '
                    'needs to be capable of outputting probabilities via '
                    'a *predict_proba* method, which takes exactly one '
                    'required parameter -- data to be predicted -- and '
                    'outputs a 2-dimensional array with probabilities.')
        else:
            is_functional = fumv.check_model_functionality(
                predictive_model, False, True)
            if not is_functional:
                raise IncompatibleModelError(
                    'With as_predictive set to False the predictive model '
                    'needs to be capable of outputting (class) predictions '
                    'via a *predict* method, which takes exactly one required '
                    'parameter -- data to be predicted -- and outputs a '
                    '1-dimensional array with (class) predictions.')

    if categorical_indices is not None:
        if isinstance(categorical_indices, list):
            if len(categorical_indices) != len(set(categorical_indices)):
                raise ValueError('The categorical_indices list contains '
                                 'duplicated entries.')

            invalid_indices = fuat.get_invalid_indices(
                dataset, np.asarray(categorical_indices))
            if invalid_indices.size:
                raise IndexError('The following indices are invalid for the '
                                 'input dataset: {}.'.format(invalid_indices))
        else:
            raise TypeError('The categorical_indices parameter must be a '
                            'Python list or None.')

    if class_names is not None:
        if isinstance(class_names, list):
            if not class_names:
                raise ValueError('The class_names list cannot be empty.')
            if len(class_names) != len(set(class_names)):
                raise ValueError('The class_names list contains '
                                 'duplicated entries.')
            for class_name in class_names:
                if not isinstance(class_name, str):
                    raise TypeError('All elements of the class_names '
                                    'list must be strings; '
                                    '*{}* is not.'.format(class_name))
        else:
            raise TypeError('The class_names parameter must be a Python list '
                            'or None.')

    if classes_number is not None:
        if isinstance(classes_number, int):
            if classes_number < 2:
                raise ValueError('The number of classes cannot be '
                                 'smaller than 2.')
        else:
            raise TypeError('The classes number parameter must be an integer '
                            'or None.')

    if feature_names is not None:
        if fuav.is_structured_array(dataset):
            features_number = len(dataset.dtype.names)
        else:
            features_number = dataset.shape[1]

        if isinstance(feature_names, list):
            if len(feature_names) != features_number:
                raise ValueError('The length of feature_names must be equal '
                                 'to the number of features in the dataset '
                                 '({}).'.format(features_number))
            if len(feature_names) != len(set(feature_names)):
                raise ValueError('The feature_names list contains '
                                 'duplicated entries.')
            for feature_name in feature_names:
                if not isinstance(feature_name, str):
                    raise TypeError('All elements of the feature_names '
                                    'list have to be strings; '
                                    '*{}* is not.'.format(feature_name))
        else:
            raise TypeError('The feature_names parameter must be a Python '
                            'list or None.')

    if unique_predictions is not None:
        if isinstance(unique_predictions, list):
            if unique_predictions:
                if len(unique_predictions) != len(set(unique_predictions)):
                    raise ValueError('The unique_predictions list contains '
                                     'duplicated entries.')

                expected_type = type(unique_predictions[0])
                incorrect_type_msg = ('One of the elements in the '
                                      'unique_predictions list is neither a '
                                      'string nor an integer.')
                if expected_type not in (int, str):
                    raise TypeError(incorrect_type_msg)
                for prediction in unique_predictions:
                    if not isinstance(prediction, expected_type):
                        raise TypeError(incorrect_type_msg)
            else:
                raise ValueError('The unique_predictions list cannot be '
                                 'empty.')
        else:
            raise TypeError('The unique_predictions parameter must be a '
                            'Python list or None.')

    is_input_ok = True
    return is_input_ok


class SurrogateTabularExplainer(abc.ABC):
    """
    An abstract parent class for implementing surrogate explainers.

    .. versionchanged:: 0.1.0
       Added support for regression models.

    .. versionadded:: 0.0.2

    An abstract class that all surrogate explainer classes should inherit from.
    It contains an ``__init__`` method and an input validator --
    ``_explain_instance_input_is_valid`` -- for the abstract
    ``explain_instance`` method. The validation of the input parameters passed
    to the ``__init__`` method is done via the
    ``fatf.transparency.predictions.surrogate_explainers._input_is_valid``
    function.

    If the ``predictive_model`` is a non-probabilistic classifier
    (``as_probabilistic=False`` and ``as_regressor=False``), it is advised to
    specify both ``classes_number`` and ``unique_predictions`` parameters to
    ensure proper functionality of the explainer. Please see the respective
    parameter descriptions for more details.

    For detailed instruction how to build your own surrogate please see the
    :ref:`how_to_tabular_surrogates` *how-to guide*.

    .. warning::

       The ``_explain_instance_input_is_valid`` method should be called in all
       implementations of the ``explain_instance`` method in the children
       classes to ensure that all of the input parameters passed to this method
       are valid.

    Parameters
    ----------
    dataset : numpy.ndarray
        A 2-dimensional numpy array with a dataset (utilised in various ways
        throughout the explainer).
    predictive_model : object
        A pre-trained (black-box) predictive model to be explained. If
        ``as_probabilistic`` (see below) is set to ``True``, it must have a
        ``predict_proba`` method that takes a data set as the only required
        input parameter and returns a 2-dimensional numpy array with
        probabilities of belonging to each class. Otherwise, if
        ``as_probabilistic`` is set to ``False``, the ``predictive_model`` must
        have a ``predict`` method that outputs a 1-dimensional array with
        (class) predictions.
    as_probabilistic : boolean, optional (default=True)
        A boolean indicating whether the global model is probabilistic. If
        ``True``, the ``predictive_model`` must have a ``predict_proba``
        method. If ``False``, the ``predictive_model`` must have a ``predict``
        method. This parameter is disregarded when ``as_regressor=True``.
    as_regressor : boolean, optional (default=False)
        .. versionadded:: 0.1.0

        A boolean indicating whether the global model is regression. If
        ``True``, the ``predictive_model`` must have a ``predict``
        method; the ``as_probabilistic`` parameter is disregarded. If
        ``False``, the model is treated as a classifier -- see
        ``as_probabilistic`` parameter.
    categorical_indices : List[column indices], optional (default=None)
        A list of column indices in the input ``dataset`` that should be
        treated as categorical features.
    class_names : List[string], optional (default=None)
        A list of strings defining the names of classes. If the predictive
        model is probabilistic, the order of the class names should correspond
        to the order of columns output by the model. For other models the
        order should correspond to lexicographical ordering of all the possible
        outputs of this model. For example, if the model outputs
        ``['a', 'c', '0']`` the class names should be given for
        ``['0', 'a', 'c']`` ordering. This parameter is disregarded when
        ``as_regressor=True``.
    classes_number : integer, optional (default=None)
        The unique number of classes modelled by the ``predictive_model``.
        If the model is probabilistic, setting this parameter is not required
        as the number of classes will be inferred from the shape of the
        predicted probabilities array. For non-probabilistic models if this
        parameter is not given, this number will be inferred from the length of
        the ``class_names`` list if provided, otherwise the input ``dataset``
        will be predicted and the unique values will be counter therein.
        Since the latter method cannot guarantee counting all the possible
        predictions, a ``UserWarning`` will be emitted encouraging the user to
        specify the number of classes via the ``classes_number`` parameter.
        For non-probabilistic models it is advised to specify this parameter.
        This parameter is disregarded when ``as_regressor=True``.
    feature_names : List[string], optional (default=None)
        A list of strings defining the names of the ``dataset`` features. The
        order of the names should correspond to the order of features in the
        ``dataset``.
    unique_predictions : List[strings or integers], optional (default=None)
        A complete list of unique predictions that the ``predictive_model`` can
        output. This parameter is only used when the ``predictive_model`` is a
        non-probabilistic classifier (``as_probabilistic=False``).
        This parameter is disregarded when ``as_regressor=True``.

    Warns
    -----
    UserWarning
        If some of the string-based columns in the input data array were not
        indicated to be categorical features by the user (via the
        ``categorical_indices`` parameter), the user is warned that they will
        be added to the list of categorical features. When the
        ``classes_number`` parameter is not specified for a non-probabilistic
        model and the number of classes cannot be inferred form the
        length of the ``classes_names`` list, the number of classes is computed
        as the unique number of predictions of the ``predictive_model`` for the
        input ``datasets``, which may not be accurate. The user is warned and
        advised to specify the ``classes_number`` parameter in this case.
        The user has provided unique predictions via the ``unique_predictions``
        parameter for a probabilistic model (``as_probabilistic=True``),
        which are not needed and will be disregarded. The unique predictions
        had to be inferred from the predictions output by the
        (non-probabilistic) ``predictive_model``, therefore may be incomplete.
        It is advised to provide this list via the ``unique_predictions``
        parameter to ensure proper functionality of the explainer.

    Raises
    ------
    IncompatibleModelError
        The ``predictive_model`` does not have the required functionality:
        ``predict_proba`` method for probabilistic models and ``predict``
        method for regressors and non-probabilistic classifiers.
    IncorrectShapeError
        The input ``dataset`` is not a 2-dimensional numpy array.
    IndexError
        Some of the column indices given in the ``categorical_indices``
        parameter are not valid for the input ``dataset``.
    RuntimeError
        The number of columns in the probabilistic matrix output by the
        ``predictive_model`` (when ``as_probabilistic=True``) is different
        to the number of features specified by the user via the
        ``features_number`` parameter. The ``predictive_model`` has output
        different unique classes to the ones specified by the user via the
        ``unique_predictions`` parameter (checked only for non-probabilistic
        models, i.e., ``as_probabilistic=False``). Either the user-specified
        or inferred number of unique predictions does not agree with the
        internal number of classes.
    TypeError
        The input ``dataset`` is not of a base (numerical and/or string)
        type. The ``as_probabilistic`` parameter is not a boolean.
        The ``as_regressor`` parameter is not a boolean.
        The ``categorical_indices`` parameter is neither a list nor ``None``.
        The ``class_names`` parameter is neither a list nor ``None`` or one of
        the elements in this list is not a string.
        The ``classes_number`` parameter is neither an integer nor ``None``.
        The ``feature_names`` parameter is neither a list nor ``None`` or one
        of the elements in this list is not a string.
        The ``unique_predictions`` parameter is neither a list nor ``None`` or
        all the elements in this list are not of string or integer type.
    ValueError
        The ``categorical_indices`` list contains duplicated entries.
        The length of the ``class_names`` list is not equal to the detected or
        given number of classes, some of the entires in this list are
        duplicated or this list is empty. The length of the ``feature_names``
        list is not the same as the number of features in the input ``dataset``
        or some of the entries in that list are duplicated. The
        ``classes_number`` parameter is smaller than 2. The
        ``unique_predictions`` list is empty or contains duplicated entries.

    Attributes
    ----------
    dataset : numpy.ndarray
        A 2-dimensional numpy array with the input ``dataset``.
    is_structured : boolean
        ``True`` if the ``dataset`` is a structured numpy array, ``False``
        otherwise.
    column_indices : List[column indices]
        A list of column indices in the order they appear in the ``dataset``.
    categorical_indices : List[column indices]
        A list of column indices that should be treat as categorical features.
    numerical_indices : List[column indices]
        A list of column indices that should be treat as numerical features.
    as_probabilistic : boolean
        ``True`` if the ``predictive_model`` should be treated as
        probabilistic and ``False`` if it should be treated as a classifier.
    as_regressor : boolean
        ``True`` if the ``predictive_model`` should be treated as
        regression and ``False`` if it should be treated as a (probabilistic)
        classifier.
    predictive_model : object
        A pre-trained (black-box) predictive model to be explained.
    predictive_function : Callable[[np.ndarray], np.ndarray]
        A function that will be used to get predictions from the input
        predictive model. It references the ``predictive_model.predict_proba``
        method for for probabilistic models (``as_probabilistic=True``) and the
        ``predictive_model.predict`` method for non-probabilistic models.
    classes_number : integer
        A number of unique classes that the ``predictive_model`` is trained to
        recognise.
    class_names : List[string]
        A list of strings defining the names of classes. If this was not
        specified by the user, the classes will be assigned names based on the
        following pattern: 'class %d'. If the ``predictive_model`` is a
        classifier (``as_probabilistic=False``) and the number of unique
        predictions is equal to the number of classes, the ``class_names`` will
        be lexicographically sorted list of the unique values output by the
        ``predictive_model``.
    feature_names : List[string]
        A list of strings defining the names of the features. If this was not
        specified by the user, the features will be assigned names based on the
        following pattern: 'feature %d'.
    unique_predictions : List[strings or integers] or None
        ``None`` for probabilistic ``predictive_model``
        (``as_probabilistic=True``) and a list of unique classes output by
        the ``predictive_model`` if it is non-probabilistic.
    """

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-few-public-methods

    def __init__(self,
                 dataset: np.ndarray,
                 predictive_model: object,
                 as_probabilistic: bool = True,
                 as_regressor: bool = False,
                 categorical_indices: Optional[List[Index]] = None,
                 class_names: Optional[List[str]] = None,
                 classes_number: Optional[int] = None,
                 feature_names: Optional[List[str]] = None,
                 unique_predictions: Optional[List[Union[str, int]]] = None
                 ) -> None:
        """
        Constructs a ``SurrogateTabularExplainer`` class.
        """
        # pylint: disable=too-many-arguments,too-many-locals,too-many-branches
        # pylint: disable=too-many-statements
        assert _input_is_valid(dataset, predictive_model, as_probabilistic,
                               as_regressor, categorical_indices, class_names,
                               classes_number, feature_names,
                               unique_predictions), 'Invalid input.'

        self.dataset = dataset
        self.is_structured = fuav.is_structured_array(dataset)

        if self.is_structured:
            column_indices = list(self.dataset.dtype.names)
        else:
            column_indices = list(range(self.dataset.shape[1]))
        self.column_indices = column_indices

        # Sort out column indices
        indices = fuat.indices_by_type(dataset)
        num_indices = set(indices[0])
        cat_indices = set(indices[1])
        all_indices = num_indices.union(cat_indices)

        if categorical_indices is None:
            categorical_indices = list(cat_indices)
            numerical_indices = list(num_indices)
        else:
            if cat_indices.difference(categorical_indices):
                msg = ('Some of the string-based columns in the input dataset '
                       'were not selected as categorical features via the '
                       'categorical_indices parameter. String-based columns '
                       'cannot be treated as numerical features, therefore '
                       'they will be also treated as categorical features '
                       '(in addition to the ones selected with the '
                       'categorical_indices parameter).')
                warnings.warn(msg, UserWarning)
                categorical_indices = list(
                    cat_indices.union(categorical_indices))
            numerical_indices = list(
                all_indices.difference(categorical_indices))

        self.categorical_indices = sorted(categorical_indices)
        self.numerical_indices = sorted(numerical_indices)

        self.as_probabilistic = as_probabilistic
        self.as_regressor = as_regressor
        self.predictive_model = predictive_model

        if self.as_regressor:
            predictive_function = (
                self.predictive_model.predict)  # type: ignore
        else:
            if self.as_probabilistic:
                predictive_function = (
                    self.predictive_model.predict_proba)  # type: ignore
            else:
                predictive_function = (
                    self.predictive_model.predict)  # type: ignore
        self.predictive_function = predictive_function

        if self.is_structured:
            row = dataset[0].reshape(-1)
        else:
            row = dataset[0].reshape(1, -1)

        if self.as_regressor:
            _unique_predictions = None
            _classes_number = None
        else:
            if self.as_probabilistic:
                _unique_predictions = None
                _classes_number = (
                    self.predictive_model.predict_proba(  # type: ignore
                        row).shape[1])
            else:
                _unique_predictions = np.unique(
                    self.predictive_model.predict(  # type: ignore
                        self.dataset))
                _classes_number = _unique_predictions.shape[0]
                _unique_predictions = _unique_predictions.tolist()

        if self.as_regressor:
            assert _classes_number is None
            classes_number = _classes_number
        else:
            if self.as_probabilistic:
                if classes_number is None:
                    classes_number = _classes_number
                else:
                    if classes_number != _classes_number:
                        raise RuntimeError(
                            'The user specified number of classes ({}) for '
                            'the provided probabilistic model is different '
                            'than the number of columns ({}) in the '
                            'probabilistic matrix output by the '
                            'model.'.format(classes_number, _classes_number))
            else:
                if classes_number is None:
                    if unique_predictions is not None:
                        logger.debug('The classes number was taken from the '
                                     'length of the unique_predictions list.')
                        classes_number = len(unique_predictions)
                    elif class_names is not None:
                        logger.debug('The classes number was taken from the '
                                     'length of the class_names list.')
                        classes_number = len(class_names)
                    else:
                        msg = ('The number of classes ({}) was inferred from '
                               'predicting the input dataset. Since this may '
                               'not be accurate please consider specifying '
                               'the number of unique classes via the '
                               'classes_number parameter.'.format(
                                   _classes_number))
                        warnings.warn(msg, UserWarning)
                        classes_number = _classes_number
        self.classes_number = classes_number

        if self.as_regressor:
            assert _unique_predictions is None
            set_unique_predictions = _unique_predictions
        else:
            if self.as_probabilistic:
                if unique_predictions is not None:
                    warnings.warn('The unique_predictions provided by the '
                                  'user will be disregarded as the '
                                  'predictive_model is probabilistic '
                                  '(as_probabilistic=True).')
                    set_unique_predictions = None
                else:
                    set_unique_predictions = unique_predictions
            else:
                # The self.classes_number can come from the length of the
                # classes_names list, the unique_predictions list, the
                # number of predictions taken from the model or it can be
                # user-defined, hence we need to verify whether it agrees with
                # the length of the unique_predictions list for consistency.
                if unique_predictions is None:
                    up_num = len(_unique_predictions)  # type: ignore
                    if up_num != self.classes_number:
                        raise RuntimeError('The inferred number of unique '
                                           'predictions ({}) does not agree '
                                           'with the internal number of '
                                           'classes. Try providing the '
                                           'unique_predictions parameter to '
                                           'fix this issue.'.format(up_num))

                    msg = ('The unique predictions ({}) were inferred from '
                           'predicting the input dataset. Since this may not '
                           'be accurate please consider specifying the list '
                           'of unique predictions via the unique_predictions '
                           'parameter.'.format(_unique_predictions))
                    warnings.warn(msg, UserWarning)
                    set_unique_predictions = _unique_predictions
                else:
                    if len(unique_predictions) != self.classes_number:
                        raise RuntimeError('The user-specified number of '
                                           'unique predictions ({}) does not '
                                           'agree with the internal number of '
                                           'classes. (The length of the '
                                           'unique_predictions list is '
                                           'different than the classes_number '
                                           'parameter.)'.format(
                                               len(unique_predictions)))

                    # Check whether the model does not output more
                    # classes than specified by the user via the
                    # unique_predictions parameter.
                    extra = set(
                        _unique_predictions).difference(  # type: ignore
                            unique_predictions)
                    if extra:
                        raise RuntimeError('The predictive_model has '
                                           'output different classes ({} '
                                           'extra) than were specified by the '
                                           'unique_predictions '
                                           'parameter.'.format(
                                               sorted(list(extra))))
                    set_unique_predictions = unique_predictions

        if set_unique_predictions is None:
            self.unique_predictions = set_unique_predictions
        else:
            self.unique_predictions = sorted(set_unique_predictions)

        if self.as_regressor:
            class_names = None
        else:
            if class_names is None:
                if self.as_probabilistic:
                    assert (self.unique_predictions is None
                            and self.classes_number is not None), (
                                'Probabilities, hence no classes.')
                    class_names = [
                        'class {}'.format(i)
                        for i in range(self.classes_number)
                    ]
                else:
                    # If unique predictions were provided by the user...
                    if unique_predictions is not None:
                        class_names = [
                            'class {}'.format(i) for i in unique_predictions
                        ]
                    # If the unique predictions were computed and the computed
                    # classes number agrees...
                    elif (unique_predictions is None
                          and _classes_number == self.classes_number):
                        class_names = [
                            'class {}'.format(i)
                            for i in _unique_predictions  # type: ignore
                        ]
                    else:
                        assert False, (  # pragma: nocover
                            'If class_names is None and the number of classes '
                            'does not agree with the detected classes '
                            '(unique_predictions is None) it means that it '
                            'had to be provided by the user, hence a '
                            'RuntimeError should have been raised.')
            else:
                if len(class_names) != self.classes_number:
                    raise ValueError('The length of the class_names list does '
                                     'not agree with the number of classes '
                                     '({}).'.format(self.classes_number))
        self.class_names = class_names

        if feature_names is None:
            feature_names = [
                'feature {}'.format(i)
                for i, _ in enumerate(self.column_indices)
            ]
        # The number of feature names is validated in the _input_is_valid
        # function.
        self.feature_names = feature_names

    def _explain_instance_input_is_valid(
            self, data_row: Union[np.ndarray, np.void]) -> bool:
        """
        Validates input parameters of the ``explain_instance`` method.

        This function checks the validity of the ``data_row``.

        For the description of exceptions raised by this method please see
        the documentation of the :func:`fatf.transparency.predictions.\
surrogate_explainers.SurrogateTabularExplainer.explain_instance` method.

        Parameters
        ----------
        data_row : Union[numpy.ndarray, numpy.void]
            A data point -- 1-dimensional numpy array.

        Returns
        -------
        is_valid : boolean
            ``True`` if the input parameters are valid, ``False`` otherwise.
        """
        is_valid = False

        if not fuav.is_1d_like(data_row):
            raise IncorrectShapeError('The data_row must either be a '
                                      '1-dimensional numpy array or numpy '
                                      'void object for structured rows.')

        are_similar = fuav.are_similar_dtype_arrays(
            self.dataset, np.array([data_row]), strict_comparison=True)
        if not are_similar:
            raise TypeError('The dtype of the data_row is different to '
                            'the dtype of the data array used to '
                            'initialise this class.')

        # If the dataset is structured and the data_row has a different
        # number of features this will be caught by the above dtype check.
        # For classic numpy arrays this has to be done separately.
        if not self.is_structured:
            if data_row.shape[0] != self.dataset.shape[1]:
                raise IncorrectShapeError('The data_row must contain the '
                                          'same number of features as the '
                                          'dataset used to initialise '
                                          'this class.')

        is_valid = True
        return is_valid

    @abc.abstractmethod
    def explain_instance(self, data_row: Union[np.ndarray, np.void]) -> Any:
        """
        Explains a ``data_row``.

        This is an abstract method that must be implemented for each child
        object of this abstract class.

        .. warning::

           The ``_explain_instance_input_is_valid`` method should be called in
           all implementations of the ``explain_instance`` method in the
           children classes to ensure that all of the input parameters passed
           to this method are valid.

        Parameters
        ----------
        data_row : Union[numpy.ndarray, numpy.void]
            A data point to be explained.

        Raises
        ------
        IncorrectShapeError
            The ``data_row`` is not a 1-dimensional numpy array-like object.
            The number of features (columns) in the ``data_row`` is different
            to the number of features in the data array used to initialise this
            object.
        NotImplementedError
            This is an abstract method, hence has not been implemented.
        TypeError
            The dtype of the ``data_row`` is different than the dtype of the
            data array used to initialise this object.

        Returns
        -------
        explanation : Any
            An explanation of the ``data_row``.
        """
        assert self._explain_instance_input_is_valid(  # pragma: nocover
            data_row), 'Invalid input'

        raise NotImplementedError('The explain_instance '  # pragma: nocover
                                  'abstract method needs to be implemented '
                                  'in the children classes.')


class TabularBlimeyLime(SurrogateTabularExplainer):
    """
    A tabular LIME explainer -- a surrogate explainer based on a linear model.

    .. versionchanged:: 0.1.0
       (1) Added support for regression models.
       (2) Changed the feature selection mechanism from k-LASSO to
       :func:`~fatf.utils.data.feature_selection.sklearn.forward_selection`
       when the number of selected features is less than 7, and
       :func:`~fatf.utils.data.feature_selection.sklearn.highest_weights`
       otherwise -- the default LIME behaviour.

    .. versionadded:: 0.0.2

    This class implements Local Interpretable Model-agnostic Explanations
    (LIME_) introduced by [RIBEIRO2016WHY]_. This implementation mirrors the
    one in the `official LIME package`_, which is available under the
    ``lime.lime_tabular.LimeTabularExplainer`` class therein.

    This explainer uses a quartile discretiser
    (:class:`fatf.utils.data.discretisation.QuartileDiscretiser`) and a normal
    sampler (:class:`fatf.utils.data.augmentation.NormalSampling`) for
    augmenting the data. The following steps are taken to generate the
    explanation (when the ``explain_instance`` method is called):

    * The input ``data_row`` is discretised using the quartile discretiser.
      The numerical features are binned and the categorical ones are left
      unchanged (selected via the ``categorical_indices`` parameter).
    * The data are sampled around the discretised ``data_row`` using the normal
      sampler. Since after the discretisation all of the features are
      categorical the bin indices are sampled based on their frequency
      in (the discretised version of) the ``dataset`` used to initialise this
      class.
    * The sampled data are reverted back to their original domain and predicted
      with the black-box model (``predictive_model`` used to initialise this
      class). This step is done via sampling each (numerical) feature value
      from the corresponding bin using the truncated normal distribution for
      which minimum (lower threshold), maximum (upper threshold), mean and
      standard deviation are computed empirically from all the data points from
      the ``dataset`` for which feature values fall into that bin. The
      categorical features are left unchanged.
    * The discretised sampled data set is binarised by comparing each row with
      the user-specified ``data_row`` (in the ``explain_instance`` method).
      This step is performed by taking XNOR logical operation between the two
      -- 1 if the feature value is the same in a row of the discretised data
      set and the ``data_row`` and 0 if it is different.
    * The Euclidean distance between the binarised sampled data and binarised
      ``data_row`` is computed and passed through an exponential kernel
      (:func:`fatf.utils.kernels.exponential_kernel`) to get similarity scores,
      which will be used as data point weights when reducing the number of
      features (see below) and training the linear regression.
    * To limit the number of features in the explanation (if enabled by the
      user) we either use *forward selection* when the number of selected
      features is less than 7 or *highest weights* otherwise. (This is
      controlled by the ``features_number`` parameter in the
      ``explain_instance`` method and by default -- ``features_number=None`` --
      all of the feature are used.)
    * A local (weighted) ridge regression (``sklearn.linear_model.Ridge``) is
      fitted to the sampled and binarised data with the target being:

      - The numerical predictions of the black-box model when the underlying
        model is a regression.
      - A vector of probabilities output by the black-box model for the
        selected class (one-vs-rest) when the underlying model is a
        probabilistic classifier. By default, one model is trained for all of
        the classes (``explained_class=None`` in the ``explain_instance``
        method), however the class to be explained can be specified by the
        user.

    .. note:: How to interpret the results?

       Because the local surrogate model is trained on the binarised sampled
       data that is parsed through the XNOR operation, the parameters extracted
       from this model (feature importances) should be interpreted as an answer
       to the following question:

           "Had this particular feature value of the explained data point
           been outside of this range (for numerical features) or had a
           different value (for categorical feature), how would that influence
           the *probability of this point belonging to the explained class*
           (probabilistic classification) / *predicted numerical value*
           (regression)?"

    This LIME implementation is limited to black-box
    **probabilistic classifiers** and **regressors** (similarly to the
    `official implementation`_). Therefore, the ``predictive_model`` must have
    a ``predict_proba`` method for probabilistic models and ``predict`` method
    for regressors. When the surrogate is built for a probabilistic classifier,
    the local model will be trained using the **one-vs-rest** approach since
    the output of the global model is an array with probabilities of each class
    (the classes to be explained can be selected using the ``explained_class``
    parameter in the ``explain_instance`` method). The column indices indicated
    as categorical features (via the ``categorical_indices`` parameter) will
    not be discretised.

    For detailed instructions on how to build a custom surrogate explainer
    (to avoid tinkering with this class) please see the
    :ref:`how_to_tabular_surrogates` *how-to guide*.

    For additional parameters, warnings and errors description please see the
    documentation of the parent class :class:`fatf.transparency.predictions.\
surrogate_explainers.SurrogateTabularExplainer`.

    .. _LIME: https://github.com/marcotcr/lime
    .. _`official LIME package`: https://github.com/marcotcr/lime
    .. _`official implementation`: https://github.com/marcotcr/lime/blob/
       master/lime/lime_tabular.py#L357

    .. [RIBEIRO2016WHY] Ribeiro, M.T., Singh, S. and Guestrin, C., 2016,
       August. Why should i trust you?: Explaining the predictions of any
       classifier. In Proceedings of the 22nd ACM SIGKDD international
       conference on knowledge discovery and data mining (pp. 1135-1144). ACM.

    Parameters
    ----------
    dataset : numpy.ndarray
        A 2-dimensional numpy array with a dataset (utilised in various ways
        throughout the explainer).
    predictive_model : object
        A pre-trained (black-box) predictive model to be explained. If
        ``as_probabilistic`` (see below) is set to ``True``, it must have a
        ``predict_proba`` method that takes a data set as the only required
        input parameter and returns a 2-dimensional numpy array with
        probabilities of belonging to each class. Otherwise, if
        ``as_probabilistic`` is set to ``False``, the ``predictive_model`` must
        have a ``predict`` method that outputs a 1-dimensional array with
        (class) predictions.
    as_regressor : boolean, optional (default=False)
        .. versionadded:: 0.1.0

        A boolean indicating whether the global model should be treated as
        regression (``True``) or probabilistic classification (``False``).
    categorical_indices : List[column indices], optional (default=None)
        A list of column indices in the input ``dataset`` that should be
        treated as categorical features.
    class_names : List[string], optional (default=None)
        A list of strings defining the names of classes. If the predictive
        model is probabilistic, the order of the class names should correspond
        to the order of columns output by the model. For other models the
        order should correspond to lexicographical ordering of all the possible
        outputs of this model. For example, if the model outputs
        ``['a', 'c', '0']`` the class names should be given for
        ``['0', 'a', 'c']`` ordering.
    feature_names : List[string], optional (default=None)
        A list of strings defining the names of the ``dataset`` features. The
        order of the names should correspond to the order of features in the
        ``dataset``.

    Raises
    ------
    ImportError
        The scikit-learn package is missing.

    Attributes
    ----------
    discretiser : fatf.utils.data.discretisation.Discretiser
        An instance of the quartile discretiser
        (:class:`fatf.utils.data.discretisation.QuartileDiscretiser`)
        initialised with the input ``dataset`` and used to discretise
        the ``data_row`` when the ``explain_instance`` method is called.
    augmenter : fatf.utils.data.augmentation.Augmentation
        An instance of the normal sampling augmenter
        (:class:`fatf.utils.data.augmentation.NormalSampling`)
        used to sample new data points around the discretised ``data_row``
        (in the ``explain_instance`` method).
    bin_sampling_values : Dictionary[dataset column index, \
Dictionary[discretised bin id, Tuple(float, float, float, float)]]
        A dictionary holding characteristics for each bin of each numerical
        feature. The characteristic are represented as a 4-tuple consisting of:
        the lower bin boundary, the upper bin boundary, the empirical mean of
        of all the values of this feature for data points (in ``dataset``)
        falling into that bin, and the empirical standard deviation (calculated
        in the same way). For the edge bins, if there are data available the
        lower edge is calculated empirically (as the minimum of the
        corresponding feature values falling into that bin), otherwise it is
        set to ``-numpy.inf``. The same applies to the upper edge, which is
        either set to ``numpy.inf`` or calculated empirically (as the maximum
        of the corresponding feature values falling into that bin).
        If there are no data points to calculate the mean and standard
        deviation for a given bin, these two values are set to ``numpy.nan``.
        (This does not influence the future reverse sampling, for which this
        attribute is used: since there were no data for a given bin, the
        frequency of data for that bin is 0, therefore no data falling into
        this bin will be sampled.)
    """

    # pylint: disable=too-few-public-methods

    def __init__(self,
                 dataset: np.ndarray,
                 predictive_model: object,
                 as_regressor: bool = False,
                 categorical_indices: Optional[List[Index]] = None,
                 class_names: Optional[List[str]] = None,
                 feature_names: Optional[List[str]] = None) -> None:
        """
        Constructs a ``TabularBlimeyLime`` class.
        """
        # pylint: disable=too-many-arguments,too-many-locals,too-many-branches

        if SKLEARN_MISSING:
            raise ImportError('The scikit-learn package is required to use '
                              'the TabularBlimeyLime explainer.')

        # The implementation of the tabular LIME explainer only supports
        # probabilistic classifiers and regressors, hence does not require
        # unique prediction values and the number of classes.
        # (as_probabilistic=True; classes_number=None; unique_predictions=None)
        super().__init__(dataset, predictive_model, True, as_regressor,
                         categorical_indices, class_names, None, feature_names,
                         None)

        self.discretiser = fudd.QuartileDiscretiser(
            dataset,
            categorical_indices=self.categorical_indices,
            feature_names=self.feature_names)

        # The discretiser transforms each numerical feature into a categorical
        # (integer) feature that indicates to which bin a particular value
        # belongs.
        dataset_discretised = self.discretiser.discretise(self.dataset)

        # Since the dataset is discretised, the NormalSampling augmenter will
        # sample unique values for each column (categorical values for
        # categorical features and bin IDs for numerical features)
        # proportionally to their frequency in the discretised dataset.
        self.augmenter = fuda.NormalSampling(
            dataset_discretised, categorical_indices=self.column_indices)

        # Get empirical characteristics (minimum, maximum, mean and standard
        # deviation) for each bin of a numerical feature. If a bin is empty
        # (no data), use the quartile boundaries.
        bin_sampling_values = {}  # type: BinSamplingValues
        for index in self.numerical_indices:
            bin_sampling_values[index] = {}

            if self.is_structured:
                discretised_feature = dataset_discretised[index]
                feature = self.dataset[index]
            else:
                discretised_feature = dataset_discretised[:, index]
                feature = self.dataset[:, index]

            # The bin IDs need to be sorted as they are retrieved from
            # dictionary keys (hence may come in a random order), therefore
            # interfering with the enumerate procedure.
            bin_ids = sorted(
                list(self.discretiser.feature_value_names[index].keys()))
            bin_boundaries = self.discretiser.feature_bin_boundaries[index]
            for bin_i, bin_id in enumerate(bin_ids):
                bin_feature_indices = (discretised_feature == bin_id)
                bin_feature_values = feature[bin_feature_indices]

                # If there is data in the bin, get its empirical mean and
                # standard deviation, otherwise use numpy nan.
                # If there are no data in a bin, the frequency of this bin
                # will be 0, therefore data will never get sampled from this
                # bin, i.e., there will be no attempt to undiscretised them.
                if bin_feature_values.size:
                    mean_val = bin_feature_values.mean()
                    std_val = bin_feature_values.std()
                else:
                    mean_val = np.nan
                    std_val = np.nan

                # Use the true bin boundaries (extracted from the discretiser).
                # For the edge bins (with -inf and +inf edges) use the
                # empirical minimum and maximum (if possible) to avoid problems
                # with reverse sampling (see the _undiscretise_data method).
                if bin_i == 0:
                    if bin_feature_values.size:
                        min_val = bin_feature_values.min()
                    else:
                        min_val = -np.inf  # pragma: nocover
                        assert False, (  # pragma: nocover
                            'Since the upper bin boundary is inclusive in '
                            'the quartile discretiser this can never happen.')
                    max_val = bin_boundaries[bin_i]
                # This is bin id count (+1) and not bind boundary count.
                elif bin_i == bin_boundaries.shape[0]:
                    min_val = bin_boundaries[bin_i - 1]
                    if bin_feature_values.size:
                        max_val = bin_feature_values.max()
                    else:
                        max_val = np.inf
                else:
                    min_val = bin_boundaries[bin_i - 1]
                    max_val = bin_boundaries[bin_i]

                bin_sampling_values[index][bin_id] = (min_val, max_val,
                                                      mean_val, std_val)
        self.bin_sampling_values = bin_sampling_values

    def _explain_instance_input_is_valid(  # type: ignore
            self, data_row: Union[np.ndarray, np.void],
            explained_class: Union[None, int, str], samples_number: int,
            features_number: Union[None, int],
            kernel_width: Union[None, float], return_models: bool) -> bool:
        """
        Validates the input parameters of the ``explain_instance`` method.

        For additional documentation of the input parameters, warnings and
        errors please see the description of :func:`fatf.transparency.\
predictions.surrogate_explainers.TabularBlimeyLime.explain_instance` method.

        Returns
        -------
        is_valid : boolean
            ``True`` if the input parameters are valid, ``False`` otherwise.
        """
        # pylint: disable=too-many-arguments,too-many-branches
        # pylint: disable=arguments-differ
        is_valid = False
        assert super()._explain_instance_input_is_valid(data_row)

        if explained_class is not None:
            if not isinstance(explained_class, (int, str)):
                raise TypeError('The explained_class parameter must be either '
                                'None, a string or an integer.')

        if isinstance(samples_number, int):
            if samples_number < 1:
                raise ValueError('The samples_number parameter must be a '
                                 'positive integer (larger than 0).')
        else:
            raise TypeError('The samples_number parameter must be an integer.')

        if features_number is not None:
            if isinstance(features_number, int):
                if features_number < 1:
                    raise ValueError('The features_number parameter must be a '
                                     'positive integer (larger than 0).')
            else:
                raise TypeError('The features_number parameter must either be '
                                'None or an integer.')

        if kernel_width is not None:
            if isinstance(kernel_width, Number):
                if kernel_width <= 0:
                    raise ValueError('The kernel_width parameter must be a '
                                     'positive float (larger than 0).')
            else:
                raise TypeError('The kernel_width parameter must either be '
                                'None or a float.')

        if not isinstance(return_models, bool):
            raise TypeError('The return_models parameter must be a boolean.')

        is_valid = True
        return is_valid

    def _undiscretise_data(self, discretised_data: np.ndarray) -> np.ndarray:
        """
        Transforms discretised data set into its original representation.

        The ``discretised_data`` are reverted back to their original domain by
        sampling each (numerical) feature value from the corresponding bin
        using the truncated normal distribution for which minimum (lower
        threshold), maximum (upper threshold), mean and standard deviation were
        computed empirically from all the data points in the ``dataset``
        (used to initialise this class) for which feature values fall into that
        bin. The categorical features are left unchanged.

        This method mimics the "un-discretisation" procedure done by the
        `official LIME implementation`_.

        .. _`official LIME implementation`: https://github.com/marcotcr/lime/
          blob/master/lime/discretize.py

        Parameters
        ----------
        discretised_data : numpy.ndarray
            A discretised data set to be reverted back to the original
            representation (domain).

        Returns
        -------
        undiscretised_data : numpy.ndarray
            The ``discretised_data`` reverted back to the original
            representation (domain).
        """
        # pylint: disable=too-many-locals
        assert fuav.is_2d_array(discretised_data), 'Not a 2-D array.'

        # Create a placeholder for undiscretised data. We copy the discretised
        # array instead of creating an empty one to preserve the values of
        # sampled categorical features, hence we do not need to copy them
        # later on. We also need to change the type of the array to correspond
        # to the original dataset.
        undiscretised_data = discretised_data.copy().astype(self.dataset.dtype)

        for index in self.numerical_indices:
            if self.is_structured:
                discretised_column = discretised_data[index]
                undiscretised_column = undiscretised_data[index]
            else:
                discretised_column = discretised_data[:, index]
                undiscretised_column = undiscretised_data[:, index]

            unique_column_values = np.unique(discretised_column)
            for bin_id, bin_values in self.bin_sampling_values[index].items():
                if bin_id in unique_column_values:
                    # Since sampling values must have been found in this bin,
                    # there should be an empirical mean (2) and
                    # standard deviation (3).
                    # yapf: disable
                    assert (self.bin_sampling_values[index][bin_id][2]
                            is not np.nan), ('No empirical mean for a bin '
                                             'without data points.')
                    assert (self.bin_sampling_values[index][bin_id][3]
                            is not np.nan), ('No empirical standard deviation '
                                             'for a bin without data points.')
                    # yapf: enable

                    bin_indices = np.where(discretised_column == bin_id)[0]
                    samples_number = bin_indices.shape[0]

                    min_, max_, mean_, std_ = bin_values
                    if std_:
                        lower_bound = (min_ - mean_) / std_
                        upper_bound = (max_ - mean_) / std_

                        unsampled = scipy.stats.truncnorm.rvs(
                            lower_bound,
                            upper_bound,
                            loc=mean_,
                            scale=std_,
                            size=samples_number)
                    else:
                        unsampled = np.array(samples_number * [mean_])

                    undiscretised_column[bin_indices] = unsampled

        return undiscretised_data

    def explain_instance(self,
                         data_row: Union[np.ndarray, np.void],
                         explained_class: Optional[Union[int, str]] = None,
                         samples_number: int = 50,
                         features_number: Optional[int] = None,
                         kernel_width: Optional[float] = None,
                         return_models: bool = False) -> ExplanationTuple:
        """
        Explains the ``data_row`` with linear regression feature importance.

        .. versionchanged:: 0.1.0
           Changed the feature selection mechanism from k-LASSO to
           :func:`~fatf.utils.data.feature_selection.sklearn.forward_selection`
           when the number of selected features is less than 7, and
           :func:`~fatf.utils.data.feature_selection.sklearn.highest_weights`
           otherwise -- the default LIME behaviour.

        For probabilistic classifiers the explanations will be produced for all
        of the classes by default. This can be changed by selecting a specific
        class with the ``explained_class`` parameter.

        The default ``kernel_width`` is computed as the square root of the
        number of features multiplied by 0.75. Also, by default, all of the
        (interpretable) features will be used to create an explanation, which
        can be limited by setting the ``features_number`` parameter. The data
        sampling around the ``data_row`` can be customised by specifying the
        number of points to be generated (``samples_number``).

        By default, this method only returns feature importance, however
        by setting ``return_models`` to ``True``, it will also return the local
        linear surrogates for further analysis and processing done outside of
        this method.

        .. note::

           The exact description of the explanation generation procedure can
           be found in the documentation of this class (:class:`fatf.\
transparency.predictions.surrogate_explainers.TabularBlimeyLime`).

        For additional parameters, warnings and errors please see the parent
        class method :func:`fatf.transparency.predictions.\
surrogate_explainers.SurrogateTabularExplainer.explain_instance`.

        Parameters
        ----------
        data_row : Union[numpy.ndarray, numpy.void]
            A data point to be explained (1-dimensional numpy array).
        explained_class : Union[integer, string], optional (default=None)
            The class to be explained -- only applicable to probabilistic
            classifiers. If ``None``, all of the classes will be
            explained. This can either be the index of the class (the column
            index of the probabilistic vector) or the class name (taken from
            ``self.class_names``).
        samples_number : integer, optional (default=50)
            The number of data points sampled from the normal augmenter, which
            will be used to fit the local surrogate model.
        features_number : integer, optional (default=None)
            The maximum number of (interpretable) features -- found with
            *forward selection* or *highest weights* -- to be used in the
            explanation (the local surrogate model is trained with this feature
            subset). By default (``None``), all of the (interpretable) features
            are used.
        kernel_width : float, optional (default=None)
            The width of the exponential kernel used when computing weights of
            the sampled data based on the distances between the sampled data
            and the ``data_row``.The default ``kernel_width``
            (``kernel_width=None``) is computed as the square root of the
            number of features multiplied by 0.75.
        return_models : boolean, optional (default=False)
            If ``True``, this method will return both the feature importance
            explanation dictionary and a dictionary holding the local models.
            Otherwise, only the first dictionary will be returned.

        Raises
        ------
        TypeError
            The ``explained_class`` parameter is neither ``None``, an integer
            or a string. The ``samples_number`` parameter is not an integer.
            The ``features_number`` parameter is neither ``None`` nor an
            integer. The ``kernel_width`` parameter is neither ``None`` nor
            a number. The ``return_models`` parameter is not a boolean.
        ValueError
            The ``samples_number`` parameter is a non-positive integer (smaller
            than 1). The ``features_number`` parameter is a non-positive
            integer (smaller than 1). The ``kernel_width`` parameter is a
            non-positive number (smaller or equal to 0).
            The ``explained_class`` specified by the user could neither be
            recognised as one of the allowed class names (``self.class_names``)
            nor an index of a class name.

        Returns
        -------
        explanations : Dictionary[string, Dictionary[string, float]]
            A dictionary holding dictionaries that contain feature
            importance -- where the feature names are taken from
            ``self.feature_names`` and the feature importances are extracted
            from local linear surrogates. These dictionaries are held under
            keys corresponding to class names (taken from
            ``self.class_names``).
        models : sklearn.linear_model.base.LinearModel, optional
            A dictionary holding locally fitted surrogate linear models
            held under class name keys (taken from ``self.class_names``).
            This dictionary is only returned when the ``return_models``
            parameter is set to ``True``.
        """
        # pylint: disable=too-many-arguments,too-many-locals,too-many-branches
        # pylint: disable=arguments-differ,too-many-statements
        assert self._explain_instance_input_is_valid(
            data_row, explained_class, samples_number, features_number,
            kernel_width, return_models), 'Invalid input.'

        dataset_features_number = len(self.column_indices)

        if kernel_width is None:
            kernel_width = np.sqrt(dataset_features_number) * 0.75

        # Discretise data row
        data_row_discretised = self.discretiser.discretise(data_row)
        # Sample around the discretised data row (in the discretised domain)
        sampled_data_discretised = self.augmenter.sample(
            data_row_discretised, samples_number=samples_number)

        # Revert back the sampled data into the original domain
        sampled_data = self._undiscretise_data(sampled_data_discretised)

        # Get predictions of the sampled data
        sampled_data_predictions = self.predictive_function(sampled_data)

        # Binarise the sampled data, i.e., XNOR (in the discretised domain)
        # The value will be 1 if the same as in the data_row and 0 if different
        binarised_data = fudt.dataset_row_masking(sampled_data_discretised,
                                                  data_row_discretised)
        binarised_data = fuat.as_unstructured(binarised_data)

        # Get similarity measure (weights) by kernelising the distance
        # (in the binary domain). The data_row is represented as an all-1
        # vector in the binarised domain as 1s indicate that it lies in all.
        # of the numerical bins and has the came categorical feature values
        # as the original data_row.
        distances = fud.euclidean_point_distance(
            np.ones(dataset_features_number), binarised_data)
        weights = fatf_kernels.exponential_kernel(
            distances, width=kernel_width)

        # Get feature names for the binarised domain
        binarised_data_feature_names = []
        for i, index in enumerate(self.column_indices):
            feature_value = data_row_discretised[index]

            if index in self.discretiser.numerical_indices:
                feature_value = feature_value.astype(int)
                binarised_data_feature_names.append(
                    self.discretiser.feature_value_names[index][feature_value])
            else:
                binarised_data_feature_names.append('*{}* = {}'.format(
                    self.feature_names[i], feature_value))

        # Get classes to be explained
        if self.as_regressor:
            classes_to_explain = [None]  # type: List[Union[None, int]]
        else:
            assert (self.classes_number is not None
                    and self.class_names is not None)
            if explained_class is None:
                classes_to_explain = list(range(self.classes_number))
            elif isinstance(explained_class, str):
                if explained_class not in self.class_names:
                    raise ValueError('The *{}* explained class name was not '
                                     'recognised. The following class names '
                                     'are allowed: {}.'.format(
                                         explained_class, self.class_names))
                # Translate the class name into a probability index
                classes_to_explain = [self.class_names.index(explained_class)]
            elif isinstance(explained_class, int):
                if (explained_class < 0  # yapf: disable
                        or explained_class >= self.classes_number):
                    raise ValueError('The explained class index is out of the '
                                     'allowed range: 0 to {} (there are {} '
                                     'classes altogether).'.format(
                                         self.classes_number - 1,
                                         self.classes_number))
                classes_to_explain = [explained_class]
            else:
                assert False, (  # pragma: nocover
                    'Cannot be anything else but None, string or int.')

        # Filter features
        if features_number is None:
            features_number = len(self.column_indices)
        if features_number < 7:
            _feature_selection_algo = 'forward selection'
        else:
            _feature_selection_algo = 'highest weights'
        logger.info('Selecting %d features with %s.', features_number,
                    _feature_selection_algo)

        # Generate the explanations
        if self.as_regressor:
            assert len(classes_to_explain) == 1
            assert classes_to_explain[0] is None
        else:
            assert self.as_probabilistic, (
                'The loop below assumes the sampled_data_predictions array to '
                'be a vector of probabilities since the '
                'self.predictive_function has to be probabilistic -- this '
                'implementation of LIME does not support non-probabilistic '
                'classifiers.')
        explanations = {}
        models = {}
        for class_index in classes_to_explain:
            if self.as_regressor:
                assert class_index is None
                class_name = None
                predictions = sampled_data_predictions
            else:
                assert self.class_names is not None and class_index is not None
                class_name = self.class_names[class_index]
                # Select the predictions of the class to be explained
                predictions = sampled_data_predictions[:, class_index]

            # Filter features
            if features_number < 7:
                selected_indices = fudfs.forward_selection(
                    binarised_data, predictions, weights, features_number)
            else:
                selected_indices = fudfs.highest_weights(
                    binarised_data, predictions, weights, features_number)
            selected_training_data = binarised_data[:, selected_indices]
            # The returned indices can either be strings (structured) or
            # integers (classic). In this case they have to be integers because
            # the discretised data set is classic.
            assert isinstance(selected_indices[0],
                              np.integer), 'Classic array.'
            selected_feature_names = [
                binarised_data_feature_names[i]  # type: ignore
                for i in selected_indices
            ]

            # Train the local (weighted) ridge regression and use our
            # linear model explainer to generate explanations.
            local_model = sklearn.linear_model.Ridge()
            local_model.fit(
                selected_training_data, predictions, sample_weight=weights)

            # Get a linear model explainer
            explainer = ftslm.SKLearnLinearModelExplainer(
                local_model, feature_names=selected_feature_names)

            # Extract feature importance explanations from the linear model
            assert explainer.feature_names is not None, 'Defined above.'

            if self.as_regressor:
                assert class_index is None and class_name is None
                explanations = dict(
                    zip(explainer.feature_names,
                        explainer.feature_importance()))
                models = local_model
            else:
                assert class_name is not None
                explanations[class_name] = dict(
                    zip(explainer.feature_names,
                        explainer.feature_importance()))
                models[class_name] = local_model

        if return_models:
            return_ = (explanations, models)  # type: ExplanationTuple
        else:
            return_ = explanations
        return return_


class TabularBlimeyTree(SurrogateTabularExplainer):
    """
    A surrogate explainer based on a decision tree.

    .. versionchanged:: 0.1.0
       Added support for regression models.

    .. versionadded:: 0.0.2

    This explainer does not use an interpretable data representation (as one
    is learnt by the tree). The data augmentation is done with *Mixup*
    (:class:`fatf.utils.data.augmentation.Mixup`) around the data point
    specified in the ``explain_instance`` method. No data weighting procedure
    is used when fitting the local surrogate model.

    When ``as_regressor`` is set to ``True``, a surrogate regression tree is
    fitted to a black-box regression. When it is set to ``False``,
    ``predictive_model`` is treated as a classifier. When the underlying
    predictive model is probabilistic (``as_probabilistic=True``), the local
    decision tree is trained as a regressor of probabilities output by the
    black-box ``predictive_model``. When the ``predictive_model`` is a
    non-probabilistic classifier, the local decision tree is a classifier that
    is either fitted as one-vs-rest for a selected class or mimics the
    classification problem by fitting a multi-class classifier.

    The explanation output by the ``explain_instance`` method is a simple
    feature importance measure extracted from the local tree. Alternatively,
    the local tree model can be returned for further processing or visualising.

    Since this explainer is based on scikit-learn's implementation of decision
    trees, it does not support structured arrays and categorical (text-based)
    features.

    For additional parameters, warnings and errors please see the documentation
    of the parent class: :class:`fatf.transparency.predictions.\
surrogate_explainers.SurrogateTabularExplainer`.

    Raises
    ------
    ImportError
        The scikit-learn package is missing.
    TypeError
        The ``dataset`` parameter is a structured array or the ``dataset``
        contains non-numerical features.

    Attributes
    ----------
    augmenter : fatf.utils.data.augmentation.Augmentation
        The augmenter class (:class:`fatf.utils.data.augmentation.Mixup`) used
        for local data sampling.
    """

    # pylint: disable=too-few-public-methods

    def __init__(self,
                 dataset: np.ndarray,
                 predictive_model: object,
                 as_probabilistic: bool = True,
                 as_regressor: bool = False,
                 categorical_indices: Optional[List[Index]] = None,
                 class_names: Optional[List[str]] = None,
                 classes_number: Optional[int] = None,
                 feature_names: Optional[List[str]] = None,
                 unique_predictions: Optional[List[Union[str, int]]] = None
                 ) -> None:
        """
        Constructs a ``TabularBlimeyTree`` class.
        """
        # pylint: disable=too-many-arguments

        if SKLEARN_MISSING:
            raise ImportError('The scikit-learn package is required to use '
                              'the TabularBlimeyTree explainer.')

        super().__init__(dataset, predictive_model, as_probabilistic,
                         as_regressor, categorical_indices, class_names,
                         classes_number, feature_names, unique_predictions)

        if self.is_structured:
            raise TypeError('The TabularBlimeyTree explainer does not support '
                            'structured data arrays as it uses scikit-learn '
                            'implementation of decision trees.')

        # Check if the dtype of the dataset is string-based. (We can check the
        #  dtype since this is a plane numpy array.)
        if fuav.is_textual_dtype(self.dataset.dtype):
            raise TypeError('The TabularBlimeyTree explainer does not support '
                            'data sets that have a string-based dtype as it '
                            'uses scikit-learn implementation of decision '
                            'trees.')

        # Get predictions for the input data set to initialise Mixup
        if self.as_regressor:
            dataset_predictions = None
        else:
            dataset_predictions = self.predictive_function(self.dataset)
            if self.as_probabilistic:
                dataset_predictions = dataset_predictions.argmax(axis=1)

        # Initialise the Mixup augmenter
        self.augmenter = fuda.Mixup(
            dataset=self.dataset,
            ground_truth=dataset_predictions,
            categorical_indices=self.categorical_indices,
            beta_parameters=(2, 5),
            int_to_float=True)

    def _explain_instance_input_is_valid(  # type: ignore
            self, data_row: Union[np.ndarray, np.void],
            explained_class: Union[int, str, None], one_vs_rest: bool,
            samples_number: int, maximum_depth: int,
            return_models: bool) -> bool:
        """
        Validates the input parameters of the ``explain_instance`` method.

        For additional documentation of the input parameters, warnings and
        errors please see the description of :func:`fatf.transparency.\
predictions.surrogate_explainers.TabularBlimeyTree.explain_instance` method.

        Returns
        -------
        is_valid : boolean
            ``True`` if the input parameters are valid, ``False`` otherwise.
        """
        # pylint: disable=too-many-arguments
        # pylint: disable=arguments-differ
        is_valid = False
        assert super()._explain_instance_input_is_valid(data_row)

        if explained_class is not None:
            if not isinstance(explained_class, (int, str)):
                raise TypeError('The explained_class parameter must be either '
                                'None, a string or an integer.')

        if not isinstance(one_vs_rest, bool):
            raise TypeError('The one_vs_rest parameter must be either '
                            'None or a boolean.')

        if isinstance(samples_number, int):
            if samples_number < 1:
                raise ValueError('The samples_number parameter must be a '
                                 'positive (larger than 0) integer.')
        else:
            raise TypeError('The samples_number parameter must be an integer.')

        if isinstance(maximum_depth, int):
            if maximum_depth < 1:
                raise ValueError('The maximum_depth parameter must be a '
                                 'positive (larger than 0) integer.')
        else:
            raise TypeError('The maximum_depth parameter must be an integer.')

        if not isinstance(return_models, bool):
            raise TypeError('The return_models parameter must be a boolean.')

        is_valid = True
        return is_valid

    def _get_local_model(self, sampled_data: np.ndarray,
                         sampled_data_predictions: np.ndarray,
                         selected_class_index: int, one_vs_rest: bool,
                         maximum_depth: int) -> ReturnTree:  # type: ignore
        """
        Fits a local tree surrogate.

        The tree is a classifier if the ``self.predictive_model`` is
        non-probabilistic (``self.as_probabilistic=False`` and
        ``self.as_regressor=False``). Otherwise, it is a regressor.

        For additional documentation of the input parameters, warnings and
        errors please see the description of :func:`fatf.transparency.\
predictions.surrogate_explainers.TabularBlimeyTree.explain_instance` method.

        Parameters
        ----------
        sampled_data : numpy.ndarray
            A data set used to fit the local decision tree.
        sampled_data_predictions : numpy.ndarray
            A ground truth used to fit the local decision tree. For
            probabilistic models the probabilities column with the
            ``selected_class_index`` parameter.
        selected_class_index : integer
            The index of the probability column for probabilistic models and
            the class index (based on the ``self.unique_predictions``
            attribute) of the possible classes for non-probabilistic models,
            which will be used as a target when fitting the local model.
        one_vs_rest : boolean
            A boolean indicating whether the local model should be fitted as
            one-vs-rest (required for probabilistic models) or as a multi-class
            classifier.
        maximum_depth : integer
            The maximum depth of the local decision tree surrogate model. The
            lower the number the smaller the decision tree, therefore making it
            (and the resulting explanations) more comprehensible.

        Returns
        -------
        local_model : sklearn.tree.tree.BaseDecisionTree
            A locally fitted decision tree classifier or regressor.
        """
        # pylint: disable=too-many-arguments

        if self.as_regressor:
            assert self.classes_number is None
            assert fuav.is_1d_array(sampled_data_predictions), 'Numbers.'

            local_model = sklearn.tree.DecisionTreeRegressor(
                max_depth=maximum_depth)
            local_model.fit(sampled_data, sampled_data_predictions)
        else:
            assert (isinstance(selected_class_index, int)
                    and self.classes_number is not None
                    and self.classes_number > selected_class_index >= 0
                    ), 'Must be a correct class index.'
            assert self.class_names is not None

            if self.as_probabilistic:
                assert one_vs_rest is True, ('Probabilistic must be '
                                             'one-vs-rest.')
                assert fuav.is_2d_array(sampled_data_predictions), (
                    'Probabilities.')

                predictions = (
                    sampled_data_predictions[:, selected_class_index])

                local_model = sklearn.tree.DecisionTreeRegressor(
                    max_depth=maximum_depth)
                local_model.fit(sampled_data, predictions)
            else:
                assert fuav.is_1d_array(sampled_data_predictions), 'Classes.'
                if one_vs_rest:
                    assert self.unique_predictions is not None, (
                        'Unique predictions list is needed for one-vs-rest '
                        'surrogate fitted for a non-probabilistic black-box '
                        'model.')

                    one_class = self.unique_predictions[selected_class_index]
                    one_index = (sampled_data_predictions == one_class)
                    if not one_index.any():
                        one_class_name = self.class_names[selected_class_index]
                        raise RuntimeError('A surrogate for the *{}* class '
                                           '(class index: {}; class name: '
                                           '{}) could not be fitted as none '
                                           'of the sampled data points were '
                                           'predicted (by the black-box '
                                           'model) as this particular '
                                           'class.'.format(
                                               one_class, selected_class_index,
                                               one_class_name))

                    predictions = np.zeros_like(
                        sampled_data_predictions, dtype=np.int16)
                    # Disable pylint's unsupported-assignment-operation (E1137)
                    predictions[one_index] = 1  # pylint: disable=E1137
                else:
                    predictions = sampled_data_predictions

                    # If all of the data points were predicted as a single
                    # class the local model will not be meaningful.
                    assert fuav.is_1d_array(predictions), ('1-D class '
                                                           'predictions.')
                    unique_predictions = np.unique(predictions)
                    if unique_predictions.shape[0] < 2:
                        raise RuntimeError('A surrogate model (classifier) '
                                           'could not be fitted as the '
                                           '(black-box) predictions for the '
                                           'sampled data are of a single '
                                           'class: *{}*.'.format(
                                               unique_predictions[0]))

                local_model = sklearn.tree.DecisionTreeClassifier(
                    max_depth=maximum_depth)
                local_model.fit(sampled_data, predictions)

        return local_model

    def explain_instance(self,
                         data_row: Union[np.ndarray, np.void],
                         explained_class: Optional[Union[int, str]] = None,
                         one_vs_rest: bool = True,
                         samples_number: int = 50,
                         maximum_depth: int = 3,
                         return_models: bool = False) -> ExplanationTuple:
        """
        Explains the ``data_row`` with decision tree feature importance.

        If the black-box model is a classifier, the explanations will be
        produced for all of the classes by default. This behaviour can be
        changed by selecting a specific class with the ``explained_class``
        parameter. For black-box classifiers, the local tree is learnt as
        a one-vs-rest (for one class at a time or only for the selected class)
        model by default. This is a requirement for probabilistic black-box
        models as the local model has to be a regression (tree) of
        probabilities for a selected class. However, when the black-box model
        is a non-probabilistic classifier, the local tree can either be learnt
        as one-vs-rest or multi-class (chosen by setting the ``one_vs_rest``
        parameter). The depth of the local tree can also be limited to improve
        its comprehensiveness by setting the ``maximum_depth`` parameter.

        The data sampling around the ``data_row`` can be customised by
        specifying the number of points to be generated (``samples_number``).
        By default, this method only returns feature importance, however
        by setting ``return_models`` to ``True``, it will also return the local
        tree surrogates for further analysis and processing done outside of
        this method.

        For additional parameters, warnings and errors please see the parent
        class method :func:`fatf.transparency.predictions.\
surrogate_explainers.SurrogateTabularExplainer.explain_instance`.

        Parameters
        ----------
        data_row : Union[numpy.ndarray, numpy.void]
            A data point to be explained (1-dimensional numpy array).
        explained_class : Union[integer, string], optional (default=None)
            The class to be explained. This parameter is ignored when the
            black-box model is a regressor. If ``None``, all of the classes
            will be explained. For probabilistic (black-box) models this can
            either be the index of the class (the column index of the
            probabilistic vector) or the class name (taken from
            ``self.class_names``). For non-probabilistic (black-box) models
            this can either be the name of the class (taken from
            ``self.class_names``), the prediction value (taken from
            ``self.unique_predictions``) or the index of any of these two
            (assuming the lexicographical ordering of the unique predictions
            output by the model).
        one_vs_rest : boolean, optional (default=True)
            A boolean indicating whether the local model should be fitted as
            one-vs-rest (required for probabilistic models) or as a multi-class
            classifier. This parameter is ignored when the black-box model is
            a regressor.
        samples_number : integer, optional (default=50)
            The number of data points sampled from the Mixup augmenter, which
            will be used to fit the local surrogate model.
        maximum_depth : integer, optional (default=3)
            The maximum depth of the local decision tree surrogate model. The
            lower the number the smaller the decision tree, therefore making it
            (and the resulting explanations) more comprehensible.
        return_models : boolean, optional (default=False)
            If ``True``, this method will return both the feature importance
            explanation dictionary and a dictionary holding the local models.
            Otherwise, only the first dictionary will be returned.

        Warns
        -----
        UserWarning
            The ``one_vs_rest`` parameter was set to ``True`` for an
            explainer that is based on a probabilistic (black-box) model.
            This is not possible, hence the ``one_vs_rest`` parameter will be
            overwritten to ``False``.
            Choosing a class to be explained (via the ``explained_class``
            parameter) is not required when requesting a multi-class local
            classifier (``one_vs_rest=False``) for a non-probabilistic
            black-box model since all of the classes will share a single
            surrogate.

        Raises
        ------
        RuntimeError
            A surrogate cannot be fitted as the (black-box) predictions for
            the sampled data are of a single class or do not have the requested
            class in case of the one-vs-rest local model (only applies to
            black-box models that are non-probabilistic classifiers
            (``self.as_probabilistic=False`` and ``self.as_regressor=False``).
        TypeError
            The ``explained_class`` parameter is neither ``None``, a string or
            an integer. The ``one_vs_rest`` parameter is not a boolean.
            The ``samples_number`` parameter is not an integer.
            The ``maximum_depth`` parameter is not an integer.
            The ``return_models`` parameter is not a boolean.
        ValueError
            The ``samples_number`` parameter is not a positive integer (larger
            than 0). The ``maximum_depth`` parameter is not a positive integer
            (larger than 0).
            The ``explained_class`` parameter is not recognised. For
            probabilistic (black-box) models this means that it could neither
            be recognised as a class name (``self.class_names``) nor an index
            of a class name. For non-probabilistic (black-box) models this
            means that it could neither be recognised as on of the possible
            predictions (``self.unique_predictions``) or a class name
            (``self.class_names``) nor as an index of either of these two.

        Returns
        -------
        explanations : Dictionary[string, Dictionary[string, float]]
            A dictionary holding dictionaries that contain feature
            importance -- where the feature names are taken from
            ``self.feature_names`` and the feature importances are extracted
            from local surrogate trees. These dictionaries are held under keys
            corresponding to class names (taken from ``self.class_names``).
        models : sklearn.tree.tree.BaseDecisionTree, optional
            A dictionary holding locally fitted surrogate decision tree models
            held under class name keys (taken from ``self.class_names``).
            This dictionary is only returned when the ``return_models``
            parameter is set to ``True``.
        """
        # pylint: disable=too-many-branches,too-many-statements
        # pylint: disable=too-many-arguments,too-many-locals
        # pylint: disable=arguments-differ
        assert self._explain_instance_input_is_valid(
            data_row, explained_class, one_vs_rest, samples_number,
            maximum_depth, return_models), 'Invalid input.'

        sampled_data = self.augmenter.sample(
            data_row, samples_number=samples_number)
        sampled_data_predictions = self.predictive_function(sampled_data)

        if self.as_regressor:
            explained_class_index = None  # type: Union[None, int]
            explained_class_name = None  # type: Union[None, str]
        else:
            assert (self.class_names is not None
                    and self.classes_number is not None)
            if self.as_probabilistic:
                if one_vs_rest is False:
                    warnings.warn(
                        'The one_vs_rest parameter cannot be set to False for '
                        'probabilistic models, since a regression tree is '
                        'fitted to the probabilities of each (or just the '
                        'selected) class. This parameter setting will be '
                        'ignored. Please see the documentation of the '
                        'TabularBlimeyTree class for more details.',
                        UserWarning)
                    one_vs_rest = True

                # Validate the explained_class parameter
                if isinstance(explained_class, str):
                    if explained_class not in self.class_names:
                        raise ValueError('The *{}* explained class name was '
                                         'not recognised. The following class '
                                         'names are allowed: {}.'.format(
                                             explained_class,
                                             self.class_names))
                    # Translate the class name into a probability index
                    explained_class_name = explained_class
                    explained_class_index = self.class_names.index(
                        explained_class)
                elif isinstance(explained_class, int):
                    if (explained_class < 0
                            or explained_class >= self.classes_number):
                        raise ValueError('The explained class index is out of '
                                         'the allowed range: 0 to {} (there '
                                         'are {} classes altogether).'.format(
                                             self.classes_number - 1,
                                             self.classes_number))
                    # Translate the probability index into a class name
                    explained_class_index = explained_class
                    explained_class_name = self.class_names[explained_class]
                elif explained_class is None:
                    explained_class_index = None
                    explained_class_name = None
                else:
                    assert False, (  # pragma: nocover
                        'Can only be None, a string or an int.')
            else:
                assert self.unique_predictions is not None, (
                    'Not probabilistic.')
                if one_vs_rest is False and explained_class is not None:
                    warnings.warn(
                        'Choosing a class to explain (via the explained_class '
                        'parameter) when the one_vs_rest parameter is set to '
                        'False is not required as a single multi-class '
                        'classification tree will be learnt regardless of the '
                        'explained_class parameter setting.', UserWarning)

                # Validate the explained_class parameter
                if explained_class is None:
                    explained_class_index = None
                    explained_class_name = None
                elif explained_class in self.unique_predictions:
                    logger.debug('Using the explained_class parameter as a '
                                 '*unique prediction* name for a classifier.')
                    explained_class_index = self.unique_predictions.index(
                        explained_class)
                    explained_class_name = (
                        self.class_names[explained_class_index])
                elif explained_class in self.class_names:
                    assert isinstance(explained_class, str), ('Class names '
                                                              'are str.')
                    logger.debug('Using the explained_class parameter as a '
                                 '*class name* for a classifier.')
                    explained_class_index = self.class_names.index(
                        explained_class)
                    explained_class_name = explained_class
                elif isinstance(explained_class, int):
                    if (explained_class < 0
                            or explained_class >= self.classes_number):
                        raise ValueError('The explained_class parameter was '
                                         'not recognised as one of the '
                                         'possible class names and when '
                                         'treated as a class name index, it '
                                         'is outside of the allowed range: 0 '
                                         'to {} (there are {} classes '
                                         'altogether).'.format(
                                             self.classes_number - 1,
                                             self.classes_number))
                    logger.debug('Using the explained_class parameter as a '
                                 'class index for a classifier.')
                    explained_class_index = explained_class
                    explained_class_name = self.class_names[explained_class]
                else:
                    raise ValueError('The explained_class was not recognised. '
                                     'The following predictions: {}; and '
                                     'class names are allowed: {}. '
                                     'Alternatively, this parameter can be '
                                     'used to indicate the index of the class '
                                     '(from the list above) to be '
                                     'explained.'.format(
                                         self.unique_predictions,
                                         self.class_names))

        explanations = {}
        models = {}  # type: ExplanationSurrogate
        if self.as_regressor:
            assert (explained_class_index is None
                    and explained_class_name is None)
            local_model = self._get_local_model(
                sampled_data, sampled_data_predictions, 0, True, maximum_depth)
            models = local_model
            explanations = dict(
                zip(self.feature_names,
                    local_model.feature_importances_))  # type: ignore
        else:
            assert self.class_names is not None and isinstance(models, dict)
            if explained_class is None:
                assert (explained_class_index is None
                        and explained_class_name is None), ('Explain all '
                                                            'classes.')

                if one_vs_rest:
                    for class_i, class_name in enumerate(self.class_names):
                        local_model = self._get_local_model(
                            sampled_data, sampled_data_predictions, class_i,
                            one_vs_rest, maximum_depth)

                        models[class_name] = local_model
                        exp = zip(
                            self.feature_names,
                            local_model.feature_importances_)  # type: ignore
                        explanations[class_name] = dict(exp)
                else:
                    assert not self.as_probabilistic, ('Multi-class local '
                                                       'model requires a '
                                                       'global classifier.')

                    local_model = self._get_local_model(
                        sampled_data, sampled_data_predictions, 0, one_vs_rest,
                        maximum_depth)

                    logger.info('A multi-class surrogate for a '
                                'non-probabilistic black-box model is the '
                                'same for all the possible classes, therefore '
                                'a single model will be trained and used for '
                                'explaining all of the classes.')

                    # The same multi-class local model is used for every class
                    for class_i, class_name in enumerate(self.class_names):
                        models[class_name] = local_model
                        exp = zip(
                            self.feature_names,
                            local_model.feature_importances_)  # type: ignore
                        explanations[class_name] = dict(exp)
            else:
                assert explained_class_index, 'Explain a single class.'
                assert explained_class_name, 'Explain a single class.'

                local_model = self._get_local_model(
                    sampled_data, sampled_data_predictions,
                    explained_class_index, one_vs_rest, maximum_depth)

                models[explained_class_name] = local_model
                explanations[explained_class_name] = dict(
                    zip(self.feature_names,
                        local_model.feature_importances_))  # type: ignore

        if return_models:
            return_ = (explanations, models)  # type: ExplanationTuple
        else:
            return_ = explanations
        return return_
