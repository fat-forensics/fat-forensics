"""
.. versionadded:: 0.0.2

The :mod:`fatf.utils.transparency.surrogate_evaluation` module implements
various surrogate model evaluation measures.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
#         Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

from typing import Callable, List, Optional, Union

import warnings

import numpy as np

from fatf.exceptions import IncompatibleModelError, IncorrectShapeError

import fatf.utils.array.tools as fuat
import fatf.utils.array.validation as fuav
import fatf.utils.data.augmentation as fuda
import fatf.utils.validation as fuv

__all__ = ['local_fidelity_score']

IndexType = Union[int, str]
PredictiveFunctionType = Callable[[np.ndarray], np.ndarray]


def _validate_input_local_fidelity(
        dataset: np.ndarray, data_row: Union[np.ndarray, np.void],
        global_predictive_function: PredictiveFunctionType,
        local_predictive_function: PredictiveFunctionType,
        metric_function: Callable[[np.ndarray, np.ndarray], float],
        explained_class_index: Union[int, None],
        explained_feature_indices: Union[List[IndexType], None],
        fidelity_radius_percentage: int, samples_number: int) -> bool:
    """
    Validates the input parameters for the ``local_fidelity_score`` function.

    This function validates input parameter of the
    :func:`fatf.utils.transparency.surrogate_evaluation.local_fidelity_score`
    function. The description of this function's input parameters, errors and
    exceptions can be found therein.

    Returns
    -------
    is_input_ok : boolean
        ``True`` if the input is valid, ``False`` otherwise.
    """
    # pylint: disable=too-many-arguments,too-many-branches,too-many-statements
    is_input_ok = False

    if not fuav.is_2d_array(dataset):
        raise IncorrectShapeError('The input dataset must be a '
                                  '2-dimensional numpy array.')
    if not fuav.is_base_array(dataset):
        raise TypeError('The input dataset must be of a base type -- numbers '
                        'and/or strings.')

    if not fuav.is_1d_like(data_row):
        raise IncorrectShapeError('The data_row must either be a '
                                  '1-dimensional numpy array or a numpy '
                                  'void object for structured data rows.')

    are_similar = fuav.are_similar_dtype_arrays(dataset, np.array([data_row]))
    if not are_similar:
        raise TypeError('The dtype of the data_row is too different from '
                        'the dtype of the dataset array.')

    # If the dataset is structured and the data_row has a different
    # number of features this will be caught by the above dtype check.
    # For classic numpy arrays this has to be done separately.
    if not fuav.is_structured_array(dataset):
        if dataset.shape[1] != data_row.shape[0]:
            raise IncorrectShapeError('The data_row must contain the same '
                                      'number of features as the dataset.')

    if callable(global_predictive_function):
        global_params_n = fuv.get_required_parameters_number(
            global_predictive_function)
        if global_params_n != 1:
            raise IncompatibleModelError(
                'The global predictive function must have exactly *one* '
                'required parameter to work with this metric.')
    else:
        raise TypeError('The global_predictive_function should be a Python '
                        'callable, e.g., a Python function.')

    if callable(local_predictive_function):
        local_params_n = fuv.get_required_parameters_number(
            local_predictive_function)
        if local_params_n != 1:
            raise IncompatibleModelError(
                'The local predictive function must have exactly *one* '
                'required parameter to work with this metric.')
    else:
        raise TypeError('The local_predictive_function should be a Python '
                        'callable, e.g., a Python function.')

    if callable(metric_function):
        if fuv.get_required_parameters_number(metric_function) != 2:
            raise TypeError('The metric_function must take exactly *two* '
                            'required parameters.')
    else:
        raise TypeError('The metric_function should be a Python callable, '
                        'e.g., a Python function.')

    # Explained class index
    global_prediction = global_predictive_function(dataset[:1])
    assert not fuav.is_structured_array(global_prediction), 'Must be plain.'
    assert global_prediction.shape[0] == 1, 'Just 1 data point was predicted.'
    if fuav.is_2d_array(global_prediction):  # A probabilistic model.
        if explained_class_index is not None:
            if isinstance(explained_class_index, int):
                if (explained_class_index >= global_prediction.shape[1]
                        or explained_class_index < 0):
                    raise ValueError('The explained_class_index parameter is '
                                     'negative or larger than the number of '
                                     'classes output by the global '
                                     'probabilistic model.')
            else:
                raise TypeError('For probabilistic global models, i.e., '
                                'global predictive functions, the '
                                'explained_class_index parameter has to be an '
                                'integer or None.')
    elif fuav.is_1d_array(global_prediction):
        if explained_class_index is not None:
            warnings.warn(
                'The explained_class_index parameter is not None and will be '
                'ignored since the global model is not probabilistic.',
                UserWarning)
    else:
        assert False, ('Global predictor must output a 1- or 2-dimensional '
                       'numpy array.')  # pragma: nocover

    if explained_feature_indices is not None:
        if isinstance(explained_feature_indices, list):
            invalid_indices = fuat.get_invalid_indices(
                dataset, np.asarray(explained_feature_indices))
            if invalid_indices.size:
                raise IndexError(
                    'The following column indices are invalid for the input '
                    'dataset: {}.'.format(invalid_indices))
        else:
            raise TypeError('The explained_feature_indices parameter must be '
                            'a Python list or None.')

    if isinstance(fidelity_radius_percentage, int):
        if fidelity_radius_percentage <= 0 or fidelity_radius_percentage > 100:
            raise ValueError('The fidelity_radius_percentage must be an '
                             'integer between 1 and 100.')
    else:
        raise TypeError('The fidelity_radius_percentage must be an integer '
                        'between 1 and 100.')

    if isinstance(samples_number, int):
        if samples_number < 1:
            raise ValueError('The samples_number must be a positive integer.')
    else:
        raise TypeError('The samples_number must be an integer.')

    is_input_ok = True
    return is_input_ok


def local_fidelity_score(
        dataset: np.ndarray,
        data_row: Union[np.ndarray, np.void],
        global_predictive_function: PredictiveFunctionType,
        local_predictive_function: PredictiveFunctionType,
        metric_function: Callable[[np.ndarray, np.ndarray], float],
        explained_class_index: Optional[int] = None,
        explained_feature_indices: Optional[List[IndexType]] = None,
        fidelity_radius_percentage: int = 5,
        samples_number: int = 50) -> float:
    """
    Computes local fidelity between a global and a local (surrogate) model.

    .. versionadded:: 0.0.2

    For a selected data point (``data_row``), it samples uniformly around it
    within a hypersphere, which radius corresponds to a percentage -- defined
    with ``fidelity_radius_percentage`` parameter -- of the maximum l-2
    distance between the specified data point and all the instances in the
    ``dataset``. (This sampling is based on
    :class:`fatf.utils.data.augmentation.LocalSphere` data augmenter.)

    .. warning:: A ``dataset`` with categorical features.

       This surrogate evaluation metric should **not** be used when the
       ``dataset`` contains *categorical features* (even when they are encoded,
       e.g., one-hot encoding) since the l-2 distance computed on mixed true
       numerical and (numerically-encoded) categorical features causes the
       local sample (computed with the
       :class:`fatf.utils.data.augmentation.LocalSphere` data augmenter) to
       be ill-defined. Feature scaling could possibly be used to overcome this
       issue, however we leave such consideration up to the user.

    The global and local predictive functions can be either: a probabilistic
    predictor, a (multi-class) classifier or a regressor.

    +-------+---------------------------------------------------------------+
    |       | Global Model                                                  |
    +-------+--------+-------------------------+--------------------+-------+
    | Local |        | |prob|                  | |clf|              | |reg| |
    | Model +--------+-------------------------+--------------------+-------+
    |       | |prob| | OK, e.g., KL-divergence | OK, e.g., log-loss | |imp| |
    |       +--------+-------------------------+--------------------+-------+
    |       | |clf|  | OK (via thresholding)   | OK                 | |imp| |
    |       +--------+-------------------------+--------------------+-------+
    |       | |reg|  | OK for a single class   | |imp|              | OK    |
    +-------+--------+-------------------------+--------------------+-------+

    .. |prob| replace:: **probabilistic**
    .. |clf| replace:: **classifier**
    .. |reg| replace:: **regressor**
    .. |imp| replace:: Not possible

    If the ``global_predictive_function`` outputs **probabilities**, the
    following should be considered for different types of a local model:

    * The local model is **probabilistic** as well:

      + a native probabilistic evaluation metric, such as the
        `Kullback–Leibler divergence`_, can be used; or
      + a thresholding can be applied or a top prediction can be chosen for
        both the local and the global probabilistic prediction and a classic
        classification performance metric can be used.

    * The local model is a **classifier** -- the probabilistic output of the
      global model has to be thresholded or the top prediction needs to be
      selected and a classic classification performance metric can be used.
    * The local model is a **regressor** -- this is only possible if the
      regressor is fitted for the probabilistic output of one of the classes.
      In this case any of the standard regression evaluation measures can be
      used.

    If the ``global_predictive_function`` is a **classifier**, the
    following should be considered for different types of a local model:

    * The local model is **probabilistic**:

      + a native performance metric, like log-loss_, can be used; or
      + the probabilistic output of the local predictor can be thresholded or
        the top label selected and compare using standard classification
        performance metrics.

    * The local model is a **classifier** as well -- any standard (multi-class)
      classification performance metric can be used.
    * Having a local **regressor** is not possible in this case.

    Finally, if the ``global_predictive_function`` is a **regressor**, the
    local model can **only** be a regressor as well, in which case any standard
    regression evaluation metric can be used.

    If the problem being modelled is multi-class (for probabilistic models and
    classifiers), the local model can either be fitted to the original
    multi-class problem or as one-vs-the-rest for a selected class. In the
    latter case, when the global model is probabilistic, the
    ``explained_class_index`` parameter may be used to specify the class
    (column index) that the ``data_row`` belongs to (according to the global
    model) -- in this case only the selected column with probabilities will be
    passed to the local fidelity score (``metric_function``) function.

    .. note:: Why to train the local model as one-vs-the-rest?

       When the local model is trained in the same output domain as the global
       model, the explanations extracted from this local model apply to all of
       the possible classes, what for some types of local models renders them
       uninformative. For example, consider training a decision tree locally
       and using the feature importance it provides. In this case we know
       which features are important in this local space but we cannot attribute
       these importances to any of the possible classes. However, a different
       type of explanation extracted from the same tree, for example, the
       logical conditions extracted from a root-to-leaf path that the selected
       ``data_row`` falls into, can be perfectly reasonable.

       If, on the other hand, the local model is trained as one-vs-the rest,
       where the "one" class is often set to be the class of the selected
       ``data_row``, any type of the explanation can be attributed to the
       selected class. In this case feature importances extracted from the
       local model can attributed to the selected class in the specified
       neighbourhood. This mode of training the local model is required when
       the global model is probabilistic and the local one is a regressor, and
       optional for all the other combinations of the two.

       The consequence of training the local model as one-vs-the-rest is the
       need for train a separate local model for every class desired to be
       explained. For some local models and explanation types this is a
       requirement. For example, when the local model is a linear regression
       (trained on probabilities of a selected class) the only possible
       explanation is feature importance, which is meaningless in other cases.

       In general, when evaluating the quality of a local surrogate, the most
       truthful measure would be the one achieved when the local model is
       trained on the same set of target classes. A good quality of a local
       one-vs-the-rest model with respect to the global model should be
       treated with caution as it only indicates that the local model excels
       at this task and may not be a good approximation of the global decisive
       process at all. Comparing quality of two local models where one is
       multi-class and the other one-vs-the-rest is relatively complex and
       should be done with caution (the former local model has a more difficult
       task to solve).

    Examples of how to define the ``metric_function`` can be found in the
    *Examples* section down below. This local fidelity evaluation is inspired
    by the local fidelity method introduced in [LAUGEL2018SPHERES]_.

    .. _`Kullback–Leibler divergence`: https://en.wikipedia.org/wiki/
       Kullback–Leibler_divergence
    .. _log-loss: https://scikit-learn.org/stable/modules/
       model_evaluation.html#log-loss
    .. [LAUGEL2018SPHERES] Laugel, T., Renard, X., Lesot, M. J., Marsala,
       C., & Detyniecki, M. (2018). Defining locality for surrogates in
       post-hoc interpretablity. Workshop on Human Interpretability for
       Machine Learning (WHI) -- International Conference on Machine Learning,
       2018.

    Examples
    --------
    The metric function should be adjusted to the type of the global and local
    predictors (and the use of the ``explained_class_index`` parameter).

    >>> import numpy as np
    >>> data = np.array([[0, 1], [1, 1], [1, 0]])
    >>> targets = np.array(['a', 'b', 'c'])

    Let us assume that the global model is probabilistic, the local model is
    a regressor and we are explaining class ``'b'`` with index ``1``.
    (The index of the class is based on the lexicographical ordering of all the
    unique target values.)

    >>> explained_class_index = 1

    >>> import fatf.utils.models.models as fatf_models
    >>> global_model = fatf_models.KNN(k=1)
    >>> global_model.fit(data, targets)

    >>> probabilities = global_model.predict_proba(data)
    >>> selected_class_probabilities = probabilities[:, explained_class_index]

    >>> local_model = fatf_models.KNN(k=1, mode='regressor')
    >>> local_model.fit(data, selected_class_probabilities)

    One way to evaluate the performance of our local (surrogate) model in this
    scenario is the *Mean Squared Error*:

    >>> def mse(global_predictions, local_predictions):
    ...     mse = np.square(global_predictions - local_predictions)
    ...     mse = mse.mean()
    ...     return mse

    >>> import fatf.utils.transparency.surrogate_evaluation as surrogate_eval
    >>> mse_fidelity_score = surrogate_eval.local_fidelity_score(
    ...     data, data[0], global_model.predict_proba, local_model.predict,
    ...     mse, explained_class_index=explained_class_index)
    >>> mse_fidelity_score
    0.0

    Alternatively, if ``scikit-learn`` is available, an ROC can be computed,
    in which case the probabilities of the selected class need to be
    thresholded:

    >>> import sklearn.metrics
    >>> def roc(global_predictions, local_predictions):
    ...     global_predictions[global_predictions >= .5] = 1
    ...     global_predictions[global_predictions < .5] = 0
    ...     global_predictions = global_predictions.astype(int)
    ...
    ...     roc = sklearn.metrics.roc_auc_score(global_predictions,
    ...                                         local_predictions)
    ...     return roc

    >>> roc_fidelity_score = surrogate_eval.local_fidelity_score(
    ...     data, data[1], global_model.predict_proba, local_model.predict,
    ...     roc, explained_class_index=explained_class_index)
    >>> roc_fidelity_score
    1.0

    If both models are classifiers trained with the same set of target classes,

    >>> local_classifier = fatf_models.KNN(k=1)
    >>> local_classifier.fit(data, targets)

    a simple *accuracy* (implemented in FAT Forensics) can be used:

    >>> import fatf.utils.metrics.metrics as fatf_metrics
    >>> import fatf.utils.metrics.tools as fatf_metrics_tools
    >>> def accuracy(global_predictions, local_predictions):
    ...     confusion_matrix = fatf_metrics_tools.get_confusion_matrix(
    ...         local_predictions, global_predictions, labels=['a', 'b', 'c'])
    ...     accuracy = fatf_metrics.accuracy(confusion_matrix)
    ...     return accuracy

    >>> accuracy_fidelity_score = surrogate_eval.local_fidelity_score(
    ...     data, data[2], global_model.predict, local_classifier.predict,
    ...     accuracy)
    >>> accuracy_fidelity_score
    1.0

    (Note ``global_model.predict`` instead of ```global_model.predict_proba``.)

    Parameters
    ----------
    dataset : numpy.ndarray
        A 2-dimensional numpy array with a dataset used to initialise the data
        sampler.
    data_row : Union[numpy.ndarray, numpy.void]
        A data point around which local fidelity is evaluated.
    global_predictive_function : Callable[[np.ndarray], np.ndarray]
        A Python callable (e.g., a function) that is responsible for predicting
        data points in the global model. This function can either be
        *probabilistic*, i.e., return a 2-dimensional numpy array with
        probabilities for every possible target class; a *regressor* (returning
        a 1-dimensional regression values array) or a *classifier* (returning a
        1-dimensional class prediction array). Regardless of the type it
        **must** allow only **one required parameter** -- a 2-dimensional data
        array to be predicted.
    local_predictive_function : Callable[[np.ndarray], np.ndarray]
        A Python callable (e.g., a function) that is responsible for predicting
        data points in the local (surrogate) model. For more details about the
        allowed function types please see the description of the
        ``global_predictive_function`` parameter.
    metric_function : Callable[[numpy.ndarray, numpy.ndarray], float]
        A Python callable (e.g., a function) that computes a (performance)
        metric between the predictions of the global model
        (``global_predictive_function``) and the predictions of the local
        (surrogate) model (``local_predictive_function``). The passed callable
        object **must** take exactly **two required parameters**: the first one
        being predictions of the global model and the latter predictions of the
        local model, and return a number (float) representing performance
        comparison of the two. This callable object has to be adjusted to the
        types of global and local predictive functions.
    explained_class_index : integer, optional (default=None)
        If the global model (``global_predictive_function``) is probabilistic,
        this parameter allows to select a single column of probabilities for a
        selected class to be passed to the ``metric_function``. This parameter
        is useful when the local (surrogate) model is a regressor predicting
        probabilities of this chosen class (the class being explained).
    explained_feature_indices : List[IndexType], optional (default=None)
        If the local (surrogate) model was trained on a subset of the features,
        this parameter allows to indicate which features should be used when
        predicting the generated data with the local model. If ``None``, all of
        the features will be used.
    fidelity_radius_percentage : integer, optional (default=5)
        The locality of the fidelity measure is enforced by limiting the
        distance from the selected ``data_row`` to generated data, which is
        used for fidelity metric evaluation. This radius (of a hyper-sphere
        around the selected ``data_row``) is defined as a percentage of the
        largest l-2 distance between any two data points in the input
        ``dataset`` within which the evaluation data will be sampled.
    samples_number : integer, optional (default=50)
        The number of samples to be generated when computing the local fidelity
        score.

    Warns
    -----
    UserWarning
        If the user specifies the ``explained_class_index`` parameter for a
        global model that is not probabilistic, this parameter is ignored,
        about what the user is warned.

    Raises
    ------
    IncompatibleModelError
        The ``global_predictive_function`` or the ``local_predictive_function``
        does not required **exactly one** parameter.
    IncorrectShapeError
        The input ``dataset`` is not a 2-dimensional numpy array. The input
        ``data_row`` is not 1-dimensional: either a 1-dimensional numpy array
        or a numpy void object for structured rows. The number of columns
        (features) in the ``data_row`` is different to the number of columns in
        the input ``dataset``.
    IndexError
        Some of the ``explained_feature_indices`` are invalid column indices
        for the input ``dataset``.
    TypeError
        The input ``dataset`` is not of a base type. The dtype of the
        ``data_row`` is too different from the dtype of the ``dataset``.
        The ``global_predictive_function`` or the ``local_predictive_function``
        is not a Python callable. The ``metric_function`` is not a Python
        callable or it does not require **exactly** two parameters.
        The ``explained_class_index`` is neither ``None`` nor an integer.
        The ``explained_feature_indices`` is neither ``None`` nor a Python
        list. The ``fidelity_radius_percentage`` is not an integer. The
        ``samples_number`` is not an integer.
    ValueError
        The ``explained_class_index`` is a negative integer or out of bounds
        for the number of classes output by the global probabilistic model
        (``global_predictive_function``). The ``fidelity_radius_percentage``
        is smaller than 1 or larger than 100. The ``samples_number`` is smaller
        than 1.

    Returns
    -------
    fidelity_score : float
        A metric of "closeness" between the global and the local predictive
        function predictions calculated using the ``metric_function`` on the
        sampled data.
    """
    # pylint: disable=too-many-arguments
    assert _validate_input_local_fidelity(
        dataset, data_row, global_predictive_function,
        local_predictive_function, metric_function, explained_class_index,
        explained_feature_indices, fidelity_radius_percentage,
        samples_number), 'Input is invalid.'

    augmentor = fuda.LocalSphere(dataset, int_to_float=False)
    sampled_data = augmentor.sample(data_row, fidelity_radius_percentage,
                                    samples_number)

    global_predictions = global_predictive_function(sampled_data)
    assert not fuav.is_structured_array(global_predictions), 'Is structured.'
    if explained_class_index is not None:
        assert fuav.is_2d_array(global_predictions), '2-D probabilities array.'
        global_predictions = global_predictions[:, explained_class_index]

    if explained_feature_indices is None:
        local_data = sampled_data
    else:
        if fuav.is_structured_array(sampled_data):
            local_data = sampled_data[explained_feature_indices]
        else:
            local_data = sampled_data[:, explained_feature_indices]
    local_predictions = local_predictive_function(local_data)

    fidelity_score = metric_function(global_predictions, local_predictions)

    return fidelity_score
