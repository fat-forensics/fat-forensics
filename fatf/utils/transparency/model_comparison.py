"""
Implementation various measures for comparing local model to global model.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
# License: new BSD

from typing import Union, List, Optional, Callable

import numpy as np

import fatf
import fatf.utils.data.augmentation as fuda
import fatf.utils.validation as fuv
import fatf.utils.models.validation as fumv
import fatf.utils.array.validation as fuav
import fatf.utils.array.tools as fuat
import fatf.utils.models.validation as fumv
from fatf.exceptions import IncompatibleModelError, IncorrectShapeError

__all__ = ['local_fidelity_score']

Index = Union[int, str]


def _input_is_valid(dataset: np.ndarray,
                    data_row: np.ndarray,
                    local_model: object,
                    global_model: object,
                    local_features: List[Index],
                    r_fid: float,
                    samples_number: int,
                    metric: Callable[[np.ndarray, np.ndarray], float],
                    global_class: int) -> bool:
    """
    Validates the input parameters of model comparison functions.

    For parameters descriptions and exceptions raised please see
    :func:`fatf.utils.transparency.model_comparison.local_fidelity_score`.

    Returns
    -------
    is_input_ok : boolean
        ``True`` if input is valid, ``False`` otherwise.
    """
    is_input_ok = False

    if not fuav.is_2d_array(dataset):
        raise IncorrectShapeError('The input dataset must be a '
                                  '2-dimensional numpy array.')
    if not fuav.is_base_array(dataset):
        raise TypeError('The input dataset must be of a base type.')

    is_structured = fuav.is_structured_array(dataset)

    if not fuav.is_1d_like(data_row):
        raise IncorrectShapeError('The data_row must either be a '
                                  '1-dimensional numpy array or numpy '
                                  'void object for structured rows.')

    are_similar = fuav.are_similar_dtype_arrays(
        dataset, np.array([data_row]), strict_comparison=True)
    if not are_similar:
        raise TypeError('The dtype of the data_row is different to '
                        'the dtype of the dataset array.')

    if local_features is not None:
        if isinstance(local_features, list):
            invalid_indices = fuat.get_invalid_indices(
                dataset, np.asarray(local_features))
            if invalid_indices.size:
                raise IndexError('The following indices are invalid for the '
                                 'input dataset: {}.'.format(invalid_indices))
        else:
            raise TypeError('The local_features parameter must be a '
                            'Python list or None.')

    # If the dataset is structured and the data_row has a different
    # number of features this will be caught by the above dtype check.
    # For classic numpy arrays this has to be done separately.
    if not is_structured:
        if data_row.shape[0] != dataset.shape[1]:
            raise IncorrectShapeError('The data_row must contain the '
                                        'same number of features as the '
                                        'dataset.')

    if not fumv.check_model_functionality(global_model, True, True):
        raise IncompatibleModelError('This functionality requires the global '
                                     'model to be capable of outputting '
                                     'predicted class probabilities via '
                                     'predict proba method.')

    if not fumv.check_model_functionality(local_model, False, True):
        raise IncompatibleModelError('This functionality requires the local '
                                     'model to be capable of outputting '
                                     'predicted class via predict method.')

    if not isinstance(r_fid, float):
            raise TypeError('r_fid must be a float.')
    else:
        if r_fid <= 0.0:
            raise ValueError('r_fid must be a positive float.')

    if isinstance(samples_number, int):
            if samples_number < 1:
                raise ValueError('The samples_number parameter must be a '
                                 'positive integer.')
    else:
        raise TypeError('The samples_number parameter must be an integer.')

    if fuv.get_required_parameters_number(metric) != 2:
        raise TypeError('The metric function must take only two required '
                         'parameters.')

    if is_structured:
        datapoint = dataset[0:1]
    else:
        datapoint = dataset[0:1, :]

    n_classes = global_model.predict_proba(datapoint).shape[1]
    if global_class > n_classes:
        raise ValueError('global_class is larger than the number of classes '
                         'outputted by the global model.')

    is_input_ok = True

    return is_input_ok


def local_fidelity_score(dataset: np.ndarray,
                         data_row: np.ndarray,
                         local_model: object,
                         global_model: object,
                         global_class: int,
                         metric: Callable[[np.ndarray, np.ndarray], float],
                         local_features: Optional[List[Index]]=None,
                         r_fid: float=0.05,
                         samples_number: int=50) -> float:
    """
    Computes local fidelity score for an instance in a dataset.

    This function implements a adapted version of the local fidelity method
    introduced by [LAUGEL2018SPHERES]_. For a specific data point,
    it samples uniformally within a hypersphere with radius corresponding to a
    percentage of the maximum l-2 distance between the instance to generate
    around and all other instances in the dataset. It then computes
    probabilities that the instances belong to the ``global_class`` that the
    data_row belongs to. It then computes an agreement between these
    predictions using accuracy.

    .. [LAUGEL2018SPHERES] Laugel, T., Renard, X., Lesot, M. J., Marsala,
       C., & Detyniecki, M. (2018). Defining locality for surrogates in
       post-hoc interpretablity. Workshop on Human Interpretability for
       Machine Learning (WHI)-International Conference on Machine Learning,
       2018.

    Parameters
    ----------
    dataset : numpy.ndarray
        A 2-dimensional numpy array with a dataset to be used for sampling.
    data_row : numpy.ndarray
        A data point.
    local_model : object
        A trained local model that must be able to output predicted class
        via ``predict`` method.
    global_model : object
        A trained global model that must be able to output predicted class
        via ``predict`` method.
    global_class : integer
        The class which the probabilities belong to that the local model was
        trained to predict.
    local_features : List[Index], Optional (default=None)
        List of indices which correspond to the features used to train the
        local model. If ``None`` then it will use all features in
        ``sampled_data``.
    r_fid : float, Optional (default=0.05)
        Radius of fideility which is the percentage of the maximum distance
        between any two dataponts in the dataset, which will be the radius
        of the l-2 hypersphere that points will be sampled in.
    samples_number : integer, Optional(default=50)
        The number of samples to be generated.
    metric : Callable[[numpy.ndarray, numpy.ndarray], float]
        A function that computes a metric between the predicted class of the
        local model, and the predicted probability of the global model. It must
        take only two required parameters and return a float. THe first
        parameter must be the local predictions and the second must be the
        global model probability predictions.

    Raises
    ------
    IncorrectShapeError
        The input dataset must be a 2-dimensional numpy array. The data_row
        must either be a 1-dimensional numpy array or numpy void object for
        structured rows. The data_row must contain the same number of features
        as the dataset.
    TypeError
        The input dataset must be of a base type. The dtype of the data_row
        must be the same dtype of the dataset array. The local_features
        parameter must be a Python list or None. r_fid must be a float. The
        samples_number paramter must be an integer. The metric function must
        take only two required parameters.

    Returns
    -------
    metric_result : float
        metric calculated between ``local_model`` predictions and
        ``global_model`` prediction probabilities for the data sampled.
    """
    assert _input_is_valid(dataset, data_row, local_model, global_model,
                           local_features, r_fid, samples_number,
                           metric, global_class), 'Input is invalid.'

    # TODO: int_to_float to be or not to be
    augmentor = fuda.LocalFidelity(dataset, int_to_float=True)
    sampled_data = augmentor.sample(data_row, r_fid, samples_number)

    if local_features is None:
        local_predictions = local_model.predict(sampled_data)
    else:
        if fuav.is_structured_array(sampled_data):
            local_data = sampled_data[local_features]
        else:
            local_data = sampled_data[:, local_features]
        local_predictions = local_model.predict(local_data)

    global_predictions = global_model.predict_proba(sampled_data)[:, global_class]

    metric_result = metric(local_predictions, global_predictions)

    return metric_result
