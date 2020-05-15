"""
The :mod:`fatf.transparency.models.feature_influence` module holds functions
for calculating feature influence for predictive models.

This module implements Partial Dependence (PD) and Individual Conditional
Expectation (ICE) -- model agnostic feature influence measurements.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
#         Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

from typing import List, Optional, Tuple, Union

import warnings

import numpy as np

import fatf.utils.array.tools as fuat
import fatf.utils.array.validation as fuav
import fatf.utils.models.validation as fumv

from fatf.exceptions import IncompatibleModelError, IncorrectShapeError

__all__ = ['individual_conditional_expectation',
           'merge_ice_arrays',
           'partial_dependence_ice',
           'partial_dependence']  # yapf: disable


def _input_is_valid(dataset: np.ndarray,
                    model: object,
                    feature_index: Union[int, str],
                    treat_as_categorical: Optional[bool],
                    steps_number: Optional[int]) -> bool:  # yapf: disable
    """
    Validates input parameters of Individual Conditional Expectation function.

    For the input parameter description, warnings and exceptions please see the
    documentation of the :func:`fatf.transparency.model.feature_influence.\
individual_conditional_expectation` function.

    Returns
    -------
    is_input_ok : boolean
        ``True`` if the input is valid, ``False`` otherwise.
    """
    is_input_ok = False

    if not fuav.is_2d_array(dataset):
        raise IncorrectShapeError('The input dataset must be a 2-dimensional '
                                  'array.')

    if not fuav.is_base_array(dataset):
        raise ValueError('The input dataset must only contain base types '
                         '(textual and numerical).')

    if not fumv.check_model_functionality(model, require_probabilities=True):
        raise IncompatibleModelError('This functionality requires the model '
                                     'to be capable of outputting '
                                     'probabilities via predict_proba method.')

    if not fuat.are_indices_valid(dataset, np.array([feature_index])):
        raise IndexError('Provided feature index is not valid for the input '
                         'dataset.')

    if isinstance(steps_number, int):
        if steps_number < 2:
            raise ValueError('steps_number has to be at least 2.')
    elif steps_number is None:
        pass
    else:
        raise TypeError('steps_number parameter has to either be None or an '
                        'integer.')

    if (not isinstance(treat_as_categorical, bool)
            and treat_as_categorical is not None):
        raise TypeError('treat_as_categorical has to either be None or a '
                        'boolean.')

    is_input_ok = True
    return is_input_ok


def _interpolate_array(
        dataset: np.ndarray,
        feature_index: Union[int, str],  # yapf: disable
        treat_as_categorical: bool,
        steps_number: Union[int, None]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a 3-D array with interpolated values for the selected feature.

    If the selected feature is numerical the interpolated values are a
    numerical array with evenly spaced numbers between the minimum and the
    maximum value in that column. Otherwise, when the feature is categorical
    the interpolated values are all the unique elements of the that column.

    To get the interpolation the original 2-D dataset is stacked on top of
    itself the number of times equal to the number of desired interpolation
    samples. Then, for every copy of that dataset the selected feature is fixed
    to consecutive values of the interpolated array (the same value for the
    whole copy of the dataset).

    Parameters
    ----------
    dataset : numpy.ndarray
        A dataset based on which interpolation will be done.
    feature_index : Union[integer, string]
        An index of the feature column in the input dataset for which the
        interpolation will be computed.
    treat_as_categorical : boolean
        Whether to treat the selected feature as categorical or numerical.
    steps_number : Union[integer, None]
        The number of evenly spaced samples between the minimum and the maximum
        value of the selected feature for which the model's prediction will be
        evaluated. This parameter applies only to numerical features, for
        categorical features regardless whether it is a number or ``None``, it
        will be ignored.

    Returns
    -------
    interpolated_data : numpy.ndarray
        Numpy array of shape (n_samples, steps_number, n_features) -- where the
        (n_samples, n_features) is the dimension of the input ``dataset`` --
        holding the input ``dataset`` augmented with the interpolated values.
    interpolated_values : numpy.ndarray
        A 1-dimensional array of shape (steps_number, ) holding the
        interpolated values. If a numerical column is selected this will be a
        series of uniformly distributed ``steps_number`` values between the
        minimum and the maximum value of that column. For categorical (textual)
        columns it will hold all the unique values from that column.
    """
    assert isinstance(dataset, np.ndarray), 'Dataset -> numpy array.'
    assert isinstance(feature_index, (int, str)), 'Feature index -> str/ int.'
    assert isinstance(treat_as_categorical, bool), 'As categorical -> bool.'
    assert steps_number is None or isinstance(steps_number, int), \
        'Steps number -> None/ int.'

    is_structured = fuav.is_structured_array(dataset)

    if is_structured:
        column = dataset[feature_index]
    else:
        column = dataset[:, feature_index]

    if treat_as_categorical:
        interpolated_values = np.unique(column)
        interpolated_values.sort()
        # Ignoring steps number -- not needed for categorical.
        steps_number = interpolated_values.shape[0]
    else:
        assert isinstance(steps_number, int), 'Steps number must be an int.'
        interpolated_values = np.linspace(column.min(), column.max(),
                                          steps_number)

        # Give float type to this column if it is a structured array
        if (is_structured
                and dataset.dtype[feature_index] != interpolated_values.dtype):
            new_types = []
            for name in dataset.dtype.names:
                if name == feature_index:
                    dtype = fuat.generalise_dtype(interpolated_values.dtype,
                                                  dataset.dtype[name])
                    new_types.append((name, dtype))
                else:
                    new_types.append((name, dataset.dtype[name]))
            dataset = dataset.astype(new_types)
        elif not is_structured and dataset.dtype != interpolated_values.dtype:
            dtype = fuat.generalise_dtype(interpolated_values.dtype,
                                          dataset.dtype)
            dataset = dataset.astype(dtype)

    interpolated_data = np.repeat(dataset[:, np.newaxis], steps_number, axis=1)
    assert len(interpolated_values) == steps_number, 'Required for broadcast.'
    if is_structured:
        for idx in range(steps_number):
            # Broadcast the new value.
            interpolated_data[:, idx][feature_index] = interpolated_values[idx]
    else:
        # Broadcast the new vector.
        interpolated_data[:, :, feature_index] = interpolated_values

    return interpolated_data, interpolated_values


def _filter_rows(include_rows: Union[None, int, List[int]],
                 exclude_rows: Union[None, int, List[int]],
                 rows_number: int) -> List[int]:
    """
    Filters row indices given the include and exclude lists.

    For the exceptions description please see the documentation of the
    :func`fatf.transparency.model.feature_influence.
    individual_conditional_expectation` function.

    Parameters
    ----------
    include_rows: Union[None, integer, List[integer]]
        Indices of rows that will be included. If this parameter is specified,
        only the selected rows will be included. If additionally
        ``exclude_rows`` is specified the selected rows will be a set
        difference between the two. This parameter can either be a *list* of
        indices, a single index (integer) or ``None`` -- all indices.
    exclude_rows: Union[None, integer, List[integer]]
        Indices of rows to be excluded. If this parameter is specified and
        ``include_rows`` is not, these indices will be excluded from all of the
        rows. If both include and exclude parameters are specified, the
        included rows will be a set difference of the two. This parameter can
        either be a *list* of indices, a single index (integer) or ``None`` --
        no indices.
    rows_number : integer
        The total number of rows from which these indices will be included/
        excluded.

    Returns
    -------
    filtered_rows : List[integer]
        Sorted list of row indices computed as a set difference between the
        ``include_rows`` parameter (or the whole set of indices for the array
        if the parameter is ``None``) and the ``exclude_rows`` parameters (or
        an empty set of indices if the parameter is ``None``).
    """

    def within_bounds(error_type, row_index):
        if row_index < 0 or row_index >= rows_number:
            raise ValueError('{} rows element {} is out of bounds. There are '
                             'only {} rows in the input dataset.'.format(
                                 error_type, row_index, rows_number))
        return True

    assert isinstance(rows_number, int) and rows_number > 0, \
        'Rows number must be a positive integer.'

    if include_rows is None:
        include_rows = list(range(rows_number))
    elif isinstance(include_rows, int):
        assert within_bounds('Include', include_rows), 'Index within bounds.'
        include_rows = [include_rows]
    elif isinstance(include_rows, list):
        for i in include_rows:
            if not isinstance(i, int):
                raise TypeError(
                    'Include rows element *{}* is not an integer.'.format(i))
            assert within_bounds('Include', i), 'Every index is within bounds.'
    else:
        raise TypeError('The include_rows parameters must be either None or a '
                        'list of integers indicating which rows should be '
                        'included in the computation.')

    if exclude_rows is None:
        exclude_rows = []
    elif isinstance(exclude_rows, int):
        assert within_bounds('Exclude', exclude_rows), 'Index within bounds.'
        exclude_rows = [exclude_rows]
    elif isinstance(exclude_rows, list):
        for i in exclude_rows:
            if not isinstance(i, int):
                raise TypeError(
                    'Exclude rows element *{}* is not an integer.'.format(i))
            assert within_bounds('Exclude', i), 'Every index is within bounds.'
    else:
        raise TypeError('The exclude_rows parameters must be either None or a '
                        'list of integers indicating which rows should be '
                        'excluded in the computation.')

    filtered_rows = sorted(set(include_rows).difference(exclude_rows))
    return filtered_rows


def individual_conditional_expectation(
        dataset: np.ndarray,
        model: object,
        feature_index: Union[int, str],
        treat_as_categorical: Optional[bool] = None,
        steps_number: Optional[int] = None,
        include_rows: Optional[Union[int, List[int]]] = None,
        exclude_rows: Optional[Union[int, List[int]]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates Individual Conditional Expectation for a selected feature.

    Based on the provided dataset and model this function computes Individual
    Conditional Expectation (ICE) of a selected feature for all target classes.
    If ``treat_as_categorical`` parameter is not provided the function will
    infer the type of the selected feature and compute the appropriate ICE.
    Otherwise, the user can specify whether the selected feature should be
    treated as a categorical or numerical feature. If the selected feature is
    numerical, you can specify the number of samples between this feature's
    minimum and maximum value for which the input model will be evaluated.
    By default this value is set to 100.

    Finally, it is possible to filter the rows of the input dataset that will
    be used to calculate ICE with ``include_rows`` and ``exclude_rows``
    parameters. If ``include_rows`` is specified ICE will only be calculated
    for these rows. If both include and exclude parameters are given, ICE will
    be computed for the set difference. Finally, if only the exclude parameter
    is specified, these rows will be subtracted from the whole dataset.

    This approach is an implementation of a method introduced by
    [GOLDSTEIN2015PEEKING]_. It is intended to be used with probabilistic
    models, therefore the input model must have a ``predict_proba`` method.

    .. [GOLDSTEIN2015PEEKING] Goldstein, A., Kapelner, A., Bleich, J. and
       Pitkin, E., 2015. Peeking inside the black box: Visualizing statistical
       learning with plots of individual conditional expectation. Journal of
       Computational and Graphical Statistics, 24(1), pp.44-65.

    Parameters
    ----------
    dataset : numpy.ndarray
        A dataset based on which ICE will be computed.
    model : object
        A fitted model which predictions will be used to calculate ICE. (Please
        see :class:`fatf.utils.models.models.Model` class documentation for the
        expected model object specification.)
    feature_index : Union[integer, string]
        An index of the feature column in the input dataset for which ICE will
        be computed.
    treat_as_categorical : boolean, optional (default=None)
        Whether to treat the selected feature as categorical or numerical.
    steps_number : integer, optional (default=None, i.e. 100)
        The number of evenly spaced samples between the minimum and the maximum
        value of the selected feature for which the model's prediction will be
        evaluated. (This parameter applies only to numerical features.)
    include_rows : Union[int, List[int]], optional (default=None)
        Indices of rows that will be included in the ICE calculation. If this
        parameter is specified, ICE will only be calculated for the selected
        rows. If additionally ``exclude_rows`` is specified the selected rows
        will be a set difference between the two. This parameter can either be
        a *list* of indices or a single index (integer).
    exclude_rows : Union[int, List[int]], optional (default=None)
        The indices of rows to be excluded from the ICE calculation. If this
        parameter is specified and ``include_rows`` is not, these indices will
        be excluded from all of the rows. If both include and exclude
        parameters are specified, the rows included in the ICE calculation will
        be a set difference of the two. This parameter can either be a *list*
        of indices or a single index (integer).

    Warns
    -----
    UserWarning
        The feature is treated as categorical but the number of steps parameter
        is provided (not ``None``). In this case the ``steps_number`` parameter
        is ignored. Also, the user is warned when the selected feature is
        detected to be categorical (textual) while the user indicated that it
        is numerical.

    Raises
    ------
    IncompatibleModelError
        The model does not have required functionality -- it needs to be able
        to output probabilities via ``predict_proba`` method.
    IncorrectShapeError
        The input dataset is not a 2-dimensional numpy array.
    IndexError
        Provided feature (column) index is invalid for the input dataset.
    TypeError
        ``treat_as_categorical`` is not ``None`` or boolean. The
        ``steps_number`` parameter is not ``None`` or integer. Either
        ``include_rows`` or ``exclude_rows`` parameter is not ``None``, an
        integer or a list of integers.
    ValueError
        The input dataset must only contain base types (textual and numerical
        values). One of the ``include_rows`` or ``exclude_rows`` indices is not
        valid for the input dataset. The ``steps_number`` is smaller than 2.

    Returns
    -------
    ice : numpy.ndarray
        An array of Individual Conditional Expectations for all of the selected
        dataset rows and the feature (dataset column) of choice. It's of the
        (n_samples, steps_number, n_classes) shape where n_samples is the
        number of rows selected from the dataset for the ICE computation,
        steps_number is the number of generated samples for the selected
        feature and n_classes is the number of classes in the target of the
        dataset. The numbers in this array represent the probability of every
        class for every selected data point when the selected feature is fixed
        to one of the values in the generated feature linespace (see below).
    feature_linespace : numpy.ndarray
        A one-dimensional array -- (steps_number, ) -- with the values for
        which the selected feature was substituted when the dataset was
        evaluated with the specified model.
    """
    # pylint: disable=too-many-arguments,too-many-locals
    assert _input_is_valid(dataset, model, feature_index, treat_as_categorical,
                           steps_number), 'Input must be valid.'

    is_structured = fuav.is_structured_array(dataset)

    if is_structured:
        column = dataset[feature_index]
    else:
        column = dataset[:, feature_index]
    assert fuav.is_1d_array(column), 'Column must be a 1-dimensional array.'

    if fuav.is_numerical_array(column):
        is_categorical_column = False
    elif fuav.is_textual_array(column):
        is_categorical_column = True
    else:
        assert False, 'Must be an array of a base type.'  # pragma: nocover

    # If needed, infer the column type.
    if treat_as_categorical is None:
        treat_as_categorical = is_categorical_column
    elif not treat_as_categorical and is_categorical_column:
        message = ('Selected feature is categorical (string-base elements), '
                   'however the treat_as_categorical was set to False. Such '
                   'a combination is not possible. The feature will be '
                   'treated as categorical.')
        warnings.warn(message, category=UserWarning)
        treat_as_categorical = True
        steps_number = None

    if treat_as_categorical and steps_number is not None:
        warnings.warn(
            'The steps_number parameter will be ignored as the feature is '
            'being treated as categorical.',
            category=UserWarning)

    # If needed, get the default steps number.
    if not treat_as_categorical and steps_number is None:
        steps_number = 100

    rows_number = dataset.shape[0]
    include_r = _filter_rows(include_rows, exclude_rows, rows_number)
    filtered_dataset = dataset[include_r]

    sampled_data, feature_linespace = _interpolate_array(
        filtered_dataset, feature_index, treat_as_categorical, steps_number)

    ice = [
        model.predict_proba(data_slice)  # type: ignore
        for data_slice in sampled_data
    ]
    ice = np.stack(ice, axis=0)

    return ice, feature_linespace


def merge_ice_arrays(ice_arrays_list: List[np.ndarray]) -> np.ndarray:
    """
    Merges multiple Individual Conditional Expectation arrays.

    This function allows you to merge Individual Conditional Expectation arrays
    into a single array as long as they were calculated for the same feature
    and for the same number of classes. This may be helpful when evaluating ICE
    for a model over multiple cross-validation folds or for multiple models.

    Parameters
    ----------
    ice_arrays_list : List[numpy.ndarray]
        A list of Individual Conditional Expectation arrays to be merged.

    Raises
    ------
    IncorrectShapeError
        One of the ICE arrays is not 3-dimensional.
    TypeError
        The ``ice_arrays_list`` input parameter is not a list.
    ValueError
        The list of ICE arrays to be merged is empty. One of the ICE arrays is
        not a numerical array. One of the ICE arrays is structured. Some of the
        ICE arrays do not share the same second (number of steps) or third
        (number of classes) dimension or type.

    Returns
    -------
    ice_arrays : numpy.ndarray
        All of the ICE arrays merged together alongside the first dimension
        (number of instances).
    """
    if isinstance(ice_arrays_list, list):
        if not ice_arrays_list:
            raise ValueError('Cannot merge 0 arrays.')

        previous_shape = None
        for ice_array in ice_arrays_list:
            if not fuav.is_numerical_array(ice_array):
                raise ValueError('The ice_array list should only contain '
                                 'numerical arrays.')
            if fuav.is_structured_array(ice_array):
                raise ValueError('The ice_array list should only contain '
                                 'unstructured arrays.')
            if len(ice_array.shape) != 3:
                raise IncorrectShapeError('The ice_array should be '
                                          '3-dimensional.')

            if previous_shape is None:
                previous_shape = (ice_array.shape[1], ice_array.shape[2],
                                  ice_array.dtype)  # yapf: disable
            elif (previous_shape[:2] != ice_array.shape[1:]
                  or previous_shape[2] != ice_array.dtype):
                raise ValueError('All of the ICE arrays need to be '
                                 'constructed for the same number of classes '
                                 'and the same number of samples for the '
                                 'selected feature (the second and the third '
                                 'dimension of the ice array).')
    else:
        raise TypeError('The ice_arrays_list should be a list of numpy arrays '
                        'that represent Individual Conditional Expectation.')

    ice_arrays = np.concatenate(ice_arrays_list, axis=0)
    return ice_arrays


def partial_dependence_ice(
        ice_array: np.ndarray,
        include_rows: Optional[Union[int, List[int]]] = None,
        exclude_rows: Optional[Union[int, List[int]]] = None) -> np.ndarray:
    """
    Calculates Partial Dependence based on Individual Conditional Expectations.

    .. note:: If you want to calculate Partial Dependence directly from a
       dataset and a classifier please see
       :func:`fatf.transparency.models.feature_influence.partial_dependence`
       function.

    Parameters
    ----------
    ice_array : numpy.ndarray
        Individual Conditional Expectation array for which Partial Dependence
        is desired.
    include_rows : Union[int, List[int]], optional (default=None)
        Indices of rows that will be included in the PD calculation. If this
        parameter is specified, PD will only be calculated for the selected
        rows. If additionally ``exclude_rows`` is specified the selected rows
        will be a set difference between the two. This parameter can either be
        a *list* of indices or a single index (integer).
    exclude_rows : Union[int, List[int]], optional (default=None)
        The indices of rows to be excluded from the PD calculation. If this
        parameter is specified and ``include_rows`` is not, these indices will
        be excluded from all of the rows. If both include and exclude
        parameters are specified, the rows included in the PD calculation will
        be a set difference of the two. This parameter can either be a *list*
        of indices or a single index (integer).

    Raises
    ------
    IncorrectShapeError
        The input array is not a 3-dimensional numpy array.
    TypeError
        Either ``include_rows`` or ``exclude_rows`` parameter is not ``None``,
        an integer or a list of integers.
    ValueError
        The ``ice_array`` is not an unstructured numpy array or it is not a
        numerical array. One of the ``include_rows`` or ``exclude_rows``
        indices is not valid for the input array.

    Returns
    -------
    partial_dependence_array : numpy.ndarray
        A 2-dimensional array of (steps_number, n_classes) shape representing
        Partial Dependence for all of the classes for selected rows (data
        points).
    """
    if fuav.is_structured_array(ice_array):
        raise ValueError('The ice_array should not be structured.')
    if not fuav.is_numerical_array(ice_array):
        raise ValueError('The ice_array should be purely numerical.')
    if len(ice_array.shape) != 3:
        raise IncorrectShapeError('The ice_array should be 3-dimensional.')

    rows_number = ice_array.shape[0]
    include_r = _filter_rows(include_rows, exclude_rows, rows_number)
    filtered_ice_array = ice_array[include_r]

    partial_dependence_array = filtered_ice_array.mean(axis=0)

    return partial_dependence_array


def partial_dependence(dataset: np.ndarray,
                       model: object,
                       feature_index: Union[int, str],
                       treat_as_categorical: Optional[bool] = None,
                       steps_number: Optional[int] = None,
                       include_rows: Optional[Union[int, List[int]]] = None,
                       exclude_rows: Optional[Union[int, List[int]]] = None
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates Partial Dependence for a selected feature.

    Partial Dependence [FRIEDMAN2001GREEDY]_ is computed as a mean value of
    Individual Conditional Expectations (c.f. :func:`fatf.transparency.models.\
feature_influence.individual_conditional_expectation`) over all the
    selected rows in the input dataset.

    The input parameters, exceptions and warnings match those used in
    :func:`fatf.transparency.models.feature_influence.\
individual_conditional_expectation` function.

    .. note:: If you wish to have access to both ICE and PDP results consider
       using :func:`fatf.transparency.models.feature_influence.\
individual_conditional_expectation` and :func:`fatf.transparency.models.\
feature_influence.partial_dependence_ice` functions to minimise the
       computational cost.

    .. [FRIEDMAN2001GREEDY] J. H. Friedman. Greedy function approximation: A
       gradient boosting machine. The Annals of Statistics, 29:1189–1232, 2001.
       URL https://projecteuclid.org/euclid.aos/1013203451. [p421, 428]

    Returns
    -------
    partial_dependence_array : numpy.ndarray
        A 2-dimensional array of (steps_number, n_classes) shape representing
        Partial Dependence for all of the classes for selected rows (data
        points).
    feature_linespace : numpy.ndarray
        A one-dimensional array -- (steps_number, ) -- with the values for
        which the selected feature was substituted when the dataset was
        evaluated with the specified model.
    """
    # pylint: disable=too-many-arguments
    ice_array, feature_linespace = individual_conditional_expectation(
        dataset,
        model,
        feature_index,
        treat_as_categorical=treat_as_categorical,
        steps_number=steps_number,
        include_rows=include_rows,
        exclude_rows=exclude_rows)

    partial_dependence_array = partial_dependence_ice(ice_array)

    return partial_dependence_array, feature_linespace
