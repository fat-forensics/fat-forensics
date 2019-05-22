"""
Functions for calculating feature influence for a predictive model.

This module implements Partial Dependence (PD) and Individual Conditional
Expectation (ICE) -- model agnostic feature influence measurements.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
#         Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

from typing import List, Optional, Tuple, Union

import warnings

import numpy as np
import scipy.stats

import fatf.utils.array.tools as fuat
import fatf.utils.array.validation as fuav
import fatf.utils.models.validation as fumv

from fatf.exceptions import IncompatibleModelError, IncorrectShapeError

__all__ = ['individual_conditional_expectation',
           'merge_ice_arrays',
           'compute_feature_distribution',
           'partial_dependence_ice',
           'partial_dependence']  # yapf: disable


def _input_is_valid(
        dataset: np.ndarray,
        feature_index: Union[int, str, List[int], List[str]],
        treat_as_categorical: Optional[Union[bool, List[bool]]],
        steps_number: Optional[Union[int, List[int]]]
) -> bool:  # yapf: disable
    """
    Validates input parameters of Individual Conditional Expectation function.

    For the input parameter description, warnings and exceptions please see the
    documentation of the :func`fatf.transparency.model.feature_influence.
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

    # Convert everything to a list, then check length and cycle through list.
    if steps_number is None or isinstance(steps_number, int):
        steps_number = [steps_number]
    if treat_as_categorical is None or isinstance(treat_as_categorical, bool):
        treat_as_categorical = [treat_as_categorical]
    if isinstance(feature_index, str) or isinstance(feature_index, int):
        feature_index = [feature_index]

    if len(feature_index) > 2 or len(feature_index) < 1:
        raise ValueError('feature_index has to be a single value or a list of '
                         'length two.')

    if not fuat.are_indices_valid(dataset, np.array(feature_index)):
        raise IndexError('Provided feature index is not valid for the input '
                         'dataset.')

    if len(steps_number) > 2 or len(steps_number) < 1:
        raise ValueError('steps_number has to be a single value or a list of '
                         'length two.')

    if len(treat_as_categorical) > 2 or len(treat_as_categorical) < 1:
        raise ValueError('treat_as_categorical has to be a single value or a '
                         'list of length two.')
    
    # Check lengths of lists agree
    if not len(treat_as_categorical) <= len(feature_index):
        raise ValueError('{} feature indices given but {} '
                         'treat_as_categorical values given. If one feature '
                         'index is given, treat_as_categorical must only be '
                         'one value.'.format(len(feature_index), 
                         len(treat_as_categorical)))
    if not len(steps_number) <= len(feature_index):
        raise ValueError('{} feature indices given but {} steps_number values '
                         'given. If one feature index is given, steps_number '
                         'must only be one value.'.format(len(feature_index),
                         len(steps_number)))

    for steps in steps_number:
        if isinstance(steps, int):
            if steps < 2:
                raise ValueError('steps_number has to be at least 2.')
        elif steps is None:
            pass
        else:
            raise TypeError('steps_number parameter has to either be None, an '
                            'integer or a list of None and integers.')
    for categorical in treat_as_categorical:
        if (not isinstance(categorical, bool) and categorical is not None):
            raise TypeError('treat_as_categorical has to either be None, a '
                            'boolean or a list of None and booleans.')

    is_input_ok = True
    return is_input_ok


def _generalise_dataset_type(
        dataset: np.ndarray,
        feature_index: Union[int, str],  # yapf: disable
        interpolated_values: np.ndarray,
        is_structured: bool) -> np.ndarray:
    """
    Generealises dataset dtype if needed.

    If the calculated interpolated_values are incompatible with the
    corresponding column in dataset, the dataset dtypes will be generalised
    to allow the interpolated array to have the correct interpolated_values.

    For example, if a feature in a dataset has the values [0, 1] of type 
    integer and the user would like to interpolate with steps size of 3, the
    interpolated values will be [0., 0.5, 1.]. However, these values are
    incompatible with type integer and as such we need to generalise the type
    of the feature column to float.

    Parameters
    ----------
    dataset : numpy.ndarray
        A dataset based on which data type generalisation will be done.
    feature_index : Union[integer, string]
        An index of the feature column in the input dataset for which the
        interpolation will be computed.
    interpolated_values : np.ndarray
        A 1-dimensional array of shape (steps_number, ) holding the
        interpolated values. If a numerical column is selected this will be a
        series of uniformly distributed ``steps_number`` values between the
        minimum and the maximum value of that column. For categorical (textual)
        columns it will hold all the unique values from that column.
    is_structured : boolean
        Indicates if the dataset is a structured array.

    Returns
    -------
    dataset : numpy.ndarray
        The input dataset with generalised dtypes. If dataset is structurued,
        then the dtype names will be the same as in the input dataset.
    """
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
    return dataset


def _get_feature_range(
        dataset: np.ndarray,
        feature_index: Union[int, str],  # yapf: disable
        treat_as_categorical: bool,
        steps_number: Union[int, None],
        is_structured: bool) -> np.ndarray:
    """
    Calculates feature range with correct step size.

    For the input parameter description, warnings and exceptions please see the
    documentation of the :func`fatf.transparency.model.feature_influence.
    _interpolate_array` function.

    Returns
    -------
    interpolated_values : np.ndarray
        A 1-dimensional array of shape (steps_number, ) holding the
        interpolated values. If a numerical column is selected this will be a
        series of uniformly distributed ``steps_number`` values between the
        minimum and the maximum value of that column. For categorical (textual)
        columns it will hold all the unique values from that column.
    steps_number : integer
        The number of evenly spaced samples between the minimum and the maximum
        value of the selected feature for which the model's prediction will be
        evaluated. This parameter applies only to numerical features, for
        categorical features regardless whether it is a number or ``None``, it
        will set as the number of unique values for the feature.
    """
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
    assert len(interpolated_values) == steps_number, 'Required for broadcast.'
    return interpolated_values, steps_number


def _interpolate_array(
        dataset: np.ndarray,
        feature_index: Union[int, str],  # yapf: disable
        treat_as_categorical: bool,
        steps_number: Union[int, None]
) -> Tuple[np.ndarray, Tuple[np.ndarray]]:
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

    interpolated_values, steps_number = _get_feature_range(
        dataset, feature_index, treat_as_categorical, steps_number,
        is_structured)
    dataset = _generalise_dataset_type(
        dataset, feature_index, interpolated_values, is_structured)

    interpolated_data = np.repeat(dataset[:, np.newaxis], steps_number, axis=1)
    if is_structured:
        for idx in range(steps_number):
            # Broadcast the new value.
            interpolated_data[:, idx][feature_index] = interpolated_values[idx]
    else:
        # Broadcast the new vector.
        interpolated_data[:, :, feature_index] = interpolated_values

    return interpolated_data, (interpolated_values)


def _interpolate_array_2d(
        dataset: np.ndarray,
        feature_index: Union[List[int], List[str]],  # yapf: disable
        treat_as_categorical: Optional[Union[bool, List[bool]]] = [None, None],
        steps_number: Optional[Union[int, List[int]]] = [None, None]
) -> Tuple[np.ndarray, Tuple[np.ndarray]]:
    """
    Generates a 4-D array with interpolated values for the selected features.

    If the selected feature is numerical the interpolated values are a
    numerical array with evenly spaced numbers between the minimum and the
    maximum value in that column. Otherwise, when the feature is categorical
    the interpolated values are all the unique elements of the that column.

    To get the interpolation the original 2-D dataset is stacked on top of
    itself the number of times equal to the number of desired interpolation
    samples. Then, for every copy of that dataset the selected feature is fixed
    to consecutive values of the interpolated array (the same value for the
    whole copy of the dataset). This action is performed twice for both
    features resulting in a 4-D array which contains every possible combination
    of the selected features and their interpolated values.

    Parameters
    ----------
    dataset : numpy.ndarray
        A dataset based on which interpolation will be done.
    feature_index : Union[List[integer], List[string]]
        list of indices of the feature columns in the input dataset for which
        the interpolation will be computed.
    treat_as_categorical : Union[boolean, List[boolean]]
        Whether to treat the selected feature as categorical or numerical.
    steps_number : Union[integer, None, List[integer]]
        The number of evenly spaced samples between the minimum and the maximum
        value of the selected feature for which the model's prediction will be
        evaluated. This parameter applies only to numerical features, for
        categorical features regardless whether it is a number or ``None``, it
        will be ignored.

    Returns
    -------
    interpolated_data : numpy.ndarray
        Numpy array of shape (n_samples, steps_number[0], steps_number[1],
        n_features) -- where the (n_samples, n_features) is the dimension of
        the input ``dataset`` -- holding the input ``dataset`` augmented with
        the interpolated values.
    interpolated_values : numpy.ndarray
        A list of 1-dimensional array of shape (steps_number[0], ) and
        (steps_number[1], ) holding the interpolated values. If a numerical
        column is selected this will be a series of uniformly distributed
        ``steps_number`` values between the minimum and the maximum value of
        that column. For categorical (textual) columns it will hold all the
        unique values from that column.
    """

    is_structured = fuav.is_structured_array(dataset)

    # Interpolate across one dimension
    sampled_data, feature_linespace = _interpolate_array(
        dataset, feature_index[0], treat_as_categorical[0],
        steps_number[0])

    assert isinstance(feature_index[1], (int, str)), \
         'Feature index -> str/ int.'
    assert isinstance(treat_as_categorical[1], bool), 'As categorical -> bool.'
    assert steps_number[1] is None or isinstance(steps_number[1], int), \
        'Steps number -> None/ int.'

    interpolated_values, steps_number[1] = _get_feature_range(
        dataset, feature_index[1], treat_as_categorical[1], steps_number[1],
        is_structured)

    sampled_data = _generalise_dataset_type(
        sampled_data, feature_index[1], interpolated_values, is_structured)

    interpolated_data = np.repeat(sampled_data[:, :, np.newaxis],
                                  steps_number[1], axis=2)

    if is_structured:
        for idx in range(steps_number[1]):
            # Broadcast the new value.
            interpolated_data[:, :, idx][feature_index[1]] = \
                interpolated_values[idx]
    else:
        # Broadcast the new vector.
        interpolated_data[:, :, :, feature_index[1]] = interpolated_values

    interpolated_values = (feature_linespace, interpolated_values)

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
        else:
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


def compute_feature_distribution(
        dataset: np.ndarray,
        feature_index: Union[int, str],
        treat_as_categorical: Optional[bool] = None,
        kde: Optional[bool] = False,
        samples: Optional[int] = None) -> np.ndarray:
    """
    Calculated distribution of values for a selected feature.

    This function computes a distribution, based off either constructing a
    histogram or fitting a Gaussian kernel via ``scipy.stats.gaussian_kde``,
    of the selected feature. If ``treat_as_categorical`` parameter is not
    provided the function will infer the type of the selected feature and
    compute the distribution. Otherwise, the user can specify whether the
    selected feature should be treated as a categorical or numerical feature.
    If the selected feature is numerical, you can specify the number of bins
    between this feature's minimum and maximum value that the function will use
    to compute a histogram. If the user does not specify the number of bins,
    Freedman Diaconis Estimator will be used to comptue the optimal number of
    bins. If ``kde`` is ``True``, instead this function will use 
    ``scipy.stats.gaussian_kde`` to fit a Gaussian kernel to the selected
    feature and the user can specify the number of samples to return, sampled
    from this kernel.

    Parameters
    ----------
    dataset : numpy.ndarray
        A dataset based on which ICE will be computed.
    feature_index : Union[integer, string]
        An index of the feature column in the input dataset for which ICE will
        be computed.
    treat_as_categorical : boolean, optional (default=None)
        Whether to treat the selected feature as categorical or numerical.
    kde : boolean, optional (default=False)
        Whether to do gaussian kernel density estimation using
        ``scipy.stats.gaussian_kde``. If false, then the data is sampled into
        bins.
    samples : integer, optional (default=None)
        If kde is True, samples in the number of points generated by sampling
        from the fitted Gaussian kernel. If kde is False, samples in the number
        of bins to use to histogram the data. If samples is None and kde is
        False, the number of bins is determined via Freedman Diaconis
        Estimator. If samples is None and kde is True, then 50 points will be
        sampled from the Gaussian Kernel.

    Warns
    -----
    UserWarning
        The feature is treated as categorical but the number of samples
        parameter is provided (not ``None``). In this case the ``samples``
        parameter is ignored. Also, the user is warned when the selected
        feature is detected to be categorical (textual) while the user 
        indicated that it is numerical.

    Raises
    ------
    IncorrectShapeError
        The input dataset is not a 2-dimensional numpy array.
    IndexError
        Provided feature (column) index is invalid for the input dataset.
    AssertionError
        kde is not a boolean, or samples is not an integer or the feature
        column is not a 1-dimensional array or feature array is not a base
        type.
    ValueError
        if ``treat_as_categorical`` is True and ``kde`` is True or the
        feature is found to be categorical and ``kde`` is set to True,
        ``kde`` cannot be used with categorical variables. 

    Returns
    -------
    values : numpy.ndarray
        Numpy array either of shape (samples, ) or (samples+1, ). If the data
        was separated into bins, then ``counts`` is an array containing the
        bin edges. If a Gaussian kernel was fitted, then ``counts`` is an array
        containing data that was sampled from the kernel used to generate the
        ``values`` array. If the feature was treated as categorical,
        ``counts`` is an array containing the unique values in the column.
    counts : numpy.ndarray
        Numpy array of shape (samples, ) returns a value of the probability
        density function at each of the bins in ``counts``. If the feature was
        treated as categorical, then ``values`` will be an array containing
        the unique values in the selected feature.
    """
    assert isinstance(kde, bool), 'kde must be a boolean.'
    if samples is not None:
        assert isinstance(samples, int), 'samples must be an integer.'

    if treat_as_categorical and kde:
        raise ValueError('treat_as_categorical was set to True and kde was '
                         'set to True. Gaussian kernel estimation cannot '
                         'be used on categorical data.')
    assert _input_is_valid(dataset, feature_index, treat_as_categorical, None)    
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

    if is_categorical_column and kde:
        raise ValueError('Selected feature is categorical '
                         '(string-base elements), however kde was set to '
                         'True. Gaussian kernel estimation cannot be used on '
                         'categorical data.')

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

    if treat_as_categorical and samples:
        warnings.warn(
            'The samples parameter will be ignored as the feature is '
            'being treated as categorical. The number of bins will be the '
            'number of unique values in the feature.',
            category=UserWarning)

    if not kde and samples is None and treat_as_categorical is False:
        samples = 'fd'
    if treat_as_categorical:
        values, counts = np.unique(column, return_counts=True)
        counts = counts / np.sum(counts)
    else:
        if kde:
            kernel = scipy.stats.gaussian_kde(column)
            values = np.linspace(column.min(), column.max(), samples)
            counts = kernel(values)
        else:
            counts, values = np.histogram(column, bins=samples,
                                                density=True)
            counts = counts / np.sum(counts)
    return values, counts


def _infer_is_categorical_steps_number(
        column: np.ndarray,
        treat_as_categorical: bool,
        steps_number: int,
) -> Tuple[bool, int]:
    """
    Checks if sampling parameters are compatible.

    This function checks if `treat_as_categorical` and `steps_number` are
    compatible with each other and the dataset. If they are not, function
    will return values that are compatible and raise warnings based off
    overwriting parameters. 

    For the input parameter description and warnings please see the
    documentation of the :func`fatf.transparency.model.feature_influence.
    individual_conditional_expectation` function.
    
    Returns
    -------
    treat_as_categorical : boolean
        Whether to treat the column as categorical or numerical.
    steps_number : integer
        The number of evenly spaced samples between the minimum and the maximum
        value of the selected feature for which the model's prediction will be
        evaluated. If `treat_as_categorical` is True then this will be None.
    """
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

    return treat_as_categorical, steps_number


def individual_conditional_expectation(
        dataset: np.ndarray,
        model: object,
        feature_index: Union[int, str, List[int], List[str]],
        mode: str = 'classifier',
        treat_as_categorical: Optional[Union[bool, List[bool]]] = None,
        steps_number: Optional[Union[int, List[int]]] = None,
        include_rows: Optional[Union[int, List[int]]] = None,
        exclude_rows: Optional[Union[int, List[int]]] = None
) -> Tuple[np.ndarray, Tuple[np.ndarray]]:
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
    By default this value is set to 100. For 2-D ICE, a list of feature
    indices, list of steps numbers and list of treat_as_categoricals can be
    provided.

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
    feature_index : Union[integer, string, List[integer], List[string]]
        An index of the feature column in the input dataset for which ICE will
        be computed. For 2-D ICE, a list of length two of feature indices can
        be specified.
    mode : string (default='classifier)
        Specifies whether the model should be treated as classifier or
        regressor.
    #TODO: what to do when parameter declaration longer than one line.
    treat_as_categorical : Union[boolean, List[boolean]], optional (default=None)
        Whether to treat the selected feature as categorical or numerical. For
        2-D ICE a list of booleans can be provided or just one boolean, where
        both treat_as_categorical will specify how to treat both features.
    steps_number : Union[integer, List[integer]], optional (default=None, i.e. 100)
        The number of evenly spaced samples between the minimum and the maximum
        value of the selected feature for which the model's prediction will be
        evaluated. For 2-D ICE, a list of integers can be provided or just one
        integer, where both the step_number will be used for both
        interpolations. (This parameter applies only to numerical features.)
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
        dataset. For 2-D ice, `ice` is of the shape (n_samples,
        steps_number[0], steps_number[1], n_classes. If `mode` parameter is
        regressor, n_classes will be 1. The numbers in this array represent the
        probability of every class for every selected data point when the
        selected feature is fixed to one of the values in the generated feature
        linespace (see below).
    # TODO: type of feature_linespace will vary depending on 2-D or not. 2-D
    cant return numpy.ndarray as the linespace can be different lengths
    feature_linespace : Tuple[numpy.ndarray]
        A tuple of one-dimension arrays -- (steps_number, ) -- with the values
        which the selected features have been substitieud when the dataset was
        evluated with the speicifed model. For 1-D ICE, this tuple will contain
        one element, for 2-D ICE will contain two elements.
    """
    # pylint: disable=too-many-arguments,too-many-locals
    assert _input_is_valid(dataset, feature_index, treat_as_categorical,
                           steps_number), 'Input must be valid.'

    is_2d = False
    if isinstance(feature_index, list) and len(feature_index) == 2:
        is_2d = True
    else:
        feature_index = [feature_index]

    if not isinstance(steps_number, list):
        steps_number = [steps_number] * 2 if is_2d else [steps_number]

    if not isinstance(treat_as_categorical, list):
        treat_as_categorical = [treat_as_categorical] * 2 if is_2d else \
            [treat_as_categorical]

    if mode not in ['classifier', 'regressor']:
        raise ValueError('Mode {} is not a valid mode. Mode should be '
                         '\'classifier\' for classification model or '
                         '\'regressor\' for regression model.'.format(mode))
    is_classifier = True if mode == 'classifier' else False

    if not fumv.check_model_functionality(
            model, require_probabilities=is_classifier):
        if is_classifier:
            raise IncompatibleModelError(
                'This functionality requires the classification model to be '
                'capable of outputting probabilities via predict_proba '
                'method.')
        else:
            raise IncompatibleModelError(
                'This functionaility requires the regression model to be '
                'capable of outputting predictions via predict method')

    function = model.predict_proba if is_classifier else model.predict

    is_structured = fuav.is_structured_array(dataset)

    column = [dataset[feature] if is_structured else dataset[:, feature]
              for feature in feature_index]

    # In order to do the same for 1-D and 2-D, make all variables a list
    # and infer treat_as_categorical and steps_number separately, then
    # unpack values.
    parameters = [_infer_is_categorical_steps_number(params[0],
                  params[1], params[2]) for params in 
                  zip(column, treat_as_categorical, steps_number)]
    treat_as_categorical, steps_number = map(list, zip(*parameters))
    rows_number = dataset.shape[0]
    include_r = _filter_rows(include_rows, exclude_rows, rows_number)
    filtered_dataset = dataset[include_r]

    if is_2d:
        sampled_data, feature_linespace = _interpolate_array_2d(
        filtered_dataset, feature_index, treat_as_categorical, steps_number)
        ice = [[
            function(data_slice)[:, np.newaxis] if not is_classifier# type:ignore
            else function(data_slice)
            for data_slice in data] for data in sampled_data
        ]
    else:
        sampled_data, feature_linespace = _interpolate_array(
            filtered_dataset, feature_index[0], treat_as_categorical[0],
            steps_number[0])
        # Predict returns one value so it's easier to treat regressor and
        # classifier the same if we add an axis to regression value.
        ice = [
            function(data_slice)[:, np.newaxis] if not is_classifier#type: ignore
            else function(data_slice)
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
            if len(ice_array.shape) != 3 and len(ice_array.shape) != 4:
                raise IncorrectShapeError(
                    'The ice_array should be 3-dimensional or 4-dimensional '
                    'for 2 feature ICE.')
            if previous_shape is None:
                previous_shape = (tuple(ice_array.shape[1:]) +
                                (ice_array.dtype,))  # yapf: disable
            elif (previous_shape[:-1] != ice_array.shape[1:]
                or previous_shape[-1] != ice_array.dtype):
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
        exclude_rows: Optional[Union[int, List[int]]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates Partial Dependence based on Individual Conditional Expectations.

    .. note:: If you want to calculate Partial Dependence directly from a
       dataset and a classifier please see
       :func:`transparency.models.feature_influence.partial_dependence`
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
    variance : numpy.ndarray
        A 1-d dimeionsal array of (steps_number, ) shape with the values for
        the variance of the predictions of data points for selected rows.
    """
    if fuav.is_structured_array(ice_array):
        raise ValueError('The ice_array should not be structured.')
    if not fuav.is_numerical_array(ice_array):
        raise ValueError('The ice_array should be purely numerical.')
    if len(ice_array.shape) != 3 and len(ice_array.shape) != 4:
        raise IncorrectShapeError('The ice_array should be 3-dimensional or '
                                  '4-dimensional for 2 feature ICE.')

    rows_number = ice_array.shape[0]
    include_r = _filter_rows(include_rows, exclude_rows, rows_number)
    filtered_ice_array = ice_array[include_r]

    partial_dependence_array = filtered_ice_array.mean(axis=0)
    variance = filtered_ice_array.var(axis=0)

    return partial_dependence_array, variance


def partial_dependence(
        dataset: np.ndarray,
        model: object,
        feature_index: Union[int, str, List[int], List[str]],
        mode: str = 'classifier',
        treat_as_categorical: Optional[Union[bool, List[bool]]] = None,
        steps_number: Optional[Union[int, List[int]]] = None,
        include_rows: Optional[Union[int, List[int]]] = None,
        exclude_rows: Optional[Union[int, List[int]]] = None
) -> Tuple[np.ndarray, Tuple[np.array], np.ndarray]:
    """
    Calculates Partial Dependence for a selected feature.

    Partial Dependence [FRIEDMAN2001GREEDY]_ is computed as a mean value of
    Individual Conditional Expectations (c.f. :func:`fatf.transparency.models.
    feature_influence.individual_conditional_expectation`) over all the
    selected rows in the input dataset.

    The input parameters, exceptions and warnings match those used in
    :func:`fatf.transparency.models.feature_influence.
    individual_conditional_expectation` function.

    .. note:: If you wish to have access to both ICE and PDP results consider
       using :func:`transparency.models.feature_influence.
       individual_conditional_expectation` and
       :func:`transparency.models.feature_influence.partial_dependence_ice`
       functions to minimise the Computational cost.

    .. [FRIEDMAN2001GREEDY] J. H. Friedman. Greedy function approximation: A
       gradient boosting machine. The Annals of Statistics, 29:1189–1232, 2001.
       URL https://doi.org/10.1214/aos/1013203451. [p421, 428]

    Returns
    -------
    partial_dependence_array : numpy.ndarray
        A 2-dimensional array of (steps_number, n_classes) shape representing
        Partial Dependence for all of the classes for selected rows (data
        points).
    feature_linespace : Tuple[numpy.ndarray]
        A tuple of one-dimension arrays -- (steps_number, ) -- with the values
        which the selected features have been substitieud when the dataset was
        evluated with the speicifed model. For 1-D ICE, this tuple will contain
        one element, for 2-D ICE will contain two elements.
    variance : numpy.ndarray
        A 1-d dimeionsal array of (steps_number, ) shape with the values for
        the variance of the predictions of data points for selected rows.
    """
    # pylint: disable=too-many-arguments
    ice_array, feature_linespace = individual_conditional_expectation(
        dataset,
        model,
        feature_index,
        mode,
        treat_as_categorical=treat_as_categorical,
        steps_number=steps_number,
        include_rows=include_rows,
        exclude_rows=exclude_rows)
    partial_dependence_array, variance = partial_dependence_ice(ice_array)
    return partial_dependence_array, feature_linespace, variance
