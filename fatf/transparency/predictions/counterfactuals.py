"""
The :mod:`fatf.transparency.predictions.counterfactuals` module implements
counterfactual explainers for predictions.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
#         Rafael Poyiadzi <rp13102@bristol.ac.uk>
# License: new BSD

# pylint: disable=too-many-lines

import inspect
import itertools
import warnings

from numbers import Number
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

import fatf.utils.models.validation as fumv
import fatf.utils.array.tools as fuat
import fatf.utils.array.validation as fuav
import fatf.utils.tools as fut

from fatf.exceptions import IncorrectShapeError

FeatureRange = Union[Tuple[float, float], List[Union[float, str]]]
Index = Union[int, str]

__all__ = ['CounterfactualExplainer', 'textualise_counterfactuals']

_NUMPY_VERSION = [int(i) for i in np.version.version.split('.')]
_NUMPY_1_13 = fut.at_least_verion([1, 13], _NUMPY_VERSION)


class CounterfactualExplainer(object):
    """
    Generates counterfactual explanations of black-box classifier predictions.

    Finds counterfactual explanations for an arbitrary black-box **classifier**
    by a brute-force grid search with a specified step size for a selected
    range of values for selected features in the dataset.

    In order to generate the counterfactuals either a ``model`` or a
    ``predictive_function`` must be given.

    If a ``dataset`` is given, then only one of the two column indices
    parameters is required: ``categorical_indices`` or ``numerical_indices``.
    If a dataset is not given, both these parameters need to be specified.

    If only some of the features are desired to appear in the counterfactuals,
    then these may be specified in the ``counterfactual_feature_indices``
    parameter.

    .. note::
       Valid feature (column) indices are either strings for structured arrays
       or integers for normal numpy arrays.

    The user may also wish to restrict the length of any generated
    counterfactual by providing the ``max_counterfactual_length`` parameter.

    If a ``dataset`` is given, then the feature ranges will be obtained by
    taking the minimum and the maximum value for numerical features and all the
    unique values for categorical (textual) features. Alternatively, the user
    may wish to define feature ranges by using the ``feature_ranges``
    parameter. All undefined feature ranges will be filled in automatically
    given that a ``dataset`` is provided. If some of the feature ranges are not
    defined and a ``dataset`` is not given an exception will be raised.
    **Counterfactuals will only be search for within these feature ranges.**
    Ranges are only required for features specified by the
    ``counterfactual_feature_indices`` parameter or all features if this
    parameter is not given.

    For a given feature combination only the counterfactual(s) closest to the
    specified data point will be retrieved. By default the distance for every
    numerical feature is taken to be :math:`abs(x_i - \\hat{x}_i)` and for
    every categorical feature it is an identity function, i.e. :math:`1` if the
    value does not agree and :math:`0` if it agrees. If custom distance
    functions are desired, the user may specify these via the
    ``distance_functions`` parameter. Each distance function has to be a
    ``Callable`` with two input parameters. Finally, the distance can be
    normalised, please see the documentation of the ``_get_distance`` method
    for details.

    Last but not least, when doing grid search through the features to discover
    counterfactual data points the user may define the step size between the
    minimum and the maximum value for the numerical features. This can be done
    selectively for every single feature separately via the ``step_sizes``
    parameter. For all of the features that step size is not defined the
    default value (:math:`1`) will be taken -- this can be changed via the
    ``default_numerical_step_size`` parameter.

    Parameters
    ----------
    model : object, optional (default=None)
        A predictive model object that has a ``predict`` method.
    predictive_function : Callable, optional (default=None)
        A function that takes in a 2-dimensional data array and returns class
        predictions. (Alternative to the ``model`` parameter.)
    dataset : numpy.ndarray, optional (default=None)
        A 2-dimensional data array representing a dataset used for the problem
        modeling. It is advised to use the same dataset as for the training of
        the ``model`` object.
    categorical_indices : List[column indices], optional (default=None)
        A list of column indices indicating which columns are categorical.
    numerical_indices : List[column indices], optional (default=None)
        A list of column indices indicating which columns are numerical.
    counterfactual_feature_indices : List[column indices], \
optional (default=None)
        A list of column indices indicating which features should be used to
        compose counterfactual examples. If `None`` all of the features will be
        used to generate counterfactuals.
    max_counterfactual_length : integer, optional (default=2)
        The maximum length of counterfactuals -- the number of features altered
        to compose a counterfactual instance. By default it is set to 2. If set
        to 0, all available features will be used.
    feature_ranges : Dictionary[column indices, ranges], \
optional (default=None)
        A dictionary with keys representing the column (feature) indices and
        the values representing feature ranges. Numerical feature ranges are
        represented as a **pair** of numbers ``(min, max)`` where the first
        number is the lower bound of the range and the second number is the
        upper bound of the range. For categorical features this should be a
        **list** of all the values that to be tested for this feature. If set
        to ``None``, a ``dataset`` has to be provided to calculate these
        ranges.
    distance_functions : Dictionary[column indices, Callable], \
optional (default=None)
        A dictionary with keys representing the column (feature) indices and
        the values representing Python functions -- a Callable that takes two
        arguments -- that will be used to calculate the distance for this
        particular feature.
    step_sizes : Dictionary[column indices, Number], \
optional (default=None)
        A dictionary with keys representing the column (feature) indices and
        the values representing step size for the grid search of this feature.
        **It is only required for the numerical features.**
    default_numerical_step_size : Number, optional (default=1)
        The default step size used with the grid search of numerical features
        when generating counterfactuals.

    Warns
    -----
    UserWarning
        The value of the ``max_counterfactual_length parameter`` is larger than
        the number of features. A step size (via the ``step_sizes`` parameter)
        is provided for one of the categorical features. Both a ``model`` and a
        ``predictive_function`` parameters are supplied. When discovering
        categorical feature ranges from the ``dataset`` there is only one
        unique value for any particular feature.

    Raises
    ------
    AttributeError
        The ``predictive_function`` parameter, if given, does not require 2
        non-optional parameters. One of the distance functions provided via the
        ``distance_functions`` parameter does not require 2 non-optional
        parameters.
    IncorrectShapeError
        The ``dataset`` array is not 2-dimensional.
    IndexError
        Some of the ``categorical_indices`` or ``numerical_indices`` are not
        valid for the input ``dataset``, when the latter is given. When both
        ``categorical_indices`` and ``numerical_indices`` parameters are given
        alongside a ``dataset`` they do not cover all of the ``dataset``
        arrays' column indices. The union of categorical and numerical indices
        does not form a series of consecutive integers when the ``dataset``
        array is a classic numpy array. Some of the
        ``counterfactual_feature_indices`` are not valid. Some of the indices
        (dictionary keys) in the ``feature_ranges``, ``distance_functions`` or
        ``step_sizes`` parameters are not valid (consistent with the provided
        column indices).
    RuntimeError
        The ``model`` object, if provided, lacks a ``predict`` method. Neither
        ``model`` nor ``predictive_function`` was specified -- one of these is
        required.
    TypeError
        The ``predictive_function`` parameter is not Python callable, i.e. a
        Python function. The ``categorical_indices`` parameter is neither a
        list nor ``None``. The ``numerical_indices`` parameter is neither a
        list nor ``None``. Some of the indices given in these two lists do not
        share a common type -- only all strings or all integers are allowed.
        The ``counterfactual_feature_indices`` parameter is neither a list nor
        ``None``. The ``max_counterfactual_length`` parameter is not an
        integer. The ``feature_ranges`` parameter is neither a dictionary nor
        nor ``None``. A feature range is not a list for a categorical feature
        or a feature range is not a tuple for a numerical feature. One of the
        numerical range tuple elements is not a number or all of the elements
        of a categorical feature range do not share the same type. The
        ``distance_functions`` parameter is not a dictionary. One of the
        distance functions defined via the ``distance_functions`` parameter is
        not a Python callable. The ``step_sizes`` parameter is not dictionary.
        One of the step sizes defined via the ``step_sizes`` parameter is not
        a number. The ``default_numerical_step_size`` parameter is not a
        number.
    ValueError
        Some of the categorical (textual) features in the ``dataset`` array
        (when given) are not indicated by the user -- given via the
        ``categorical_indices`` parameter -- to be categorical (it is not
        possible to treat textual fields as numerical features. Some of the
        categorical features in the ``dataset`` array are selected to be
        numerical via the ``numerical_indices`` parameter. Some of the feature
        ranges are missing and need to be computed from a ``dataset`` but none
        is given. ``categorical_indices`` and ``numerical_indices``
        parameters were not provided in the absence of a ``dataset``. The
        ``dataset`` array is not of a base type (strings and/or numbers). Both
        ``categorical_indices`` and ``numerical_indices`` parameters are empty
        lists. Both of these lists share some common indices. The
        ``counterfactual_feature_indices`` parameter is an empty list. The
        ``max_counterfactual_length`` parameter is not a non-negative integer.
        The ``feature_ranges`` parameter is an empty dictionary. One of the
        categorical ranges provided is an empty list. One of the numerical
        ranges is a tuple of length different than 2 or the second element of
        the range tuple is not strictly larger than the first one. The
        ``distance_functions`` parameter is an empty dictionary. The
        ``step_sizes`` parameter is an empty dictionary. Some of the step sizes
        specified via the ``step_sizes`` dictionary are not strictly positive
        numbers. The ``default_numerical_step_size`` parameter is not a
        strictly positive number. When discovering feature ranges from the
        ``dataset`` there is only one value for a numerical feature meaning
        that a range cannot be created.

    Attributes
    ----------
    predict : Callable
        A function used to predict the class of counterfactuals.
    all_indices : Set[column indices]
        A set of all the column (feature) indices in the data set from which
        counterfactuals are generated.
    categorical_indices : Set[column indices]
        A set of categorical columns (feature) indices in the data set.
    numerical_indices : Set[column indices]
        A set of numerical columns (feature) indices in the data set.
    cf_feature_indices : Set[column indices]
        A set of column (feature) indices that will be used to generate
        counterfactuals -- only alterations of these features will be searched
        to generate counterfactuals.
    feature_ranges : Dictionary[column indices, ranges]
        A dictionary with ranges for all of the ``cf_feature_indices``.
    max_counterfactual_length : Number
        The maximum length -- the number of features altered -- of a
        counterfactual instance.
    distance_functions : Dictionary[column indices, Callable]
        A dictionary with distance functions for all of the
        ``cf_feature_indices``.
    step_sizes : Dictionary[column indices, Numbers]
        A dictionary with step sizes for all of the numerical features in the
        ``cf_feature_indices``.
    """
    # pylint: disable=useless-object-inheritance,too-many-instance-attributes
    # pylint: disable=too-few-public-methods

    __all__ = ['explain_instance']

    # Whether out-of-range warning has been issued for a particular feature.
    # Used to avoid duplicated feature warnings.
    _feature_warned = dict()  # type: Dict[Index, bool]

    def __init__(self,
                 model: Optional[object] = None,
                 predictive_function: Optional[Callable] = None,
                 dataset: Optional[np.ndarray] = None,
                 categorical_indices: Optional[List[Index]] = None,
                 numerical_indices: Optional[List[Index]] = None,
                 counterfactual_feature_indices: Optional[List[Index]] = None,
                 max_counterfactual_length: int = 2,
                 feature_ranges: Optional[Dict[Index, FeatureRange]] = None,
                 distance_functions: Optional[Dict[Index, Callable]] = None,
                 step_sizes: Optional[Dict[Index, float]] = None,
                 default_numerical_step_size: float = 1.0) -> None:
        """
        Initialises a counterfactual explainer.
        """
        # pylint: disable=too-many-arguments,too-many-locals,too-many-branches
        # pylint: disable=too-many-statements

        # Validate input
        assert _validate_input_one(
            model, predictive_function, dataset, categorical_indices,
            numerical_indices), 'The input must be valid.'

        # Select a predictive function
        if predictive_function is None:
            self.predict = model.predict  # type: ignore
        else:
            self.predict = predictive_function

        # Choose categorical and numerical indices
        if dataset is not None:
            num_ind, cat_ind = fuat.indices_by_type(dataset)
            num_ind = set(num_ind.tolist())
            cat_ind = set(cat_ind.tolist())
            all_indices = num_ind.union(cat_ind)

            if categorical_indices is None and numerical_indices is not None:
                _numerical_indices = set(numerical_indices)
                _categorical_indices = all_indices.difference(
                    _numerical_indices)
                # They have to be valid indices for the array
                if _numerical_indices.difference(all_indices):
                    raise IndexError('Some of the numerical_indices are not '
                                     'valid for the given array.')
            elif categorical_indices is not None and numerical_indices is None:
                _categorical_indices = set(categorical_indices)
                _numerical_indices = all_indices.difference(
                    _categorical_indices)
                # They have to be valid indices for the array
                if _categorical_indices.difference(all_indices):
                    raise IndexError('Some of the categorical_indices are not '
                                     'valid for the given array.')
            elif (categorical_indices is not None
                  and numerical_indices is not None):
                _categorical_indices = set(categorical_indices)
                _numerical_indices = set(numerical_indices)
                if all_indices.difference(_categorical_indices).difference(
                        _numerical_indices):
                    raise IndexError('The numerical_indices and the '
                                     'categorical_indices parameters do not '
                                     'cover all of the columns in the given '
                                     'dataset array.')
            else:
                _categorical_indices = cat_ind
                _numerical_indices = num_ind

            if (cat_ind.difference(_categorical_indices)
                    and categorical_indices is not None):
                raise ValueError('Some of the categorical indices (textual '
                                 'columns) in the array were not indicated to '
                                 'be categorical by the user. Textual columns '
                                 'must not be treated as numerical features.')
            if (cat_ind.intersection(_numerical_indices)
                    and numerical_indices is not None):
                raise ValueError('Some of the categorical fields in the input '
                                 'data set were indicated to be numerical '
                                 'indices via the numerical_indices '
                                 'parameter. Textual columns must not be '
                                 'treated as numerical features.')
        else:
            if categorical_indices is None or numerical_indices is None:
                raise ValueError('If a dataset is not given, both '
                                 'categorical_indices and numerical_indices '
                                 'parameters have to be defined.')
            _categorical_indices = set(categorical_indices)
            _numerical_indices = set(numerical_indices)
            all_indices = _categorical_indices.union(_numerical_indices)
        # ...unstructured numpy array's indices should be consecutive numbers
        if isinstance(next(iter(all_indices)), int):
            if all_indices != set(range(max(all_indices) + 1)):
                raise IndexError('The union of categorical and numerical '
                                 'indices does not form a series of '
                                 'consecutive integers. This is required for '
                                 'an classic (unstructured) numpy array.')
        self.all_indices = all_indices
        self.categorical_indices = _categorical_indices
        self.numerical_indices = _numerical_indices

        assert _validate_input_two(
            _numerical_indices, _categorical_indices, all_indices,
            counterfactual_feature_indices, max_counterfactual_length,
            feature_ranges, distance_functions, step_sizes,
            default_numerical_step_size), 'The input must be valid.'

        # Get feature indices to be used for counterfactual generation
        if counterfactual_feature_indices is None:
            self.cf_feature_indices = self.all_indices
        else:
            self.cf_feature_indices = set(counterfactual_feature_indices)

        # Sort out feature ranges
        if feature_ranges is None:
            feature_ranges = self._get_feature_ranges(dataset)
            assert self._validate_auto_ranges(feature_ranges), 'Bad ranges.'
        else:
            # Get needed ranges
            needed_ranges_indices = self.cf_feature_indices.difference(
                feature_ranges.keys())
            missing_ranges = self._get_feature_ranges(
                dataset, column_indices=needed_ranges_indices)
            assert self._validate_auto_ranges(missing_ranges), 'Bad ranges.'

            # Merge feature_ranges and missing_ranges dictionaries
            feature_ranges = {**missing_ranges, **feature_ranges}
        self.feature_ranges = feature_ranges
        # ...validate ranges availability
        for i in self.cf_feature_indices:
            assert i in feature_ranges, 'A cf feature is missing a range.'

        # The maximum number of features that a counterfactual will be based on
        if max_counterfactual_length == 0:
            max_counterfactual_length = len(self.all_indices)
        else:
            if max_counterfactual_length > len(self.all_indices):
                warnings.warn(
                    'The value of the max_counterfactual_length parameter is '
                    'larger than the number of features. It will be clipped.',
                    UserWarning)
                max_counterfactual_length = len(self.all_indices)
        self.max_counterfactual_length = max_counterfactual_length

        # Sort out step sizes
        if step_sizes is None:
            step_sizes = {
                numerical_feature_idx: default_numerical_step_size
                for numerical_feature_idx in self.numerical_indices
                if numerical_feature_idx in self.cf_feature_indices
            }
        else:
            step_sizes = dict(step_sizes)  # Make a copy since we are editing
            steps_indices = set(step_sizes.keys())
            cat_steps_idcs = steps_indices.difference(self.numerical_indices)
            if cat_steps_idcs:
                warnings.warn(
                    'Step size was provided for one of the categorical '
                    'features. Ignoring these ranges.', UserWarning)
                steps_indices = steps_indices.difference(cat_steps_idcs)
                for idx in cat_steps_idcs:
                    del step_sizes[idx]

            missing_steps = self.cf_feature_indices.difference(steps_indices)
            for missing_step_index in missing_steps:
                step_sizes[missing_step_index] = default_numerical_step_size
        self.step_sizes = step_sizes

        # Sort out distance functions
        if distance_functions is None:
            distance_functions = dict()
            for idx in self.all_indices:
                if idx in self.categorical_indices:
                    distance_functions[idx] = _categorical_distance
                else:
                    distance_functions[idx] = _numerical_distance
        else:
            distance_functions = dict(distance_functions)  # Copy, gets edited
            functions_indices = set(distance_functions.keys())
            missing_functions = self.all_indices.difference(functions_indices)
            for idx in missing_functions:
                if idx in self.categorical_indices:
                    distance_functions[idx] = _categorical_distance
                else:
                    distance_functions[idx] = _numerical_distance
        self.distance_functions = distance_functions

    def _get_feature_ranges(self,
                            dataset: Union[None, np.ndarray],
                            column_indices: Optional[Set[Index]] = None
                            ) -> Dict[Index, FeatureRange]:
        """
        Calculates ranges of selected features based on the ``dataset`` array.

        For numerical feature the range is a ``(min, max)`` tuple with the
        minimum and the maximum value for a given column. For a categorical
        feature the range is a list of all the unique values in that column.

        Parameters
        ----------
        dataset : numpy.ndarray (or None)
            A dataset used to compute ranges for the selected features.
        column_indices : Set[column indices], optional (default=None)
            If ``None``, ranges of all the counterfactual features
            (defined by the ``cf_feature_indices`` attribute) will be computed.
            Otherwise, ranges will be computed for the indicated columns.

        Raises
        ------
        ValueError
            A dataset is not given. This is usually raised when some of the
            feature ranges are missing but a dataset was not provided to
            calculate them.

        Returns
        -------
        ranges : Dictionary[column indices, ranges]
            A dictionary with the keys corresponding to the column indices
            (integers for a classic numpy array and string for a structured
            array) and the values representing ranges of a given column
            feature.
        """
        ranges = dict()  # type: Dict[Index, FeatureRange]
        if column_indices is not None and not column_indices:
            return ranges

        if dataset is None:
            raise ValueError('A dataset is needed to fill in feature ranges '
                             'for features selected for counterfactuals that '
                             'were not provided with ranges via '
                             'feature_ranges parameter. If you do not want to '
                             'provide a dataset please specify counterfactual '
                             'feature ranges via feature_ranges parameter.')
        assert isinstance(dataset, np.ndarray), 'Dataset is not a numpy array.'
        # Needed object attributes are in place
        assert self.all_indices is not None, 'All indices missing.'
        assert self.numerical_indices is not None, \
            'Numerical indices missing.'
        assert self.categorical_indices is not None, \
            'Categorical indices missing.'
        assert self.cf_feature_indices is not None, \
            'Counterfactual indices missing.'
        if column_indices is not None:
            assert isinstance(column_indices, set), \
                'Column indices is not a set.'
            # Requested indices are valid
            assert not column_indices.difference(self.all_indices), \
                'Requested indices are invalid.'

        dataset_is_structured = fuav.is_structured_array(dataset)

        if column_indices is None:
            column_indices = self.cf_feature_indices

        for column_index in column_indices:
            if column_index in self.numerical_indices:
                if dataset_is_structured:
                    column_min = np.nanmin(dataset[column_index])
                    column_max = np.nanmax(dataset[column_index])
                else:
                    column_min = np.nanmin(dataset[:, column_index])
                    column_max = np.nanmax(dataset[:, column_index])
                column_range = (column_min, column_max)
            elif column_index in self.categorical_indices:
                if dataset_is_structured:
                    column_unique = np.unique(dataset[column_index])
                else:
                    column_unique = np.unique(dataset[:, column_index])
                column_range = column_unique.tolist()
            else:
                assert False, 'Invalid column index.'  # pragma: nocover
            ranges[column_index] = column_range
        return ranges

    def _validate_auto_ranges(
            self, discovered_ranges: Dict[Index, FeatureRange]) -> bool:
        """
        Validates the feature ranges discovered from a dataset.

        For the description of the ``UserWarning`` and the ``ValueError``
        please see the documentation of the class.

        Parameters
        ----------
        discovered_ranges : Dictionary[column indices, ranges]
            A dictionary with feature ranges calculated based on the input
            ``dataset``.

        Returns
        -------
        are_ranges_valid : boolean
            ``True`` if the ranges are valid, ``False`` otherwise.
        """
        assert self.categorical_indices is not None, 'Required for validation.'
        are_ranges_valid = False

        for rng in discovered_ranges:
            if rng in self.categorical_indices:
                if len(discovered_ranges[rng]) < 2:
                    warnings.warn(
                        'There is only one unique value detected for the '
                        'categorical feature *{}*: {}.'.format(
                            rng, discovered_ranges[rng]), UserWarning)
            else:
                assert (discovered_ranges[rng][0]  # type: ignore
                        <= discovered_ranges[rng][1]), 'Incompatible range.'
                if discovered_ranges[rng][0] == discovered_ranges[rng][1]:
                    raise ValueError('The minimum and the maximum detected '
                                     'value for feature *{}* are the same '
                                     '({}). Impossible to create a '
                                     'range.'.format(
                                         rng, discovered_ranges[rng][1]))
        are_ranges_valid = True
        return are_ranges_valid

    def _get_distance(self,
                      instance_one: Union[np.ndarray, np.void],
                      instance_two: Union[np.ndarray, np.void],
                      normalise: bool = False) -> float:
        """
        Calculates a distance between two 1-dimensional data points.

        The distance can be normalised by squaring the distance between every
        single feature and then taking the square root of the sum (like
        Euclidean distance).

        Parameters
        ----------
        instance_one : numpy.ndarray or numpy.void
            A 1-dimensional numpy array representing a data point.
        instance_two : numpy.ndarray or numpy.void
            A 1-dimensional numpy array representing a data point.
        normalise : boolean, optional (default=False)
            Whether to normalise the distance.

        Returns
        -------
        distance : Number
            A number representing the distance between the two input data
            points.
        """
        assert fuav.is_1d_like(instance_one), 'Should be 1-dimensional.'
        assert fuav.is_1d_like(instance_two), 'Should be 1-dimensional.'
        assert isinstance(normalise, bool), 'Should be a boolean.'
        distances = []
        for feature in self.all_indices:
            idx_dist = self.distance_functions[feature](instance_one[feature],
                                                        instance_two[feature])
            assert isinstance(idx_dist, Number), 'Distances must be numerical.'
            distances.append(idx_dist)
        if normalise:
            distance = np.sqrt(np.power(distances, 2).sum())
        else:
            distance = np.sum(distances)
        return distance

    def _get_neighbouring_instances(
            self, instance: Union[np.ndarray, np.void],
            features_combination: Tuple[Union[str, int], ...]) -> np.ndarray:
        """
        Generates all neighbouring instances with ranges of selected features.

        Generates instances with all possible value combinations for selected
        feature indices. The possible values are taken from the feature ranges
        calculated when the object was created.

        Parameters
        ----------
        instance : numpy.ndarray or numpy.void
            A 1-dimensional numpy array representing a data point for which
            neighbouring instances are desired.
        features_combination : Tuple(column indices)
            A tuple with feature indices for which possible value combinations
            will be generated.

        Returns
        -------
        cf_instances : numpy.ndarray
            A 2-dimensional numpy array with neighbouring data points or an
            empty array if none could be generated.
        """
        # pylint: disable=too-many-locals,too-many-branches
        assert features_combination, 'Must be at least one feature.'
        warning_msg = ('The value ({}) of *{}* feature for this instance is '
                       'out of the specified {}.')
        possible_features_ranges = []
        for feature in features_combination:
            if feature in self.categorical_indices:
                feature_value = instance[feature]
                feature_ranges = self.feature_ranges[feature]
                # Get all other possible categorical values
                feature_range = np.array(
                    [i for i in feature_ranges if i != feature_value])
                possible_features_ranges.append(feature_range)

                if (feature_value not in feature_ranges
                        and not self._feature_warned[feature]):
                    self._feature_warned[feature] = True
                    complement = 'values: {}'.format(feature_ranges)
                    warning_msg = warning_msg.format(feature_value, feature,
                                                     complement)
                    warnings.warn(warning_msg, UserWarning)

            else:
                feature_range_min = self.feature_ranges[feature][0]
                feature_range_max = self.feature_ranges[feature][1]
                feature_value = instance[feature]

                if ((feature_value < feature_range_min
                     or feature_value > feature_range_max)
                        and not self._feature_warned[feature]):
                    self._feature_warned[feature] = True
                    complement = 'min-max range: {}-{}'.format(
                        feature_range_min, feature_range_max)
                    warning_msg = warning_msg.format(feature_value, feature,
                                                     complement)
                    warnings.warn(warning_msg, UserWarning)

                feature_range = np.arange(feature_range_min, feature_range_max,
                                          self.step_sizes[feature])
                possible_features_ranges.append(feature_range)

        # Generalise the type if necessary
        if fuav.is_structured_array(np.array([instance])):
            new_types = []
            for name in instance.dtype.names:
                if name in features_combination:
                    index = features_combination.index(name)
                    dtype = fuat.generalise_dtype(
                        possible_features_ranges[index].dtype,
                        instance.dtype[name])
                    new_types.append((name, dtype))
                else:
                    new_types.append((name, instance.dtype[name]))
            instance = instance.astype(new_types)
        else:
            dtype = possible_features_ranges[0].dtype
            for i in possible_features_ranges:
                dtype = fuat.generalise_dtype(dtype, i.dtype)
            dtype = fuat.generalise_dtype(dtype, instance.dtype)
            instance = instance.astype(dtype)

        # Create alternative data points
        cf_instances = []
        for value_combination in itertools.product(*possible_features_ranges):
            cf_instance = instance.copy()
            for cf_index, feature in enumerate(features_combination):
                cf_instance[feature] = value_combination[cf_index]
            cf_instances.append(cf_instance)

        # Combine alternative data points into a single numpy array
        if cf_instances:
            cf_instances = np.stack(cf_instances)
        else:
            cf_instances = np.array([])

        return cf_instances

    def explain_instance(
            self,
            instance: Union[np.ndarray, np.void],
            counterfactual_class: Optional[Union[int, str]] = None,
            normalise_distance: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Finds counterfactual data points, their class and distance.

        Returns a numpy array with counterfactual instances, their predicted
        classes and distances for a given data ``instance``. The counterfactual
        class can be selected by the user, otherwise all possible classes are
        considered.

        Parameters
        ----------
        instance : numpy.ndarray or numpy.void
            A 1-dimensional numpy array representing a data point for which
            counterfactuals are desired.
        counterfactual_class : string or integer, optional (default=None)
            A class of counterfactual instances. If ``None`` counterfactuals of
            all classes other than the predicted class of the input
            ``instance`` will be returned.
        normalise_distance : boolean, optional (default=False)
            Whether to normalise the distance, cf. the ``_get_distance`` method
            for more details.

        Raises
        ------
        IncorrectShapeError
            The input ``instance`` is not a 1-dimensional numpy array.
        IndexError
            The indices that were used to initialise this class are not valid
            for the given input ``instance``.
        TypeError
            The ``counterfactual_class`` parameter is neither string not
            integer. The ``normalise_distance`` parameter is not a boolean.
        ValueError
            The input ``instance`` is not of a base type (string and/or
            integer).

        Warns
        -----
        UserWarning
            When generating counterfactuals the value of one of the features
            for the specified input ``instance`` is outside of the specified
            range for this feature.

        Returns
        -------
        counterfactuals : numpy.ndarray
            A 2-dimensional numpy array with counterfactual data points.
        counterfactuals_distances : numpy.ndarray
            A 1-dimensional numpy array with distances from the input
            ``instance`` to every counterfactual data point.
        counterfactuals_predictions : numpy.ndarray
            A 1-dimensional numpy array with predictions for every
            counterfactual data point.
        """
        # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        if not fuav.is_1d_like(instance):
            raise IncorrectShapeError('The instance to be explained should be '
                                      'a 1-dimensional numpy array or a row '
                                      'of a structured array (numpy.void).')
        instance_2d = np.array([instance])
        if not fuav.is_base_array(instance_2d):
            raise ValueError('The instance should be of a base type -- a '
                             'mixture of numerical and textual types.')
        if not fuat.are_indices_valid(instance_2d,
                                      np.array(list(self.all_indices))):
            raise IndexError('The indices used to initialise this class are '
                             'not valid for this data point.')
        if counterfactual_class is not None:
            if not isinstance(counterfactual_class, (int, str)):
                raise TypeError('The counterfactual class should be either an '
                                'integer or a string.')
        if not isinstance(normalise_distance, bool):
            raise TypeError('The normalise_distance parameter should be a '
                            'boolean.')

        # Prepare out-of-range warnings
        self._feature_warned = {key: False for key in self.cf_feature_indices}

        if counterfactual_class is None:
            # Predict the class, counterfactuals will be of any different class
            current_prediction = self.predict(instance_2d)
            assert current_prediction.shape[0] == 1, 'Just one prediction.'
            current_class = current_prediction[0]
        else:
            # The counterfactual class is defined
            current_class = None

        counterfactuals = []
        counterfactuals_distances = []
        counterfactuals_predictions = []
        for cf_length in range(self.max_counterfactual_length):
            cf_features_combinations = itertools.combinations(
                self.cf_feature_indices, cf_length + 1)  # +1 as counts from 0
            for cf_features_combination in cf_features_combinations:
                cf_instances = self._get_neighbouring_instances(
                    instance, cf_features_combination)

                if cf_instances.size:
                    cf_predictions = self.predict(cf_instances)

                    # Identify counterfactuals based on the cf class
                    if current_class is None:
                        is_cf = cf_predictions == counterfactual_class
                    else:
                        is_cf = cf_predictions != current_class

                    # Filter the counterfactuals
                    cf_instances = cf_instances[is_cf]
                    cf_predictions = cf_predictions[is_cf]
                    distances = []
                    for cf_instance in cf_instances:
                        distance = self._get_distance(
                            instance,
                            cf_instance,
                            normalise=normalise_distance)
                        distances.append(distance)

                    if distances:
                        dists = np.array(distances)
                        dists_min = dists.min()
                        dists_min_mask = dists == dists_min

                        counterfactuals.append(cf_instances[dists_min_mask])
                        counterfactuals_distances.append(dists[dists_min_mask])
                        counterfactuals_predictions.append(
                            cf_predictions[dists_min_mask])

        if counterfactuals:
            # Make sure that all of the structured arrays share the same type.
            # Otherwise it is impossible to concatenate.
            if fuav.is_structured_array(counterfactuals[0]):
                new_types = []
                for name in counterfactuals[0].dtype.names:
                    dtype = counterfactuals[0].dtype[name]
                    for i in counterfactuals:
                        dtype = fuat.generalise_dtype(dtype, i.dtype[name])
                    new_types.append((name, dtype))
                for i, cfs in enumerate(counterfactuals):
                    counterfactuals[i] = cfs.astype(new_types)

            # Put counterfactuals together
            counterfactuals = np.concatenate(counterfactuals)
            counterfactuals_distances = np.concatenate(
                counterfactuals_distances)
            counterfactuals_predictions = np.concatenate(
                counterfactuals_predictions)

            # Remove duplicates
            if _NUMPY_1_13:  # pragma: nocover
                counterfactuals, uidx = np.unique(
                    counterfactuals, return_index=True, axis=0)
            else:  # pragma: nocover
                is_structured = fuav.is_structured_array(counterfactuals)
                uidx = []
                for i, row in enumerate(counterfactuals):
                    if is_structured:
                        same_rows = counterfactuals == row
                    else:
                        same_rows = (counterfactuals == row).all(axis=1)
                    if same_rows.sum() > 1:
                        duplicates = set(np.where(same_rows)[0].tolist())
                        if not duplicates.intersection(uidx):
                            uidx.append(i)
                    else:
                        uidx.append(i)
                counterfactuals = counterfactuals[uidx]
            counterfactuals_distances = counterfactuals_distances[uidx]
            counterfactuals_predictions = counterfactuals_predictions[uidx]

            # Sort them to get the closest ones first
            sorting = np.argsort(counterfactuals_distances)
            counterfactuals = counterfactuals[sorting]
            counterfactuals_distances = counterfactuals_distances[sorting]
            counterfactuals_predictions = counterfactuals_predictions[sorting]
        else:
            assert not counterfactuals_distances, 'Should be an empty list.'
            assert not counterfactuals_predictions, 'Should be an empty list.'
            counterfactuals = np.ndarray((0, 0))
            counterfactuals_distances = np.ndarray((0, ))
            counterfactuals_predictions = np.ndarray((0, ))

        return (counterfactuals, counterfactuals_distances,
                counterfactuals_predictions)


def _categorical_distance(first_value: Union[float, str],
                          second_value: Union[float, str]) -> int:
    """
    Defines the default categorical distance.

    This distance simply checks whether values of both features are identical
    -- 0 distance -- or different -- 1 distance.

    Parameters
    ----------
    first_value : number or string
        The first value to compare.
    second_value : number or string
        The second value to compare.

    Returns
    -------
    distance : integer
        A distance between the two input parameters.
    """
    assert isinstance(first_value, (Number, str)), 'Must be string or number.'
    assert isinstance(second_value, (Number, str)), 'Must be string or number.'

    distance = int(first_value != second_value)
    return distance


def _numerical_distance(first_value: float, second_value: float) -> float:
    """
    Defines the default numerical distance.

    This distance is simply the absolute value of the difference between the
    two input parameters.

    Parameters
    ----------
    first_value : number
        The first value to compare.
    second_value : number
        The second value to compare.

    Returns
    -------
    distance : number
        A distance between the two input parameters.
    """
    assert isinstance(first_value, Number), 'Must be a number.'
    assert isinstance(second_value, Number), 'Must be a number.'

    distance = abs(first_value - second_value)
    return distance


def _validate_input_one(model: Union[object, None],
                        predictive_function: Union[Callable, None],
                        dataset: Union[np.ndarray, None],
                        categorical_indices: Union[List[Index], None],
                        numerical_indices: Union[List[Index], None]) -> bool:
    """
    Validates the first part of input given to initialise the cf class.

    For the input parameters, warnings and the exceptions raised in this
    function please see the documentation of :class:`fatf.transparency.
    predictions.counterfactuals.CounterfactualExplainer` class.

    Returns
    -------
    input_is_valid : boolean
        ``True`` if the input is valid, ``False`` otherwise.
    """
    # pylint: disable=too-many-branches

    input_is_valid = False

    # Validate model/predictive function
    if model is not None:
        if not fumv.check_model_functionality(model):
            raise RuntimeError('The model object requires a "predict" method '
                               'to be used with this explainer.')
    if predictive_function is not None:
        if not callable(predictive_function):
            raise TypeError('The predictive_function parameter should be a '
                            'Python function.')
        # The predictive function is to have one non-optional parameter
        required_param_n = 0
        params = inspect.signature(predictive_function).parameters
        for param in params:
            if params[param].default is params[param].empty:
                required_param_n += 1
        if required_param_n != 1:
            raise AttributeError('The predictive function requires exactly 1 '
                                 'non-optional parameter: a data array to be '
                                 'predicted.')

    if model is None and predictive_function is None:
        raise RuntimeError('You either need to specify a model or a '
                           'predictive_function parameter to initialise a '
                           'counterfactual explainer.')
    if model is not None and predictive_function is not None:
        warnings.warn(
            'Both a model and a predictive_function parameters were supplied. '
            'A predictive functions takes the precedence during the '
            'execution.', UserWarning)

    # Validate data
    if dataset is not None:
        if not fuav.is_base_array(dataset):
            raise ValueError('The dataset has to be of a base type (strings '
                             'and/or numbers).')
        if not fuav.is_2d_array(dataset):
            raise IncorrectShapeError('The data array has to be '
                                      '2-dimensional.')

    # Validate categorical and numerical indices
    if categorical_indices is not None:
        if not isinstance(categorical_indices, list):
            raise TypeError('categorical_indices parameter either has to be a '
                            'list of indices or None.')
    if numerical_indices is not None:
        if not isinstance(numerical_indices, list):
            raise TypeError('numerical_indices parameter either has to be a '
                            'list of indices or None.')
    # If both lists are given both cannot be empty, only one can be empty and
    # they have to be disjoin
    if categorical_indices is not None and numerical_indices is not None:
        all_indices = categorical_indices + numerical_indices
        if not all_indices:
            raise ValueError('Both categorical_indices and numerical_indices '
                             'parameters cannot be empty lists. If you want '
                             'them to be inferred from a data array please '
                             'leave these parameters set to None.')
        if set(categorical_indices).intersection(numerical_indices):
            raise ValueError('Some of the indices in the categorical_indices '
                             'and numerical_indices parameters are repeated.')

        indices_type = type(all_indices[0])
        incorrect_type = False
        # They have to be integers (non-structured) or strings (structured)
        for i in all_indices:
            if not isinstance(i, indices_type):
                incorrect_type = True
                break
        if incorrect_type or indices_type not in (str, int):
            raise TypeError('Some of the indices given in the '
                            'categorical_indices and/or numerical_indices '
                            'parameters do not share the same type. It is '
                            'expected that indices for a classic numpy array '
                            'will all be integers and for a structured numpy '
                            'array they will be strings.')

    input_is_valid = True
    return input_is_valid


def _validate_input_two(
        numerical_indices: Set[Index],
        categorical_indices: Set[Index],
        all_indices: Set[Index],
        counterfactual_feature_indices: Union[List[Index], None],
        max_counterfactual_length: int,
        feature_ranges: Union[Dict[Index, FeatureRange], None],
        distance_functions: Union[Dict[Index, Callable], None],
        step_sizes: Union[Dict[Index, float], None],
        default_numerical_step_size: float) -> bool:  # yapf: disable
    """
    Validates the second part of input given to initialise the cf class.

    For the input parameters, warnings and the exceptions raised in this
    function please see the documentation of :class:`fatf.transparency.
    predictions.counterfactuals.CounterfactualExplainer` class.

    Returns
    -------
    input_is_valid : boolean
        ``True`` if the input is valid, ``False`` otherwise.
    """
    # pylint: disable=too-many-branches,too-many-arguments,too-many-locals
    # pylint: disable=too-many-statements

    input_is_valid = False

    assert isinstance(numerical_indices, set), 'Must be a set of indices.'
    assert isinstance(categorical_indices, set), 'Must be a set of indices.'
    assert isinstance(all_indices, set), 'Must be a set of indices.'
    assert (numerical_indices.union(categorical_indices) == all_indices
            and not numerical_indices.intersection(categorical_indices)), \
        'Indices must be valid.'

    # Validate that counterfactual_feature_indices are valid or None
    if counterfactual_feature_indices is not None:
        if not isinstance(counterfactual_feature_indices, list):
            raise TypeError('counterfactual_feature_indices parameter either '
                            'has to be a list of indices or None.')
        if not counterfactual_feature_indices:
            raise ValueError('counterfactual_feature_indices parameter cannot '
                             'be an empty list. If you want all of the '
                             'features to be used for counterfactuals '
                             'generation leave this parameter unset or set it '
                             'explicitly to None.')
        if set(counterfactual_feature_indices).difference(all_indices):
            raise IndexError('counterfactual_feature_indices list contains '
                             'invalid indices.')

    # max_counterfactual_length should be non-negative int
    if not isinstance(max_counterfactual_length, int):
        raise TypeError('The max_counterfactual_length parameter should be an '
                        'integer.')
    if max_counterfactual_length < 0:
        raise ValueError('The max_counterfactual_length parameter should be a '
                         'non-negative integer. If you want to generate '
                         'counterfactuals with a full length (number of '
                         'features), set this parameter to 0.')

    # Feature ranges should either be None or a non-empty dictionary,
    # with valid indices as keys and 2-tuples or lists as values
    if feature_ranges is not None:
        if not isinstance(feature_ranges, dict):
            raise TypeError('The feature_ranges parameter has to be a '
                            'dictionary or None.')
        if not feature_ranges:
            raise ValueError('The feature_ranges parameter cannot be an empty '
                             'dictionary.')

        if set(feature_ranges.keys()).difference(all_indices):
            raise IndexError('Some of the indices (dictionary keys) in '
                             'the feature_ranges parameter are not valid.')
        # Verify supplied ranges
        for column_index, column_range in feature_ranges.items():
            if column_index in categorical_indices:
                if not isinstance(column_range, list):
                    raise TypeError('Categorical column range should be a '
                                    'list of values to be used for the '
                                    'counterfactuals generation process.')
                if not column_range:
                    raise ValueError('A list specifying the possible '
                                     'values of a categorical feature '
                                     'should not be empty.')
                range_type = type(column_range[0])
                for i in column_range:
                    if not isinstance(i, range_type):
                        raise TypeError('The possible values defined for '
                                        'the *{}* feature do not share '
                                        'the same type.'.format(column_index))
            elif column_index in numerical_indices:
                if not isinstance(column_range, tuple):
                    raise TypeError('Numerical column range should be a '
                                    'pair of numbers defining the lower '
                                    'and the upper limits of the range.')
                if len(column_range) != 2:
                    raise ValueError('Numerical column range tuple should '
                                     'just contain 2 numbers: the lower '
                                     'and the upper bounds of the range '
                                     'to be searched.')
                if (not isinstance(column_range[0], Number)
                        or not isinstance(column_range[1], Number)):
                    raise TypeError('Both the lower and the upper bound '
                                    "defining column's range should numbers.")
                if column_range[1] <= column_range[0]:
                    raise ValueError('The second element of a tuple '
                                     'defining a numerical range should '
                                     'be strictly larger than the first '
                                     'element.')
            else:
                assert False, 'Unknown index.'  # pragma: nocover

    # Distance functions need to be defined for valid indices and
    # they need to be Callable with two parameters
    if distance_functions is not None:
        if not isinstance(distance_functions, dict):
            raise TypeError('The distance_functions parameter has to be a '
                            'dictionary.')
        if not distance_functions:
            raise ValueError('The distance_functions parameter cannot be an '
                             'empty dictionary.')

        if set(distance_functions.keys()).difference(all_indices):
            raise IndexError('Some of the indices (dictionary keys) in the '
                             'distance_functions parameter are invalid.')

        for key in distance_functions:
            if not callable(distance_functions[key]):
                raise TypeError('All of the distance functions defined via '
                                'the distance_functions parameter have to be '
                                'Python callable.')
            required_param_n = 0
            params = inspect.signature(distance_functions[key]).parameters
            for param in params:
                if params[param].default is params[param].empty:
                    required_param_n += 1
            if required_param_n != 2:
                raise AttributeError('Every distance function requires '
                                     'exactly 2 non-optional parameters.')

    # Check whether the keys in step sizes dictionary are valid indices and
    # step sizes have to be positive numbers.
    if step_sizes is not None:
        if not isinstance(step_sizes, dict):
            raise TypeError('The step_sizes parameter has to be a '
                            'dictionary.')
        if not step_sizes:
            raise ValueError('The step_sizes parameter cannot be an empty '
                             'dictionary.')
        if set(step_sizes.keys()).difference(all_indices):
            raise IndexError('Some of the indices (dictionary keys) in '
                             'the step_sizes parameter are not valid.')
        for i in step_sizes.values():
            if not isinstance(i, Number):
                raise TypeError('All of the step values contained in the '
                                'step_sizes must be numbers.')
            if i <= 0:
                raise ValueError('All of the step values contained in the '
                                 'step_sizes must be positive numbers.')

    # The default numerical step size has to be a positive number
    if not isinstance(default_numerical_step_size, Number):
        raise TypeError('The default_numerical_step_size parameter has to be '
                        'a number.')
    if default_numerical_step_size <= 0:
        raise ValueError('The default_numerical_step_size parameter has to be '
                         'a positive number.')

    input_is_valid = True
    return input_is_valid


def textualise_counterfactuals(
        instance: Union[np.ndarray, np.void],
        counterfactuals: np.ndarray,
        instance_class: Optional[Union[int, str]] = None,
        counterfactuals_distances: Optional[np.ndarray] = None,
        counterfactuals_predictions: Optional[np.ndarray] = None) -> str:
    """
    Translates a counterfactuals array into a textual description.

    Parameters
    ----------
    instance : numpy.ndarray or numpy.void
        A 1-dimensional numpy array representing a data point for which
        counterfactuals were calculated.
    counterfactuals : numpy.ndarray
        A 2-dimensional numpy array with the counterfactual data points.
    instance_class : string or integer, optional (default=None)
        The class of the input ``instance``.
    counterfactuals_distances : numpy.ndarray, optional (default=None)
        A 1-dimensional numpy array with distances from the input ``instance``
        to every counterfactual data point.
    counterfactuals_predictions : numpy.ndarray, optional (default=None)
        A 1-dimensional numpy array with predictions for every counterfactual
        data point.

    Raises
    ------
    IncorrectShapeError
        ``instance`` is not a 1-dimensional numpy array or any of
        ``counterfactuals``, ``counterfactuals_distances`` or
        ``counterfactuals_predictions`` is not a 2-dimensional numpy array.
    IndexError
        The ``counterfactuals`` and the ``instance`` column indices disagree.
    TypeError
        The ``instance_class`` parameter is neither an integer nor a string.
        The types of the ``instance`` and the ``counterfactuals`` arrays are
        different.
    ValueError
        The ``instance`` is not of a base type (strings and/or numbers);
        the ``counterfactuals`` is not of a base type; the
        ``counterfactuals_distances`` is not a purely numerical array; or
        ``counterfactuals_predictions`` is not a base array. Either the length
        of the ``counterfactuals_distances`` array or of the
        ``counterfactuals_predictions`` array is not the same as the number of
        rows in the ``counterfactuals`` array. The type of the
        ``instance_class`` is different than the type of the
        ``counterfactuals_predictions`` array.

    Returns
    -------
    textualisation : string
        A string representation of the ``counterfactuals``.
    """
    # pylint: disable=too-many-branches,too-many-statements,too-many-locals

    if not fuav.is_1d_like(instance):
        raise IncorrectShapeError('The instance has to be a 1-dimensional '
                                  'numpy array.')
    instance_2d = np.array([instance])
    if not fuav.is_base_array(instance_2d):
        raise ValueError('The instance has to be of a base type (strings '
                         'and/or numbers).')
    #
    if not fuav.is_2d_array(counterfactuals):
        raise IncorrectShapeError('The counterfactuals array should be a '
                                  '2-dimensional numpy array.')
    if not fuav.is_base_array(counterfactuals):
        raise ValueError('The counterfactuals array has to be of a base type '
                         '(strings and/or numbers).')
    #
    if not fuav.are_similar_dtype_arrays(instance_2d, counterfactuals):
        raise TypeError('The type of the instance and the counterfactuals '
                        'arrays do not agree.')
    # These have to be valid indices for the instance
    if not fuav.is_structured_array(counterfactuals):
        if counterfactuals.shape[1] != instance.shape[0]:
            raise IndexError('The counterfactuals and instance column indices '
                             'do not agree. (The two arrays have different '
                             'number of columns.)')
    #
    if instance_class is not None:
        if not isinstance(instance_class, (int, str, np.int32, np.int64)):
            raise TypeError('The instance_class has to be either an integer '
                            'or a string.')
    #
    if counterfactuals_distances is not None:
        if not fuav.is_1d_array(counterfactuals_distances):
            raise IncorrectShapeError('The counterfactuals_distances array '
                                      'should be a 1-dimensional array.')
        if not fuav.is_numerical_array(counterfactuals_distances):
            raise ValueError('The counterfactuals_distances array should be '
                             'purely numerical.')
        if counterfactuals.shape[0] != counterfactuals_distances.shape[0]:
            raise ValueError('The counterfactuals_distances array should be '
                             'of the same length as the number of rows in the '
                             'counterfactuals array.')
    #
    if counterfactuals_predictions is not None:
        if not fuav.is_1d_array(counterfactuals_predictions):
            raise IncorrectShapeError('The counterfactuals_predictions array '
                                      'should be a 1-dimensional array.')
        if not fuav.is_base_array(counterfactuals_predictions):
            raise ValueError('The counterfactuals_predictions array should be '
                             'of a base type (numbers and/or strings).')
        if counterfactuals.shape[0] != counterfactuals_predictions.shape[0]:
            raise ValueError('The counterfactuals_predictions array should be '
                             'of the same length as the number of rows in the '
                             'counterfactuals array.')
        if instance_class is not None:
            if not fuav.are_similar_dtype_arrays(
                    np.array([instance_class]), counterfactuals_predictions):
                raise ValueError('The type of the instance_class is different '
                                 'than the type of the '
                                 'counterfactuals_predictions array.')

    # Sort the counterfactuals in case they are not sorted
    if counterfactuals_distances is not None:
        ordering = np.argsort(counterfactuals_distances)
        counterfactuals = counterfactuals[ordering]
        counterfactuals_distances = counterfactuals_distances[ordering]
        if counterfactuals_predictions is not None:
            counterfactuals_predictions = counterfactuals_predictions[ordering]

    # Get feature names
    if fuav.is_structured_array(counterfactuals):
        feature_names = counterfactuals.dtype.names
    else:
        feature_names = list(range(counterfactuals.shape[1]))

    if instance_class is None:
        inctance_class_str = ''
    else:
        inctance_class_str = ' (of class *{}*)'.format(instance_class)
    output = ['Instance{}:'.format(inctance_class_str),
              '{}\n'.format(instance),
              'Feature names: {}\n'.format(feature_names)]  # yapf: disable

    cf_template = '    feature *{}*: *{}* -> *{}*'

    for i in range(counterfactuals.shape[0]):
        if counterfactuals_predictions is None:
            cf_instance_class_str = ''
        else:
            cf_instance_class_str = ' (of class *{}*)'.format(
                counterfactuals_predictions[i])
        output.append(
            'Counterfactual instance{}:'.format(cf_instance_class_str))

        if counterfactuals_distances is not None:
            output.append('Distance: {}'.format(counterfactuals_distances[i]))

        for feature_name in feature_names:
            i_val = instance[feature_name]
            c_val = counterfactuals[i][feature_name]
            if i_val != c_val:
                output.append(cf_template.format(feature_name, i_val, c_val))
        output[-1] += '\n'

    # Remove the trailing new-line
    if output[-1][-1] == '\n':
        output[-1] = output[-1][:-1]

    textualisation = '\n'.join(output)
    return textualisation
