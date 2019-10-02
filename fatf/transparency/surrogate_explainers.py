"""
Default surrogate explainers.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
# License: new BSD

import abc
from typing import Dict, Union, Optional, List, Callable
import warnings

import numpy as np

import sklearn.linear_model
import sklearn.tree

import scipy.stats

import fatf.utils.data.feature_selection.sklearn as fudfs
import fatf.transparency.sklearn.linear_model as ftslm
import fatf.transparency.sklearn.tools as ftst

import fatf.utils.data.augmentation as fuda
import fatf.utils.data.transformation as fudt
import fatf.utils.data.discretisation as fudd
import fatf.utils.distances as fud
import fatf.utils.kernels as fuk
import fatf.utils.transparency.explainers as fute
import fatf.utils.array.validation as fuav
import fatf.utils.models.validation as fumv
import fatf.utils.array.tools as fuat

from fatf.exceptions import IncompatibleModelError, IncorrectShapeError

__all__ = ['TabularLime', 'TabularBlimeyTree']

Index = Union[int, str]


def _input_is_valid(dataset: np.ndarray,
                    global_model: object,
                    probabilistic: bool,
                    categorical_indices: Optional[List[Index]],
                    class_names: Optional[List[str]],
                    feature_names: Optional[List[str]]) -> bool:
    """
    Validates the input parameters of the surrogate explainers.

    For the input parameter description, warnings and exceptions please see
    the documentation of the :func`fatf.transparency.sklearn.\
    SurrogateExplainer`
    __init__` function.

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
        raise TypeError('The input dataset must only contain base types '
                        '(textual and numerical).')
    if probabilistic:
        if not fumv.check_model_functionality(global_model, True, True):
            raise IncompatibleModelError(
                'Probabilistic functionality requires the global model to be '
                'capable of outputting probabilities via predict_proba method.')
    else:
        if not fumv.check_model_functionality(global_model, False, True):
            raise IncompatibleModelError(
                'Non-probabilistic functionality requires the global model to '
                'be capable of outputting probabilities via predict method.')

    if categorical_indices is not None:
        if isinstance(categorical_indices, list):
            invalid_indices = fuat.get_invalid_indices(
                dataset, np.asarray(categorical_indices))
            if invalid_indices.size:
                raise IndexError('The following indices are invalid for the '
                                 'input dataset: {}.'.format(invalid_indices))
        else:
            raise TypeError('The categorical_indices parameter must be a '
                            'Python list or None.')

    if class_names is not None:
        if not isinstance(class_names, list):
            raise TypeError('The class_names parameter must be None or a '
                            'list.')
        else:
            for class_name in class_names:
                if (class_name is not None
                        and not isinstance(class_name, str)):
                    raise TypeError('The class_name has to be either None or '
                                    'a string or a list of strings.')

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
                    raise TypeError('The feature name has to be either None '
                                    'or a string or a list of strings.')

    is_input_ok = True
    return is_input_ok


class SurrogateExplainer(abc.ABC):
    """
    An abstract class for implementing surrogate explainers.

    An abstract class that all surrogate explainer classes should inherit from. It
    contains abstract ``__init__`` and ``explain_instance`` methods and an input
    validator -- ``_validate_explainer_instance_input`` -- for the
    ``explain_instance`` method. The validation of the input parameters to the
    initialisation method is done via the ``fatf.transparency.sklearn.\
    surrogate_explainers._input_is_valid`` function.

    .. note::
       The ``_input_is_valid`` method should be called in all implementations
       of the ``explain_instance`` method in the children classes to ensure
       that all the input parameters of this method are valid.

    Parameters
    ----------
    dataset : numpy.ndarray
        A 2-dimensional numpy array with the dataset.
    global_model : object
        A pretrained global model. This must contain method ``predict_proba``
        that will return a numpy array for probabilities of instances belonging
        to each of the classses or method ``predict`` that will return the
        class prediction for each datapoint.
    probabilistic: boolean
        A boolean describing whether the global model is probabilistic. If
        ``True`` then ``global_model`` must have method ``predict_proba``, if
        ``False`` then ``global_model`` must have method ``predict``. In all
        cases the local model will be trained using an one v all method.
    categorical_indices : List[column indices]
        A list of column indices that should be treat as categorical features.
    class_names : List[string], optional (default=None)
        A list of strings defining the names of classes.
    feature_names : List[string], optional (default=None)
        A list of strings defining the names of the features.

    Warns
    -----
    UserWarning
        If some of the string-based columns in the input data array were not
        indicated to be categorical features by the user (via the
        ``categorical_indices`` parameter) the user is warned that they will be
        added to the list of categorical features.

    Raises
    ------
    IncorrectShapeError
        The parameter ``dataset`` is not a 2-dimensional numpy array.
    TypeError
        The parameter ``dataset`` is not of base (numerical and/or string)
        type. The ``categorical_indices`` parameter is neither a list nor
        ``None``. The
        ``feature_names`` parameter is neither a list nor ``None``. One of the
        values in ``feature_names`` is neither a string nor ``None``. The
        ``class_names`` parameter is neither a list nor ``None``. One of the
        values in ``class_names`` is neither a string nor ``None``.
    IndexError
        Some of the column indices given in the ``categorical_indices``
        parameter are not valid for the input ``dataset``.
    IncompatibleModelError
        The parameter ``global_model`` does not have required functionality
        -- it needs to be able to output probabilities via ``predict_proba``
        method.
    ValueError
        The length of parameter ``feature_names`` is not the same as the number
        of features in ``dataset``.

    Attributes
    ----------
    dataset : numpy.ndarray
        A 2-dimensional numpy array with the dataset.
    is_structured : boolean
        ``True`` if the ``dataset`` is a structured numpy array, ``False``
        otherwise.
    global_model : object
        A pretrained global model. This must contain method ``predict_proba``
        that will return a numpy array for probabilities of instances belonging
        to each of the classses or method ``predict`` that will return the
        class prediction for each datapoint.
    categorical_indices : List[column indices]
        A list of column indices that should be treat as categorical features.
    numerical_indices : List[column indices]
        A list of column indices that should be treat as numerical features.
    prediction_function : Callable[np.ndarray, np.ndarray]
        Function that will be used to get predictions from the global model.
    class_names : List[string]
        A list of strings defining the names of classes.
    feature_names : List[string]
        A list of strings defining the names of the features.
    n_classes : integer
        Number of classes that ``global_model`` produces prediction
        probabilities for.
    indices : List[column indices]
        A list of all indices in ``dataset``.
    """
    def __init__(self,
                 dataset: np.ndarray,
                 global_model: object,
                 probabilistic: bool = True,
                 categorical_indices: Optional[List[Index]] = None,
                 class_names: Optional[List[str]] = None,
                 feature_names: Optional[List[str]] = None):
        """
        Constructs a ``SurrogateExplainer`` class.
        """
        assert _input_is_valid(dataset, global_model, probabilistic,
                               categorical_indices, class_names,
                               feature_names), 'Input is not valid.'

        self.dataset = dataset
        self.is_structured = fuav.is_structured_array(dataset)

        # Sort out column indices
        indices = fuat.indices_by_type(dataset)
        cat_indices = set(indices[1])
        num_indices = set(indices[0])
        all_indices = num_indices.union(cat_indices)

        if categorical_indices is None:
            categorical_indices = cat_indices
            numerical_indices = num_indices
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
                categorical_indices = cat_indices.union(categorical_indices)
            numerical_indices = all_indices.difference(categorical_indices)

        self.categorical_indices = sorted(list(categorical_indices))
        self.numerical_indices = sorted(list(numerical_indices))

        # Initialise objects
        self.global_model = global_model
        if probabilistic:
            self.prediction_function = global_model.predict_proba
        else:
            self.prediction_function = global_model.predict

        self.probabilistic = probabilistic

        if self.is_structured:
            row = np.array([dataset[0]], dtype=self.dataset.dtype)
        else:
            row = dataset[0].reshape(1, -1)
        self.n_classes = self.global_model.predict_proba(row).shape[1]

        if fuav.is_structured_array(self.dataset):
            self.indices = self.dataset.dtype.names
        else:
            self.indices = np.arange(0, self.dataset.shape[1], 1)

        # pre-process class_names and feature_names
        if feature_names is None:
            feature_names = [None] * len(self.indices)
        self.feature_names = []
        for i, feature_name in enumerate(feature_names):
            if feature_name is None:
                self.feature_names.append('feature {}'.format(i))
            else:
                self.feature_names.append(feature_name)

        if class_names is None:
            class_names = [None] * self.n_classes
        self.class_names = []
        for i, class_name in enumerate(class_names):
            if class_name is None:
                self.class_names.append('class {}'.format(i))
            else:
                self.class_names.append(class_name)

    def _explain_instance_input_is_valid(
            self,
            data_row: Union[np.ndarray, np.void]) -> bool:
        """
        Validates input parameters of the ``explain_instance`` method.

        This function checks the validity of ``data_row``.

        Raises
        ------
        IncorrectShapeError
            The ``data_row`` is not a 1-dimensional numpy array-like object.
            The number of features (columns) in the ``data_row`` is different
            to the number of features in the data array used to initialise this
            object.
        TypeError
            The dtype of the ``data_row`` is different than the dtype of the
            data array used to initialise this object.

        Returns
        -------
        is_valid : boolean
            ``True`` if input parameters are valid, ``False`` otherwise.
        """
        is_valid = False

        if data_row is not None:
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
    def explain_instance(
            self,
            data_row: Union[np.ndarray, np.void]
        ) -> Dict[str, Dict[Index, np.float64]]:
        """
        Samples a given number of data points based on the initialisation data.

        This is an abstract method that must be implemented for each child
        object. This method should provide two modes of operation:

        - if ``data_row`` is ``None``, the sample should be from the
          distribution of the whole dataset that was used to initialise this
          class; and

        - if ``data_row`` is a numpy array with a data point, the sample should
          be from the vicinity of this data point.

        Parameters
        ----------
        data_row : Union[numpy.ndarray, numpy.void], optional (default=None)
            A data point. If given, the sample will be generated around that
            point.
        samples_number : integer, optional (default=50)
            The number of samples to be generated.

        Raises
        ------
        NotImplementedError
            This is an abstract method and has not been implemented.

        Returns
        -------
        explanation : Dict[]
            Explanation.
        """
        assert self._explain_instance_input_is_valid(data_row), \
            'Invalid sample method input.'

        raise NotImplementedError('explain_instance method needs to be '
                                  'overwritten.')


class TabularLIME(SurrogateExplainer):
    """
    An implementation of LIME explainer.

    Samples from the discretise dataset so the sample data has the same
    proportions of bins as the original dataset. Then computes distances
    between sampled points and ``data_row`` and uses K-LASSO to select
    ``features_number`` number of features to use in the explanation.
    Then fits a local ridge regression model. The weightings of this ridge
    regression is then interpreted as feature importances.

    Indicies in the parameter ``categorical_indices`` will not be discretised.

    For additional parameters, warnings and errors please see the parent class
    :class:`fatf.transparency.sklearn.surrogate_explainers.SurrogateExplainer`.

    Parameters
    ----------

    Raises
    ------

    Attributes
    ----------
    discretiser : fatf.utils.data.discretisation.QuartileDiscretiser
        Quartile discretiser to be used to discretise dataset and ``data_row``
        when ``explain_instance`` is called.
    augmentor : fatf.utils.data.augmentation.NormalSampling
        Augmentor for performing normal sampling on the data.
    discretised_dataset : numpy.ndarray
        A discretised version of the dataset.
    bin_sampling_values : Dictionary[Index, Dictionary[value, Tuple(float, float, float, float)]]
        A dictionary specifying for each feature, a dictionary that maps the
        discretised values in the dataset to the (minimum, maximum, mean,
        standard deviation) for the original data bin corresponding to the
        value. This is used to go from discretised sampled values to the
        original data space.
    local_models : List[object]
        Stores the local models that were used to get explanations in the
        most recent call of ``explain_insatnce``. Will be an empty list if
        ``explain_instance`` has not been called yet.
    """
    def __init__(self,
                 dataset: np.ndarray,
                 global_model: object,
                 probabilistic: bool = True,
                 categorical_indices: Optional[List[Index]] = None,
                 class_names: Optional[List[str]] = None,
                 feature_names: Optional[List[str]] = None):
        super().__init__(
                dataset, global_model, probabilistic, categorical_indices,
                class_names, feature_names)

        self.discretiser = fudd.QuartileDiscretiser(
            dataset,
            categorical_indices=self.categorical_indices,
            feature_names=self.feature_names)

        self.discretised_dataset = self.discretiser.discretise(self.dataset)

        # Normal sampler will just sample all indices to be the proportion that
        # is present in the dataset, as categorical_indices will be treated as
        # categorical and numerical_indices will be treated as categorical as
        # they will be discretised.
        self.augmentor = fuda.NormalSampling(
            self.discretised_dataset,
            categorical_indices=self.categorical_indices+self.numerical_indices)

        self.bin_sampling_values = {}
        for index in self.numerical_indices:
            self.bin_sampling_values[index] = {}
            if self.is_structured:
                discretised_feature = self.discretised_dataset[index]
                feature = self.dataset[index]
            else:
                discretised_feature = self.discretised_dataset[:, index]
                feature = self.dataset[:, index]

            unique_values = np.unique(discretised_feature)
            for value in unique_values:
                indices = (discretised_feature == value).nonzero()[0]
                bin_values = feature[indices]
                stats = (bin_values.min(), bin_values.max(), bin_values.mean(),
                         bin_values.std())
                self.bin_sampling_values[index][value] = stats

        self.local_models = []

    def _explain_instance_input_is_valid(
            self,
            data_row: Union[np.ndarray, np.void],
            samples_number: Optional[int],
            features_number: Optional[int],
            kernel_width: Optional[float]) -> bool:
        """
        Validates explain_instance input parameters for the class.

        For additional documentation of parameters, warnings and errors
        please see the description of :func:`fatf.transparency.sklearn.\
        surrogate_explainers.TabularLime.explain_instance`.

        Returns
        -------
        is_valid : boolean
        ``True`` if input is valid, ``False`` otherwise.
        """
        is_valid = False
        assert super()._explain_instance_input_is_valid(data_row)

        if not isinstance(samples_number, int):
            raise TypeError('samples_number must be an integer.')
        else:
            if samples_number < 1:
                raise ValueError('samples_number must be a positive integer '
                                 'larger than 0.')

        if not isinstance(features_number, int):
            raise TypeError('features_number must be an integer.')
        else:
            if features_number < 1:
                raise ValueError('features_number must be a positive integer '
                                 'larger than 0.')

        if kernel_width is not None:
            if not isinstance(kernel_width, float):
                raise TypeError('kernel_width must be None or a float.')
            else:
                if kernel_width <= 0.0:
                    raise ValueError('kernel_width must be None or a positive '
                                     'float larger than 0.')

        is_valid = True
        return is_valid

    def _undiscretise_feature(self,
                              index: Index,
                              values: np.ndarray,
                              random_state: int) -> np.ndarray:
        """
        Transforms ``values`` from discretised data space to original. Adapted
        from: https://github.com/marcotcr/lime/blob/master/lime/discretize.py

        Parameters
        ----------
        index : Index
            The index that values are a feature for
        values : numpy.ndarray
            Values from the feature column
        random_state : int
            Used for testing purposes. Used in sampling the truncnorm for
            undiscretising data.
        Returns
        -------
        undiscretised_feature : numpy.ndarray
            The feature values in the original data space.

        # TODO: too similar to LIME rewrite
        """
        mins = np.array(
            [self.bin_sampling_values[index][v][0]for v in values])
        maxs = np.array(
            [self.bin_sampling_values[index][v][1] for v in values])
        means = np.array(
            [self.bin_sampling_values[index][v][2] for v in values])
        stds = np.array(
            [self.bin_sampling_values[index][v][3] for v in values])
        nonzero_stds = (stds != 0)
        a = (mins[nonzero_stds] - means[nonzero_stds]) / (stds[nonzero_stds])
        b = (maxs[nonzero_stds] - means[nonzero_stds]) / (stds[nonzero_stds])
        undiscretised_feature = means
        undiscretised_feature[np.where(nonzero_stds)] = \
            scipy.stats.truncnorm.rvs(
                    a,
                    b,
                    loc=means[nonzero_stds],
                    scale=stds[nonzero_stds],
                    random_state=random_state)

        return undiscretised_feature

    def explain_instance(
            self,
            data_row: Union[np.ndarray, np.void],
            samples_number: Optional[int] = 50,
            features_number: Optional[int] = None,
            kernel_width: Optional[float] = None,
            random_state: Optional[int] = 42
    ) -> Dict[str, Dict[Index, np.float64]]:
        """
        Explains instance ``data_row`` using LIME algorithm.

        For additional parameters, warnings and errors please see the parent
        class function :func:`fatf.transparency.sklearn.surrogate_explainers.\
        SurrogateExplainer.explain_instance`.

        Parameters
        ----------
        samples_number : integer, optional (default=50)
            The number of samples to be generated.
        features_number : integer, optional (default=None)
            Number of features to find the K-LASSO to train the local model
            using. Default is ``None``, meaning all features will be used.
        kernel_width : float, optional (default=None)
            Kernel width to use in exponential kernel when computing distance
            between sampled data and ``data_row``.
        random_state : integer, optional (default=42)
            Used for testing purposes. The random state to be used in the
            DecisionTreeRegressor that is the local model. Also used for
            sampling the truncnorm for undiscretising data.
        """
        assert self._explain_instance_input_is_valid(
            data_row, samples_number, features_number, kernel_width), \
            'Invalid explain_instance method input.'

         # Create an array to hold the samples.
        dataset_features = (len(self.categorical_indices) +
                            len(self.numerical_indices))

        if kernel_width is None:
            kernel_width = np.sqrt(dataset_features) * .75

        if self.is_structured:
            shape = (samples_number, )  # type: Tuple[int, ...]
        else:
            shape = (samples_number, dataset_features)
        samples = np.zeros(shape, dtype=self.dataset.dtype)

        discretised_data_row = self.discretiser.discretise(data_row)
        discretised_sampled_data = self.augmentor.sample(
            discretised_data_row, samples_number=samples_number)

        # Create sampled_data in the original space
        for index in self.numerical_indices:
            if self.is_structured:
                values = discretised_sampled_data[index]
            else:
                values = discretised_sampled_data[:, index]

            undiscretised_feature = self._undiscretise_feature(
                index, values, random_state)

            if self.is_structured:
                samples[index] = undiscretised_feature
            else:
                samples[:, index] = undiscretised_feature

        # Get feature names for binarised data
        discretised_value_names = self.discretiser.feature_value_names
        binarised_feature_names = []
        for i, index in enumerate(self.indices):
            data_row_value = discretised_data_row[index]
            if index in discretised_value_names.keys():
                binarised_feature_names.append(
                    discretised_value_names[index][int(data_row_value)])
            elif index in self.categorical_indices:
                binarised_feature_names.append('*{}* = {}'.format(
                    self.feature_names[i], data_row_value))
        lime_explanation = {}

         # binarised data will be 1 if value is the same as in data_row
        # else 0
        binarised_data = fudt.dataset_row_masking(
            discretised_sampled_data, discretised_data_row)
        # kernalised distances between data_row and the binarised sampled
        # data.
        distances = fud.euclidean_array_distance(
                np.ones((1, dataset_features)),
                binarised_data).flatten()
        weights = fuk.exponential_kernel(distances, width=kernel_width)

        for i in range(self.n_classes):
            if self.probabilistic:
                predictions = self.prediction_function(samples)[:, i]
            else:
                predictions = self.prediction_function(samples)
                class_indx = np.where(predictions==i)
                rest_indx = np.where(predictions!=i)
                predictions[class_indx] = 1
                predictions[rest_indx] = 0

            # Choose indices using k-LASSO
            lasso_indices = fudfs.lasso_path(binarised_data,
                                             predictions,
                                             weights,
                                             features_number)
            if self.is_structured:
                local_training_data = fuat.as_unstructured(
                    binarised_data[lasso_indices])
                feature_name_indices = [binarised_data.dtype.names.index(i)
                                        for i in lasso_indices]
            else:
                local_training_data = binarised_data[:, lasso_indices]
                feature_name_indices = lasso_indices

            feature_names = [binarised_feature_names[name]
                             for name in feature_name_indices]

            # Train the local ridge regression and use our linear model
            # explainer to generate explanations.
            local_model = sklearn.linear_model.Ridge(random_state=random_state)
            local_model.fit(local_training_data, predictions)
            self.local_models.append(local_model)

            explainer = ftslm.SKLearnLinearModelExplainer(
                local_model,
                feature_names=feature_names)

            # TODO: maybe put this in SkLearnLinearModelExplainer.\
            # feature_importance
            lime_explanation[self.class_names[i]] = dict(
                zip(feature_names,
                    explainer.feature_importance()))

        return lime_explanation


class TabularBlimeyTree(SurrogateExplainer):
    """
    An implemented of bLIMEy explainer.

    Use ``Mixup`` (:class:`fatf.utils.data.augmentation.Mixup`) sampling
    around instance. Then fits a local decision tree regressor on the sampled
    data. Extracts feature importances from the decision tree and exports this
    as an explanation.

    For additional parameters, warnings and errors please see the parent class
    :class:`fatf.transparency.sklearn.surrogate_explainers.SurrogateExplainer`.

    Raises
    ------
    TypeError
        ``dataset`` is a structured array. ``dataset`` contains non-numerical
        dtype.

    Attributes
    ----------
    local_models : List[object]
        Stores the local models that were used to get explanations in the
        most recent call of ``explain_insatnce``. Will be an empty list if
        ``explain_instance`` has not been called yet.
    augmentor : fatf.utils.data.augmentation.Mixup
        The augmentor used to sample locally around ``data_row`` in each call
        to ``explain_instance``.
    """
    def __init__(self,
                 dataset: np.ndarray,
                 global_model: object,
                 probabilistic: bool = True,
                 categorical_indices: Optional[List[Index]] = None,
                 class_names: Optional[List[str]] = None,
                 feature_names: Optional[List[str]] = None):
        super().__init__(
                dataset, global_model, probabilistic, categorical_indices,
                class_names, feature_names)

        if self.is_structured:
            raise TypeError('TabularBlimeyTree does not support structured '
                            'arrays as it uses sci-kit learn implementation '
                            'of decision trees.')
        # Check if dtype is string based.
        if fuav.is_textual_dtype(dataset.dtype):
            raise TypeError('TabularBlimeyTree does not support string dtype '
                            'as it uses sci-kit learn implementation of '
                            'decision trees.')

        # List to store all the local_models constructed in the latest call
        # of explain_instance.
        self.local_models = []

        # Get predictions for dataset for mixup
        dataset_predictions = self.global_model.predict(self.dataset)

        # Initialise Mixup augmentor
        self.augmentor = fuda.Mixup(
            dataset=self.dataset,
            ground_truth=dataset_predictions,
            categorical_indices=self.categorical_indices,
            beta_parameters=(2, 5),
            int_to_float=True)

    def _explain_instance_input_is_valid(
            self,
            data_row: Union[np.ndarray, np.void],
            samples_number: Optional[int],
            maximum_depth: Optional[int]) -> bool:
        """
        Validates explain_instance input parameters for the class.

        For additional documentation of parameters, warnings and errors
        please see the description of :func:`fatf.transparency.sklearn.\
        surrogate_explainers.TabularLime.explain_instance`.

        Returns
        -------
        is_valid : boolean
        ``True`` if input is valid, ``False`` otherwise.
        """
        is_valid = False
        assert super()._explain_instance_input_is_valid(data_row)

        if not isinstance(samples_number, int):
            raise TypeError('samples_number must be an integer.')
        else:
            if samples_number < 1:
                raise ValueError('samples_number must be a positive integer '
                                 'larger than 0.')

        if not isinstance(maximum_depth, int):
            raise TypeError('maximum_depth must be an integer.')
        else:
            if maximum_depth < 1:
                raise ValueError('maximum_depth must be a positive integer '
                                 'larger than 0.')

        is_valid = True
        return is_valid

    def explain_instance(
            self,
            data_row: Union[np.ndarray, np.void],
            samples_number: Optional[int] = 50,
            maximum_depth: Optional[int] = 3,
            random_state: Optional[int] = 42
    ) -> Dict[str, Dict[str, np.float64]]:
        """
        Explains instance ``data_row`` using bLIMEy algorithm with a
        decision tree as the local surrogate model.

        For additional parameters, warnings and errors please see the parent
        class function :func:`fatf.transparency.sklearn.surrogate_explainers.\
        SurrogateExplainer.explain_instance`.

        Parameters
        ----------
        samples_number : integer, optional (default=50)
            The number of samples to be generated.
        maximum_depth : integer, optional (default=3)
            Maximum depth of the decision tree local surrogate model, can be
            used to limit the number of splits and simplify the resulting
            local model.
        random_state : integer, optional (default=42)
            Used for testing purposes. The random state to be used in the
            DecisionTreeRegressor that is the local model.
        """
        assert self._explain_instance_input_is_valid(
            data_row, samples_number, maximum_depth), \
                'Invalid explain_instance method input.'

        sampled_data = self.augmentor.sample(data_row,
                                             samples_number=samples_number)
        blimey_explanation = {}

        for i in range(self.n_classes):
            if self.probabilistic:
                predictions = self.prediction_function(sampled_data)[:, i]
            else:
                predictions = self.prediction_function(sampled_data)
                class_indx = np.where(predictions==i)
                rest_indx = np.where(predictions!=i)
                predictions[class_indx] = 1
                predictions[rest_indx] = 0

            local_model = sklearn.tree.DecisionTreeRegressor(
                max_depth=maximum_depth,
                random_state=random_state)

            local_model.fit(sampled_data, predictions)
            self.local_models.append(local_model)
            blimey_explanation[self.class_names[i]] = dict(
                zip(self.feature_names,
                    list(local_model.feature_importances_)))

            # TODO: Decision tree explainer

        return blimey_explanation
