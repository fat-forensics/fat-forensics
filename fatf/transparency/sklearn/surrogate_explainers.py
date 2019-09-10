"""
Default surrogate explainers.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
# License: new BSD

import abc
from typing import Dict, Union, Optional, List, Callable
import warnings

import numpy as np

import fatf.utils.data.feature_selection.sklearn as fudfs
import fatf.transparency.sklearn.linear_model as ftslm
import fatf.transparency.sklearn.tools as ftst

import fatf.utils.data.augmentation as fuda
import fatf.utils.data.discretisation as fudd
import fatf.utils.distances as fud
import fatf.utils.kernels as fatf_kernels
import fatf.utils.transparency.explainers as fute
import fatf.utils.array.validation as fuav
import fatf.utils.models.validation as fumv
import fatf.utils.array.tools as fuat
import fatf.utils.data.transformation as fudt

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
        cat_indices = set(indices[1])
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


class TabularLime(SurrogateExplainer):
    """
    An implemented of LIME explainer.

    Normal sampling around instance if ``sample_around_instance`` is
    ``True`` else samples around the mean of the dataset. Then computes
    distances between sampled points and ``data_row`` and uses K-LASSO to
    select ``features_number`` number of features to use in the explanation.
    Then fits a local ridge regression model. The weightings of this ridge
    regression is then interpreted as feature importances.

    For additional parameters, warnings and errors please see the parent class
    :class:`fatf.transparency.sklearn.surrogate_explainers.SurrogateExplainer`.

    Parameters
    ----------

    Raises
    ------

    Attributes
    ----------

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

    def _explain_instance_input_is_valid(
            self,
            data_row: Union[np.ndarray, np.void],
            samples_number: Optional[int],
            sample_around_instance: Optiona[bool],
            features_number: Optional[int]) -> bool:
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
        assert super()._validate_sample_input(data_row)

        is_valid = True
        return is_valid

    def explain_instance(
            self,
            data_row: Union[np.ndarray, np.void],
            samples_number: Optional[int] = 50,
            sample_around_instance: Optiona[bool] = True,
            features_number: Optional[int] = None
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
        sample_around_instance : bool, optional (default=True)
            Boolean whether to normally sample around the instance or the mean
            of the dataset. If ``True`` then data is sampled around the
            instance, if ``False`` then data is sampled around the mean of the
            dataset.
        features_number : integer, optional (default=None)
            Number of features to find the K-LASSO to train the local model
            using. Default is ``None``, meaning all features will be used.
        """
        assert self._validate_sample_input(data_row), \
            'Invalid explain_instance method input.'


class TabularBlimeyTree(SurrogateExplainer):
    """
    Parameters
    ----------

    Raises
    ------

    Attributes
    ----------

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
        assert super()._validate_sample_input(data_row)

        is_valid = True
        return is_valid

    def explain_instance(
            self,
            data_row: Union[np.ndarray, np.void],
            samples_number: Optional[int] = 50,
            maximum_depth: Optional[int] = 3
    ) -> Dict[str, Dict[Index, np.float64]]:
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
        """
        assert self._validate_sample_input(data_row), \
            'Invalid explain_instance method input.'
