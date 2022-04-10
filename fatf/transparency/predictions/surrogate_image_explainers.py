"""
The :mod:`fatf.transparency.predictions.surrogate_image_explainers` module
implements a bLIMEy version of the LIME surrogate image explainer.

.. versionadded:: 0.1.1

The ``scikit-learn``, ``scikit-image`` and ``Pillow`` packages are required for
the surrogate image explainer to work.
"""
# Author: Kacper Sokol <K.Sokol@bristol.ac.uk>
# License: new BSD

from numbers import Number
from typing import Callable, Dict, List, Optional, Tuple, Union

import logging
import warnings

import scipy.spatial

import numpy as np

from fatf.exceptions import IncompatibleModelError

import fatf.utils.data.instance_augmentation as fatf_augmentation
import fatf.utils.kernels as fatf_kernels
import fatf.utils.models.models as fatf_models
import fatf.utils.models.processing as fatf_processing
import fatf.utils.models.validation as fatf_validation

__all__ = ['ImageBlimeyLime']

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

try:
    # pylint: disable=ungrouped-imports
    import sklearn.linear_model

    import fatf.transparency.sklearn.linear_model as fatf_linear_explainer
    import fatf.utils.data.occlusion as fatf_occlusion
    import fatf.utils.data.segmentation as fatf_segmentation
except ImportError as _exc:
    _err = (  # pylint: disable=invalid-name
        'The ImageBlimeyLime surrogate image explainer requires scikit-learn, '
        'scikit-image and Pillow to be installed.\n\n{}')
    raise ImportError(_err.format(str(_exc)))

Explanation = Dict[str, Number]
ExplanationTuple = Union[Explanation, Tuple[Explanation, fatf_models.Model]]
RGBcolour = Tuple[int, int, int]
ColourFn = Callable[[np.ndarray], np.ndarray]


class ImageBlimeyLime(object):  # pylint: disable=useless-object-inheritance
    """
    Implements a surrogate image explainer equivalent to LIME.

    .. versionadded:: 0.1.1

    By default this explainer uses *quickshift* segmentation
    (the :class:`fatf.utils.data.segmentation.QuickShift` class) and
    mean-colour occlusion
    (the :class:`fatf.utils.data.occlusion.Occlusion` class).
    It uses the cosine distance transformed thorough the exponential kernel
    to generate similarity scores between the binary representation of the
    explained instance and the data sample.
    It works with both crisp and probabilistic classifiers;
    it assumes the latter by default (``as_probabilistic=True``).

    Parameters
    ----------
    image : numpy.ndarray
        A numpy array representing an image to be explained.
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
        method.
    class_names : List[string, integer], optional (default=None)
        A list of strings or integer corresponding to the names of classes.
        If the predictive model is probabilistic, the order of the class names
        should correspond to the order of columns output by the model.
        For crisp models the order is irrelevant.
    segmentation_mask : numpy.ndarray, optional (default=None)
        A numpy array representing an image to be used for generating the
        segmentation. If this parameter is not provided, the ``image`` will
        be used to generate the segmentation.
    segments_merge_list : list(integer) or list(list(integer)), \
optional (default=None)
        A collection or a set of collections of segment ids to be merged.
        See the documentation of the
        :func:`fatf.utils.data.segmentation.Segmentation.merge_segments` method
        for more details.
    ratio : number, optional (default=0.2)
        Balances color-space proximity and image-space proximity for
        the **quickshift** segmenter.
        Higher values give more weight to color-space.
        Between 0 and 1.
    kernel_size : number, optional (default=4)
        Width of Gaussian kernel used in smoothing the sample density for
        the **quickshift** segmenter.
        Higher means fewer clusters.
    max_dist : number, optional (default=200)
        Cut-off point for data distances for the **quickshift** segmenter.
        Higher means fewer clusters.
    colour : string, integer, tuple(integer, integer, integer), \
optional (default=None)
        An occlusion colour specifier.
        By default (``colour=None``) the mean colouring strategy is used.
        See the documentation of the
        :func:`fatf.utils.data.occlusion.Occlusion.set_colouring_strategy`
        method for more details.

    Raises
    ------
    IncompatibleModelError
        The ``predictive_model`` does not have the required functionality:
        ``predict_proba`` method for probabilistic models and ``predict``
        method crisp classifiers.
    RuntimeError
        The number of class names provided via the ``class_names`` parameter
        does not agree with the number of classes output by the probabilistic
        model.
    TypeError
        The ``as_probabilistic`` parameter is not a boolean.
        The ``class_names`` parameter is neither a list nor ``None``.
        Some of the elements in the ``class_names`` list are neither a string
        nor an integer.
    ValueError
        The ``class_names`` list is empty or it contains duplicates.

    Attributes
    ----------
    image : numpy.ndarray
        A numpy array representing an image to be explained.
    segmentation_mask : numpy.ndarray
        A numpy array representing an image used to perform segmentation.
    segmenter : fatf.utils.data.segmentation.Segmentation
        A *quickshift* image segmenter
        (:class:`fatf.utils.data.segmentation.QuickShift`).
    occluder : fatf.utils.data.occlusion.Occlusion
        An image occluder (:class:`fatf.utils.data.occlusion.Occlusion`).
    as_probabilistic : boolean
        ``True`` if the ``predictive_model`` should be treated as
        probabilistic and ``False`` if it should be treated as a classifier.
    predictive_model : object
        A pre-trained (black-box) predictive model to be explained.
    predictive_function : Callable[[numpy.ndarray], numpy.ndarray]
        A function that will be used to get predictions from the explained
        predictive model. It references the ``predictive_model.predict_proba``
        method for for probabilistic models (``as_probabilistic=True``) and the
        ``predictive_model.predict`` method for crisp classifiers.
    image_prediction : Union[string, integer]
        The prediction of the explained image. For probabilistic models it is
        the index of the class assigned to this instance by the explained
        model; for crisp classifier it is the predicted class.
    classes_number : integer or None
        The number of modelled classes for probabilistic models;
        ``None`` for crisp classifiers unless ``class_names`` was provided.
    class_names : List[string] or None
        A list of class names that can be predicted by the explained model.
        For probabilistic models these are in order they appear in the
        probability vector output by the model.
        There is no particular order for crisp predictors.
    surrogate_data_sample : numpy.ndarray or None
        A binary data sample generated during the last call of the
        ``explain_instance`` method.
    surrogate_data_predictions : numpy.ndarray or None
        Predictions of the explained model for the binary data sample (reversed
        to the image representation) generated during the last call of the
        ``explain_instance`` method.
    similarities : numpy.ndarray or None
        Similarities between the explained instance and the sampled data
        computed in the binary domain using the cosine distance transformed
        thorough the exponential kernel and generated during the last call of
        the ``explain_instance`` method.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self,
                 image: np.ndarray,
                 predictive_model: object,
                 as_probabilistic: bool = True,
                 class_names: Optional[Union[List[str], List[int]]] = None,
                 segmentation_mask: Optional[np.ndarray] = None,
                 segments_merge_list: Union[None, List[int], List[
                     List[int]]] = None,
                 ratio: float = 0.2,
                 kernel_size: float = 4,
                 max_dist: float = 200,
                 colour: Optional[Union[str, int, RGBcolour]] = None):
        """Constructs a bLIMEy LIME image explainer."""
        # pylint: disable=too-many-arguments,too-many-locals,too-many-branches
        # pylint: disable=too-many-statements
        # The image and the segmentation mask in numpy representation
        self.image = image.copy()
        if segmentation_mask is None:
            self.segmentation_mask = self.image.copy()
        else:
            self.segmentation_mask = segmentation_mask.copy()

        if not isinstance(as_probabilistic, bool):
            raise TypeError(
                'The as_probabilistic parameter must be a boolean.')
        self.as_probabilistic = as_probabilistic

        if self.as_probabilistic:
            is_functional = fatf_validation.check_model_functionality(
                predictive_model, True, False)
            if not is_functional:
                raise IncompatibleModelError(
                    'With as_probabilistic set to True the predictive model '
                    'needs to be capable of outputting probabilities via '
                    'a *predict_proba* method, which takes exactly one '
                    'required parameter -- data to be predicted -- and '
                    'outputs a 2-dimensional array with probabilities.')
        else:
            is_functional = fatf_validation.check_model_functionality(
                predictive_model, False, False)
            if not is_functional:
                raise IncompatibleModelError(
                    'With as_probabilistic set to False the predictive model '
                    'needs to be capable of outputting (class) predictions '
                    'via a *predict* method, which takes exactly one required '
                    'parameter -- data to be predicted -- and outputs a '
                    '1-dimensional array with (class) predictions.')
        self.predictive_model = predictive_model

        if self.as_probabilistic:
            predictive_function = \
                self.predictive_model.predict_proba  # type: ignore
            image_prediction = predictive_function([self.image])[0]
            classes_number = image_prediction.shape[0]
            image_prediction = int(np.argmax(image_prediction))
        else:
            predictive_function = self.predictive_model.predict  # type: ignore
            classes_number = None
            image_prediction = predictive_function([self.image])[0]
        self.predictive_function = predictive_function
        self.image_prediction = image_prediction
        self.classes_number = classes_number

        if class_names is not None:
            if isinstance(class_names, list):
                if not class_names:
                    raise ValueError('The class_names list cannot be empty.')
                if len(class_names) != len(set(class_names)):
                    raise ValueError('The class_names list contains '
                                     'duplicated entries.')
                _chosen_type = type(class_names[0])
                if _chosen_type is int or _chosen_type is str:
                    _chosen_error = False
                    for class_name in class_names:
                        if not isinstance(class_name, _chosen_type):
                            _chosen_error = True
                            break
                else:
                    _chosen_error = True
                    class_name = class_names[0]
                if _chosen_error:
                    raise TypeError('All elements of the class_names '
                                    'list must be strings or integers; '
                                    '*{}* is not.'.format(class_name))
                if self.classes_number is None:
                    self.classes_number = len(class_names)
                else:
                    if self.classes_number != len(class_names):
                        raise RuntimeError('The number of class names does '
                                           'not correspond to the shape of '
                                           'the model predictions.')
            else:
                raise TypeError('The class_names parameter must be a Python '
                                'list or None.')
        self.class_names = class_names

        logger.debug('Building segmentation.')
        self.segmenter = fatf_segmentation.QuickShift(
            self.image,
            segmentation_mask=self.segmentation_mask,
            ratio=ratio,
            kernel_size=kernel_size,
            max_dist=max_dist)
        if segments_merge_list is not None:
            self.segmenter.merge_segments(segments_merge_list, inplace=True)

        logger.debug('Building occlusion.')
        self.occluder = fatf_occlusion.Occlusion(
            self.image, self.segmenter.segments, colour=colour)

        # Placeholder to memorise the last data sample for training surrogates
        self.surrogate_data_sample = None  # type: Union[None, np.ndarray]
        self.surrogate_data_predictions = None  # type: Union[None, np.ndarray]
        self.similarities = None  # type: Union[None, np.ndarray]

    def set_occlusion_colour(self, colour):
        """
        Sets the occlusion colour.

        See the documentation of the
        :func:`fatf.utils.data.occlusion.Occlusion.set_colouring_strategy`
        method for more details.
        """
        self.occluder.set_colouring_strategy(colour)

    def explain_instance(self,
                         explained_class: Optional[Union[int, str]] = None,
                         samples_number: int = 50,
                         batch_size: int = 50,
                         kernel_width: float = .25,
                         colour: Optional[Union[str, int, RGBcolour]] = None,
                         reuse_sample: bool = False,
                         return_model: bool = False) -> ExplanationTuple:
        """
        Explains the image used to initialise this class.

        Parameters
        ----------
        explained_class : Union[integer, string], optional (default=None)
            The class to be explained. By default (``explained_class=None``)
            the class predicted by the explained model for the explained image
            will be used.
            For probabilistic models this can be the index of the class in the
            probability vector output by the explained model or the name of the
            class if ``class_names`` parameter was provided while initialising
            this class.
            For crisp classifiers this has to be one of the values predicted by
            the explained model.
        samples_number : integer, optional (default=50)
            The number of data points sampled from the random binary generator
            to be used for fitting the local surrogate model.
        batch_size : integer, optional (default=50)
            The number of images to be processed in one iteration. Since this
            step is computationally expensive -- images need to be generated
            and occluded according to the binary data sample, and then
            predicted by the explained model -- the data points can be
            processed in fixed-size batches.
        kernel_width : float, optional (default=0.25)
            The width of the exponential kernel used when computing weights of
            the binary sampled data based on the cosine distances between them
            and the explained image.
        colour : string, integer, tuple(integer, integer, integer), \
optional (default=None)
            An occlusion colour specifier.
            By default (``colour=None``) the colour specified when initialising
            this class is used.
            See the documentation of the
            :func:`fatf.utils.data.occlusion.Occlusion.set_colouring_strategy`
            method for more details.
        reuse_sample : boolean, optional (default=False)
            Whether to generate a new binary data sample or reuse the one
            generated with the last call of this method.
        return_models : boolean, optional (default=False)
            If ``True``, this method will return both the feature importance
            explanation and the local surrogate model.
            Otherwise, only the explanation is returned.

        Warns
        -----
        UserWarning
            Informs the user if none of the sampled data were predicted with
            the explained class when explaining a crisp model -- such a
            situation will most probably result in unreliable explanations.

        Raises
        ------
        IndexError
            The name of the class chosen to be explained could not be located
            among the class names provided upon initialising this object.
            The index of the explained class -- when explaining a probabilistic
            model -- is invalid.
        RuntimeError
            Some of the cosine distances could not be computed due to a
            numerical error.
            The data sample cannot be reused without calling this method at
            least once beforehand.
            A class name cannot be used when explaining a probabilistic model
            without initialising this object with class names.
        TypeError
            The ``return_model`` or ``reuse_sample`` parameter is not a
            boolean.
            The ``explained_class`` parameter is neither of ``None``, a string
            or an integer.

        Returns
        -------
        explanations : Dictionary[string, float]
            A dictionary containing image segment importance (extracted
            from the local linear surrogate).
        models : sklearn.linear_model.base.LinearModel, optional
            A locally fitted surrogate linear model.
            This model is only returned when ``return_model=True``.
        """
        # pylint: disable=too-many-arguments,too-many-locals,too-many-branches
        # pylint: disable=too-many-statements
        if not isinstance(return_model, bool):
            raise TypeError('The return_model parameter should be a boolean.')
        if not isinstance(reuse_sample, bool):
            raise TypeError('The reuse_sample parameter should be a boolean.')

        if not (explained_class is None or isinstance(explained_class,
                                                      (int, str))):
            raise TypeError('The explained_class parameter must be either of '
                            'None, a string or an integer.')
        if explained_class is None:
            explained_class = self.image_prediction
        assert explained_class is not None, 'Filtered out before.'

        if self.as_probabilistic:
            assert self.classes_number is not None, 'Inferred form the model.'
            if isinstance(explained_class, int):
                if (explained_class < 0
                        or explained_class >= self.classes_number):
                    raise IndexError('The explained class index is invalid.')
            else:
                if self.class_names is None:
                    raise RuntimeError(
                        'It is not possible to use a name for the explained '
                        'class without initialising this explainer with a '
                        'list of class names (the *class_names* parameter).')
                if explained_class not in self.class_names:
                    raise IndexError(
                        'The name of the explained class could not be '
                        'found in the list of class names used to '
                        'initialise this explainer (the *class_names* '
                        'parameter).')
                explained_class = self.class_names.index(
                    explained_class)  # type: ignore
        else:
            if self.class_names is not None:
                if explained_class not in self.class_names:
                    raise IndexError(
                        'The name of the explained class could not be found '
                        'in the list of class names used to initialise this '
                        'explainer (the *class_names* parameter).')

        if reuse_sample:
            logger.debug('Reusing the sample.')
            if (self.surrogate_data_sample is None
                    or self.surrogate_data_predictions is None
                    or self.similarities is None):
                raise RuntimeError('You need to explain an instance before '
                                   'being able to reuse its (random) sample.')
        else:
            # Generate binary samples in the interpretable domain
            logger.debug('Generating a sample.')
            self.surrogate_data_sample = \
                fatf_augmentation.random_binary_sampler(
                    self.segmenter.segments_number, samples_number)

            # Build interpretable representation of the explained instance
            explained_instance_ir = np.ones(
                shape=(1, self.surrogate_data_sample.shape[1]), dtype=np.int8)

            # Get distances to the sampled data
            logger.debug('Computing distances.')
            distances = scipy.spatial.distance.cdist(
                explained_instance_ir,
                self.surrogate_data_sample,
                'cosine').flatten()  # yapf: disable
            # all-0 vectors nan-out cosine similarity
            _all_zero = self.surrogate_data_sample.sum(axis=1)
            _all_zero_mask = (_all_zero == 0)
            if _all_zero_mask.any():
                assert np.isnan(distances[_all_zero_mask]).all(), 'nans.'
                logger.debug('Setting the distance to all-0 vectors to 1.')
                distances[_all_zero_mask] = 1  # similarity is 0
            assert not np.isnan(distances).any(), 'Do not expect any nans.'

            # Kernelise the distance
            logger.debug('Transforming distances into similarities.')
            self.similarities = fatf_kernels.exponential_kernel(
                distances, width=kernel_width)

            if colour is None:
                transformation_fn = \
                    self.occluder.occlude_segments_vectorised  # type: ColourFn
            else:
                transformation_fn = lambda data: \
                    self.occluder.occlude_segments_vectorised(  # noqa: E731
                        data, colour=colour)

            # Transform to images and predict the sampled data
            iter_ = fatf_processing.batch_data(
                self.surrogate_data_sample,
                batch_size=batch_size,
                transformation_fn=transformation_fn)
            sample_predictions = []
            logger.debug('Reconstructing and predicting images.')
            for batch in iter_:
                sample_predictions.append(self.predictive_function(batch))
            if self.as_probabilistic:
                self.surrogate_data_predictions = np.vstack(sample_predictions)
            else:
                self.surrogate_data_predictions = np.hstack(sample_predictions)

        # Fit surrogate model
        logger.debug('Fitting the surrogate.')
        if self.as_probabilistic:
            surrogate = sklearn.linear_model.Ridge()
            predictions = self.surrogate_data_predictions[:, explained_class]
        else:
            surrogate = sklearn.linear_model.RidgeClassifier()
            y_mask = (self.surrogate_data_predictions == explained_class)
            if not y_mask.any():
                warnings.warn(
                    'None of the sampled data points were predicted by the '
                    'model with the explained class. The explanation may be '
                    'untrustworthy or the name of the explained class has '
                    'been missspelled!', UserWarning)
            predictions = np.zeros(
                shape=(self.surrogate_data_predictions.shape[0], ), dtype=int)
            predictions[y_mask] = 1
        surrogate.fit(
            self.surrogate_data_sample,
            predictions,
            sample_weight=self.similarities)

        # Get names of the interpretable components
        ir_names = [
            'Segment #{}'.format(i)
            for i in range(1, self.segmenter.segments_number + 1)
        ]
        # Get a linear model explainer
        explainer = fatf_linear_explainer.SKLearnLinearModelExplainer(
            surrogate, feature_names=ir_names)
        assert isinstance(explainer.feature_names, list)
        feature_importance = explainer.feature_importance()
        if not self.as_probabilistic:
            assert feature_importance.shape[0] == 1, 'Single-output clf.'
            feature_importance = feature_importance[0]
        explanation = dict(
            zip(explainer.feature_names, feature_importance))

        if return_model:
            return_ = (explanation, surrogate)  # type: ExplanationTuple
        else:
            return_ = explanation
        return return_
