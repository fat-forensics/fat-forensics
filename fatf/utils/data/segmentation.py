"""
The :mod:`fatf.utils.data.segmentation` module implements image segmenters.

.. versionadded:: 0.1.1
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

# pylint: disable=too-many-lines

from numbers import Number
from typing import List, Optional, Tuple, Union

import abc
import logging
import warnings

import numpy as np

from fatf.exceptions import IncorrectShapeError

import fatf.utils.array.validation as fuav

try:
    import skimage.color as ski_colour
    import skimage.segmentation as ski_segmentation
except ImportError:
    raise ImportError(
        'scikit-image Python package is not installed on your system. '
        'You must install it in order to use the fatf.utils.data.segmentation '
        'functionality. '
        'One possibility is to install scikit-image alongside this package '
        'via auxiliary dependencies with: pip install fat-forensics[all].')

try:
    from PIL import Image, ImageFont, ImageDraw
except ImportError:
    raise ImportError(
        'PIL Python package is not installed on your system. '
        'You must install it in order to use the fatf.utils.data.segmentation '
        'functionality. '
        'One possibility is to install PIL alongside this package via '
        'auxiliary dependencies with: pip install fat-forensics[all].')

__all__ = ['get_segment_mask',
           'Segmentation',
           'Slic',
           'QuickShift']  # yapf: disable

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

RGBcolour = Tuple[int, int, int]


def _validate_image_array(image: np.ndarray, image_name: str) -> bool:
    """
    Checks whether a numpy array has properties expected of images.

    A numpy representation of an image should be a non-structured, 2- or
    3-dimensional, integer-valued array with elements in the 0--255 range.

    Parameters
    ----------
    image : numpy.ndarray
        A 2- or 3-dimensional numpy array representing an image.
    image_name : string
        A name of the validated array to be used in error messages.

    Raises
    ------
    IncorrectShapeError
        The input ``image`` is neither a 2- nor 3-dimensional numpy array.
    TypeError
        The input ``image`` is either a structured numpy array or it is not a
        numerical array.
    ValueError
        The elements of the input ``image`` are not integers in
        the 0--255 range.

    Returns
    -------
    is_valid : boolean
        Indicates whether the input ``image`` is a valid numpy array.
    """
    is_valid = False

    assert (isinstance(image_name, str)
            and image_name), 'image_name must be a non-empty string.'

    # Validate the image
    if fuav.is_structured_array(image):
        raise TypeError(('The input {} must not be a structured '
                         'numpy array.').format(image_name))
    if not fuav.is_numerical_array(image):
        raise TypeError(
            'The input {} must be of a numerical type.'.format(image_name))
    # Ensure that we are dealing with integers within the 0--255 range
    _image_is_int = image.dtype.kind in 'iu'
    _image_min, _image_max = image.min(), image.max()
    if _image_min < 0 or _image_max > 255 or not _image_is_int:
        raise ValueError(('The numpy representation of the input {} '
                          'should have its values (integers) between the '
                          '0--255 range.').format(image_name))
    # Ensure 2- or 3-dimensional
    _image_in_shape = len(image.shape) in (2, 3)
    if not _image_in_shape:
        raise IncorrectShapeError(
            'The input {} must be a 2- or 3-dimensional numpy array.'.format(
                image_name))

    is_valid = True
    return is_valid


def _validate_input(image: np.ndarray,
                    segmentation_mask: Union[None, np.ndarray]) -> bool:
    """
    Validates the input parameters of a segmentation class.

    For the description of the input parameters and exceptions raised by this
    function, please see the documentation of the
    :func:`~fatf.utils.data.segmentation._validate_image_array` function.

    Raises
    ------
    IncorrectShapeError
        The width and height of ``image`` and ``segmentation_mask`` do not
        agree.

    Returns
    -------
    is_valid : boolean
        ``True`` if input is valid, ``False`` otherwise.
    """
    is_valid = False

    assert _validate_image_array(image, 'image'), 'image is invalid.'

    if segmentation_mask is not None:
        assert _validate_image_array(
            segmentation_mask,
            'image segmentation mask'), 'image segmentation mask is invalid.'

        # Check shape
        if image.shape[:2] != segmentation_mask.shape[:2]:
            raise IncorrectShapeError(
                'The width and height of the input image and the segmentation '
                'mask do not agree.')

    is_valid = True
    return is_valid


def _validate_segmentation(segments: np.ndarray, image: np.ndarray) -> bool:
    """
    Checks whether a segmentation array is valid.

    A numpy representation of a segmentation of an image should be
    a non-structured, 2-dimensional, integer-valued array with a continuous
    sequence of unique elements starting at 1.

    Parameters
    ----------
    segments : numpy.ndarray
        A 2-dimensional numpy array representing a segmentation.
    image : numpy.ndarray
        A 2- or 3-dimensional numpy array representing an image.

    Raises
    ------
    IncorrectShapeError
        The ``segments`` array is not 2-dimensional.
        The the height or width the ``segments`` array does not agree with
        the dimensions of the segmented image.
    TypeError
        The ``segments`` array is either a structured numpy array or
        it is not an integer-valued array.
    ValueError
        The unique elements of the ``segments`` array do not form a continuous
        sequence starting at 1.

    Returns
    -------
    is_valid : boolean
        Indicates whether the ``segments`` array is valid.
    """
    is_valid = False

    # Validate image
    assert _validate_image_array(image, 'image'), 'image is invalid.'

    # Validate segments
    if fuav.is_structured_array(segments):
        raise TypeError('The segmentation array must not be a structured '
                        'numpy array.')
    if not fuav.is_2d_array(segments):
        raise IncorrectShapeError('The segmentation array must be a 2-'
                                  'dimensional numpy array.')
    _segments_is_int = segments.dtype.kind in 'iu'
    if not fuav.is_numerical_array(segments) or not _segments_is_int:
        raise TypeError('The segmentation array must be of integer type.')
    _segments_min, _segments_max = segments.min(), segments.max()
    _segments_unique = np.unique(segments)
    _segments_is_continuous = True
    for i in range(_segments_min, _segments_max + 1):
        if i not in _segments_unique:
            _segments_is_continuous = False
            break
    if _segments_min != 1 or not _segments_is_continuous:
        raise ValueError('The segmentation array should encode unique '
                         'segments with a continuous sequence of integers '
                         'starting at 1.')

    # Check shape
    if segments.shape[:2] != image.shape[:2]:
        raise IncorrectShapeError(
            'The width and height of the segmentation array and '
            'the input image do not agree.')

    is_valid = True
    return is_valid


def _validate_colour(colour: Union[None, RGBcolour]) -> bool:
    """
    Validates RGB colour triplet.

    ``colour`` must either be ``None`` or a 3-tuple of integers within the
    0--255 range.

    Parameters
    ----------
    colour : tuple(integer, integer, integer) or None
        RGB colour triplet.

    Raises
    ------
    TypeError
        The ``colour`` parameter is neither a tuple nor a ``None``;
        or one of its elements is not an integer.
    ValueError
        The ``colour`` parameter is not a 3-tuple, one of its elements
        is outside of the 0--255 range.

    Returns
    -------
    is_valid : boolean
        Indicates whether the ``colour`` RGB triplet is valid.
    """
    is_valid = False
    if colour is not None:
        if not isinstance(colour, tuple):
            raise TypeError('The colour must either be None or a thriplet '
                            'representing an RGB colour.')
        if len(colour) != 3:
            raise ValueError('The colour tuple must be a triplet.')
        for i in colour:
            if not isinstance(i, int):
                raise TypeError(
                    'Each element of the colour tuple must be an integer.')
            if i < 0 or i > 255:
                raise ValueError('Each RGB value must be between 0 and 255.')
    is_valid = True
    return is_valid


class Segmentation(abc.ABC):
    """
    An abstract class implementing image segmentation functionality.

    .. versionadded:: 0.1.1

    An abstract class that all segmentation classes should inherit from.
    It contains an abstract ``_segment`` method to be implemented by
    individual segmenters.
    This methods should return a 2-dimensional numpy array assigning each
    pixel of an image to a segment by using unique integers from a sequence
    starting at 1.
    The ``kwargs`` attribute can be used to collect optional parameters
    upon initialising this class that can be used within the ``_segment``
    method.

    This class is designed for images represented as numpy arrays with their
    values in the 0--255 range:

    - 2-dimensional arrays for grayscale (0--255 range) and
      black-and-white (0 and 255 valued) images; and
    - 3-dimensional arrays for colour images.

    The segmentation stored by this class can be overwritten either with the
    ``set_segments`` method or by directly setting the ``segments`` attribute,
    both of which will perform the necessary validation.

    Parameters
    ----------
    image : numpy.ndarray
        A numpy array representing an image to be segmented.
    segmentation_mask : numpy.ndarray, optional (default=None)
        A numpy array representing an image to be used for generating the
        segmentation. If this parameter is not provided, the ``image`` will
        be used to generate the segmentation.
    **kwargs : dictionary
        A list of named parameters saved to the ``kwargs`` attribute,
        which can be used to pass configuration options to the ``_segment``
        method.

    Warns
    -----
    UserWarning
        Inform the user that only a single segment was found.

    Raises
    ------
    IncorrectShapeError
        The input ``image`` is neither a 2- nor 3-dimensional numpy array.
        The width and height of ``image`` and ``segmentation_mask`` do not
        agree.
        The segmentation array is not 2-dimensional.
        The the height or width the segmentation array does not agree with
        the dimensions of the segmented image.
    RuntimeError
        A black-and-white image does not use 0 as black and 1 or 255 as white.
    TypeError
        The input ``image`` is either a structured numpy array or it is not a
        numerical array.
        The segmentation array is either a structured numpy array or
        it is not an integer-valued array.
    ValueError
        The elements of the input ``image`` are not integers in
        the 0--255 range.
        The unique elements of the segmentation array do not form a continuous
        sequence starting at 1.

    Attributes
    ----------
    image : numpy.ndarray
        A numpy array representing an image to be segmented.
    segmentation_mask : numpy.ndarray
        A numpy array representing an image used to perform segmentation.
    is_rgb : boolean
        Indicates whether the ``image`` is RGB or black-and-white.
    kwargs : dictionary
        A list of named parameters stored as a dictionary;
        it is used  to pass configuration options to the ``_segment`` method.
    segments : numpy.ndarray
        A 2-dimensional numpy array representing segmentation of the ``image``.
    segments_number : integer
        The number of segments.
    """
    GRAYSCALE_TRANSFORMATION = np.asarray([0.2989, 0.5870, 0.1140])

    def __init__(self,
                 image: np.ndarray,
                 segmentation_mask: Optional[np.ndarray] = None,
                 **kwargs):
        """Constructs a ``Segmentation`` abstract class."""
        assert _validate_input(image, segmentation_mask), 'Invalid input.'

        # The image and the segmentation mask in numpy representation
        self.image = image.copy()  # (np.array(image) * 255).astype(np.uint8)
        if segmentation_mask is None:
            self.segmentation_mask = self.image.copy()
        else:
            self.segmentation_mask = segmentation_mask.copy()

        # Check whether the image is RGB, greyscale or black-and-white
        self.is_rgb = len(self.image.shape) == 3

        # If {0, 1} black-and-white, scale to {0, 255}
        if not self.is_rgb:
            # For the image
            _unique_intensities = set(np.unique(self.image))
            _unique_intensities_n = len(_unique_intensities)
            if _unique_intensities_n in (1, 2):
                logger.info('Assuming a black-and-white image.')
                if 1 in _unique_intensities:
                    logger.info('Rescale 0/1 black-and-white image to 0/255.')
                    _bnw_mask = (self.image == 1)
                    self.image[_bnw_mask] = 255
                if _unique_intensities.difference((0, 1, 255)):
                    raise RuntimeError('Black-and-white images must use 0 as '
                                       'black and 1 or 255 as white.')
        # Repeat the same for the mask
        if len(self.segmentation_mask.shape) != 3:
            _unique_intensities = set(np.unique(self.segmentation_mask))
            _unique_intensities_n = len(_unique_intensities)
            if _unique_intensities_n in (1, 2):
                logger.info('Assuming a black-and-white segmentation mask.')
                print(_unique_intensities, _unique_intensities_n)
                if 1 in _unique_intensities:
                    logger.info('Rescale 0/1 black-and-white segmentation '
                                'mask to 0/255.')
                    _bnw_mask = (self.segmentation_mask == 1)
                    self.segmentation_mask[_bnw_mask] = 255
                if _unique_intensities.difference((0, 1, 255)):
                    raise RuntimeError(
                        'Black-and-white segmentation masks must use 0 as '
                        'black and 1 or 255 as white.')

        # Memorise optional arguments used for the _segment method
        self.kwargs = kwargs

        # Segments map
        self._segments = self._segment()
        assert _validate_segmentation(self._segments,
                                      self.image), 'Invalid segments.'

        # Number of segments
        self.segments_number = np.unique(self._segments).shape[0]

        if self.segments_number == 1:
            warnings.warn(
                'The segmentation returned only **one** segment. '
                'Consider tweaking the parameters to generate a reasonable '
                'segmentation.', UserWarning)

    @abc.abstractmethod
    def _segment(self) -> np.ndarray:
        """
        Segments ``self.image``.

        This methods must be implemented with the desired segmentation
        algorithm.
        It should return a two-dimensional numpy array whose shape corresponds
        to the width and height of ``self.image`` assigning a segment id to
        each of its pixels.
        The segment ids should start at 1 and be a continuous series of
        integers.
        Use the ``self.kwargs`` dictionary to pass (optional) configuration
        parameters to the segmentation function.

        Raises
        ------
        NotImplementedError
            Raised when the ``_segment`` method is not overwritten by the child
            class.

        Returns
        -------
        segmentation : numpy.ndarray
            A two-dimensional numpy array encoding segment id for each pixel
            of the segmented image.
        """
        raise NotImplementedError(  # pragma: nocover
            'Overwrite this method with your implementation of a bespoke '
            'segmentation algorithm.')

        # pylint: disable=unreachable
        segmentation = None  # Use self.kwargs # pragma: nocover
        return segmentation  # pragma: nocover

    @property
    def segments(self) -> np.ndarray:
        """Retrieves the segments."""
        return self._segments

    @segments.setter
    def segments(self, segments: np.ndarray):
        """Setups the segments manually."""
        assert _validate_segmentation(segments, self.image), 'Bad segments.'
        if np.unique(segments).shape[0] == 1:
            warnings.warn('The segmentation has only **one** segment.',
                          UserWarning)
        self._segments = segments

    def set_segments(self, segments: np.ndarray):
        """
        Manually overwrites the segmentation with custom ``segments``.

        ``segments`` must be a non-structured, 2-dimensional, integer-valued
        numpy array with a continuous sequence of unique elements starting
        at 1, which indicate the segment assignment of each pixel.
        The dimension of ``segments`` must agree with the width and height of
        the segmented image.

        .. note::
           The same can be achieved by directly setting the ``self.segments``
           with ``my_segmenter.segments = segments``.
           (A dedicated *setter* method takes care of validating the
           correctness of ``segments``.)

        Parameters
        ----------
        segments : numpy.ndarray
            A 2-dimensional numpy array representing a segmentation.

        Raises
        ------
        IncorrectShapeError
            The ``segments`` array is not 2-dimensional.
            The the height or width the ``segments`` array does not agree with
            the dimensions of the segmented image.
        TypeError
            The ``segments`` array is either a structured numpy array or
            it is not an integer-valued array.
        ValueError
            The unique elements of the ``segments`` array do not form a
            continuous sequence starting at 1.
        """
        assert _validate_segmentation(segments, self.image), 'Bad segments.'
        if np.unique(segments).shape[0] == 1:
            warnings.warn('The segmentation has only **one** segment.',
                          UserWarning)
        self._segments = segments

    def mark_boundaries(self,
                        mask: bool = False,
                        image: Optional[np.ndarray] = None,
                        colour: Optional[RGBcolour] = None) -> np.ndarray:
        """
        Marks segment boundaries atop the image used to initialise this class.

        The boundaries can either be overlaid on top of the image or
        segmentation mask (``mask=True``) used to initialise this class.
        Alternatively, an external ``image`` of the same dimensions can be
        supplied.

        .. note::
           If the image is grayscale, it will be converted to RGB to display
           the segment boundaries.

        Parameters
        ----------
        mask : boolean, optional (default=False)
            If ``True``, plot the segment boundaries on top of
            the segmentation mask;
            if ``False``, plot the segment boundaries atop the image.
        image : numpy.ndarray, optional (default=None)
            If provided, the segment boundaries will be overlaid atop this
            ``image`` instead of the one used to initialise this segmenter.
        colour : tuple(integer, integer, integer), optional (default=None)
            If provided, the segment boundaries will be plotted with this
            RGB colour.

        Raises
        ------
        IncorrectShapeError
            The the height or width the ``image`` array does not agree with
            the dimensions of the class image.
        TypeError
            The ``mask`` parameter is not a boolean.
            The ``colour`` parameter is neither a tuple nor a ``None``;
            or one of its elements is not an integer.
        ValueError
            The ``colour`` parameter is not a 3-tuple, one of its elements
            is outside of the 0--255 range.

        Returns
        -------
        marked_image : numpy.ndarray
            A numpy array holding the image with overlaid segment boundaries.
        """
        assert self._segments is not None, 'The segmenter was not initialised.'
        if not isinstance(mask, bool):
            raise TypeError('The mask parameter must be a boolean.')

        assert _validate_colour(colour), 'Invalid colour.'
        if colour is None:
            _colour = colour
        else:
            _colour = tuple([i / 255 for i in colour])  # Avoids a UserWarning

        if image is None:
            if mask:
                canvas = self.segmentation_mask
            else:
                canvas = self.image
        else:
            assert _validate_image_array(image, 'image'), 'Invalid image.'
            if image.shape[:2] != self.image.shape[:2]:
                raise IncorrectShapeError(
                    'The width and height of the input image do not agree '
                    'with the dimensions of the original image.')
            canvas = image

        bnd_float = ski_segmentation.mark_boundaries(
            canvas, self._segments, color=_colour)
        marked_image = (bnd_float * 255).astype(np.uint8)
        assert _validate_image_array(
            marked_image,
            'image with boundaries'), 'Invalid integer-based image.'
        return marked_image

    def number_segments(
            self,
            segments_subset: Optional[Union[int, List[int]]] = None,
            mask: bool = False,
            image: Optional[np.ndarray] = None,
            colour: Optional[RGBcolour] = None) -> np.ndarray:
        """
        Plots segment numbers on top of the image.

        The numbering can either be overlaid on top of the image or
        segmentation mask (``mask=True``) used to initialise this class.
        Alternatively, an external ``image`` of the same dimensions can be
        supplied.
        By default all the segments are numbered; a selected subset of segments
        can be numbered by providing the ``segments_subset`` parameter.
        The colour of the numbers can be specified via the ``colour`` parameter
        by passing an RGB triplet.

        .. note::
           The numbers may not be printed within the bounds of their respective
           segments when these are not convex.

        Parameters
        ----------
        segments_subset : intiger or list(integer), optional (default=None)
            A number of a specific segment to be numbered or a list of segments
            to be numbered. By default (``None``) all the segments are
            numbered.
        mask : boolean, optional (default=False)
            If ``True``, number the segmentation mask;
            if ``False``, number the image (default).
        image : numpy.ndarray, optional (default=None)
            If provided, this ``image`` will be numbered instead of the one
            used to initialise this segmenter.
        colour : tuple(integer, integer, integer), optional (default=None)
            If provided, the numbers will be plotted with this RGB colour.

        Raises
        ------
        IncorrectShapeError
            The the height or width the ``image`` array does not agree with
            the dimensions of the class image.
        TypeError
            The ``mask`` parameter is not a boolean.
            The ``colour`` parameter is neither a tuple nor a ``None``;
            or one of its elements is not an integer.
            The ``segments_subset`` parameter is neither ``None``, an integer,
            or a list of integers; one of the segment ids in this list is not
            an integer.
        ValueError
            The ``colour`` parameter is not a 3-tuple, one of its elements
            is outside of the 0--255 range.
            One of the segment ids provided via ``segments_subset`` is invalid
            for the class segmentation, the list of segments is empty or some
            of its elements are duplicated.

        Returns
        -------
        numbered_image : numpy.ndarray
            A numpy array holding the image with the selected subset of
            segments numbered.
        """
        # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        assert self._segments is not None, 'The segmenter was not initialised.'
        unique_segments = np.unique(self._segments)

        if segments_subset is None:
            segments_subset_ = unique_segments
        else:
            if isinstance(segments_subset, int):
                if segments_subset not in unique_segments:
                    raise ValueError(
                        ('The segment id {} does not correspond to any of '
                         'the known segments ({}).').format(
                             segments_subset, unique_segments.tolist()))
                segments_subset_ = np.asarray([segments_subset])
            elif isinstance(segments_subset, list):
                if not segments_subset:
                    raise ValueError('The list of segments cannot be empty.')
                if len(segments_subset) != len(set(segments_subset)):
                    raise ValueError('The list of segments has duplicates.')
                for i in segments_subset:
                    if not isinstance(i, int):
                        raise TypeError(
                            'The segment id {} is not an integer.'.format(i))
                    if i not in unique_segments:
                        raise ValueError(
                            ('The segment id {} does not correspond to any of '
                             'the known segments ({}).').format(
                                 i, unique_segments.tolist()))
                segments_subset_ = np.asarray(segments_subset)
            else:
                raise TypeError('Segments subset must be either of None, '
                                'an integer or a list of integers.')

        if not isinstance(mask, bool):
            raise TypeError('The mask parameter must be a boolean.')

        assert _validate_colour(colour), 'Invalid colour.'

        if image is None:
            if mask:
                canvas = self.segmentation_mask
            else:
                canvas = self.image
            is_rgb = self.is_rgb
        else:
            assert _validate_image_array(image, 'image'), 'Invalid image.'
            if image.shape[:2] != self.image.shape[:2]:
                raise IncorrectShapeError(
                    'The width and height of the input image do not agree '
                    'with the dimensions of the original image.')
            canvas = image
            is_rgb = len(image.shape) == 3
        if not is_rgb:
            canvas = ski_colour.gray2rgb(canvas)
            assert _validate_image_array(
                canvas, 'grayscale->RGB image'), 'Invalid image.'
            canvas = canvas.astype(np.uint8)
        # canvas = self.mark_boundaries(image=canvas, colour=colour)

        # font = ImageFont.truetype('~/Library/Fonts/Calibri.ttf', 11)
        font = ImageFont.load_default()

        numbered_canvas = Image.fromarray(canvas)
        numbered_canvas_draw = ImageDraw.Draw(numbered_canvas)

        for segment_id in segments_subset_:
            segment_id_mask = (self._segments == segment_id)
            segment_id_indices = np.argwhere(segment_id_mask)

            # segment_x_left = segment_id_indices[:, 1].min().astype(int)
            segment_y_top = segment_id_indices[:, 0].min().astype(int)

            eligible_y_ind = np.where(
                segment_id_indices[:, 0] == segment_y_top)[0]
            segment_x_middle = segment_id_indices[eligible_y_ind].min(
                axis=0)[1]

            numbered_canvas_draw.text((segment_x_middle, segment_y_top),
                                      '{}'.format(segment_id),
                                      fill=colour,
                                      font=font)

        numbered_image = np.asarray(numbered_canvas)
        assert _validate_image_array(
            numbered_image, 'numbered image'), 'Invalid numbered image.'

        return numbered_image

    def highlight_segments(
            self,
            segments_subset: Optional[Union[int, List[int]]] = None,
            mask: bool = False,
            image: Optional[np.ndarray] = None,
            colour: Optional[Union[RGBcolour, List[RGBcolour]]] = None
    ) -> np.ndarray:
        """
        Highlights image segments by translucently colouring them.

        The highlighting can either be applied on top of the image or
        segmentation mask (``mask=True``) used to initialise this class.
        Alternatively, an external ``image`` of the same dimensions can be
        supplied.
        By default all the segments are highlighted; a selected subset of
        segments can be highlighted by providing the ``segments_subset``
        parameter.
        The segments are highlighted with different colours by default
        (``colour=None``);
        alternatively, a single colour can be supplied with the ``colour``
        parameter.
        It is also possible to specify a unique colour for each segment
        by setting ``colour`` to a list of RGB triplets;
        in this case the list must have the same length as the number of
        segments being highlighted.

        Parameters
        ----------
        segments_subset : intiger or list(integer), optional (default=None)
            A number of a specific segment or a list of segments to be
            highlighted. By default (``None``) all the segments are
            highlighted.
        mask : boolean, optional (default=False)
            If ``True``, highlight the segmentation mask;
            if ``False``, highlight the image (default).
        image : numpy.ndarray, optional (default=None)
            If provided, this ``image`` will be highlighted instead of the one
            used to initialise this segmenter.
        colour : tuple(integer, integer, integer) or \
list(tuple(integer, integer, integer)), optional (default=None)
            If provided, the regions will be highlighted with a single RGB
            colour or each segment will be highlighted with its unique colour.
            By default (``None``) every segment receives a unique colour.

        Raises
        ------
        IncorrectShapeError
            The the height or width the ``image`` array does not agree with
            the dimensions of the class image.
        TypeError
            The ``mask`` parameter is not a boolean.
            The ``colour`` parameter is neither of ``None``, a tuple or a list
            of tuples; or one of its elements is not an integer.
            The ``segments_subset`` parameter is neither ``None``, an integer,
            or a list of integers; one of the segment ids in this list is not
            an integer.
        ValueError
            If ``colour`` is provided as a list, the list is either empty or
            the number of colours provided is not the same as the number of
            segments chosen to be highlighted.
            A colour is not a 3-tuple or one of its elements is outside of the
            0--255 range.
            One of the segment ids provided via ``segments_subset`` is invalid
            for the class segmentation, the list of segments is empty or some
            of its elements are duplicated.

        Returns
        -------
        image_highlighted : numpy.ndarray
            A numpy array holding the image with the selected subset of
            segments highlighted.
        """
        # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        assert self._segments is not None, 'The segmenter was not initialised.'
        unique_segments = np.unique(self._segments)

        if segments_subset is None:
            segments_subset_ = unique_segments
        else:
            if isinstance(segments_subset, int):
                if segments_subset not in unique_segments:
                    raise ValueError(
                        ('The segment id {} does not correspond to any of '
                         'the known segments ({}).').format(
                             segments_subset, unique_segments.tolist()))
                segments_subset_ = np.asarray([segments_subset])
            elif isinstance(segments_subset, list):
                if not segments_subset:
                    raise ValueError('The list of segments cannot be empty.')
                if len(segments_subset) != len(set(segments_subset)):
                    raise ValueError('The list of segments has duplicates.')
                for i in segments_subset:
                    if not isinstance(i, int):
                        raise TypeError(
                            'The segment id {} is not an integer.'.format(i))
                    if i not in unique_segments:
                        raise ValueError(
                            ('The segment id {} does not correspond to any of '
                             'the known segments ({}).').format(
                                 i, unique_segments.tolist()))
                segments_subset_ = np.asarray(segments_subset)
            else:
                raise TypeError('Segments subset must be either of None, '
                                'an integer or a list of integers.')

        if not isinstance(mask, bool):
            raise TypeError('The mask parameter must be a boolean.')

        if isinstance(colour, tuple):
            assert _validate_colour(colour), 'Invalid colour.'
            colour = [colour]
        elif isinstance(colour, list):
            if not colour:
                raise ValueError('The colour list cannot be empty.')
            if len(colour) != segments_subset_.shape[0]:
                raise ValueError('If colours are provided as a list, their '
                                 'number must match the number of segments '
                                 'chosen to be highlighted.')
            for clr in colour:
                assert _validate_colour(clr), 'Invalid colour.'
        else:
            if colour is not None:
                raise TypeError('The colour can be either of an RGB tuple, '
                                'a list of RGB tuples or None.')

        if image is None:
            if mask:
                canvas = self.segmentation_mask
            else:
                canvas = self.image
            is_rgb = self.is_rgb
        else:
            assert _validate_image_array(image, 'image'), 'Invalid image.'
            if image.shape[:2] != self.image.shape[:2]:
                raise IncorrectShapeError(
                    'The width and height of the input image do not agree '
                    'with the dimensions of the original image.')
            canvas = image
            is_rgb = len(image.shape) == 3
        if not is_rgb:
            canvas = ski_colour.gray2rgb(canvas)

        highlight_mask = np.zeros(shape=self._segments.shape, dtype=int)
        for i, segments in enumerate(segments_subset_):
            s_mask = (self._segments == segments)
            highlight_mask[s_mask] = i + 1

        # This step converts the image to grayscale first...
        image_highlighted_ = ski_colour.label2rgb(
            highlight_mask,
            image=canvas,
            colors=colour,
            bg_label=0,
            bg_color=None,
            kind='overlay')
        image_highlighted_ = (image_highlighted_ * 255).astype(np.uint8)
        assert _validate_image_array(
            image_highlighted_, 'highlighted image'), 'Bad highlighted image.'

        # ... so we need to restore the colour to the background
        image_highlighted = canvas.copy()
        colour_mask = highlight_mask.astype(bool)
        image_highlighted[colour_mask] = image_highlighted_[colour_mask]

        return image_highlighted

    def _stain_segments(
            self,
            segments_subset: Optional[Union[int, List[int]]] = None,
            mask: bool = False,
            image: Optional[np.ndarray] = None,
            colour: Optional[Union[str, List[str]]] = None) -> np.ndarray:
        """
        Stain selected segments of the image with red, green or blue tint.

        The staining can either be applied on top of the image or
        segmentation mask (``mask=True``) used to initialise this class.
        Alternatively, an external RGB ``image`` of the same dimensions can be
        supplied.
        By default all the segments are stained in *blue*; a selected subset of
        segments can be stained by providing the ``segments_subset`` parameter.
        The ``colour`` can be either of ``'r'``, ``'g'`` or ``'b'`` --
        respectively for red, green and blue -- or a list of thereof, which has
        the same length as the number of segments specified via
        ``segments_subset``.

        .. note::
           This method works only with RGB images.

        Parameters
        ----------
        segments_subset : intiger or list(integer), optional (default=None)
            A number of a specific segment or a list of segments to be stained.
            By default (``None``) all the segments are stained.
        mask : boolean, optional (default=False)
            If ``True``, stain the segmentation mask;
            if ``False``, stain the image (default).
        image : numpy.ndarray, optional (default=None)
            If provided, this ``image`` will be stained instead of the one
            used to initialise this segmenter.
        colour : string or list(string), optional (default=None)
            Either of ``'r'``, ``'g'`` or ``'b'`` for red, green or blue
            respectively; or list thereof of the length equal to the subset of
            segments being stained.
            By default (``None``) every segment is stained in *blue*.
            If provided as string, the regions will be stained in a single
            colour; if provided as a list, each segment will be stained with
            its unique colour.

        Raises
        ------
        IncorrectShapeError
            The the height or width the ``image`` array does not agree with
            the dimensions of the class image or the ``image`` is not RGB.
        RuntimeError
            The class has been initialised with a black-and-white or
            grayscale image.
        TypeError
            The ``mask`` parameter is not a boolean.
            The ``segments_subset`` parameter is neither ``None``, an integer,
            or a list of integers; one of the segment ids in this list is not
            an integer.
            The ``colour`` is neither a string nor a list of strings.
        ValueError
            One of the segment ids provided via ``segments_subset`` is invalid
            for the class segmentation, the list of segments is empty or some
            of its elements are duplicated.
            One of the colour strings is neither of ``'r'``, ``'g'`` or
            ``'b'``.
            The colour list is empty or its length is different to the number
            of segments selected to be stained.

        Returns
        -------
        image_stained : numpy.ndarray
            A numpy array holding the image with the selected subset of
            segments stained.
        """
        # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        _accepted_colours = ('r', 'g', 'b')
        _colour_map = {'r': 0, 'g': 1, 'b': 2}

        if not self.is_rgb:
            raise RuntimeError('Staining segments of an image can only be '
                               'performed on RGB images.')

        assert self._segments is not None, 'The segmenter was not initialised.'
        unique_segments = np.unique(self._segments)

        if segments_subset is None:
            segments_subset_ = unique_segments.astype(int).tolist()
        else:
            if isinstance(segments_subset, int):
                if segments_subset not in unique_segments:
                    raise ValueError(
                        ('The segment id {} does not correspond to any of '
                         'the known segments ({}).').format(
                             segments_subset, unique_segments.tolist()))
                segments_subset_ = [segments_subset]
            elif isinstance(segments_subset, list):
                if not segments_subset:
                    raise ValueError('The list of segments cannot be empty.')
                if len(segments_subset) != len(set(segments_subset)):
                    raise ValueError('The list of segments has duplicates.')
                for i in segments_subset:
                    if not isinstance(i, int):
                        raise TypeError(
                            'The segment id {} is not an integer.'.format(i))
                    if i not in unique_segments:
                        raise ValueError(
                            ('The segment id {} does not correspond to any of '
                             'the known segments ({}).').format(
                                 i, unique_segments.tolist()))
                segments_subset_ = segments_subset
            else:
                raise TypeError('Segments subset must be either of None, '
                                'an integer or a list of integers.')

        if not isinstance(mask, bool):
            raise TypeError('The mask parameter must be a boolean.')

        segments_subset_n = len(segments_subset_)
        if colour is None:
            colour = segments_subset_n * ['b']
        elif isinstance(colour, str):
            if colour not in _accepted_colours:
                raise ValueError(('One of the provided colour strings ({}) is '
                                  "not 'r', 'g' or 'b'.").format(colour))
            colour = segments_subset_n * [colour]
        elif isinstance(colour, list):
            if not colour:
                raise ValueError('The colour list cannot be empty.')
            if len(colour) != segments_subset_n:
                raise ValueError('If colours are provided as a list, their '
                                 'number must match the number of segments '
                                 'chosen to be highlighted.')
            for clr in colour:
                if clr not in _accepted_colours:
                    raise ValueError(
                        ('One of the provided colour strings ({}) is not '
                         "'r', 'g' or 'b'.").format(clr))
        else:
            raise TypeError("The colour can be either of 'r', 'g' or 'b' "
                            'strings, a list thereof or None.')

        if image is None:
            if mask:
                canvas = self.segmentation_mask
            else:
                canvas = self.image
        else:
            assert _validate_image_array(image, 'image'), 'Invalid image.'
            if len(image.shape) != 3:
                raise IncorrectShapeError(
                    'The user-provided image is not RGB.')
            if image.shape[:2] != self.image.shape[:2]:
                raise IncorrectShapeError(
                    'The width and height of the input image do not agree '
                    'with the dimensions of the original image.')
            canvas = image

        image_stained = canvas.copy()
        max_value = np.max(image_stained)
        for id_, clr in zip(segments_subset_, colour):
            pixel_mask = get_segment_mask(id_, self._segments)
            colour_channel = _colour_map[clr]
            image_stained[pixel_mask, colour_channel] = max_value

        return image_stained

    def grayout_segments(
            self,
            segments_subset: Optional[Union[int, List[int]]] = None,
            mask: bool = False,
            image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Grays out a selected subset of segments in the RGB image.

        The graying out can either be applied on top of the image or
        segmentation mask (``mask=True``) used to initialise this class.
        Alternatively, an external RGB ``image`` of the same dimensions can be
        supplied.
        By default all the segments are grayed out; a selected subset of
        segments can be grayed out by providing the ``segments_subset``
        parameter.

        Parameters
        ----------
        segments_subset : intiger or list(integer), optional (default=None)
            A number of a specific segment or a list of segments to be
            grayed out. By default (``None``) all the segments are grayed out.
        mask : boolean, optional (default=False)
            If ``True``, gray out the segmentation mask;
            if ``False``, gray out the image (default).
        image : numpy.ndarray, optional (default=None)
            If provided, this ``image`` will be grayed out instead of the one
            used to initialise this segmenter.

        Raises
        ------
        IncorrectShapeError
            The the height or width the ``image`` array does not agree with
            the dimensions of the class image or the ``image`` is not RGB.
        RuntimeError
            The class has been initialised with a black-and-white or
            grayscale image.
        TypeError
            The ``mask`` parameter is not a boolean.
            The ``segments_subset`` parameter is neither ``None``, an integer,
            or a list of integers; one of the segment ids in this list is not
            an integer.
        ValueError
            One of the segment ids provided via ``segments_subset`` is invalid
            for the class segmentation, the list of segments is empty or some
            of its elements are duplicated.

        Returns
        -------
        image_grayscale : numpy.ndarray
            A numpy array holding the image with the selected subset of
            segments grayed out.
        """
        # pylint: disable=too-many-branches
        assert self._segments is not None, 'The segmenter was not initialised.'

        if not self.is_rgb:
            raise RuntimeError('Graying out segments of an image can only be '
                               'performed on RGB images.')

        unique_segments = np.unique(self._segments)

        if segments_subset is None:
            segments_subset = unique_segments.astype(int).tolist()
        else:
            if isinstance(segments_subset, int):
                if segments_subset not in unique_segments:
                    raise ValueError(
                        ('The segment id {} does not correspond to any of '
                         'the known segments ({}).').format(
                             segments_subset, unique_segments.tolist()))
                segments_subset = [segments_subset]
            elif isinstance(segments_subset, list):
                if not segments_subset:
                    raise ValueError('The list of segments cannot be empty.')
                if len(segments_subset) != len(set(segments_subset)):
                    raise ValueError('The list of segments has duplicates.')
                for i in segments_subset:
                    if not isinstance(i, int):
                        raise TypeError(
                            'The segment id {} is not an integer.'.format(i))
                    if i not in unique_segments:
                        raise ValueError(
                            ('The segment id {} does not correspond to any of '
                             'the known segments ({}).').format(
                                 i, unique_segments.tolist()))
                segments_subset = segments_subset
            else:
                raise TypeError('Segments subset must be either of None, '
                                'an integer or a list of integers.')

        if not isinstance(mask, bool):
            raise TypeError('The mask parameter must be a boolean.')

        if image is None:
            if mask:
                canvas = self.segmentation_mask
            else:
                canvas = self.image
        else:
            assert _validate_image_array(image, 'image'), 'Invalid image.'
            if len(image.shape) != 3:
                raise IncorrectShapeError(
                    'The user-provided image is not RGB.')
            if image.shape[:2] != self.image.shape[:2]:
                raise IncorrectShapeError(
                    'The width and height of the input image do not agree '
                    'with the dimensions of the original image.')
            canvas = image

        # Convert RGB into a grayscale representation
        image_grayscale_ = np.dot(canvas, self.GRAYSCALE_TRANSFORMATION)
        image_grayscale_ = np.repeat(
            image_grayscale_[:, :, np.newaxis], 3, axis=2).astype(np.uint8)

        # grayscale_image_ = np.clip(
        #     (0.1 * grayscale_image_ + 200), 0, 255)
        # grayscale_image = np.dstack(
        #     [np.zeros((grayscale_image_.shape[0],
        #                grayscale_image_.shape[1],
        #                2)),
        #      grayscale_image_]).astype(np.uint8)

        # Filter out segments
        image_grayscale = canvas.copy()
        grayscale_mask = get_segment_mask(segments_subset, self._segments)
        image_grayscale[grayscale_mask] = image_grayscale_[grayscale_mask]

        return image_grayscale

    def merge_segments(self,
                       segments_grouping: Union[List[int], List[List[int]]],
                       inplace: bool = True,
                       segments: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Merges segments based on the provided grouping.

        The merging can either be applied to the segmentation stored in the
        class to a segmentation passed as a parameter (``segments``).
        By default (``inplace=True``) the segmentation stored in the class
        will be updated to the merged segmentation.

        Parameters
        ----------
        segments_grouping : list(integer) or list(list(integer))
            A collection or a set of collections of segment ids to be merged.
        inplace : boolean, optional (default=True)
            If ``True``, overwrite the segmentation stored in the class.
        segments : numpy.ndarray, optional (default=None)
            If provided, the merging will be performed on this segmentation
            instead of the one stored in the class.

        Raises
        ------
        IncorrectShapeError
            The ``segments`` array is not 2-dimensional.
            The the height or width the ``segments`` array does not agree with
            the dimensions of the segmented image.
        TypeError
            The ``segments`` array is either a structured numpy array or
            it is not an integer-valued array.
            The inplace parameter is not a boolean.
            The segments grouping is not a list of integers or lists.
            One of the segment ids is not an integer.
        ValueError
            The unique elements of the ``segments`` array do not form a
            continuous sequence starting at 1.
            The segments grouping is an empty lists or the list has duplicates.
            One of the segment ids is invalid or appears across different
            groupings.

        Returns
        -------
        merged_segments : numpy.ndarray
            A 2-dimensional numpy array holding the merged segmentation.
        """
        # pylint: disable=too-many-branches,too-many-locals
        assert self._segments is not None, 'The segmenter was not initialised.'
        if segments is None:
            segments_ = self._segments
        else:
            assert _validate_segmentation(segments,
                                          self.image), 'Invalid segmentation.'
            segments_ = segments

        if not isinstance(inplace, bool):
            raise TypeError('The inplace parameter must be a boolean.')

        unique_segments = np.unique(segments_)
        if isinstance(segments_grouping, list):
            if not segments_grouping:
                raise ValueError(
                    'The segments grouping cannot be an empty list.')

            if isinstance(segments_grouping[0], int):
                if len(segments_grouping) != len(set(segments_grouping)):
                    raise ValueError('The segments grouping has duplicates.')
                for i in segments_grouping:
                    if not isinstance(i, int):
                        raise TypeError(
                            'The segment id {} is not an integer.'.format(i))
                    if i not in unique_segments:
                        raise ValueError(
                            ('The segment id {} does not correspond to any of '
                             'the known segments ({}).').format(
                                 i, unique_segments.tolist()))
                segments_grouping_ = [segments_grouping]
            elif isinstance(segments_grouping[0], list):
                _item_collector = []
                for i in segments_grouping:
                    if not isinstance(i, list):
                        raise TypeError(
                            'The nested elements of segments grouping are not '
                            'consistent. If one is a list, all must be lists.')
                    if len(i) != len(set(i)):
                        raise ValueError(
                            'The segments grouping has duplicates.')
                    for j in i:
                        if not isinstance(j, int):
                            raise TypeError(
                                'The segment id {} is not an integer.'.format(
                                    j))
                        if j not in unique_segments:
                            raise ValueError(
                                ('The segment id {} does not correspond to '
                                 'any of the known segments ({}).').format(
                                     j, unique_segments.tolist()))
                        if j in _item_collector:
                            raise ValueError(
                                ('The segment id {} is duplicated across '
                                 'grouping lists.').format(j))
                        _item_collector.append(j)
                segments_grouping_ = segments_grouping  # type: ignore
            else:
                raise TypeError('The segments grouping must either be a list '
                                'of integers or a list of lists.')
        else:
            raise TypeError('Segments grouping must be a list.')

        merged_segments_ = segments_.copy()
        for group in segments_grouping_:
            mask = get_segment_mask(group, segments_)  # type: ignore
            # use the smallest id to avoid collisions
            merged_segments_[mask] = min(group)

        # Remap segment ids to ensure continuous numbering starting at 1
        merged_segments = np.full(
            merged_segments_.shape, -1, dtype=merged_segments_.dtype)
        for new_id, old_id in enumerate(np.unique(merged_segments_)):
            mask = (merged_segments_ == old_id)
            merged_segments[mask] = new_id + 1
        assert not (merged_segments == -1).any(), 'Internal remapping error.'
        assert _validate_segmentation(merged_segments,
                                      self.image), 'Invalid segmentation.'

        if inplace:
            self.set_segments(merged_segments)

        return merged_segments


class Slic(Segmentation):
    """
    Wraps the slic segmentation algorithm implemented in scikit-image.

    .. versionadded:: 0.1.1

    This class provides an interface for the slic segmentation implemented by
    the :func:`skimage.segmentation.slic` function.

    For the documentation see the specification of the
    :class:`fatf.utils.data.segmentation.Segmentation` abstract class.
    The initialisation parameters specific to the slic segmenter are
    documented below.

    Parameters
    ----------
    n_segments : integer, optional (default=10)
        The number of segments desired of slic.

    Raises
    ------
    TypeError
        The number of segments parameter is not an integer.
    ValueError
        The number of segments parameter is less than 2.
    """

    def __init__(self,
                 image: np.ndarray,
                 segmentation_mask: Optional[np.ndarray] = None,
                 n_segments: int = 10):
        """Constructs a ``slic`` segmenter."""
        super().__init__(image, segmentation_mask, n_segments=n_segments)

    def _segment(self):
        """
        Wraps the :func:`skimage.segmentation.slic` function.

        Raises
        ------
        TypeError
            The number of segments parameter is not an integer.
        ValueError
            The number of segments parameter is less than 2.

        Returns
        -------
        segments : numpy.ndarray
            Segments of the image (segmentation mask).
        """
        assert 'n_segments' in self.kwargs, 'Parameter missing.'
        n_segments = self.kwargs.get('n_segments')
        if not isinstance(n_segments, int):
            raise TypeError('The n_segments parameter must be an integer.')
        if n_segments < 2:
            raise ValueError('The n_segments parameter must be at least 2.')

        segments = ski_segmentation.slic(
            self.segmentation_mask, start_label=1, **self.kwargs)
        return segments


class QuickShift(Segmentation):
    """
    Wraps the quickshift segmentation algorithm implemented in scikit-image.

    .. versionadded:: 0.1.1

    This class provides an interface for the quickshift segmentation
    implemented by the :func:`skimage.segmentation.quickshift` function.

    For the documentation see the specification of the
    :class:`fatf.utils.data.segmentation.Segmentation` abstract class.
    The initialisation parameters specific to the quickshift segmenter are
    documented below.
    The parameter values for ``ratio``, ``kernel_size`` and ``max_dist`` are
    by default set to the values used by the official LIME_ implementation.

    .. _LIME: https://github.com/marcotcr/lime

    Parameters
    ----------
    ratio : number, optional (default=0.2)
        Balances color-space proximity and image-space proximity.
        Higher values give more weight to color-space.
        Between 0 and 1.
    kernel_size : number, optional (default=4)
        Width of Gaussian kernel used in smoothing the sample density.
        Higher means fewer clusters.
    max_dist : number, optional (default=200)
        Cut-off point for data distances. Higher means fewer clusters.

    Raises
    ------
    TypeError
        The ratio, kernel size or max dist parameter is not an integer.
    ValueError
        The ratio parameter is outside of the 0--1 range.
    """

    def __init__(self,
                 image: np.ndarray,
                 segmentation_mask: Optional[np.ndarray] = None,
                 ratio: float = 0.2,
                 kernel_size: float = 4,
                 max_dist: float = 200):
        """Constructs a ``quickshift`` segmenter."""
        # pylint: disable=too-many-arguments
        super().__init__(
            image,
            segmentation_mask,
            ratio=ratio,
            kernel_size=kernel_size,
            max_dist=max_dist)

    def _segment(self):
        """
        Wraps the :func:`skimage.segmentation.quickshift` function.

        Raises
        ------
        TypeError
            The ratio, kernel size or max dist parameter is not an integer.
        ValueError
            The ratio parameter is outside of the 0--1 range.

        Returns
        -------
        segments : numpy.ndarray
            Segments of the image (segmentation mask).
        """
        assert ('ratio' in self.kwargs and 'kernel_size' in self.kwargs
                and 'max_dist' in self.kwargs), 'Parameters missing.'
        ratio = self.kwargs.get('ratio')
        if not isinstance(ratio, Number):
            raise TypeError('Ratio should be a number.')
        if ratio < 0 or ratio > 1:
            raise ValueError('Ratio must be between 0 and 1.')
        kernel_size = self.kwargs.get('kernel_size')
        if not isinstance(kernel_size, Number):
            raise TypeError('Kernel size should be a number.')
        max_dist = self.kwargs.get('max_dist')
        if not isinstance(max_dist, Number):
            raise TypeError('Max dist should be a number.')

        segments = ski_segmentation.quickshift(self.segmentation_mask,
                                               **self.kwargs)
        segments = segments + 1  # quickshift starts segment numbering at 0
        return segments


def get_segment_mask(segments_subset: Union[int, List[int]],
                     segmentation: np.ndarray) -> np.ndarray:
    """
    Generates a boolean mask for pixels belonging to the specified segments.

    .. versionadded:: 0.1.1

    The mask holds ``True`` where the pixel belongs to one of the specified
    segments.

    Parameters
    ----------
    segments_subset : intiger or list(integer)
        A number of a specific segment or a list of segments for which a mask
        will be created.
    segmentation : np.ndarray
        A 2-dimensional numpy array defining segmentation of an image
        (each unique integer -- in sequence starting at 1 -- indicates segment
        id of the pixel at this coordinate).

    Raises
    ------
    IncorrectShapeError
        The ``segmentation`` array is not 2-dimensional.
    TypeError
        The ``segments_subset`` parameter is neither an integer nor a list of
        integers; one of the segment ids in this list is not an integer.
        The ``segmentation`` array is either a structured numpy array or
        it is not an integer-valued array.
    ValueError
        One of the segment ids provided via ``segments_subset`` is invalid
        for the ``segmentation`` or some of its elements are duplicated.
        The unique elements of the ``segments`` array do not form a continuous
        sequence starting at 1.

    Returns
    -------
    segment_mask : numpy.ndarray
        A boolean numpy array of the same shape as ``segmentation`` indicating
        the pixels (``True``) belonging to the specified segments.
    """
    # Validate segments
    assert _validate_segmentation(
        segmentation, np.zeros(shape=segmentation.shape,
                               dtype=np.int8)), 'Invalid segmentation array.'
    unique_segments = np.unique(segmentation)

    if isinstance(segments_subset, int):
        if segments_subset not in unique_segments:
            raise ValueError(
                ('The segment id {} does not correspond to any of '
                 'the known segments ({}).').format(segments_subset,
                                                    unique_segments.tolist()))
        segments_subset_ = np.asarray([segments_subset])
    elif isinstance(segments_subset, list):
        if len(segments_subset) != len(set(segments_subset)):
            raise ValueError('The list of segments has duplicates.')
        for i in segments_subset:
            if not isinstance(i, int):
                raise TypeError(
                    'The segment id {} is not an integer.'.format(i))
            if i not in unique_segments:
                raise ValueError(
                    ('The segment id {} does not correspond to any of '
                     'the known segments ({}).').format(
                         i, unique_segments.tolist()))
        segments_subset_ = np.asarray(segments_subset)
    else:
        raise TypeError('Segments subset must either be an integer '
                        'or a list of integers.')

    # Get a 2-D mask where True indicates pixels from the selected segments
    segment_mask = np.zeros(shape=segmentation.shape, dtype=bool)
    for segment_id in segments_subset_:
        mask = (segmentation == segment_id)
        segment_mask[mask] = True
    return segment_mask
