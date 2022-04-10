"""
The :mod:`fatf.utils.data.occlusion` module implements functions to generate
partial occlusion of images represented as numpy arrays.

.. versionadded:: 0.1.1
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

from typing import Callable, Dict, List, Optional, Tuple, Union

import logging
import random
import warnings

import numpy as np

from fatf.exceptions import IncorrectShapeError

import fatf.utils.array.validation as fuav
import fatf.utils.data.segmentation as fuds

__all__ = ['Occlusion']

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

RGBcolour = Tuple[int, int, int]
SegmentColour = Union[int, RGBcolour]


class Occlusion(object):  # pylint: disable=useless-object-inheritance
    """
    Implements functionality for (partiall) occlusion a segmented image.

    .. versionadded:: 0.1.1

    This class allows to occlude specified segments of an image with
    a selected colouring strategy.
    This class is designed for images represented as numpy arrays with their
    values in the 0--255 range:

    - 2-dimensional arrays for grayscale (0--255 range) and
      black-and-white (0 and 255 valued) images; and
    - 3-dimensional arrays for colour images.

    The colouring strategy used by this class can be overwritten either with
    the ``set_colouring_strategy`` method or by directly setting the
    ``colouring_strategy`` attribute, both of which will perform the
    necessary validation.

    Parameters
    ----------
    image : numpy.ndarray
        A numpy array representing an image to be occluded.
    segments : numpy.ndarray
        A 2-dimensional numpy array defining segmentation of an image
        (each unique integer -- in sequence starting at 1 -- indicates segment
        id of the pixel at this coordinate).
    colour : string, integer or tuple(integer, integer, integer), \
optional (default=None)
        One of the colour specifier strings: ``'mean'``, ``'black'``,
        ``'white'``, ``'red'``, ``'green'``, ``'blue'``, ``'pink'``,
        ``'random'``, ``'random-patch'``, ``'randomise'`` or
        ``'randomise-patch'``.
        For black-and-white and grayscale images it can be an integer
        specifying respectively black or white (0 or 255)
        and intensity (0--255).
        RGB images, on the other hand, accept RGB triplets.
        See the documentation of the
        :func:`fatf.utils.data.occlusion.Occlusion.set_colouring_strategy`
        method for more details.
        By default (``None``) the mean-colour occlusion is selected for colour
        and grayscale images, and black occlusion for black-and-white images.

    Raises
    ------
    IncorrectShapeError
        The input ``image`` is neither a 2- nor 3-dimensional numpy array.
        The ``segments`` array is not 2-dimensional.
        The the height or width the ``segments`` array does not agree with
        the dimensions of the segmented image.
    RuntimeError
        A black-and-white image does not use 0 as black and 1 or 255 as white.
        The mean-colour occlusion was selected for a black-and-white image.
    TypeError
        The input ``image`` is either a structured numpy array or it is not a
        numerical array.
        The ``segments`` array is either a structured numpy array or
        it is not an integer-valued array.
        The colour specifier is neither ``None``, a string, an integer or
        a tuple; or one of the tuple elements is not an integer.
    ValueError
        The elements of the input ``image`` are not integers in
        the 0--255 range.
        The unique elements of the ``segments`` array do not form a continuous
        sequence starting at 1.
        The colour string specifier refers to an unknown colouring
        strategy name.
        An colour integer specifier for a black-and-white image is neither
        0 for black nor 1 or 255 for white.
        A colour component for an RGB image or pixel intensity for a grayscale
        image is outside of the 0--255 range.
        An RGB colour specifier is not a 3-tuple.

    Attributes
    ----------
    image : numpy.ndarray
        A numpy array representing an image to be occluded.
    segments : numpy.ndarray
        A 2-dimensional numpy array representing segmentation of the ``image``.
    is_rgb : boolean
        Indicates whether the ``image`` is RGB or grayscale/black-and-white.
    is_bnw : boolean
        Indicates whether the ``image`` is black-and-white.
    unique_segments : numpy.ndarray
        A 1-D numpy array holding unique segmentation ids.
    segments_number : integer
        The number of segments.
    colouring_strategy : callable(image indices)
        A function that returns a numpy array with the colour(s) of patches
        for the requested indices (mask) of the occluded ``image``.
    """

    def __init__(self,
                 image: np.ndarray,
                 segments: np.ndarray,
                 colour: Optional[Union[str, int, RGBcolour]] = None):
        """Constructs an occluder object."""
        assert fuds._validate_image_array(  # pylint: disable=protected-access
            image, 'image'), 'Invalid image.'
        self.image = image.copy()

        assert fuds._validate_segmentation(  # pylint: disable=protected-access
            segments, image), 'Bad segments.'
        self.segments = segments.copy()

        self.is_rgb = len(self.image.shape) == 3

        # If {0, 1} black-and-white, scale to {0, 255}
        if self.is_rgb:
            is_bnw = False
        else:
            _unique_intensities = set(np.unique(self.image))
            _unique_intensities_n = len(_unique_intensities)
            if _unique_intensities_n in (1, 2):
                is_bnw = True
                logger.info('Assuming a black-and-white image.')
                if 1 in _unique_intensities:
                    logger.info('Rescale 0/1 black-and-white image to 0/255.')
                    _bnw_mask = (self.image == 1)
                    self.image[_bnw_mask] = 255
                if _unique_intensities.difference((0, 1, 255)):
                    raise RuntimeError('Black-and-white images must use 0 as '
                                       'black and 1 or 255 as white.')
            else:
                is_bnw = False
        self.is_bnw = is_bnw

        self.unique_segments = np.unique(self.segments)
        self.segments_number = self.unique_segments.shape[0]

        if self.segments_number == 1:
            warnings.warn('The segmentation has only **one** segment.',
                          UserWarning)

        # This must be called as last since it requires other class attributes
        if colour is None:
            if self.is_bnw:
                colour = 'black'
            else:
                colour = 'mean'
        self._colouring_strategy = self._generate_colouring_strategy(colour)

    @property
    def colouring_strategy(self) -> Callable:
        """Retrieves the colouring strategy."""
        return self._colouring_strategy

    @colouring_strategy.setter
    def colouring_strategy(self, colour: Union[str, int, RGBcolour]):
        """Setups the colouring strategy."""
        self._colouring_strategy = self._generate_colouring_strategy(colour)

    def set_colouring_strategy(self, colour: Union[str, int, RGBcolour]):
        """
        Selects the colouring strategy.

        For colour images this can be specified with one of the following
        strings:

        * ``'mean'`` -- the mean RGB colour of each segment;
        * ``'black'`` -- RGB black (0, 0, 0);
        * ``'white'`` -- RGB white (255, 255, 255);
        * ``'red'`` -- RGB white (255, 0, 0);
        * ``'green'`` -- RGB white (0, 255, 0);
        * ``'blue'`` -- RGB white (0, 0, 255);
        * ``'pink'`` -- RGB white (255, 192, 203);
        * ``'random'`` -- a single random RGB colour for all the segments;
        * ``'random-patch'`` -- a separate random RGB colour for each segment;
        * ``'randomise'`` -- for each occlusion procedure randomly select
          a single RGB occlusion colour for all the segments; or
        * ``'randomise-patch'`` -- for each occlusion procedure randomly select
          a separate random RGB colour for each segment.

        Alternatively, it can be a user-defined RGB colour provided as a
        3-tuple with values in the 0--255 range.
        For black-and-white and grayscale images the string specifier can be
        one of:

        * ``'black'`` -- black (0);
        * ``'white'`` -- white (255);
        * ``'random'`` -- a single random 0--255 intensity for all the segments
          for grayscale images and one of 0 or 255 for black-and-white images;
        * ``'random-patch'`` -- a separate random intensity for each segment;
        * ``'randomise'`` -- for each occlusion procedure randomly select
          a single occlusion intensity for all the segments; or
        * ``'randomise-patch'`` -- for each occlusion procedure randomly select
          a separate random intensity for each segment.

        Alternatively, it can be a user-defined intensity provided as an
        integer with values in the 0--255 range for grayscale images or 0 or
        255 for black-and-white images.

        The colouring strategy is constructed as a callable object (a function)
        that takes in a boolean numpy array of the same dimensions as
        ``segments`` and returns a numpy array with the colours generated
        according to the desired colouring strategy.

        .. note::
           When ``None`` is passed as the ``colour``, the mean-colour occlusion
           is selected.

        Parameters
        ----------
        colour : string, integer, tuple(integer, integer, integer) or None
            A colour specifier.

        Raises
        ------
        RuntimeError
            The mean-colour occlusion was selected for a black-and-white image.
        TypeError
            The colour specifier is neither ``None``, a string, an integer or
            a tuple; or one of the tuple elements is not an integer.
        ValueError
            The colour string specifier refers to an unknown colouring
            strategy name.
            An colour integer specifier for a black-and-white image is neither
            0 for black nor 1 or 255 for white.
            A colour component for an RGB image or pixel intensity for a
            grayscale image is outside of the 0--255 range.
            An RGB colour specifier is not a 3-tuple.
        """
        self._colouring_strategy = self._generate_colouring_strategy(colour)

    def _randomise_patch(self, mask: np.ndarray) -> np.ndarray:
        """
        Generates a random colour for each segment selected to be occluded
        by the ``mask``.

        Parameters
        ----------
        mask : numpy.ndarray
            A boolean numpy array of the same shape as ``segments``, indicating
            the pixels (``True``) for which a random colour patch should be
            generated.

        Returns
        -------
        randomise_patch : numpy.ndarray
            A numpy array of (number of pixels to be occluded X number of
            colour channels) dimensions holding random colour patches for the
            segments selected to be occluded.
        """
        assert fuav.is_2d_array(mask), 'Mask must 2-D numpy array.'
        assert mask.shape == self.segments.shape, 'Mask must be segments-like.'
        assert mask.dtype.kind == 'b', 'Mask must be binary.'

        randomise_patch = self.image.copy()
        unique_segments = np.unique(self.segments[mask])
        for id_ in unique_segments:
            segment_mask = (self.segments == id_)
            if self.is_rgb:
                segment_colour = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)
                )  # type: SegmentColour # yapf: disable
            else:
                if self.is_bnw:
                    segment_colour = random.choice([0, 255])
                else:
                    segment_colour = random.randint(0, 255)
            randomise_patch[segment_mask] = segment_colour
        randomise_patch = randomise_patch[mask]
        return randomise_patch

    def _generate_colouring_strategy(self,
                                     colour: Union[None, str, int, RGBcolour]
                                     ) -> Callable[[np.ndarray], np.ndarray]:
        """
        Builds a callable form a specifier of the colouring strategy.

        See the documentation of the
        :func:`fatf.utils.data.occlusion.Occlusion.set_colouring_strategy`
        method for more details.
        """
        # pylint: disable=too-many-branches,too-many-statements
        colouring_strategy = None  # type: Union[None, Callable]
        if self.is_rgb:
            _colouring_strategies = {
                'mean': None,
                'black': (0, 0, 0),
                'white': (255, 255, 255),
                'red': (255, 0, 0),
                'green': (0, 255, 0),
                'blue': (0, 0, 255),
                'pink': (255, 192, 203),
                'random': (random.randint(0, 255),
                           random.randint(0, 255),
                           random.randint(0, 255)),
                'random-patch': None,
                'randomise': None,
                'randomise-patch': None
            }  # type: Dict[str, Union[None, SegmentColour]] # yapf: disable
        else:
            if self.is_bnw:
                _rnd = random.choice([0, 255])
            else:
                _rnd = random.randint(0, 255)
            _colouring_strategies = {
                'mean': None,
                'black': 0,
                'white': 255,
                'random': _rnd,
                'random-patch': None,
                'randomise': None,
                'randomise-patch': None
            }  # yapf: disable

        if isinstance(colour, tuple) and self.is_rgb:
            assert fuds._validate_colour(  # pylint: disable=protected-access
                colour), 'Invalid colour.'
            colouring_strategy = lambda _: colour  # noqa: E731
        elif isinstance(colour, int) and not self.is_rgb:
            if self.is_bnw:
                if colour in (1, 255):
                    colour = 255
                elif colour != 0:
                    raise ValueError(
                        'The colour should be 0 for black, or 1 or 255 for '
                        'white for black-and-white images.')
            else:
                if colour < 0 or colour > 255:
                    raise ValueError('The colour should be an integer between '
                                     '0 and 255 for grayscale images.')
            colouring_strategy = lambda _: colour  # noqa: E731
        elif isinstance(colour, str) or colour is None:
            if colour is None or colour == 'mean':
                mosaic = self.image.copy()
                for id_ in self.unique_segments:
                    segment_mask = (self.segments == id_)
                    if self.is_rgb:
                        segment_colour = (self.image[segment_mask, 0].mean(),
                                          self.image[segment_mask, 1].mean(),
                                          self.image[segment_mask, 2].mean()
                                          )  # type: SegmentColour
                    else:
                        if self.is_bnw:
                            raise RuntimeError(
                                'Mean occlusion is not supported for '
                                'black-and-white images.')
                        segment_colour = self.image[segment_mask].mean()
                    mosaic[segment_mask] = segment_colour
                mosaic = mosaic.astype(np.uint8)
                colouring_strategy = lambda mask: mosaic[mask]  # noqa: E731
            elif colour == 'random-patch':
                random_patch = self.image.copy()
                for id_ in self.unique_segments:
                    segment_mask = (self.segments == id_)
                    if self.is_rgb:
                        segment_colour = (random.randint(0, 255),
                                          random.randint(0, 255),
                                          random.randint(0, 255))
                    else:
                        if self.is_bnw:
                            segment_colour = random.choice([0, 255])
                        else:
                            segment_colour = random.randint(0, 255)
                    random_patch[segment_mask] = segment_colour
                colouring_strategy = lambda mask: (  # noqa: E731
                    random_patch[mask])
            elif colour == 'randomise':
                if self.is_rgb:
                    colouring_strategy = lambda _: (  # noqa: E731
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255))  # yapf: disable
                else:
                    if self.is_bnw:
                        colouring_strategy = lambda _: (  # noqa: E731
                            random.choice([0, 255]))
                    else:
                        colouring_strategy = lambda _: (  # noqa: E731
                            random.randint(0, 255))
            elif colour == 'randomise-patch':
                colouring_strategy = self._randomise_patch
            elif colour in _colouring_strategies:
                colouring_strategy = lambda _: (  # noqa: E731
                    _colouring_strategies[colour])  # type: ignore
            else:
                raise ValueError((
                    'Unknown colouring strategy name: {}.\n'
                    'Choose one of the following: {}.').format(
                        colour, sorted(list(
                            _colouring_strategies.keys()))))  # yapf: disable
        else:
            raise TypeError('The colour can either be a string specifier; or '
                            'an RGB thriplet for RGB images and an integer '
                            'for or grayscale and black-and-white images.')

        return colouring_strategy  # type: ignore

    def occlude_segments(
            self,
            segments_subset: Union[int, List[int]],
            image: Optional[np.ndarray] = None,
            colour: Optional[Union[str, int, RGBcolour]] = None) -> np.ndarray:
        """
        Generates the image with a specified subset of segments occluded.

        The occlusion is applied on top of the image used to initialise this
        class; alternatively, an external ``image`` of the same type and
        dimensions can be supplied.
        If a colouring strategy different to the one of the class is desired,
        it can be specified via the ``colour`` parameter.

        Parameters
        ----------
        segments_subset : intiger or list(integer)
            An id of a specific segment or a list of segment ids to be
            occluded.
        image : numpy.ndarray, optional (default=None)
            If provided, this ``image`` will be occluded instead of the one
            used to initialise this class.
        colour : string, integer, tuple(integer, integer, integer), \
optional (default=None)
            A colour specifier.
            By default (colour=None) the colouring strategy of the class is
            used.
            See the documentation of the
            :func:`fatf.utils.data.occlusion.Occlusion.set_colouring_strategy`
            method for more details.

        Raises
        ------
        IncorrectShapeError
            The input ``image`` is neither a 2- nor 3-dimensional numpy array.
            The the height, width or the number of channels in the ``image``
            array does not agree with the same parameters of the class image.
        TypeError
            The ``segments_subset`` parameter is neither an integer nor a list
            of integers; one of the segment ids in this list is not an integer.
        ValueError
            One of the segment ids provided via ``segments_subset`` is invalid
            for the class segmentation or some of its elements are duplicated.

        Returns
        -------
        image_occluded : numpy.ndarray
            A numpy array holding the image with the selected subset of
            segments occluded.
        """
        # pylint: disable=too-many-branches
        if isinstance(segments_subset, int):
            if segments_subset not in self.unique_segments:
                raise ValueError(
                    ('The segment id {} does not correspond to any of '
                     'the known segments ({}).').format(
                         segments_subset, self.unique_segments.tolist()))
            segments_subset = [segments_subset]
        elif isinstance(segments_subset, list):
            if len(segments_subset) != len(set(segments_subset)):
                raise ValueError('The list of segments has duplicates.')
            for i in segments_subset:
                if not isinstance(i, int):
                    raise TypeError(
                        'The segment id {} is not an integer.'.format(i))
                if i not in self.unique_segments:
                    raise ValueError(
                        ('The segment id {} does not correspond to any of '
                         'the known segments ({}).').format(
                             i, self.unique_segments.tolist()))
        else:
            raise TypeError('Segments subset must be either '
                            'an integer or a list of integers.')

        if image is None:
            canvas = self.image
        else:
            assert (
                fuds._validate_image_array(  # pylint: disable=protected-access
                    image, 'image')), 'Invalid image.'  # yapf: disable
            if image.shape != self.image.shape:
                raise IncorrectShapeError(
                    'The width, height or number of channels of the input '
                    'image does not agree with the same parameters of the '
                    'original image.')
            canvas = image

        if colour is None:
            colouring_strategy = self._colouring_strategy
        else:
            colouring_strategy = self._generate_colouring_strategy(colour)

        occlusion_mask = fuds.get_segment_mask(segments_subset, self.segments)
        image_occluded = canvas.copy()
        image_occluded[occlusion_mask] = colouring_strategy(occlusion_mask)
        return image_occluded

    def occlude_segments_vectorised(
            self,
            vectorised_segments_subset: np.ndarray,
            image: Optional[np.ndarray] = None,
            colour: Optional[Union[str, int, RGBcolour]] = None) -> np.ndarray:
        """
        Generates multiple images with a selected subsets of segments occluded.

        The segments to be occluded are provided as boolean vectors;
        either a 1-D numpy array of length equal to the number of segments
        to produce a single occluded image, or a 2-D array where each row
        represents a separate occlusion pattern.
        In this format the n-th element or column corresponds to the the
        n+1 segment id;
        1 indicates that the segment should be preserved and 0 that it should
        be occluded.

        The occlusion is applied on top of the image used to initialise this
        class; alternatively, an external ``image`` of the same type and
        dimensions can be supplied.
        If a colouring strategy different to the one of the class is desired,
        it can be specified via the ``colour`` parameter.

        Parameters
        ----------
        vectorised_segments_subset : numpy.ndarray
            A 1-D boolean occlusion vector of the length equal to the number of
            segments or a 2-D boolean matrix of the (number of occlusion images
            to generate X number of segments) shape.
        image : numpy.ndarray, optional (default=None)
            If provided, this ``image`` will be occluded instead of the one
            used to initialise this class.
        colour : string, integer, tuple(integer, integer, integer), \
optional (default=None)
            A colour specifier.
            By default (``colour=None``) the colouring strategy of the class is
            used.
            See the documentation of the
            :func:`fatf.utils.data.occlusion.Occlusion.set_colouring_strategy`
            method for more details.

        Raises
        ------
        IncorrectShapeError
            The ``vectorised_segments_subset`` numpy array is neither 1- nor
            2-dimensional.
            The number of elements in ``vectorised_segments_subset`` (when it
            is 1-D) does not correspond to the number of segments.
            The number of columns in ``vectorised_segments_subset`` (when it is
            2-D) does not correspond to the number of segments.
            The input ``image`` is neither a 2- nor 3-dimensional numpy array.
            The the height, width or the number of channels in the ``image``
            array does not agree with the same parameters of the class image.
        TypeError
            The ``vectorised_segments_subset`` numpy array is not boolean.

        Returns
        -------
        image_occluded : numpy.ndarray
            A numpy array holding the image(s) with the selected subset(s) of
            segments occluded.
        """
        # pylint: disable=too-many-branches
        if image is None:
            canvas = self.image
        else:
            assert (  # yapf: disable
                fuds._validate_image_array(  # pylint: disable=protected-access
                    image, 'image')), 'Invalid image.'
            if image.shape != self.image.shape:
                raise IncorrectShapeError(
                    'The width, height or number of channels of the input '
                    'image does not agree with the same parameters of the '
                    'original image.')
            canvas = image

        if colour is None:
            colouring_strategy = self._colouring_strategy
        else:
            colouring_strategy = self._generate_colouring_strategy(colour)

        if fuav.is_structured_array(vectorised_segments_subset):
            raise TypeError('The vector representation of segments cannot be '
                            'a structured numpy array.')
        if not fuav.is_numerical_array(vectorised_segments_subset):
            raise TypeError('The vector representation of segments should be '
                            'a numerical numpy array.')
        if fuav.is_1d_array(vectorised_segments_subset):
            if vectorised_segments_subset.shape[0] != self.segments_number:
                raise IncorrectShapeError(
                    ('The number of elements ({}) in the vector '
                     'representation of segments should correspond to the '
                     'unique number of segments ({}).').format(
                         vectorised_segments_subset.shape[0],
                         self.segments_number))
            samples = 1
            vectorised_segments_subset = np.asarray(
                [vectorised_segments_subset])
        elif fuav.is_2d_array(vectorised_segments_subset):
            if vectorised_segments_subset.shape[1] != self.segments_number:
                raise IncorrectShapeError(
                    ('The number of columns ({}) in the vector representation '
                     'of segments should correspond to the unique number of '
                     'segments ({}).').format(
                         vectorised_segments_subset.shape[1],
                         self.segments_number))
            samples = vectorised_segments_subset.shape[0]
        else:
            raise IncorrectShapeError(
                'The vector representation of segments should be a 1- or '
                '2-dimensional numpy array.')
        _unique_entries = set(np.unique(vectorised_segments_subset).astype(
            int)).difference((0, 1))  # yapf: disable
        if _unique_entries:
            raise TypeError('The vector representation of segments should be '
                            'binary numpy array.')

        # image_occluded = canvas.copy()
        image_occluded = np.repeat(canvas[np.newaxis, :], samples, axis=0)
        for i, vec in enumerate(vectorised_segments_subset):
            # Get ids of segments to be occluded (0s) from a vector form
            # 1 is added as segments are numbered from 1, not 0
            segments_subset = np.where(vec == 0)[0] + 1
            occlusion_mask = fuds.get_segment_mask(segments_subset.tolist(),
                                                   self.segments)
            image_occluded[i, occlusion_mask] = colouring_strategy(
                occlusion_mask)
        if samples == 1:
            image_occluded = image_occluded[0]

        return image_occluded
