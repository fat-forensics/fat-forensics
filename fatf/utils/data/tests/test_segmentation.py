"""
Tets the :mod:`fatf.utils.data.segmentation` module.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import importlib
import pytest
import sys

import numpy as np

from fatf.exceptions import IncorrectShapeError

import fatf

import fatf.utils.data.segmentation as fuds
import fatf.utils.testing.imports as futi

try:
    import skimage
    import PIL
except ImportError:  # pragma: no cover
    pytest.skip(
        'Skipping tests of image segmentation -- scikit-image or Pillow is '
        'not installed.',
        allow_module_level=True)
else:
    del skimage
    del PIL

ARRAY_1D = np.array([0, 1])
ARRAY_2D_NUM = np.array([[-.1, 1], [1, 2]])
ARRAY_2D_STR = np.array([[-.1, '1'], [1, 2]])
ARRAY_IMAGE_2D = np.array([[0, 255], [1, 2]])
ARRAY_IMAGE_3D = np.ones(shape=(2, 2, 3), dtype=np.uint8)
ARRAY_STRUCT = np.array([(-.1, 1)], dtype=[('a', np.float64), ('b', np.int8)])


def test_imports():
    """Tests importing the :mod:`fatf.utils.data.segmentation` module."""
    # Present
    # scikit-image
    assert 'fatf.utils.data.segmentation' in sys.modules
    with futi.module_import_tester('skimage', when_missing=False):
        importlib.reload(fatf.utils.data.segmentation)
    # Pillow
    assert 'fatf.utils.data.segmentation' in sys.modules
    with futi.module_import_tester('PIL', when_missing=False):
        importlib.reload(fatf.utils.data.segmentation)
    assert 'fatf.utils.data.segmentation' in sys.modules

    # Missing scikit-image
    import_msg = (
        'scikit-image Python package is not installed on your system. '
        'You must install it in order to use the fatf.utils.data.segmentation '
        'functionality. '
        'One possibility is to install scikit-image alongside this package '
        'via auxiliary dependencies with: pip install fat-forensics[all].')
    with futi.module_import_tester('skimage', when_missing=True):
        with pytest.raises(ImportError) as exin:
            importlib.reload(fatf.utils.data.segmentation)
        assert str(exin.value) == import_msg
    assert 'fatf.utils.data.segmentation' in sys.modules

    # Missing PIL
    import_msg = (
        'PIL Python package is not installed on your system. '
        'You must install it in order to use the fatf.utils.data.segmentation '
        'functionality. '
        'One possibility is to install PIL alongside this package via '
        'auxiliary dependencies with: pip install fat-forensics[all].')
    with futi.module_import_tester('PIL', when_missing=True):
        with pytest.raises(ImportError) as exin:
            importlib.reload(fatf.utils.data.segmentation)
        assert str(exin.value) == import_msg
    assert 'fatf.utils.data.segmentation' in sys.modules


def test_validate_image_array():
    """Tests :func:`fatf.utils.data.segmentation._validate_image_array`."""
    err_msg = 'The input xyz must not be a structured numpy array.'
    with pytest.raises(TypeError) as exin:
        fuds._validate_image_array(ARRAY_STRUCT, 'xyz')
    assert str(exin.value) == err_msg

    err_msg = 'The input xyz must be of a numerical type.'
    with pytest.raises(TypeError) as exin:
        fuds._validate_image_array(ARRAY_2D_STR, 'xyz')
    assert str(exin.value) == err_msg

    err_msg = ('The numpy representation of the input xyz should have its '
               'values (integers) between the 0--255 range.')
    with pytest.raises(ValueError) as exin:
        fuds._validate_image_array(ARRAY_2D_NUM, 'xyz')
    assert str(exin.value) == err_msg

    err_msg = 'The input xyz must be a 2- or 3-dimensional numpy array.'
    with pytest.raises(IncorrectShapeError) as exin:
        fuds._validate_image_array(ARRAY_1D, 'xyz')
    assert str(exin.value) == err_msg

    assert fuds._validate_image_array(ARRAY_IMAGE_2D, 'xyz')
    assert fuds._validate_image_array(ARRAY_IMAGE_3D, 'xyz')


def test_validate_input():
    """Tests :func:`fatf.utils.data.segmentation._validate_input`."""
    err_msg = ('The width and height of the input image and the segmentation '
               'mask do not agree.')
    with pytest.raises(IncorrectShapeError) as exin:
        fuds._validate_input(ARRAY_IMAGE_2D, np.array([[1, 2, 3], [1, 2, 3]]))
    assert str(exin.value) == err_msg

    assert fuds._validate_input(ARRAY_IMAGE_2D, ARRAY_IMAGE_2D)
    assert fuds._validate_input(ARRAY_IMAGE_3D, ARRAY_IMAGE_3D)
    assert fuds._validate_input(ARRAY_IMAGE_3D, ARRAY_IMAGE_2D)
    assert fuds._validate_input(ARRAY_IMAGE_2D, ARRAY_IMAGE_3D)


def test_validate_segmentation():
    """Tests :func:`fatf.utils.data.segmentation._validate_segmentation`."""
    err_msg = 'The segmentation array must not be a structured numpy array.'
    with pytest.raises(TypeError) as exin:
        fuds._validate_segmentation(ARRAY_STRUCT, ARRAY_IMAGE_2D)
    assert str(exin.value) == err_msg

    err_msg = 'The segmentation array must be a 2-dimensional numpy array.'
    with pytest.raises(IncorrectShapeError) as exin:
        fuds._validate_segmentation(ARRAY_1D, ARRAY_IMAGE_2D)
    assert str(exin.value) == err_msg

    err_msg = 'The segmentation array must be of integer type.'
    with pytest.raises(TypeError) as exin:
        fuds._validate_segmentation(ARRAY_2D_STR, ARRAY_IMAGE_2D)
    assert str(exin.value) == err_msg
    with pytest.raises(TypeError) as exin:
        fuds._validate_segmentation(ARRAY_2D_NUM, ARRAY_IMAGE_2D)
    assert str(exin.value) == err_msg

    err_msg = ('The segmentation array should encode unique segments with a '
               'continuous sequence of integers starting at 1.')
    with pytest.raises(ValueError) as exin:
        fuds._validate_segmentation(ARRAY_IMAGE_2D, ARRAY_IMAGE_2D)
    assert str(exin.value) == err_msg
    with pytest.raises(ValueError) as exin:
        fuds._validate_segmentation(np.array([[0, 1], [1, 2]]), ARRAY_IMAGE_2D)
    assert str(exin.value) == err_msg
    with pytest.raises(ValueError) as exin:
        fuds._validate_segmentation(np.array([[0, 0], [0, 0]]), ARRAY_IMAGE_2D)
    assert str(exin.value) == err_msg
    with pytest.raises(ValueError) as exin:
        fuds._validate_segmentation(np.array([[1, 1], [2, 4]]), ARRAY_IMAGE_2D)
    assert str(exin.value) == err_msg

    err_msg = ('The width and height of the segmentation array and the input '
               'image do not agree.')
    with pytest.raises(IncorrectShapeError) as exin:
        fuds._validate_segmentation(
            np.array([[1, 2], [1, 2], [1, 2]]), ARRAY_IMAGE_2D)
    assert str(exin.value) == err_msg

    assert fuds._validate_segmentation(
        np.array([[1, 1], [1, 1]]), ARRAY_IMAGE_2D)
    assert fuds._validate_segmentation(
        np.array([[1, 2], [3, 4]]), ARRAY_IMAGE_2D)


def test_validate_colour():
    """Tests :func:`fatf.utils.data.segmentation._validate_colour`."""
    fuds._validate_colour(None)

    err_msg = ('The colour must either be None or a thriplet representing an '
               'RGB colour.')
    with pytest.raises(TypeError) as exin:
        fuds._validate_colour('tuple')
    assert str(exin.value) == err_msg

    err_msg = 'The colour tuple must be a triplet.'
    with pytest.raises(ValueError) as exin:
        fuds._validate_colour((1, ))
    assert str(exin.value) == err_msg
    with pytest.raises(ValueError) as exin:
        fuds._validate_colour((1, 2))
    assert str(exin.value) == err_msg

    err_msg = 'Each element of the colour tuple must be an integer.'
    with pytest.raises(TypeError) as exin:
        fuds._validate_colour((1, 2, 'three'))
    assert str(exin.value) == err_msg
    with pytest.raises(TypeError) as exin:
        fuds._validate_colour((1, 2, 3.0))
    assert str(exin.value) == err_msg

    err_msg = 'Each RGB value must be between 0 and 255.'
    with pytest.raises(ValueError) as exin:
        fuds._validate_colour((0, 2, 256))
    assert str(exin.value) == err_msg
    with pytest.raises(ValueError) as exin:
        fuds._validate_colour((-1, 2, 255))
    assert str(exin.value) == err_msg


class TestSegmentation(object):
    """
    Tests the :class:`fatf.utils.data.segmentation.Segmentation`
    abstract class.
    """

    class BrokenSegmentation(fuds.Segmentation):
        """A broken image segmentation implementation."""

    class BaseSegmentation(fuds.Segmentation):
        """A dummy image segmentation implementation."""

        def _segment(self):
            """Dummy segment method."""
            return np.ones(shape=self.image.shape[:2], dtype=np.uint8)

    class BinarySegmentation(fuds.Segmentation):
        """A dummy image segmentation implementation."""

        def _segment(self):
            """Dummy segment method."""
            segments = np.ones(shape=self.image.shape[:2], dtype=np.uint8)
            segments[0, 0] = 2
            return segments

    def test_segmentation_class_init(self, caplog):
        """
        Tests :class:`fatf.utils.data.segmentation.Segmentation` class init.
        """
        log_img_1 = 'Assuming a black-and-white image.'
        log_img_2 = 'Rescale 0/1 black-and-white image to 0/255.'
        rntm_img = ('Black-and-white images must use 0 as black and 1 or 255 '
                    'as white.')
        log_mask_1 = 'Assuming a black-and-white segmentation mask.'
        log_mask_2 = 'Rescale 0/1 black-and-white segmentation mask to 0/255.'
        rntm_mask = ('Black-and-white segmentation masks must use 0 as black '
                     'and 1 or 255 as white.')
        assert len(caplog.records) == 0

        abstract_method_error = ("Can't instantiate abstract class "
                                 '{} with abstract methods _segment')
        with pytest.raises(TypeError) as exin:
            self.BrokenSegmentation(ARRAY_IMAGE_2D)
        assert str(
            exin.value) == abstract_method_error.format('BrokenSegmentation')

        # Warning
        wrn_msg = ('The segmentation returned only **one** segment. '
                   'Consider tweaking the parameters to generate a reasonable '
                   'segmentation.')
        with pytest.warns(UserWarning) as warning:
            segmenter = self.BaseSegmentation(
                ARRAY_IMAGE_3D, ARRAY_IMAGE_2D, kw1=1, kw2='2')
        assert len(warning) == 1
        assert str(warning[0].message) == wrn_msg
        with pytest.warns(UserWarning) as warning:
            segmenter_ = self.BaseSegmentation(ARRAY_IMAGE_2D, x=4)
        assert len(warning) == 1
        assert str(warning[0].message) == wrn_msg

        # Attributes
        segments = np.ones(shape=ARRAY_IMAGE_3D.shape[:2], dtype=np.uint8)

        # With a segmentation mask
        assert np.array_equal(segmenter.image, ARRAY_IMAGE_3D)
        assert np.array_equal(segmenter.segmentation_mask, ARRAY_IMAGE_2D)
        assert segmenter.is_rgb
        assert np.array_equal(sorted(list(segmenter.kwargs)), ['kw1', 'kw2'])
        assert segmenter.kwargs['kw1'] == 1 and segmenter.kwargs['kw2'] == '2'
        assert np.array_equal(segmenter._segments, segments)
        assert segmenter.segments_number == 1

        # Without a segmentation mask
        assert np.array_equal(segmenter_.image, ARRAY_IMAGE_2D)
        assert np.array_equal(segmenter_.segmentation_mask, ARRAY_IMAGE_2D)
        assert not segmenter_.is_rgb
        assert np.array_equal(sorted(list(segmenter_.kwargs)), ['x'])
        assert segmenter_.kwargs['x'] == 4
        assert np.array_equal(segmenter_._segments, segments)
        assert segmenter_.segments_number == 1

        # Logging -- image
        img_black = np.zeros(shape=(2, 2), dtype=np.uint8)
        img_white_1 = np.ones(shape=(2, 2), dtype=np.uint8)
        img_white_255 = np.full((2, 2), 255, dtype=np.uint8)
        img_bnw_1 = np.ones(shape=(2, 2), dtype=np.uint8)
        img_bnw_1[0, 0] = 0
        img_bnw_1[1, 1] = 0
        img_bnw_255 = np.full((2, 2), 255, dtype=np.uint8)
        img_bnw_255[0, 0] = 0
        img_bnw_255[1, 1] = 0

        assert len(caplog.records) == 0
        _ = self.BinarySegmentation(img_black, ARRAY_IMAGE_2D)
        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == 'INFO'
        assert caplog.records[0].getMessage() == log_img_1

        assert len(caplog.records) == 1
        segmenter = self.BinarySegmentation(img_white_1, ARRAY_IMAGE_2D)
        assert np.array_equal(segmenter.image, img_white_255)
        assert len(caplog.records) == 3
        assert caplog.records[1].levelname == 'INFO'
        assert caplog.records[1].getMessage() == log_img_1
        assert caplog.records[2].levelname == 'INFO'
        assert caplog.records[2].getMessage() == log_img_2

        assert len(caplog.records) == 3
        segmenter = self.BinarySegmentation(img_white_255, ARRAY_IMAGE_2D)
        assert np.array_equal(segmenter.image, img_white_255)
        assert len(caplog.records) == 4
        assert caplog.records[3].levelname == 'INFO'
        assert caplog.records[3].getMessage() == log_img_1

        assert len(caplog.records) == 4
        segmenter = self.BinarySegmentation(img_bnw_1, ARRAY_IMAGE_2D)
        assert np.array_equal(segmenter.image, np.array([[0, 255], [255, 0]]))
        assert len(caplog.records) == 6
        assert caplog.records[4].levelname == 'INFO'
        assert caplog.records[4].getMessage() == log_img_1
        assert caplog.records[5].levelname == 'INFO'
        assert caplog.records[5].getMessage() == log_img_2

        assert len(caplog.records) == 6
        segmenter = self.BinarySegmentation(img_bnw_255, ARRAY_IMAGE_2D)
        assert np.array_equal(segmenter.image, img_bnw_255)
        assert len(caplog.records) == 7
        assert caplog.records[6].levelname == 'INFO'
        assert caplog.records[6].getMessage() == log_img_1

        # Error -- image
        img_bnw_0_42 = np.zeros(shape=(2, 2), dtype=np.uint8)
        img_bnw_0_42[0, 0] = 42
        img_bnw_0_42[1, 1] = 42
        img_bnw_1_42 = np.ones(shape=(2, 2), dtype=np.uint8)
        img_bnw_1_42[0, 0] = 42
        img_bnw_1_42[1, 1] = 42

        assert len(caplog.records) == 7
        with pytest.raises(RuntimeError) as exin:
            self.BinarySegmentation(img_bnw_0_42, ARRAY_IMAGE_2D)
        assert str(exin.value) == rntm_img
        assert len(caplog.records) == 8
        assert caplog.records[7].levelname == 'INFO'
        assert caplog.records[7].getMessage() == log_img_1

        assert len(caplog.records) == 8
        with pytest.raises(RuntimeError) as exin:
            self.BinarySegmentation(img_bnw_1_42, ARRAY_IMAGE_2D)
        assert str(exin.value) == rntm_img
        assert len(caplog.records) == 10
        assert caplog.records[8].levelname == 'INFO'
        assert caplog.records[8].getMessage() == log_img_1
        assert caplog.records[9].levelname == 'INFO'
        assert caplog.records[9].getMessage() == log_img_2

        # Logging -- segmentation mask
        assert len(caplog.records) == 10
        _ = self.BinarySegmentation(ARRAY_IMAGE_2D, img_black)
        assert len(caplog.records) == 11
        assert caplog.records[10].levelname == 'INFO'
        assert caplog.records[10].getMessage() == log_mask_1

        assert len(caplog.records) == 11
        segmenter = self.BinarySegmentation(ARRAY_IMAGE_2D, img_white_1)
        assert np.array_equal(segmenter.segmentation_mask, img_white_255)
        assert len(caplog.records) == 13
        assert caplog.records[11].levelname == 'INFO'
        assert caplog.records[11].getMessage() == log_mask_1
        assert caplog.records[12].levelname == 'INFO'
        assert caplog.records[12].getMessage() == log_mask_2

        assert len(caplog.records) == 13
        segmenter = self.BinarySegmentation(ARRAY_IMAGE_2D, img_white_255)
        assert np.array_equal(segmenter.segmentation_mask, img_white_255)
        assert len(caplog.records) == 14
        assert caplog.records[13].levelname == 'INFO'
        assert caplog.records[13].getMessage() == log_mask_1

        assert len(caplog.records) == 14
        segmenter = self.BinarySegmentation(ARRAY_IMAGE_2D, img_bnw_1)
        assert np.array_equal(segmenter.segmentation_mask,
                              np.array([[0, 255], [255, 0]]))
        assert len(caplog.records) == 16
        assert caplog.records[14].levelname == 'INFO'
        assert caplog.records[14].getMessage() == log_mask_1
        assert caplog.records[15].levelname == 'INFO'
        assert caplog.records[15].getMessage() == log_mask_2

        assert len(caplog.records) == 16
        segmenter = self.BinarySegmentation(ARRAY_IMAGE_2D, img_bnw_255)
        assert np.array_equal(segmenter.segmentation_mask, img_bnw_255)
        assert len(caplog.records) == 17
        assert caplog.records[16].levelname == 'INFO'
        assert caplog.records[16].getMessage() == log_mask_1

        # Error -- segmentation mask
        assert len(caplog.records) == 17
        with pytest.raises(RuntimeError) as exin:
            self.BinarySegmentation(ARRAY_IMAGE_2D, img_bnw_0_42)
        assert str(exin.value) == rntm_mask
        assert len(caplog.records) == 18
        assert caplog.records[17].levelname == 'INFO'
        assert caplog.records[17].getMessage() == log_mask_1

        assert len(caplog.records) == 18
        with pytest.raises(RuntimeError) as exin:
            self.BinarySegmentation(ARRAY_IMAGE_2D, img_bnw_1_42)
        assert str(exin.value) == rntm_mask
        assert len(caplog.records) == 20
        assert caplog.records[18].levelname == 'INFO'
        assert caplog.records[18].getMessage() == log_mask_1
        assert caplog.records[19].levelname == 'INFO'
        assert caplog.records[19].getMessage() == log_mask_2

    def test_set_segments(self):
        """
        Tests :class:`fatf.utils.data.segmentation.Segmentation` segment setup.
        """
        segments = np.ones(shape=ARRAY_IMAGE_3D.shape[:2], dtype=np.uint8)
        segments[0, 0] = 2
        segments_ = segments.copy()
        segments_[0, 1] = 2
        segments_one = segments.copy()
        segments_one[0, 0] = 1

        segmenter = self.BinarySegmentation(ARRAY_IMAGE_3D)

        assert np.array_equal(segmenter._segments, segments)
        assert np.array_equal(segmenter.segments, segments)
        segmenter.segments = segments_
        assert np.array_equal(segmenter._segments, segments_)
        assert np.array_equal(segmenter.segments, segments_)

        segmenter.segments = segments
        assert np.array_equal(segmenter._segments, segments)
        assert np.array_equal(segmenter.segments, segments)
        segmenter.set_segments(segments_)
        assert np.array_equal(segmenter._segments, segments_)
        assert np.array_equal(segmenter.segments, segments_)

        segmenter.segments = segments
        assert np.array_equal(segmenter._segments, segments)
        assert np.array_equal(segmenter.segments, segments)
        segmenter._segments = segments_
        assert np.array_equal(segmenter._segments, segments_)
        assert np.array_equal(segmenter.segments, segments_)

        segmenter.segments = segments
        assert np.array_equal(segmenter._segments, segments)
        assert np.array_equal(segmenter.segments, segments)

        wrn_msg = 'The segmentation has only **one** segment.'
        with pytest.warns(UserWarning) as warning:
            segmenter.segments = segments_one
        assert np.array_equal(segmenter._segments, segments_one)
        assert np.array_equal(segmenter.segments, segments_one)
        assert len(warning) == 1
        assert str(warning[0].message) == wrn_msg

        segmenter.segments = segments
        assert np.array_equal(segmenter._segments, segments)
        assert np.array_equal(segmenter.segments, segments)
        with pytest.warns(UserWarning) as warning:
            segmenter.set_segments(segments_one)
        assert np.array_equal(segmenter._segments, segments_one)
        assert np.array_equal(segmenter.segments, segments_one)
        assert len(warning) == 1
        assert str(warning[0].message) == wrn_msg

        segmenter.segments = segments
        assert np.array_equal(segmenter._segments, segments)
        assert np.array_equal(segmenter.segments, segments)
        segmenter._segments = segments_one
        assert np.array_equal(segmenter._segments, segments_one)
        assert np.array_equal(segmenter.segments, segments_one)

    def test_mark_boundaries(self):
        """
        Tests
        :func:`fatf.utils.data.segmentation.Segmentation.mark_boundaries`.
        """
        segmenter = self.BinarySegmentation(ARRAY_IMAGE_3D)

        msg = 'The mask parameter must be a boolean.'
        with pytest.raises(TypeError) as exin:
            segmenter.mark_boundaries('bool')
        assert str(exin.value) == msg

        msg = ('The width and height of the input image do not agree '
               'with the dimensions of the original image.')
        with pytest.raises(IncorrectShapeError) as exin:
            segmenter.mark_boundaries(image=np.array([[1, 2, 3], [4, 5, 6]]))
        assert str(exin.value) == msg

        # Default params
        boundaries_ = np.array(
            [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [1, 1, 1]]], dtype=np.uint8)
        boundaries = segmenter.mark_boundaries()
        assert np.array_equal(boundaries, boundaries_)
        # Mask
        boundaries = segmenter.mark_boundaries(mask=True)
        assert np.array_equal(boundaries, boundaries_)
        # Custom image
        boundaries = segmenter.mark_boundaries(image=ARRAY_IMAGE_3D)
        assert np.array_equal(boundaries, boundaries_)

        # Custom colour
        boundaries_ = np.array(
            [[[255, 0, 0], [255, 0, 0]], [[255, 0, 0], [1, 1, 1]]],
            dtype=np.uint8)
        boundaries = segmenter.mark_boundaries(colour=(255, 0, 0))
        assert np.array_equal(boundaries, boundaries_)

    def test_number_segments(self):
        """
        Tests
        :func:`fatf.utils.data.segmentation.Segmentation.number_segments`.
        """
        segmenter = self.BinarySegmentation(ARRAY_IMAGE_3D)

        msg = ('Segments subset must be either of None, '
               'an integer or a list of integers.')
        with pytest.raises(TypeError) as exin:
            segmenter.number_segments(segments_subset='list')
        assert str(exin.value) == msg

        msg = ('The segment id 0 does not correspond to any of '
               'the known segments ([1, 2]).')
        with pytest.raises(ValueError) as exin:
            segmenter.number_segments(segments_subset=0)
        assert str(exin.value) == msg

        msg = 'The list of segments cannot be empty.'
        with pytest.raises(ValueError) as exin:
            segmenter.number_segments(segments_subset=[])
        assert str(exin.value) == msg

        msg = 'The list of segments has duplicates.'
        with pytest.raises(ValueError) as exin:
            segmenter.number_segments(segments_subset=[1, 2, 1])
        assert str(exin.value) == msg

        msg = 'The segment id 1 is not an integer.'
        with pytest.raises(TypeError) as exin:
            segmenter.number_segments(segments_subset=[1, 2, '1'])
        assert str(exin.value) == msg

        msg = ('The segment id 4 does not correspond to any of '
               'the known segments ([1, 2]).')
        with pytest.raises(ValueError) as exin:
            segmenter.number_segments(segments_subset=[2, 4, 1])
        assert str(exin.value) == msg

        msg = 'The mask parameter must be a boolean.'
        with pytest.raises(TypeError) as exin:
            segmenter.number_segments(mask='bool')
        assert str(exin.value) == msg

        msg = ('The width and height of the input image do not agree '
               'with the dimensions of the original image.')
        with pytest.raises(IncorrectShapeError) as exin:
            segmenter.number_segments(image=np.ones(shape=(4, 4), dtype=int))
        assert str(exin.value) == msg

        # Colour image, no params -- all segments
        numbered_ = np.ones(shape=(2, 2, 3), dtype=np.uint8)
        numbered = segmenter.number_segments()
        assert np.array_equal(numbered, numbered_)

        # Colour image -- 1 segments
        numbered = segmenter.number_segments(segments_subset=2)
        assert np.array_equal(numbered, numbered_)

        # Colour image -- 1 segments, segmentation image
        numbered = segmenter.number_segments(mask=True, segments_subset=[2])
        assert np.array_equal(numbered, numbered_)

        # Grayscale image -- image takes precedence over mask
        numbered_ = np.array(
            [[[0, 0, 0], [255, 255, 255]], [[1, 1, 1], [2, 2, 2]]],
            dtype=np.uint8)
        numbered = segmenter.number_segments([2], True, ARRAY_IMAGE_2D,
                                             (255, 255, 0))
        assert np.array_equal(numbered, numbered_)

    def test_highlight_segments(self):
        """
        Tests
        :func:`fatf.utils.data.segmentation.Segmentation.highlight_segments`.
        """
        segmenter = self.BinarySegmentation(ARRAY_IMAGE_3D)

        msg = ('Segments subset must be either of None, '
               'an integer or a list of integers.')
        with pytest.raises(TypeError) as exin:
            segmenter.highlight_segments(segments_subset='list')
        assert str(exin.value) == msg

        msg = ('The segment id 0 does not correspond to any of '
               'the known segments ([1, 2]).')
        with pytest.raises(ValueError) as exin:
            segmenter.highlight_segments(segments_subset=0)
        assert str(exin.value) == msg

        msg = 'The list of segments cannot be empty.'
        with pytest.raises(ValueError) as exin:
            segmenter.highlight_segments(segments_subset=[])
        assert str(exin.value) == msg

        msg = 'The list of segments has duplicates.'
        with pytest.raises(ValueError) as exin:
            segmenter.highlight_segments(segments_subset=[1, 2, 1])
        assert str(exin.value) == msg

        msg = 'The segment id 1 is not an integer.'
        with pytest.raises(TypeError) as exin:
            segmenter.highlight_segments(segments_subset=[1, 2, '1'])
        assert str(exin.value) == msg

        msg = ('The segment id 4 does not correspond to any of '
               'the known segments ([1, 2]).')
        with pytest.raises(ValueError) as exin:
            segmenter.highlight_segments(segments_subset=[2, 4, 1])
        assert str(exin.value) == msg

        msg = 'The mask parameter must be a boolean.'
        with pytest.raises(TypeError) as exin:
            segmenter.highlight_segments(mask='bool')
        assert str(exin.value) == msg

        msg = ('The width and height of the input image do not agree '
               'with the dimensions of the original image.')
        with pytest.raises(IncorrectShapeError) as exin:
            segmenter.highlight_segments(
                image=np.ones(shape=(4, 4), dtype=int))
        assert str(exin.value) == msg

        msg = ('The colour can be either of an RGB tuple, '
               'a list of RGB tuples or None.')
        with pytest.raises(TypeError) as exin:
            segmenter.highlight_segments(colour='triplet')
        assert str(exin.value) == msg

        msg = 'The colour list cannot be empty.'
        with pytest.raises(ValueError) as exin:
            segmenter.highlight_segments(colour=[])
        assert str(exin.value) == msg

        msg = ('If colours are provided as a list, their number must match '
               'the number of segments chosen to be highlighted.')
        with pytest.raises(ValueError) as exin:
            segmenter.highlight_segments(colour=['colour'])
        assert str(exin.value) == msg

        # Colour image, no params -- all segments
        highlighted_ = np.array(
            [[[0, 0, 77], [77, 0, 0]], [[77, 0, 0], [77, 0, 0]]],
            dtype=np.uint8)
        highlighted = segmenter.highlight_segments()
        assert np.array_equal(highlighted, highlighted_)

        # Colour image -- 1 segments
        highlighted_ = np.ones(shape=(2, 2, 3), dtype=np.uint8)
        highlighted_[0, 0] = (77, 0, 0)
        highlighted = segmenter.highlight_segments(segments_subset=2)
        assert np.array_equal(highlighted, highlighted_)

        # Colour image -- 1 segments, segmentation image
        highlighted = segmenter.highlight_segments(
            mask=True, segments_subset=[2])
        assert np.array_equal(highlighted, highlighted_)

        # Grayscale image -- image takes precedence over mask
        highlighted_ = np.array(
            [[[51, 51, 0], [255, 255, 255]], [[1, 1, 1], [2, 2, 2]]],
            dtype=np.uint8)
        highlighted = segmenter.highlight_segments([2], True, ARRAY_IMAGE_2D,
                                                   (255, 255, 0))
        assert np.array_equal(highlighted, highlighted_)
        highlighted = segmenter.highlight_segments([2], True, ARRAY_IMAGE_2D,
                                                   [(255, 255, 0)])
        assert np.array_equal(highlighted, highlighted_)

    def test_stain_segments(self):
        """
        Tests
        :func:`fatf.utils.data.segmentation.Segmentation._stain_segments`.
        """
        segmenter = self.BinarySegmentation(ARRAY_IMAGE_2D)
        msg = ('Staining segments of an image can only be performed on '
               'RGB images.')
        with pytest.raises(RuntimeError) as exin:
            segmenter._stain_segments()
        assert str(exin.value) == msg

        arr_ = ARRAY_IMAGE_3D.copy()
        arr_[0, 0, 0] = 42
        segmenter = self.BinarySegmentation(arr_)

        msg = ('Segments subset must be either of None, '
               'an integer or a list of integers.')
        with pytest.raises(TypeError) as exin:
            segmenter._stain_segments(segments_subset='list')
        assert str(exin.value) == msg

        msg = ('The segment id 0 does not correspond to any of '
               'the known segments ([1, 2]).')
        with pytest.raises(ValueError) as exin:
            segmenter._stain_segments(segments_subset=0)
        assert str(exin.value) == msg

        msg = 'The list of segments cannot be empty.'
        with pytest.raises(ValueError) as exin:
            segmenter._stain_segments(segments_subset=[])
        assert str(exin.value) == msg

        msg = 'The list of segments has duplicates.'
        with pytest.raises(ValueError) as exin:
            segmenter._stain_segments(segments_subset=[1, 2, 1])
        assert str(exin.value) == msg

        msg = 'The segment id 1 is not an integer.'
        with pytest.raises(TypeError) as exin:
            segmenter._stain_segments(segments_subset=[1, 2, '1'])
        assert str(exin.value) == msg

        msg = ('The segment id 4 does not correspond to any of '
               'the known segments ([1, 2]).')
        with pytest.raises(ValueError) as exin:
            segmenter._stain_segments(segments_subset=[2, 4, 1])
        assert str(exin.value) == msg

        msg = 'The mask parameter must be a boolean.'
        with pytest.raises(TypeError) as exin:
            segmenter._stain_segments(mask='bool')
        assert str(exin.value) == msg

        msg = 'The user-provided image is not RGB.'
        with pytest.raises(IncorrectShapeError) as exin:
            segmenter._stain_segments(image=np.ones(shape=(4, 4), dtype=int))
        assert str(exin.value) == msg

        msg = ('The width and height of the input image do not agree '
               'with the dimensions of the original image.')
        with pytest.raises(IncorrectShapeError) as exin:
            segmenter._stain_segments(
                image=np.ones(shape=(4, 4, 3), dtype=int))
        assert str(exin.value) == msg

        msg = ("The colour can be either of 'r', 'g' or 'b' strings, "
               'a list thereof or None.')
        with pytest.raises(TypeError) as exin:
            segmenter._stain_segments(colour=42)
        assert str(exin.value) == msg

        msg = "One of the provided colour strings (k) is not 'r', 'g' or 'b'."
        with pytest.raises(ValueError) as exin:
            segmenter._stain_segments(colour='k')
        assert str(exin.value) == msg
        with pytest.raises(ValueError) as exin:
            segmenter._stain_segments(colour=['r', 'k'])
        assert str(exin.value) == msg

        msg = 'The colour list cannot be empty.'
        with pytest.raises(ValueError) as exin:
            segmenter._stain_segments(colour=[])
        assert str(exin.value) == msg

        msg = ('If colours are provided as a list, their number must match '
               'the number of segments chosen to be highlighted.')
        with pytest.raises(ValueError) as exin:
            segmenter._stain_segments(colour=['colour'])
        assert str(exin.value) == msg

        # Colour image, no params -- all segments
        highlighted_ = np.array(
            [[[42, 1, 42], [1, 1, 42]], [[1, 1, 42], [1, 1, 42]]],
            dtype=np.uint8)
        highlighted = segmenter._stain_segments()
        assert np.array_equal(highlighted, highlighted_)

        # Colour image -- 1 segments
        highlighted_ = np.ones(shape=(2, 2, 3), dtype=np.uint8)
        highlighted_[0, 0] = (42, 1, 42)
        highlighted = segmenter._stain_segments(segments_subset=2)
        assert np.array_equal(highlighted, highlighted_)

        # Colour image -- 1 segments, segmentation image
        highlighted = segmenter._stain_segments(mask=True, segments_subset=[2])
        assert np.array_equal(highlighted, highlighted_)

        # Colour image -- image takes precedence over mask
        highlighted_ = np.ones(shape=(2, 2, 3), dtype=np.uint8)
        highlighted_[0, 0] = (84, 84, 1)
        #
        arr__ = arr_.copy()
        arr__[0, 0, 1] = 84
        highlighted = segmenter._stain_segments([2], True, arr__, 'r')
        assert np.array_equal(highlighted, highlighted_)
        highlighted = segmenter._stain_segments([2], True, arr__, ['r'])
        assert np.array_equal(highlighted, highlighted_)

    def test_grayout_segment(self):
        """
        Tests
        :func:`fatf.utils.data.segmentation.Segmentation.grayout_segment`.
        """
        segmenter = self.BinarySegmentation(ARRAY_IMAGE_2D)
        msg = ('Graying out segments of an image can only be '
               'performed on RGB images.')
        with pytest.raises(RuntimeError) as exin:
            segmenter.grayout_segments()
        assert str(exin.value) == msg

        segmenter = self.BinarySegmentation(ARRAY_IMAGE_3D)

        msg = ('Segments subset must be either of None, '
               'an integer or a list of integers.')
        with pytest.raises(TypeError) as exin:
            segmenter.grayout_segments(segments_subset='list')
        assert str(exin.value) == msg

        msg = ('The segment id 0 does not correspond to any of '
               'the known segments ([1, 2]).')
        with pytest.raises(ValueError) as exin:
            segmenter.grayout_segments(segments_subset=0)
        assert str(exin.value) == msg

        msg = 'The list of segments cannot be empty.'
        with pytest.raises(ValueError) as exin:
            segmenter.grayout_segments(segments_subset=[])
        assert str(exin.value) == msg

        msg = 'The list of segments has duplicates.'
        with pytest.raises(ValueError) as exin:
            segmenter.grayout_segments(segments_subset=[1, 2, 1])
        assert str(exin.value) == msg

        msg = 'The segment id 1 is not an integer.'
        with pytest.raises(TypeError) as exin:
            segmenter.grayout_segments(segments_subset=[1, 2, '1'])
        assert str(exin.value) == msg

        msg = ('The segment id 4 does not correspond to any of '
               'the known segments ([1, 2]).')
        with pytest.raises(ValueError) as exin:
            segmenter.grayout_segments(segments_subset=[2, 4, 1])
        assert str(exin.value) == msg

        msg = 'The mask parameter must be a boolean.'
        with pytest.raises(TypeError) as exin:
            segmenter.grayout_segments(mask='bool')
        assert str(exin.value) == msg

        msg = 'The user-provided image is not RGB.'
        with pytest.raises(IncorrectShapeError) as exin:
            segmenter.grayout_segments(image=np.ones(shape=(4, 4), dtype=int))
        assert str(exin.value) == msg

        msg = ('The width and height of the input image do not agree '
               'with the dimensions of the original image.')
        with pytest.raises(IncorrectShapeError) as exin:
            segmenter.grayout_segments(
                image=np.ones(shape=(4, 4, 3), dtype=int))
        assert str(exin.value) == msg

        # Colour image, no params -- all segments
        highlighted_ = np.zeros(shape=(2, 2, 3), dtype=np.uint8)
        highlighted = segmenter.grayout_segments()
        assert np.array_equal(highlighted, highlighted_)

        # Colour image -- 1 segments
        highlighted_ = np.ones(shape=(2, 2, 3), dtype=np.uint8)
        highlighted_[0, 0] = (0, 0, 0)
        highlighted = segmenter.grayout_segments(segments_subset=2)
        assert np.array_equal(highlighted, highlighted_)

        # Colour image -- 1 segments, segmentation image
        highlighted = segmenter.grayout_segments(
            mask=True, segments_subset=[2])
        assert np.array_equal(highlighted, highlighted_)

        # Colour image -- image takes precedence over mask
        highlighted = segmenter.grayout_segments([2], True, ARRAY_IMAGE_3D)
        assert np.array_equal(highlighted, highlighted_)

    def test_merge_segment(self):
        """
        Tests
        :func:`fatf.utils.data.segmentation.Segmentation.merge_segment`.
        """
        segmenter = self.BinarySegmentation(ARRAY_IMAGE_3D)

        msg = 'The inplace parameter must be a boolean.'
        with pytest.raises(TypeError) as exin:
            segmenter.merge_segments([], inplace='bool')
        assert str(exin.value) == msg

        msg = 'Segments grouping must be a list.'
        with pytest.raises(TypeError) as exin:
            segmenter.merge_segments(segments_grouping='list')
        assert str(exin.value) == msg

        msg = 'The segments grouping cannot be an empty list.'
        with pytest.raises(ValueError) as exin:
            segmenter.merge_segments([])
        assert str(exin.value) == msg

        msg = ('The segments grouping must either be a list '
               'of integers or a list of lists.')
        with pytest.raises(TypeError) as exin:
            segmenter.merge_segments(['int/list'])
        assert str(exin.value) == msg

        msg = 'The segments grouping has duplicates.'
        with pytest.raises(ValueError) as exin:
            segmenter.merge_segments([42, 1, 42])
        assert str(exin.value) == msg

        msg = 'The segment id 1 is not an integer.'
        with pytest.raises(TypeError) as exin:
            segmenter.merge_segments([1, '1', 42])
        assert str(exin.value) == msg

        msg = ('The segment id 42 does not correspond to any of the known '
               'segments ([1, 2]).')
        with pytest.raises(ValueError) as exin:
            segmenter.merge_segments([1, 42, 2])
        assert str(exin.value) == msg

        msg = ('The nested elements of segments grouping are not '
               'consistent. If one is a list, all must be lists.')
        with pytest.raises(TypeError) as exin:
            segmenter.merge_segments([[1, 2], 42])
        assert str(exin.value) == msg

        msg = 'The segments grouping has duplicates.'
        with pytest.raises(ValueError) as exin:
            segmenter.merge_segments([[1], [2, 42, 2]])
        assert str(exin.value) == msg

        msg = 'The segment id 1 is not an integer.'
        with pytest.raises(TypeError) as exin:
            segmenter.merge_segments([[1], ['1', 42]])
        assert str(exin.value) == msg

        msg = ('The segment id 42 does not correspond to any of the known '
               'segments ([1, 2]).')
        with pytest.raises(ValueError) as exin:
            segmenter.merge_segments([[1], [42, 2]])
        assert str(exin.value) == msg

        msg = 'The segment id 1 is duplicated across grouping lists.'
        with pytest.raises(ValueError) as exin:
            segmenter.merge_segments([[1], [2, 1]])
        assert str(exin.value) == msg

        segments_ = np.ones(shape=ARRAY_IMAGE_2D.shape, dtype=np.uint8)
        segments_[0, 0] = 2

        # No mergin
        segments = segmenter.merge_segments([[1], [2]], inplace=False)
        assert np.array_equal(segmenter.segments, segments_)
        assert np.array_equal(segments, segments_)

        # In-place
        wrn_msg = 'The segmentation has only **one** segment.'
        with pytest.warns(UserWarning) as warning:
            segments = segmenter.merge_segments([[1, 2]])
        assert len(warning) == 1
        assert str(warning[0].message) == wrn_msg
        assert np.array_equal(segmenter.segments,
                              np.ones(shape=(2, 2), dtype=np.uint8))
        assert np.array_equal(segments, np.ones(shape=(2, 2), dtype=np.uint8))

        segmenter = self.BinarySegmentation(ARRAY_IMAGE_3D)
        with pytest.warns(UserWarning) as warning:
            segments = segmenter.merge_segments([1, 2])
        assert len(warning) == 1
        assert str(warning[0].message) == wrn_msg
        assert np.array_equal(segmenter.segments,
                              np.ones(shape=(2, 2), dtype=np.uint8))
        assert np.array_equal(segments, np.ones(shape=(2, 2), dtype=np.uint8))

        # Custom segmentation -- leave one out
        segments = segmenter.merge_segments(  # yapf: disable
            [1, 3],
            inplace=False,
            segments=np.array([[1, 2], [1, 3]]))
        assert np.array_equal(segmenter.segments,
                              np.ones(shape=(2, 2), dtype=np.uint8))
        assert np.array_equal(segments, np.array([[1, 2], [1, 1]]))


def test_slic():
    """Tests the :class:`fatf.utils.data.segmentation.Slic` class."""
    msg = 'The n_segments parameter must be an integer.'
    with pytest.raises(TypeError) as exin:
        fuds.Slic(ARRAY_IMAGE_3D, n_segments='1')
    assert str(exin.value) == msg
    with pytest.raises(TypeError) as exin:
        fuds.Slic(ARRAY_IMAGE_3D, n_segments=1.1)
    assert str(exin.value) == msg

    msg = 'The n_segments parameter must be at least 2.'
    with pytest.raises(ValueError) as exin:
        fuds.Slic(ARRAY_IMAGE_3D, n_segments=0)
    assert str(exin.value) == msg
    with pytest.raises(ValueError) as exin:
        fuds.Slic(ARRAY_IMAGE_3D, n_segments=-1)
    assert str(exin.value) == msg

    slic = fuds.Slic(ARRAY_IMAGE_3D)
    assert len(slic.kwargs) == 1
    assert 'n_segments' in slic.kwargs
    assert slic.kwargs['n_segments'] == 10
    assert np.array_equal(slic.segments,
                          np.array([[1, 2], [3, 4]], dtype=np.uint8))


def test_quickshift():
    """Tests the :class:`fatf.utils.data.segmentation.QuickShift` class."""
    msg = 'Ratio should be a number.'
    with pytest.raises(TypeError) as exin:
        fuds.QuickShift(ARRAY_IMAGE_3D, ratio='1')
    assert str(exin.value) == msg
    msg = 'Ratio must be between 0 and 1.'
    with pytest.raises(ValueError) as exin:
        fuds.QuickShift(ARRAY_IMAGE_3D, ratio=-.00001)
    assert str(exin.value) == msg
    with pytest.raises(ValueError) as exin:
        fuds.QuickShift(ARRAY_IMAGE_3D, ratio=1.00001)
    assert str(exin.value) == msg

    msg = 'Kernel size should be a number.'
    with pytest.raises(TypeError) as exin:
        fuds.QuickShift(ARRAY_IMAGE_3D, kernel_size='1')
    assert str(exin.value) == msg

    msg = 'Max dist should be a number.'
    with pytest.raises(TypeError) as exin:
        fuds.QuickShift(ARRAY_IMAGE_3D, max_dist='1')
    assert str(exin.value) == msg

    wrn_msg = ('The segmentation returned only **one** segment. '
               'Consider tweaking the parameters to generate a reasonable '
               'segmentation.')
    with pytest.warns(UserWarning) as warning:
        quickshift = fuds.QuickShift(ARRAY_IMAGE_3D)
    assert len(warning) == 1
    assert str(warning[0].message) == wrn_msg

    assert len(quickshift.kwargs) == 3
    assert 'ratio' in quickshift.kwargs
    assert 'kernel_size' in quickshift.kwargs
    assert 'max_dist' in quickshift.kwargs
    assert quickshift.kwargs['ratio'] == .2
    assert quickshift.kwargs['kernel_size'] == 4
    assert quickshift.kwargs['max_dist'] == 200
    assert np.array_equal(quickshift.segments,
                          np.ones(shape=(2, 2), dtype=np.uint8))


def test_get_segment_mask():
    """
    Tests the :func:`fatf.utils.data.segmentation.get_segment_mask` function.
    """
    msg = 'Segments subset must either be an integer or a list of integers.'
    with pytest.raises(TypeError) as exin:
        fuds.get_segment_mask('int/list', np.array([[1, 2], [2, 1]]))
    assert str(exin.value) == msg

    msg = ('The segment id 0 does not correspond to any of the known segments '
           '([1, 2]).')
    with pytest.raises(ValueError) as exin:
        fuds.get_segment_mask(0, np.array([[1, 2], [2, 1]]))
    assert str(exin.value) == msg

    msg = 'The list of segments has duplicates.'
    with pytest.raises(ValueError) as exin:
        fuds.get_segment_mask([0, -1, 0], np.array([[1, 2], [2, 1]]))
    assert str(exin.value) == msg

    msg = 'The segment id 1 is not an integer.'
    with pytest.raises(TypeError) as exin:
        fuds.get_segment_mask([1, '1', -1], np.array([[1, 2], [2, 1]]))
    assert str(exin.value) == msg

    msg = ('The segment id 42 does not correspond to any of the known '
           'segments ([1, 2]).')
    with pytest.raises(ValueError) as exin:
        fuds.get_segment_mask([1, 2, 42], np.array([[1, 2], [2, 1]]))
    assert str(exin.value) == msg

    # Test int
    mask = fuds.get_segment_mask(1, np.array([[1, 2], [2, 3]]))
    assert np.array_equal(mask, np.array([[True, False], [False, False]]))
    mask = fuds.get_segment_mask(2, np.array([[1, 2], [2, 3]]))
    assert np.array_equal(mask, np.array([[False, True], [True, False]]))

    # Test list
    mask = fuds.get_segment_mask([2], np.array([[1, 2], [2, 3]]))
    assert np.array_equal(mask, np.array([[False, True], [True, False]]))
    mask = fuds.get_segment_mask([2, 3], np.array([[1, 2], [2, 3]]))
    assert np.array_equal(mask, np.array([[False, True], [True, True]]))
    mask = fuds.get_segment_mask([3, 1, 2], np.array([[1, 2], [2, 3]]))
    assert np.array_equal(mask, np.array([[True, True], [True, True]]))
