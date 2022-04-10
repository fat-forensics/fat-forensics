"""
Tets the :mod:`fatf.utils.data.occlusion` module.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import pytest

import numpy as np

from fatf.exceptions import IncorrectShapeError

import fatf

try:
    import fatf.utils.data.segmentation
except ImportError:  # pragma: no cover
    pytest.skip(
        'Skipping tests of image occlusion -- scikit-image or Pillow is '
        'not installed (and is required for the segmentation dependency).',
        allow_module_level=True)

import fatf.utils.data.occlusion as fudo

ONES = np.ones(shape=(2, 2), dtype=int)
SEGMENTS = np.ones(shape=(2, 2), dtype=int)
SEGMENTS[1, 1] = 2

ARRAY_IMAGE_BNW1 = np.array([[0, 1], [1, 0]])
ARRAY_IMAGE_2D = np.array([[0, 255], [1, 2]])
ARRAY_IMAGE_3D = np.ones(shape=(2, 2, 3), dtype=np.uint8)
ARRAY_STRUCT = np.array([(-.1, 1)], dtype=[('a', np.float64), ('b', np.int8)])


class TestOcclusion(object):
    """Tests the :class:`fatf.utils.data.occlusion.Occlusion` class."""

    def test_occlusion_class_init(self, caplog):
        """
        Tests :class:`fatf.utils.data.occlusion.Occlusion` class init.
        """
        log_1 = 'Assuming a black-and-white image.'
        log_2 = 'Rescale 0/1 black-and-white image to 0/255.'

        assert len(caplog.records) == 0
        err = ('Black-and-white images must use 0 as '
               'black and 1 or 255 as white.')
        with pytest.raises(RuntimeError) as exin:
            fudo.Occlusion(
                np.array([[2, 255], [255, 2]], dtype=int),
                np.ones(shape=(2, 2), dtype=int))
        assert str(exin.value) == err
        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == 'INFO'
        assert caplog.records[0].getMessage() == log_1

        assert len(caplog.records) == 1
        with pytest.raises(RuntimeError) as exin:
            fudo.Occlusion(
                np.array([[2, 1], [1, 2]], dtype=int),
                np.ones(shape=(2, 2), dtype=int))
        assert str(exin.value) == err
        assert len(caplog.records) == 3
        assert caplog.records[1].levelname == 'INFO'
        assert caplog.records[1].getMessage() == log_1
        assert caplog.records[2].levelname == 'INFO'
        assert caplog.records[2].getMessage() == log_2

        # Colour image
        wrn_msg = 'The segmentation has only **one** segment.'
        with pytest.warns(UserWarning) as warning:
            occlusion = fudo.Occlusion(ARRAY_IMAGE_3D, ONES)
        assert len(warning) == 1
        assert str(warning[0].message) == wrn_msg
        #
        assert np.array_equal(occlusion.image, ARRAY_IMAGE_3D)
        assert np.array_equal(occlusion.segments, ONES)
        assert occlusion.is_rgb
        assert not occlusion.is_bnw
        assert np.array_equal(occlusion.unique_segments, [1])
        assert occlusion.segments_number == 1
        assert np.array_equal(
            occlusion._colouring_strategy(ONES),
            occlusion._generate_colouring_strategy('mean')(ONES))
        assert np.array_equal(
            occlusion.colouring_strategy(ONES),
            occlusion._generate_colouring_strategy('mean')(ONES))

        # Grayscale image
        occlusion = fudo.Occlusion(ARRAY_IMAGE_2D, SEGMENTS, 'white')
        assert np.array_equal(occlusion.image, ARRAY_IMAGE_2D)
        assert np.array_equal(occlusion.segments, SEGMENTS)
        assert not occlusion.is_rgb
        assert not occlusion.is_bnw
        assert np.array_equal(occlusion.unique_segments, [1, 2])
        assert occlusion.segments_number == 2
        assert np.array_equal(
            occlusion._colouring_strategy(ONES),
            occlusion._generate_colouring_strategy('white')(ONES))
        assert np.array_equal(
            occlusion.colouring_strategy(ONES),
            occlusion._generate_colouring_strategy('white')(ONES))

        # Black-and-white image
        assert len(caplog.records) == 3
        occlusion = fudo.Occlusion(ARRAY_IMAGE_BNW1, SEGMENTS)
        assert len(caplog.records) == 5
        assert caplog.records[3].levelname == 'INFO'
        assert caplog.records[3].getMessage() == log_1
        assert caplog.records[4].levelname == 'INFO'
        assert caplog.records[4].getMessage() == log_2
        #
        assert np.array_equal(occlusion.image,
                              np.array([[0, 255], [255, 0]], dtype=np.uint8))
        assert np.array_equal(occlusion.segments, SEGMENTS)
        assert not occlusion.is_rgb
        assert occlusion.is_bnw
        assert np.array_equal(occlusion.unique_segments, [1, 2])
        assert occlusion.segments_number == 2
        assert np.array_equal(
            occlusion._colouring_strategy(ONES),
            occlusion._generate_colouring_strategy('black')(ONES))
        assert np.array_equal(
            occlusion.colouring_strategy(ONES),
            occlusion._generate_colouring_strategy('black')(ONES))

    def test_colouring_strategy(self):
        """
        Tests ``colouring_strategy`` getters and setter for the
        :class:`fatf.utils.data.occlusion.Occlusion` class.
        """
        occlusion = fudo.Occlusion(ARRAY_IMAGE_3D, SEGMENTS)

        assert occlusion._colouring_strategy == occlusion.colouring_strategy
        assert np.array_equal(
            occlusion.colouring_strategy(ONES),
            occlusion._generate_colouring_strategy('mean')(ONES))

        occlusion.colouring_strategy = 'black'
        assert occlusion._colouring_strategy == occlusion.colouring_strategy
        assert np.array_equal(
            occlusion.colouring_strategy(ONES),
            occlusion._generate_colouring_strategy('black')(ONES))

        occlusion.set_colouring_strategy('white')
        assert occlusion._colouring_strategy == occlusion.colouring_strategy
        assert np.array_equal(
            occlusion.colouring_strategy(ONES),
            occlusion._generate_colouring_strategy('white')(ONES))

    def test_randomise_patch(self):
        """
        Tests :func:`fatf.utils.data.occlusion.Occlusion._randomise_patch`.
        """
        fatf.setup_random_seed()
        mask_ = np.array([[1, 0], [0, 1]], dtype=bool)

        # Colour
        occlusion = fudo.Occlusion(ARRAY_IMAGE_3D, SEGMENTS)
        assert np.array_equal(
            occlusion._randomise_patch(mask_),
            np.array([[125, 114, 71], [52, 44, 216]], dtype=np.uint8))
        # ..check the default
        assert np.array_equal(
            occlusion._colouring_strategy(ONES),
            occlusion._generate_colouring_strategy('mean')(ONES))

        # Grayscale
        occlusion = fudo.Occlusion(ARRAY_IMAGE_2D, SEGMENTS)
        assert np.array_equal(
            occlusion._randomise_patch(mask_),
            np.array([119, 13], dtype=np.uint8))
        # ..check the default
        assert np.array_equal(
            occlusion._colouring_strategy(ONES),
            occlusion._generate_colouring_strategy('mean')(ONES))

        # Black-and-white
        occlusion = fudo.Occlusion(
            np.array([[0, 255], [255, 0]], dtype=np.uint8), SEGMENTS)
        assert np.array_equal(
            occlusion._randomise_patch(mask_),
            np.array([0, 255], dtype=np.uint8))
        # ..check the default
        assert np.array_equal(
            occlusion._colouring_strategy(ONES),
            occlusion._generate_colouring_strategy('black')(ONES))

    def test_generate_colouring_strategy(self):
        """
        Tests :func:`fatf.utils.data.occlusion.Occlusion.\
_generate_colouring_strategy`.
        """
        occlusion = fudo.Occlusion(ARRAY_IMAGE_3D, SEGMENTS)

        # Errors
        msg = ('The colour can either be a string specifier; or '
               'an RGB thriplet for RGB images and an integer '
               'for or grayscale and black-and-white images.')
        with pytest.raises(TypeError) as exin:
            occlusion._generate_colouring_strategy(['list'])
        assert str(exin.value) == msg

        # int for colour
        with pytest.raises(TypeError) as exin:
            occlusion._generate_colouring_strategy(33)
        assert str(exin.value) == msg

        # tuple for grayscale/black-and-white
        occlusion = fudo.Occlusion(ARRAY_IMAGE_2D, SEGMENTS)
        with pytest.raises(TypeError) as exin:
            occlusion._generate_colouring_strategy((4, 2, 0))
        assert str(exin.value) == msg
        with pytest.raises(TypeError) as exin:
            occlusion._generate_colouring_strategy(2.0)
        assert str(exin.value) == msg

        # Colour
        occlusion = fudo.Occlusion(ARRAY_IMAGE_3D, SEGMENTS)

        # string
        msg = ('Unknown colouring strategy name: colour.\n'
               "Choose one of the following: ['black', 'blue', 'green', "
               "'mean', 'pink', 'random', 'random-patch', 'randomise', "
               "'randomise-patch', 'red', 'white'].")
        with pytest.raises(ValueError) as exin:
            occlusion._generate_colouring_strategy('colour')
        assert str(exin.value) == msg
        # functional -- mean
        clr = occlusion._generate_colouring_strategy(None)(ONES)
        assert np.array_equal(clr, np.ones(shape=(2, 2, 2, 3), dtype=np.uint8))
        clr = occlusion._generate_colouring_strategy('mean')(ONES)
        assert np.array_equal(clr, np.ones(shape=(2, 2, 2, 3), dtype=np.uint8))

        one_ = np.zeros(shape=(2, 2), dtype=bool)
        one_[1, 1] = True
        fatf.setup_random_seed()
        # functional -- random
        clr = occlusion._generate_colouring_strategy('random')(ONES)
        assert np.array_equal(clr, (57, 12, 140))
        # functional -- random-patch
        clr = occlusion._generate_colouring_strategy('random-patch')(one_)
        assert np.array_equal(clr, np.array([[16, 15, 47]], dtype=np.uint8))
        # functional -- randomise
        clr = occlusion._generate_colouring_strategy('randomise')(one_)
        assert np.array_equal(clr, (101, 214, 112))
        # functional -- randomise-patch
        clr = occlusion._generate_colouring_strategy('randomise-patch')(one_)
        assert np.array_equal(clr, np.array([[81, 216, 174]], dtype=np.uint8))
        # functional -- black
        clr = occlusion._generate_colouring_strategy('black')(one_)
        assert np.array_equal(clr, (0, 0, 0))
        # functional -- white
        clr = occlusion._generate_colouring_strategy('white')(one_)
        assert np.array_equal(clr, (255, 255, 255))
        # functional -- red
        clr = occlusion._generate_colouring_strategy('red')(one_)
        assert np.array_equal(clr, (255, 0, 0))
        # functional -- green
        clr = occlusion._generate_colouring_strategy('green')(one_)
        assert np.array_equal(clr, (0, 255, 0))
        # functional -- blue
        clr = occlusion._generate_colouring_strategy('blue')(one_)
        assert np.array_equal(clr, (0, 0, 255))
        # functional -- pink
        clr = occlusion._generate_colouring_strategy('pink')(one_)
        assert np.array_equal(clr, (255, 192, 203))

        # tuple
        clr = occlusion._generate_colouring_strategy((42, 24, 242))(one_)
        assert np.array_equal(clr, (42, 24, 242))

        # Grayscale
        occlusion = fudo.Occlusion(ARRAY_IMAGE_2D, SEGMENTS)
        # int
        msg = ('Unknown colouring strategy name: colour.\n'
               "Choose one of the following: ['black', 'mean', 'random', "
               "'random-patch', 'randomise', 'randomise-patch', 'white'].")
        with pytest.raises(ValueError) as exin:
            occlusion._generate_colouring_strategy('colour')
        assert str(exin.value) == msg

        msg = ('The colour should be an integer between '
               '0 and 255 for grayscale images.')
        with pytest.raises(ValueError) as exin:
            occlusion._generate_colouring_strategy(-1)
        assert str(exin.value) == msg
        with pytest.raises(ValueError) as exin:
            occlusion._generate_colouring_strategy(256)
        assert str(exin.value) == msg

        clr = occlusion._generate_colouring_strategy(42)(one_)
        assert clr == 42

        # string
        clr = occlusion._generate_colouring_strategy(None)(ONES)
        assert np.array_equal(
            clr,
            np.array([[[85, 2], [85, 2]], [[85, 2], [85, 2]]], dtype=np.uint8))
        clr = occlusion._generate_colouring_strategy('mean')(ONES)
        assert np.array_equal(
            clr,
            np.array([[[85, 2], [85, 2]], [[85, 2], [85, 2]]], dtype=np.uint8))

        fatf.setup_random_seed()
        # functional -- random
        clr = occlusion._generate_colouring_strategy('random')(ONES)
        assert clr == 57
        # functional -- random-patch
        clr = occlusion._generate_colouring_strategy('random-patch')(one_)
        assert np.array_equal(clr, np.array([125], dtype=np.uint8))
        # functional -- randomise
        clr = occlusion._generate_colouring_strategy('randomise')(one_)
        assert clr == 71
        # functional -- randomise-patch
        clr = occlusion._generate_colouring_strategy('randomise-patch')(one_)
        assert np.array_equal(clr, np.array([44], dtype=np.uint8))
        # functional -- black
        clr = occlusion._generate_colouring_strategy('black')(one_)
        assert clr == 0
        # functional -- white
        clr = occlusion._generate_colouring_strategy('white')(one_)
        assert clr == 255

        # Black-and-white
        occlusion = fudo.Occlusion(
            np.array([[0, 255], [0, 255]], dtype=np.uint8), SEGMENTS)

        # int
        msg = ('The colour should be 0 for black, or 1 or 255 for '
               'white for black-and-white images.')
        with pytest.raises(ValueError) as exin:
            occlusion._generate_colouring_strategy(42)
        assert str(exin.value) == msg

        clr = occlusion._generate_colouring_strategy(0)(one_)
        assert clr == 0
        clr = occlusion._generate_colouring_strategy(1)(one_)
        assert clr == 255
        clr = occlusion._generate_colouring_strategy(255)(one_)
        assert clr == 255

        # string
        msg = 'Mean occlusion is not supported for black-and-white images.'
        with pytest.raises(RuntimeError) as exin:
            occlusion._generate_colouring_strategy(None)
        assert str(exin.value) == msg
        with pytest.raises(RuntimeError) as exin:
            occlusion._generate_colouring_strategy('mean')
        assert str(exin.value) == msg

        fatf.setup_random_seed()
        # functional -- random
        clr = occlusion._generate_colouring_strategy('random')(ONES)
        assert clr == 0
        # functional -- random-patch
        clr = occlusion._generate_colouring_strategy('random-patch')(one_)
        assert np.array_equal(clr, np.array([0], dtype=np.uint8))
        # functional -- randomise
        clr = occlusion._generate_colouring_strategy('randomise')(one_)
        assert clr == 0
        # functional -- randomise-patch
        clr = occlusion._generate_colouring_strategy('randomise-patch')(one_)
        assert np.array_equal(clr, np.array([0], dtype=np.uint8))
        # functional -- black
        clr = occlusion._generate_colouring_strategy('black')(one_)
        assert clr == 0
        # functional -- white
        clr = occlusion._generate_colouring_strategy('white')(one_)
        assert clr == 255

    def test_occlude_segments(self):
        """
        Tests :func:`fatf.utils.data.occlusion.Occlusion.occlude_segments`.
        """
        occlusion = fudo.Occlusion(ARRAY_IMAGE_3D, SEGMENTS)

        msg = ('Segments subset must be either '
               'an integer or a list of integers.')
        with pytest.raises(TypeError) as exin:
            occlusion.occlude_segments('list')
        assert str(exin.value) == msg

        msg = ('The segment id 0 does not correspond to any of '
               'the known segments ([1, 2]).')
        with pytest.raises(ValueError) as exin:
            occlusion.occlude_segments(segments_subset=0)
        assert str(exin.value) == msg

        msg = 'The list of segments has duplicates.'
        with pytest.raises(ValueError) as exin:
            occlusion.occlude_segments(segments_subset=[1, 2, 1])
        assert str(exin.value) == msg

        msg = 'The segment id 1 is not an integer.'
        with pytest.raises(TypeError) as exin:
            occlusion.occlude_segments(segments_subset=[1, 2, '1'])
        assert str(exin.value) == msg

        msg = ('The segment id 4 does not correspond to any of '
               'the known segments ([1, 2]).')
        with pytest.raises(ValueError) as exin:
            occlusion.occlude_segments(segments_subset=[2, 4, 1])
        assert str(exin.value) == msg

        msg = ('The width, height or number of channels of the input '
               'image does not agree with the same parameters of the '
               'original image.')
        with pytest.raises(IncorrectShapeError) as exin:
            occlusion.occlude_segments([],
                                       image=np.ones(shape=(4, 4), dtype=int))
        assert str(exin.value) == msg

        # No image
        ocl = occlusion.occlude_segments(segments_subset=[])
        assert np.array_equal(ocl, ARRAY_IMAGE_3D)

        # External image with external colour
        img_ = np.ones(shape=(2, 2, 3), dtype=np.uint8)
        img_[0, 0, 0] = 42
        img_[1, 1, 1] = 42
        img_[0, 1, 2] = 42
        ocl_ = np.zeros(shape=(2, 2, 3), dtype=np.uint8)
        ocl_[1, 1] = (1, 42, 1)

        ocl = occlusion.occlude_segments([1], image=img_, colour='black')
        assert np.array_equal(ocl, ocl_)
        ocl = occlusion.occlude_segments(1, image=img_, colour='black')
        assert np.array_equal(ocl, ocl_)

    def test_occlude_segments_vectorised(self):
        """
        Tests :func:`fatf.utils.data.occlusion.Occlusion.\
occlude_segments_vectorised`.
        """
        occlusion = fudo.Occlusion(ARRAY_IMAGE_3D, SEGMENTS)

        msg = ('The width, height or number of channels of the input '
               'image does not agree with the same parameters of the '
               'original image.')
        with pytest.raises(IncorrectShapeError) as exin:
            occlusion.occlude_segments_vectorised(
                None, image=np.ones(shape=(4, 4), dtype=int))
        assert str(exin.value) == msg

        err = ('The vector representation of segments should be a 1- or '
               '2-dimensional numpy array.')
        with pytest.raises(IncorrectShapeError) as exin:
            occlusion.occlude_segments_vectorised(np.array([[[1, 2, 3]]]))
        assert str(exin.value) == err

        err = ('The vector representation of segments cannot be '
               'a structured numpy array.')
        with pytest.raises(TypeError) as exin:
            occlusion.occlude_segments_vectorised(ARRAY_STRUCT)
        assert str(exin.value) == err

        err = ('The vector representation of segments should be '
               'a numerical numpy array.')
        with pytest.raises(TypeError) as exin:
            occlusion.occlude_segments_vectorised(np.array(['1', '2']))
        assert str(exin.value) == err

        err = ('The number of elements (3) in the vector representation of '
               'segments should correspond to the unique number of segments '
               '(2).')
        with pytest.raises(IncorrectShapeError) as exin:
            occlusion.occlude_segments_vectorised(np.array([1, 2, 3]))
        assert str(exin.value) == err

        err = ('The number of columns (3) in the vector representation '
               'of segments should correspond to the unique number of '
               'segments (2).')
        with pytest.raises(IncorrectShapeError) as exin:
            occlusion.occlude_segments_vectorised(np.array([[1, 2, 3]]))
        assert str(exin.value) == err

        err = ('The vector representation of segments should be binary '
               'numpy array.')
        with pytest.raises(TypeError) as exin:
            occlusion.occlude_segments_vectorised(np.array([[1, 2]]))
        assert str(exin.value) == err

        # 1-D mask
        ocl = occlusion.occlude_segments_vectorised(np.array([1, 1]))
        assert np.array_equal(ocl, ARRAY_IMAGE_3D)
        ocl = occlusion.occlude_segments_vectorised(
            np.array([1, 0]), colour='black')
        ocl_ = np.ones(shape=(2, 2, 3), dtype=np.uint8)
        ocl_[1, 1] = (0, 0, 0)
        assert np.array_equal(ocl, ocl_)

        # 1-D mask -- colour
        ocl = occlusion.occlude_segments_vectorised(
            np.array([1.0, 0.0]), colour='black')
        assert np.array_equal(ocl, ocl_)

        # 2-D mask -- colour
        ocl = occlusion.occlude_segments_vectorised(
            np.array([[1.0, 0.0], [1, 0]]), colour='black')
        assert np.array_equal(ocl, np.array([ocl_, ocl_]))

        # 2-D mask -- external image
        ocl = occlusion.occlude_segments_vectorised(
            np.array([[1.0, 0.0], [1, 0]]), image=ocl_, colour='black')
        assert np.array_equal(ocl, np.array([ocl_, ocl_]))
