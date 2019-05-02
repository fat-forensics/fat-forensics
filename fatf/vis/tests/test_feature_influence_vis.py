"""
Tests feature influence (ICE and PD) plotting functions.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
#         Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import matplotlib.legend
import pytest

import matplotlib.pyplot as plt
import numpy as np

import fatf.utils.testing.vis as futv
import fatf.vis.feature_influence as fvfi

from fatf.exceptions import IncorrectShapeError

FAKE_ICE_ARRAY = np.array([
    [[0.1, 0.9],
     [0.5, 0.5],
     [0.2, 0.8],
     [0.3, 0.7],
     [0.4, 0.6],
     [0.6, 0.4]],

    [[0.0, 1.0],
     [0.7, 0.3],
     [0.8, 0.2],
     [0.9, 0.1],
     [1.0, 0.0],
     [0.9, 0.1]],

    [[0.8, 0.2],
     [0.7, 0.3],
     [0.6, 0.4],
     [0.5, 0.5],
     [0.4, 0.6],
     [0.3, 0.7]],

    [[0.2, 0.8],
     [0.1, 0.9],
     [0.0, 1.0],
     [0.1, 0.9],
     [0.2, 0.8],
     [0.3, 0.7]]])  # yapf: disable
FAKE_ICE_ARRAY_2D = np.array([
   [[[0.50, 0.50, 0.00],
    [0.90, 0.07, 0.03],
    [0.20, 0.30, 0.40]],
   [[0.50, 0.50, 0.00],
    [0.33, 0.33, 0.34],
    [0.20, 0.30, 0.40]],
   [[0.33, 0.33, 0.34],
    [0.90, 0.07, 0.03],
    [0.20, 0.30, 0.40]]],
  [[[0.33, 0.33, 0.34],
    [0.50, 0.50, 0.00],
    [0.20, 0.30, 0.40]],
   [[0.40, 0.20, 0.40],
    [0.00, 0.66, 0.33],
    [0.10, 0.40, 0.50]],
   [[0.33, 0.54, 0.13],
    [0.91, 0.04, 0.05],
    [0.21, 0.39, 0.60]]],
  [[[0.42, 0.40, 0.18],
    [0.09, 0.90, 0.01],
    [0.39, 0.21, 0.40]],
   [[0.30, 0.40, 0.20],
    [0.33, 0.33, 0.34],
    [0.10, 0.30, 0.50]],
   [[0.33, 0.33, 0.34],
    [0.70, 0.27, 0.03],
    [0.10, 0.30, 0.50]]]])   # yapf: disable
FAKE_PD_ARRAY = np.array([[0.50, 0.50, 0.00],
                          [0.33, 0.33, 0.34],
                          [0.90, 0.07, 0.03],
                          [0.33, 0.33, 0.34],
                          [0.90, 0.07, 0.03],
                          [0.20, 0.30, 0.40]])  # yapf: disable
FAKE_PD_ARRAY_2D = np.array([
   [[0.50, 0.50, 0.00],
    [0.90, 0.07, 0.03],
    [0.20, 0.30, 0.40]],
   [[0.50, 0.50, 0.00],
    [0.33, 0.33, 0.34],
    [0.20, 0.30, 0.40]],
   [[0.33, 0.33, 0.34],
    [0.90, 0.07, 0.03],
    [0.20, 0.30, 0.40]]]) # yapf: disable
FAKE_LINESPACE = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
FAKE_LINESPACE_2D = (np.array([0., 0.5, 1.]), np.array([0.32, 0.41, 0.5]))
FAKE_LINESPACE_2D_CAT = (np.array([0., 0.5, 1.]),
                         np.array(['a', 'b', 'c']))
FAKE_LINESPACE_STRING = np.array(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
FAKE_VARIANCE = np.array([[0.20, 0.05, 0.00],
                          [0.10, 0.05, 0.15],
                          [0.05, 0.01, 0.01],
                          [0.26, 0.15, 0.21],
                          [0.08, 0.05, 0.01],
                          [0.15, 0.10, 0.17]])
FAKE_LINESPACE_CAT = np.array(['a', 'b', 'c', 'd', 'e', 'f'])

STRUCT_ARRAY = np.array([(4, 2), (2, 4)], dtype=[('a', int), ('b', int)])
NON_NUMERICAL_ARRAY = np.array([[4, 'a'], [2, 'b']])
NUMERICAL_2D_ARRAY = np.array([[4, 2], [2, 4], [4, 2]])
NUMERICAL_3D_ARRAY = np.array([[[4, 3], [4, 2], [4, 2]],
                              [[8, 1], [7, 5], [4, 2]],
                              [[4, 3], [4, 2], [4, 2]],
                              [[4, 2], [2, 4], [4, 2]]])
NUMERICAL_4D_ARRAY = np.array([[[[4, 3], [4, 2], [4, 2]],
                              [[8, 1], [7, 5], [4, 2]],
                              [[4, 1], [5, 7], [4, 2]]],
                              [[[4, 3], [4, 2], [4, 2]],
                              [[4, 3], [2, 4], [4, 2]],
                              [[4, 2], [2, 4], [4, 2]]]])

def test_validate_input():
    """
    Tests :func:`fatf.vis.feature_influence._validate_input`.
    """
    msg = 'test_partial_dependence is not a boolean.'
    with pytest.raises(TypeError) as exin:
        fvfi._validate_input(None, None, None, None, None, None, None, None,
                             False, 1)
    assert str(exin.value) == msg

    msg = 'variance_area is not a boolean.'
    with pytest.raises(TypeError) as exin:
        fvfi._validate_input(None, None, None, None, None, None, None, None,
                             1, True)
    assert str(exin.value) == msg

    msg = 'The input array cannot be a structured array.'
    with pytest.raises(ValueError) as exin:
        fvfi._validate_input(STRUCT_ARRAY, None, None, None, None, None, None,
                             None, False, False)
    assert str(exin.value) == msg

    msg = 'The input array has to be a numerical array.'
    with pytest.raises(ValueError) as exin:
        fvfi._validate_input(NON_NUMERICAL_ARRAY, None, None, None, None, None,
                             None, None, False, False)
    assert str(exin.value) == msg

    # For Individual Conditional Expectation
    msg = ('plot_individual_condtional_expectation expects a 3-dimensional '
           'array of shape (n_samples, n_steps, n_classes) or for 2-D '
           'individual conditional expectation, a shape of (n_samples, '
           'n_steps_1, n_steps_2, n_classes).')
    with pytest.raises(IncorrectShapeError) as exin:
        fvfi._validate_input(NUMERICAL_2D_ARRAY, NUMERICAL_2D_ARRAY[0, :], 
                             None, None, None, None, None, None, False, False)
    assert str(exin.value) == msg

    # For Partial Dependence
    msg = ('plot_partial_depenedence expects a 2-dimensional array of shape '
           '(n_steps, n_classes) or for 2-D partial dependence, a shape of '
           '(n_steps_1, n_steps_2, n_classes).')
    with pytest.raises(IncorrectShapeError) as exin:
        fvfi._validate_input(NUMERICAL_2D_ARRAY[0, :], NUMERICAL_2D_ARRAY[0, :],
                             None, None, None, None, None, None, False, True)
    assert str(exin.value) == msg

    # Linespace
    msg = 'The linespace array cannot be a structured array.'
    with pytest.raises(ValueError) as exin:
        fvfi._validate_input(NUMERICAL_2D_ARRAY, STRUCT_ARRAY, None, None,
                             None, None, None, None, False, True)
    assert str(exin.value) == msg
    #
    msg = ('The linespace array has to be a 1-dimensional array of shape '
           '(n_steps, ).')
    with pytest.raises(IncorrectShapeError) as exin:
        fvfi._validate_input(NUMERICAL_3D_ARRAY, NUMERICAL_2D_ARRAY, None,
                             None, None, None, None, None, False, False)
    assert str(exin.value) == msg
    # Linespace vector not matching ICE/ PDP dimensions
    msg = ('The length of the linespace array ({}) does not agree with the '
           'number of linespace steps ({}) in the input array.')
    with pytest.raises(ValueError) as exin:
        fvfi._validate_input(NUMERICAL_2D_ARRAY, NUMERICAL_2D_ARRAY[0, :],
                             None, None, None, None, None, None, False, True)
    assert str(exin.value) == msg.format(2, 3)
    with pytest.raises(ValueError) as exin:
        fvfi._validate_input(NUMERICAL_3D_ARRAY, NUMERICAL_2D_ARRAY[0, :],
                             None, None, None, None, None, None, False, False)
    assert str(exin.value) == msg.format(2, 3)
    with pytest.raises(ValueError) as exin:
        fvfi._validate_input(NUMERICAL_4D_ARRAY, (NUMERICAL_2D_ARRAY[:, 0], 
                             NUMERICAL_2D_ARRAY[0, :]), None, None, None, None,
                             None, None, False, False)
    assert str(exin.value) == msg.format(2, 3)

    msg = ('A 3-dimenionsal individual conditional expectation array was '
           'given but 2 feature linespaces. To plot 2 feature individual '
           'conditional expectation, a 4-dimensional array must be provided.')
    with pytest.raises(ValueError) as exin:
        fvfi._validate_input(NUMERICAL_3D_ARRAY, (NUMERICAL_2D_ARRAY[:, 0], 
                             NUMERICAL_2D_ARRAY[0, :]), None, None, None, None,
                             None, None, False, False)
    assert str(exin.value) == msg

    # Variance vector not matching ICE/ PDP dimensions
    msg = ('The length of the variance array ({}) does agree with the number '
           'of linespace steps ({}) in the input array.')
    with pytest.raises(ValueError) as exin:
        fvfi._validate_input(NUMERICAL_2D_ARRAY, NUMERICAL_2D_ARRAY[:, 0], 1,
                             None, None, None, None, NUMERICAL_2D_ARRAY[0, :], 
                             False, True)
    assert str(exin.value) == msg.format(2, 3)
    # variance_area is True but no variance vector supplied
    msg = ('Variance vector has not been given but variance_area has been '
           'given as True. To plot the variance please specify a variance '
           'vector.')
    with pytest.raises(ValueError) as exin:
        fvfi._validate_input(NUMERICAL_2D_ARRAY, NUMERICAL_2D_ARRAY[:, 0], 1,
                             None, None, None, None, None, True, True)
    assert str(exin.value) == msg
    # Index
    msg = 'Class index has to be an integer.'
    with pytest.raises(TypeError) as exin:
        fvfi._validate_input(NUMERICAL_3D_ARRAY, NUMERICAL_2D_ARRAY[:, 0],
                             None, None, None, None, None, None, False, False)
    assert str(exin.value) == msg
    #
    msg = ('Class index {} is not a valid index for the input array. There '
           'are only {} classes available. For plotting data computed using '
           'a regression model, use class_index=0.')
    with pytest.raises(IndexError) as exin:
        fvfi._validate_input(NUMERICAL_3D_ARRAY, NUMERICAL_2D_ARRAY[:, 0], -1,
                             None, None, None, None, None, False, False)
    assert str(exin.value) == msg.format(-1, 2)
    with pytest.raises(IndexError) as exin:
        fvfi._validate_input(NUMERICAL_2D_ARRAY, NUMERICAL_2D_ARRAY[:, 0], 2,
                             None, None, None, None, None, False, True)
    assert str(exin.value) == msg.format(2, 2)

    # Feature name
    msg = ('The feature name has to be either None or a string or a list of '
           'strings.')
    with pytest.raises(TypeError) as exin:
        fvfi._validate_input(NUMERICAL_2D_ARRAY, NUMERICAL_2D_ARRAY[:, 0], 1,
                             None, 42, None, None, None, False, True)
    assert str(exin.value) == msg
    with pytest.raises(TypeError) as exin:
        fvfi._validate_input(NUMERICAL_2D_ARRAY, NUMERICAL_2D_ARRAY[:, 0], 1,
                             None, ['1', 42], None, None, None, False, True)
    assert str(exin.value) == msg
    # Class name
    msg = 'The class name has to be either None or a string.'
    with pytest.raises(TypeError) as exin:
        fvfi._validate_input(NUMERICAL_3D_ARRAY, NUMERICAL_2D_ARRAY[:, 0], 0,
                             None, None, 42, None, None, False, False)
    assert str(exin.value) == msg

    # Plot axis
    msg = ('The plot axis has to be either None or a matplotlib.pyplot.Axes '
           'type object.')
    with pytest.raises(TypeError) as exin:
        fvfi._validate_input(NUMERICAL_2D_ARRAY, NUMERICAL_2D_ARRAY[:, 0], 1,
                             None, 'feature name', None, 42, None, False, True)
    assert str(exin.value) == msg

    # Treat as categorical
    msg = ('treat_as_categorical has to either be None, a boolean or a list '
           'of None and booleans.')
    with pytest.raises(TypeError) as exin:
        fvfi._validate_input(None, None, 1, 10, None, None, None, None,
                             False, True)
    assert str(exin.value) == msg
    with pytest.raises(TypeError) as exin:
        fvfi._validate_input(None, None, 1, [False, 10], None, None, None, None,
                             False, True)
    assert str(exin.value) == msg

    # Variance cannot be structured
    msg = ('The variance array cannot be a structured array.')
    with pytest.raises(ValueError) as exin:
        fvfi._validate_input(NUMERICAL_2D_ARRAY, NUMERICAL_2D_ARRAY[:, 0],
                             1, False, 'feature name', 'class name', None,
                            STRUCT_ARRAY, False, True)
    assert str(exin.value) == msg

    # Variance has to be numerical
    msg = ('The variance array has to be a numerical array.')
    with pytest.raises(ValueError) as exin:
        fvfi._validate_input(NUMERICAL_2D_ARRAY, NUMERICAL_2D_ARRAY[:, 0],
                             1, False, 'feature name', 'class name', None,
                            NON_NUMERICAL_ARRAY, False, True)
    assert str(exin.value) == msg

    dist = (NUMERICAL_2D_ARRAY[:, 0], NUMERICAL_2D_ARRAY[:, 0])
    # Testing for 2 feature PD and ICE
    msg = ('Too many values given for treat_as_categorical.')
    with pytest.raises(ValueError) as exin:
        fvfi._validate_input(NUMERICAL_2D_ARRAY, NUMERICAL_2D_ARRAY[:, 0],
                             1, [False, False, False], 'feature name',
                             'class name', None, None, False, False)
    assert str(exin.value) == msg

    msg = ('Too many values given for feature_name.')
    with pytest.raises(ValueError) as exin:
        fvfi._validate_input(NUMERICAL_2D_ARRAY, NUMERICAL_2D_ARRAY[:, 0],
                             1, False, ['a', 'b', 'c', 'd'],
                             'class name', None, None, False, False)
    assert str(exin.value) == msg
    
    msg = ('feature_linespace must be a tuple of numpy.ndarray with max length '
           '2 for use in 2 feature partial dependence and individual '
           'conditional expectation.')
    with pytest.raises(ValueError) as exin:
        fvfi._validate_input(NUMERICAL_2D_ARRAY, (NUMERICAL_2D_ARRAY[:, 0],
                             NUMERICAL_2D_ARRAY[:, 0], 
                             NUMERICAL_2D_ARRAY[:, 0]), 1, False, 
                             'feature_name', 'class name', None, None, False,
                             False)
    assert str(exin.value) == msg

    msg = ('A 3-dimension partial dependence array was provided but only one '
           'feature name. In order to use feature names you must provide a '
           'list of length 2 containing strings or None.')
    with pytest.raises(ValueError) as exin:
        fvfi._validate_input(NUMERICAL_4D_ARRAY[0], dist,
                             1, False, 'feature name', 'class name', None,
                             None, False, True)
    assert str(exin.value) == msg

    msg = ('A 4-dimension individual conditional expectation array was '
           'provided but only one feature name. In order to use feature names '
           'you must provide a list of length 2 containing strings or None.')
    with pytest.raises(ValueError) as exin:
        fvfi._validate_input(NUMERICAL_4D_ARRAY, dist,
                             1, False, 'feature name', 'class name', None,
                             None, False, False)
    assert str(exin.value) == msg

    msg = ('A 3-dimensional partial dependence array was provided but only '
           'one feature linespace. A 3-dimensional array can only be used in '
           'plotting 2 feature partial depedence, and as such a tuple of 2 '
           'feature linespaces must be given.')
    with pytest.raises(ValueError) as exin:
        fvfi._validate_input(NUMERICAL_3D_ARRAY, (NUMERICAL_2D_ARRAY[:, 0]),
                             1, False, 'feature name', 'class name', None,
                             None, False, True)
    assert str(exin.value) == msg

    msg = ('A 4-dimensional individual conditional expectation array was '
           'given but only one feature linespace. A 4-dimensional array '
           'can only be used in plotting 2 feature individual conditional '
           'expectation, and as such a tuple of 2 feature linespaces must '
           'be given.')
    with pytest.raises(ValueError) as exin:
        fvfi._validate_input(NUMERICAL_4D_ARRAY, (NUMERICAL_2D_ARRAY[:, 0]),
                             1, False, 'feature name', 'class name', None,
                             None, False, False)
    assert str(exin.value) == msg

    msg = ('feature_linespace must be a tuple of numpy.ndarray')
    with pytest.raises(TypeError) as exin:
        fvfi._validate_input(NUMERICAL_4D_ARRAY, (NUMERICAL_2D_ARRAY[:, 0], 1),
                             1, False, None, 'class name', None,
                             None, False, False)
    assert str(exin.value) == msg

    msg= ('A variance array was provided but a 3-dimensional array was also '
          'given. To plot a 2-feature partial depenedence, variance cannot '
          'be used. The variance array will be ignored.')
    with pytest.warns(UserWarning) as warning:
        fvfi._validate_input(NUMERICAL_4D_ARRAY[0], dist, 1, [False, False],
                             None, None, None, NUMERICAL_2D_ARRAY[:, 0], False,
                             True)
    assert len(warning) == 1
    assert str(warning[0].message) == msg

    # All OK
    assert fvfi._validate_input(NUMERICAL_2D_ARRAY, NUMERICAL_2D_ARRAY[:, 0],
                                1, False, 'feature name', 'class name', None, 
                                NUMERICAL_2D_ARRAY[:, 0], True, True)
    fig, my_plot = plt.subplots(1, 1)
    assert fvfi._validate_input(NUMERICAL_3D_ARRAY, NUMERICAL_2D_ARRAY[:, 0],
                                1, False, 'feature name', 'class name', my_plot, None,
                                False, False)
    assert fvfi._validate_input(NUMERICAL_4D_ARRAY[0], dist, 1, [False, False],
                                [None, None], 'class name', None, None,
                                False, True)
    assert fvfi._validate_input(NUMERICAL_4D_ARRAY, dist, 1, [False, False],
                                None, 'class name', None, None,
                                False, False)
    assert fvfi._validate_input(NUMERICAL_4D_ARRAY, dist, 1, False,
                                ['feature name', 'feature_name'], 'class',
                                None, None, False, False)
    plt.close(fig=fig)


def test_validate_feature():
    """
    Tests :func:`fatf.vis.feature_influence._validate_feature` function.
    """
    # treat_as_categorical
    msg = ('treat_as_categorical is not a boolean.')
    with pytest.raises(TypeError) as exin:
        fvfi._validate_feature(None, 'false', None, None, False, None, None)
    assert str(exin.value) == msg

    # Structured feature_distribution
    msg = ('The {} element in feature_distribution array cannot be a '
           'structured array.')
    dist = (NUMERICAL_2D_ARRAY[:, 0], STRUCT_ARRAY)
    with pytest.raises(TypeError) as exin:
        fvfi._validate_feature(dist, False, None, None, False, None, None)
    assert str(exin.value) == msg.format(1)

    # Non-numerical counts
    msg = ('The 1 element of feature_distribution has to be a numerical '
           'array.')
    dist = (NUMERICAL_2D_ARRAY[:, 0], NON_NUMERICAL_ARRAY)
    with pytest.raises(ValueError) as exin:
        fvfi._validate_feature(dist, False, None, None, False, None, None)
    assert str(exin.value) == msg

    # test_feature_categorical not boolean
    msg = ('test_feature_linespace is not a boolean.')
    with pytest.raises(TypeError) as exin:
        fvfi._validate_feature(dist, False, None, None, 'false', None, None)
    assert str(exin.value) == msg

    # Invalid plot_axis
    msg = ('The plot axis has to be either None or a matplotlib.'
           'pyplot.Axes type object.')
    with pytest.raises(TypeError) as exin:
        fvfi._validate_feature(None, False, None, None, False, None, 'plot')
    assert str(exin.value) == msg

    # Invalid feature name
    msg = ('The feature name has to be either None or a string.')
    with pytest.raises(TypeError) as exin:
        fvfi._validate_feature(None, False, 12, None, False, None, None)
    assert str(exin.value) == msg

    # Invalid feature distribution type
    msg = ('Feature distribution has to be a tuple.')
    with pytest.raises(TypeError) as exin:
        fvfi._validate_feature(0, False, None, None, False, None, None)
    assert str(exin.value) == msg

    # Invalid orientation
    msg = ('Orientation must be either \'vertical\' or \'horizontal\'.')
    with pytest.raises(ValueError) as exin:
        fvfi._validate_feature(0, False, None, None, False, 'None', None)
    assert str(exin.value) == msg

    msg = ('Orientation must be either None or a string with value '
           '\'vertical\' or \'horizontal\'.')
    with pytest.raises(TypeError) as exin:
        fvfi._validate_feature(0, False, None, None, False, 2, None)
    assert str(exin.value) == msg

    # Invalid feature distribution
    msg = ('Feature distribution has to be a tuple of length 2 where the first '
           'element is a values array and the second element is a counts '
           'array.')
    with pytest.raises(ValueError) as exin:
        fvfi._validate_feature(tuple([np.array([])]*4), False, None, None,
                               False, None, None)
    assert str(exin.value) == msg

    # List of none np.array as feature distribution
    msg = ('The {} element in feature_distribution array must be of type '
           'np.ndarray.')
    with pytest.raises(TypeError) as exin:
        fvfi._validate_feature((np.array([]), 1), False, None, None, False,
        None, None)
    assert str(exin.value) == msg.format(1)

    # Invalid shapes of feature distribution arrays
    msg = ('Values shape {} and counts shape {} do not agree. In order to '
           'define histograms, values has to be of shape '
           '(counts.shape[0]+1, ). In order to define Gaussian Kernel, values '
           'and counts must be of the same shape.')
    with pytest.raises(ValueError) as exin:
        fvfi._validate_feature(
            (np.array([1, 2, 3,]), np.array([1])), False, None, None, False,
            None, None)
    assert str(exin.value) == msg.format(3, 1)

    # Categorical data but different shaped counts and values
    msg = ('For categorical data, values and counts array must be of the same '
           'shape.')
    with pytest.raises(ValueError) as exin:
        fvfi._validate_feature(
            (np.array([1, 2, 3,]), np.array([1])), True, None, None, False,
            None, None)
    assert str(exin.value) == msg

    # Distribution above 1 should be rejected for histogram
    dist  = (np.array([0., .2, .4, .6, .8, 1.]), 
             np.array([0.1, 0.2, 0.5, 0.1, 1.1,]))
    msg = ('Distribution cannot have value more than 1.0')
    with pytest.raises(ValueError) as exin:
        fvfi._validate_feature(dist, False, 'feat', None, False, None, None)
    assert str(exin.value) == msg

    # Distribution above 1 should be rejected for categorical data
    dist = (np.array([0, 0.2, 0.4, 0.6, 0.8, 1]),
            np.array([0.1, 0.1, 0.3, 0.2, 0.2, 1.1]))
    msg = ('Distribution cannot have value more than 1.0')
    with pytest.raises(ValueError) as exin:
        fvfi._validate_feature(dist, True, 'feat', None, False, None, None)
    assert str(exin.value) == msg

    # If called from ICE or PD plotting functions and feature_distribution
    # does not agree with feature_linespace
    dist = (np.array([-1, -0.8, -0.6, -0.4, -0.2, 0]),
            np.array([0.1, 0.1, 0.3, 0.2, 0.2, 0.1]))
    msg = ('To plot the feature distribution of categorical features, the '
           'values array in feature_distribution[0] must contain all values '
           'of feature_linespace in it.')
    with pytest.raises(ValueError) as exin:
        fvfi._validate_feature(dist, True, 'feat', FAKE_LINESPACE, True, None,
                               None)
    assert str(exin.value) == msg


def test_prepare_a_canvas():
    """
    Tests :func:`fatf.vis.feature_influence._prepare_a_canvas`.

    This test checks for the plot title, x range, x label, y range and y label.
    """
    title = 'plot title'
    title_custom = 'custom plot title'
    class_index = 0
    x_range = [-5, 3]
    y_range = [-0.05, 1.05]
    class_name_n = None
    class_name_s = 'class name'
    feature_name_n = None
    feature_name_s = 'feature name'

    # Plotting from scratch
    axis = None
    #
    figure, plot = fvfi._prepare_a_canvas(
        title, axis, class_index, class_name_s, feature_name_n, x_range)
    assert isinstance(figure, plt.Figure)
    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        plot)
    # ...check title
    assert p_title == title
    # ...check x range
    assert np.array_equal(p_x_range, x_range)
    # ...check x label
    assert p_x_label == "Selected Feature's Linespace"
    # ...check y range
    assert np.array_equal(p_y_range, y_range)
    # ...check y label
    assert p_y_label == '{} class probability'.format(class_name_s)
    #
    figure, plot = fvfi._prepare_a_canvas(
        title, axis, class_index, class_name_n, feature_name_s, x_range)
    assert isinstance(figure, plt.Figure)
    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        plot)
    # ...check title
    assert p_title == title
    # ...check x range
    assert np.array_equal(p_x_range, x_range)
    # ...check x label
    assert p_x_label == feature_name_s
    # ...check y range
    assert np.array_equal(p_y_range, y_range)
    # ...check y label
    assert p_y_label == '{} (class index) class probability'.format(
        class_index)

    # Plotting on an existing axis
    fig, axis = plt.subplots(1, 1)
    #
    axis.set_xlim(np.array([-3, 3]))
    msg = ('The x-axis range of the plot given in the plot_axis parameter '
           'differs from the x-axis range of this plot.')
    with pytest.raises(ValueError) as exin:
        fvfi._prepare_a_canvas(title, axis, class_index, class_name_n,
                               feature_name_n, x_range)
    assert str(exin.value) == msg
    #
    axis.set_xlim(np.array(x_range))
    axis.set_ylim(np.array([0, 1]))
    msg = ('The y-axis range of the plot given in the plot_axis parameter '
           'differs from the y-axis range of this plot.')
    with pytest.raises(ValueError) as exin:
        fvfi._prepare_a_canvas(title, axis, class_index, class_name_n,
                               feature_name_n, x_range)
    assert str(exin.value) == msg
    #
    axis.set_ylim(np.array(y_range))
    axis.set_title(title_custom)
    #
    # Do not extend plot title; new feature name and new class name.
    figure, plot = fvfi._prepare_a_canvas('', axis, class_index, class_name_s,
                                          feature_name_s, x_range)
    assert figure is None
    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        plot)
    # ...check title
    assert p_title == title_custom
    # ...check x range
    assert np.array_equal(p_x_range, x_range)
    # ...check x label
    assert p_x_label == feature_name_s
    # ...check y range
    assert np.array_equal(p_y_range, y_range)
    # ...check y label
    assert p_y_label == '{} class probability'.format(class_name_s)
    #
    # Do not extend plot title; no new feature name & existing feature name and
    # no new class name and existing class name.
    figure, plot = fvfi._prepare_a_canvas('', plot, class_index, class_name_n,
                                          feature_name_n, x_range)
    assert figure is None
    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        plot)
    # ...check title
    assert p_title == title_custom
    # ...check x range
    assert np.array_equal(p_x_range, x_range)
    # ...check x label
    assert p_x_label == feature_name_s
    # ...check y range
    assert np.array_equal(p_y_range, y_range)
    # ...check y label
    assert p_y_label == '{} class probability'.format(class_name_s)
    #
    # Do not extend plot title; no new feature name & no existing feature name
    # and no new class name and no existing class name.
    axis.set_ylabel(None)
    axis.set_xlabel(None)
    figure, plot = fvfi._prepare_a_canvas(
        'extension', axis, class_index, class_name_n, feature_name_n, x_range)
    assert figure is None
    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        plot)
    # ...check title
    assert p_title == '{} &\nextension'.format(title_custom)
    # ...check x range
    assert np.array_equal(p_x_range, x_range)
    # ...check x label
    assert p_x_label == "Selected Feature's Linespace"
    # ...check y range
    assert np.array_equal(p_y_range, y_range)
    # ...check y label
    assert p_y_label == '{} (class index) class probability'.format(
        class_index)
    plt.close(fig=figure) 

    # 2-D feature distribution
    y_range = [0.1, 1.5]
    figure, plot = fvfi._prepare_a_canvas(
        '2-D Partial', None, 1, 'class 1', ['feature1', 'feature2'], x_range, 
        plot_distribution=False, is_2d=True, y_range=y_range)
    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        plot)
    assert p_title == '2-D Partial'
    assert p_x_label == 'feature1'
    assert p_y_label == 'feature2'
    assert np.array_equal(p_x_range, x_range)
    assert np.array_equal(p_y_range, y_range)
    plt.close(fig=figure)

    # 2-D with feature distributions 
    figure, plot = fvfi._prepare_a_canvas(
        'title', None, 1, 'class 1', 'age', x_range, 
        plot_distribution=True, is_2d=True, y_range=y_range)
    assert len(plot) == 3
    (plot_axis, dist_axis_x, dist_axis_y) = plot
    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        plot_axis)
    assert np.array_equal(p_x_range, x_range)
    assert np.array_equal(p_y_range, y_range)
    assert p_title == 'title'
    assert p_x_label == 'age'
    assert p_y_label == 'feature 1'
    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        dist_axis_x)
    assert np.array_equal(p_x_range, x_range)
    assert np.array_equal(p_y_range, np.array([-0.05, 1.05]))
    assert p_title == ''
    assert p_x_label == ''
    assert p_y_label == ''
    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        dist_axis_y)
    assert np.array_equal(p_y_range, y_range)
    assert np.array_equal(p_x_range, np.array([-0.05, 1.05]))
    assert p_title == ''
    assert p_x_label == ''
    assert p_y_label == ''
    plt.close(fig=figure)

    # 2-D with feature distributions 
    figure, plot = fvfi._prepare_a_canvas(
        'title', None, 1, 'class 1', None, x_range, 
        plot_distribution=True, is_2d=True, y_range=y_range)
    assert len(plot) == 3
    (plot_axis, dist_axis_x, dist_axis_y) = plot
    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        plot_axis)
    assert np.array_equal(p_x_range, x_range)
    assert np.array_equal(p_y_range, y_range)
    assert p_title == 'title'
    assert p_x_label == 'feature 0'
    assert p_y_label == 'feature 1'
    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        dist_axis_x)
    assert np.array_equal(p_x_range, x_range)
    assert np.array_equal(p_y_range, np.array([-0.05, 1.05]))
    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        dist_axis_y)
    assert np.array_equal(p_y_range, y_range)
    assert np.array_equal(p_x_range, np.array([-0.05, 1.05]))
    plt.close(fig=figure)


def test_plot_feature_distribution():
    """
    Tests feature distribution plotting.

    Tests
    :func:`fatf.vis.feature_influence.plot_feature_distribution` function.
    """
    dist  = (np.array([0., .2, .4, .6, .8, 1.]), 
             np.array([0.1, 0.2, 0.5, 0.1, 0.1]))
    # Without passing axis
    fig, axis = fvfi.plot_feature_distribution(dist, False, 'feat')
    assert isinstance(fig, plt.Figure)
    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        axis)
    # ...check title
    assert p_title == 'Feature Distribution for feat'
    # ...check x range
    assert np.array_equal(p_x_range, [0., 1.])
    # ...check x label
    assert p_x_label == 'Histogram of values for {}'.format('feat')
    # ...check y label
    assert p_y_label == 'Density'
    # ...check y range
    assert np.array_equal(p_y_range, [0, 1.05])
    bars = axis.patches
    texts = axis.texts
    assert len(bars) == dist[0].shape[0] - 1
    assert len(texts) == dist[0].shape[0] - 1

    heights = np.array([bar.get_height() for bar in bars])
    alphas = np.array([bar.get_alpha() for bar in bars])
    colours = np.array([bar.get_facecolor() for bar in bars])
    correct_colours = np.repeat(
        np.array([0.25, 0.41, 0.88, 0.6])[np.newaxis, :], len(bars), axis=0)
    labels = np.array([text.get_text() for text in axis.get_xticklabels()])
    text_positions = np.array([text.get_position() for text in texts])
    correct_text_positions = np.stack([dist[0][:-1]+0.1,
                                       dist[1]], axis=1)
    text_string = [text.get_text() for text in texts]
    correct_text_string = ['%.2f'%num for num in dist[1]]
    assert np.array_equal(dist[1], heights)
    assert np.array_equal(np.array([0.6]*len(dist[1])), alphas)
    assert np.allclose(correct_colours, colours, atol=1e-2)
    assert np.array_equal(correct_text_positions, text_positions)
    assert text_string == correct_text_string
    plt.close(fig=fig) 

    # Numerical feature histogram
    fig, axis = plt.subplots()
    _, axis = fvfi.plot_feature_distribution(dist, False, None, None, axis)
    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        axis)
    # ...check title
    assert p_title == ''
    # ...check x range
    assert np.array_equal(p_x_range, [0., 1.,])
    # ...check y range
    assert np.array_equal(p_y_range, [0., 1.05])
    bars = axis.patches
    texts = axis.texts
    assert len(bars) == dist[0].shape[0] - 1
    assert len(texts) == dist[0].shape[0] - 1
    assert np.array_equal(dist[1], heights)
    assert np.array_equal(np.array([0.6]*len(dist[1])), alphas)
    assert np.allclose(correct_colours, colours, atol=1e-2)
    assert np.array_equal(correct_text_positions, text_positions)
    assert text_string == correct_text_string
    plt.close(fig=fig)

    # Numerical Categorical feature
    fig, axis = plt.subplots()
    dist = (np.array([0, 0.2, 0.4, 0.6, 0.8, 1]),
            np.array([0.1, 0.1, 0.3, 0.2, 0.2, 0.1]))
    _, axis = fvfi.plot_feature_distribution(dist, True, None, None, axis)
    x_ticklabels = [x.get_text() for x in axis.get_xticklabels()]
    assert x_ticklabels == ['%.1f'%num for num in dist[0]]
    bars = axis.patches
    texts = axis.texts
    assert len(bars) == dist[0].shape[0]
    assert len(texts) == dist[0].shape[0]

    heights = np.array([bar.get_height() for bar in bars])
    alphas = np.array([bar.get_alpha() for bar in bars])
    colours = np.array([bar.get_facecolor() for bar in bars])
    correct_colours = np.repeat(
        np.array([0.25, 0.41, 0.88, 0.6])[np.newaxis, :], len(bars), axis=0)
    labels = np.array([text.get_text() for text in axis.get_xticklabels()])
    text_positions = np.array([text.get_position() for text in texts])
    x_ticks = np.linspace(0, FAKE_LINESPACE.shape[0], len(FAKE_LINESPACE))
    correct_text_positions = np.stack([x_ticks,
                                       dist[1]], axis=1)
    text_string = [text.get_text() for text in texts]
    correct_text_string = ['%.2f'%num for num in dist[1]]
    assert np.array_equal(dist[1], heights)
    assert np.array_equal(np.array([0.6]*len(dist[1])), alphas)
    assert np.allclose(correct_colours, colours, atol=1e-2)
    assert np.array_equal(correct_text_positions, text_positions)
    assert text_string == correct_text_string
    plt.close(fig=fig) 

    # Numerical feature kde
    fig, axis = plt.subplots()
    dist = (np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
            np.array([0.1, 0.2, 0.4, 0.4, 0.3, 0.2, 0.1, 0.05, 0.05, 0.05]))
    _, axis = fvfi.plot_feature_distribution(dist, False, None, None, axis)
    assert len(axis.lines) == 1

    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        axis)
    # ...check title
    assert p_title == ''
    # ...check x range
    assert np.array_equal(p_x_range, [0., 0.9])
    # ...check y range
    assert np.array_equal(p_y_range, [-0.05, 1.05])

    l_data, l_colour, l_alpha, l_label, l_width = futv.get_line_data(
        axis.lines[0])
    assert np.array_equal(np.stack(dist, axis=1), l_data)
    assert l_colour == 'royalblue'
    assert l_alpha == 0.6
    assert l_width == 3.0
    plt.close(fig=fig)

    # Categorical feature
    fig, axis = plt.subplots()
    dist = (np.array(['a', 'b', 'c', 'd', 'e', 'f']),
            np.array([0.1, 0.1, 0.3, 0.2, 0.2, 0.1]))
    _, axis = fvfi.plot_feature_distribution(dist, True, None, None, axis)
    x_ticklabels = [x.get_text() for x in axis.get_xticklabels()]
    assert x_ticklabels == list(dist[0])

    bars = axis.patches
    texts = axis.texts
    assert len(bars) == dist[0].shape[0]
    assert len(texts) == dist[0].shape[0]

    heights = np.array([bar.get_height() for bar in bars])
    alphas = np.array([bar.get_alpha() for bar in bars])
    colours = np.array([bar.get_facecolor() for bar in bars])
    correct_colours = np.repeat(
        np.array([0.25, 0.41, 0.88, 0.6])[np.newaxis, :], len(bars), axis=0)
    labels = np.array([text.get_text() for text in axis.get_xticklabels()])
    text_positions = np.array([text.get_position() for text in texts])
    x_ticks = np.linspace(0, FAKE_LINESPACE.shape[0], len(FAKE_LINESPACE))
    correct_text_positions = np.stack([x_ticks,
                                       dist[1]], axis=1)
    text_string = [text.get_text() for text in texts]
    correct_text_string = ['%.2f'%num for num in dist[1]]
    assert np.array_equal(dist[1], heights)
    assert np.array_equal(np.array([0.6]*len(dist[1])), alphas)
    assert np.allclose(correct_colours, colours, atol=1e-2)
    assert np.array_equal(correct_text_positions, text_positions)
    assert text_string == correct_text_string
    plt.close(fig=fig)

    # KDE where one value is more than 1
    fig, axis = plt.subplots()
    dist = (np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
            np.array([0.1, 0.2, 0.4, 1.2, 0.3, 0.2, 0.1, 0.05, 0.05, 0.05]))
    _, axis = fvfi.plot_feature_distribution(dist, False, None, None, axis)
    assert len(axis.lines) == 1

    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        axis)
    # ...check title
    assert p_title == ''
    # ...check x range
    assert np.array_equal(p_x_range, [0., 0.9])
    # ...check y range
    assert np.array_equal(p_y_range, [-0.05, 1.25])

    l_data, l_colour, l_alpha, l_label, l_width = futv.get_line_data(
        axis.lines[0])
    assert np.array_equal(np.stack(dist, axis=1), l_data)
    assert l_colour == 'royalblue'
    assert l_alpha == 0.6
    assert l_width == 3.0
    plt.close(fig=fig)

    # Horizontal plotting for use in 2-D ICE and PD
    dist  = (np.array([0., .2, .4, .6, .8, 1.]), 
             np.array([0.1, 0.2, 0.5, 0.1, 0.1]))
    # Without passing axis
    fig, axis = fvfi.plot_feature_distribution(dist, False, 'feat',
                                               orientation='horizontal')
    assert isinstance(fig, plt.Figure)
    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        axis)
    # ...check title
    assert p_title == 'Feature Distribution for feat'
    # ...check x range
    assert np.array_equal(p_x_range, [0., 1.05])
    # ...check x label
    assert p_x_label == 'feat'
    # ...check y label
    assert p_y_label == 'Histogram of values for feat'
    # ...check y range
    assert np.array_equal(p_y_range, [0, 1.])
    bars = axis.patches
    texts = axis.texts
    assert len(bars) == dist[0].shape[0] - 1
    assert len(texts) == dist[0].shape[0] - 1

    heights = np.array([bar.get_width() for bar in bars])
    alphas = np.array([bar.get_alpha() for bar in bars])
    colours = np.array([bar.get_facecolor() for bar in bars])
    correct_colours = np.repeat(
        np.array([0.25, 0.41, 0.88, 0.6])[np.newaxis, :], len(bars), axis=0)
    labels = np.array([text.get_text() for text in axis.get_xticklabels()])
    text_positions = np.array([text.get_position() for text in texts])
    correct_text_positions = np.stack([dist[1],
                                       dist[0][:-1]+0.1], axis=1)
    text_string = [text.get_text() for text in texts]
    correct_text_string = ['%.2f'%num for num in dist[1]]
    assert np.array_equal(dist[1], heights)
    assert np.array_equal(np.array([0.6]*len(dist[1])), alphas)
    assert np.allclose(correct_colours, colours, atol=1e-2)
    assert np.array_equal(correct_text_positions, text_positions)
    assert text_string == correct_text_string
    plt.close(fig=fig) 

    # Numerical feature histogram
    fig, axis = plt.subplots()
    _, axis = fvfi.plot_feature_distribution(dist, False, None, 'horizontal',
                                             axis)
    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        axis)
    # ...check title
    assert p_title == ''
    # ...check y range
    assert np.array_equal(p_y_range, [0., 1.,])
    # ...check x range
    assert np.array_equal(p_x_range, [0., 1.05])
    bars = axis.patches
    texts = axis.texts
    assert len(bars) == dist[0].shape[0] - 1
    assert len(texts) == dist[0].shape[0] - 1
    assert np.array_equal(dist[1], heights)
    assert np.array_equal(np.array([0.6]*len(dist[1])), alphas)
    assert np.allclose(correct_colours, colours, atol=1e-2)
    assert np.array_equal(correct_text_positions, text_positions)
    assert text_string == correct_text_string
    plt.close(fig=fig)

    # Numerical Categorical feature
    fig, axis = plt.subplots()
    dist = (np.array([0, 0.2, 0.4, 0.6, 0.8, 1]),
            np.array([0.1, 0.1, 0.3, 0.2, 0.2, 0.1]))
    _, axis = fvfi.plot_feature_distribution(dist, True, None, 'horizontal',
                                             axis)
    y_ticklabels = [x.get_text() for x in axis.get_yticklabels()]
    assert y_ticklabels == ['%.1f'%num for num in dist[0]]

    bars = axis.patches
    texts = axis.texts
    assert len(bars) == dist[0].shape[0]
    assert len(texts) == dist[0].shape[0]

    heights = np.array([bar.get_width() for bar in bars])
    alphas = np.array([bar.get_alpha() for bar in bars])
    colours = np.array([bar.get_facecolor() for bar in bars])
    correct_colours = np.repeat(
        np.array([0.25, 0.41, 0.88, 0.6])[np.newaxis, :], len(bars), axis=0)
    labels = np.array([text.get_text() for text in axis.get_xticklabels()])
    text_positions = np.array([text.get_position() for text in texts])
    x_ticks = np.linspace(0, FAKE_LINESPACE.shape[0], len(FAKE_LINESPACE))
    correct_text_positions = np.stack([dist[1],
                                       x_ticks], axis=1)
    text_string = [text.get_text() for text in texts]
    correct_text_string = ['%.2f'%num for num in dist[1]]
    assert np.array_equal(dist[1], heights)
    assert np.array_equal(np.array([0.6]*len(dist[1])), alphas)
    assert np.allclose(correct_colours, colours, atol=1e-2)
    assert np.array_equal(correct_text_positions, text_positions)
    assert text_string == correct_text_string
    plt.close(fig=fig) 

    # Numerical feature kde
    fig, axis = plt.subplots()
    dist = (np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
            np.array([0.1, 0.2, 0.4, 0.4, 0.3, 0.2, 0.1, 0.05, 0.05, 0.05]))
    _, axis = fvfi.plot_feature_distribution(dist, False, None, 'horizontal', axis)
    assert len(axis.lines) == 1

    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        axis)
    # ...check title
    assert p_title == ''
    assert p_y_label == 'KDE fit to feature'
    # ...check y range
    assert np.array_equal(p_y_range, [0., 0.9])
    # ...check x range
    assert np.array_equal(p_x_range, [-0.05, 1.05])

    l_data, l_colour, l_alpha, l_label, l_width = futv.get_line_data(
        axis.lines[0])
    assert np.array_equal(np.stack(dist[::-1], axis=1), l_data)
    assert l_colour == 'royalblue'
    assert l_alpha == 0.6
    assert l_width == 3.0
    plt.close(fig=fig)

    # Categorical feature
    fig, axis = plt.subplots()
    dist = (np.array(['a', 'b', 'c', 'd', 'e', 'f']),
            np.array([0.1, 0.1, 0.3, 0.2, 0.2, 0.1]))
    _, axis = fvfi.plot_feature_distribution(dist, True, None, 'horizontal',
                                             axis)
    y_ticklabels = [x.get_text() for x in axis.get_yticklabels()]
    assert y_ticklabels == list(dist[0])
    bars = axis.patches
    texts = axis.texts
    assert len(bars) == dist[0].shape[0]
    assert len(texts) == dist[0].shape[0]

    heights = np.array([bar.get_width() for bar in bars])
    alphas = np.array([bar.get_alpha() for bar in bars])
    colours = np.array([bar.get_facecolor() for bar in bars])
    correct_colours = np.repeat(
        np.array([0.25, 0.41, 0.88, 0.6])[np.newaxis, :], len(bars), axis=0)
    labels = np.array([text.get_text() for text in axis.get_xticklabels()])
    text_positions = np.array([text.get_position() for text in texts])
    x_ticks = np.linspace(0, FAKE_LINESPACE.shape[0], len(FAKE_LINESPACE))
    correct_text_positions = np.stack([dist[1],
                                       x_ticks], axis=1)
    text_string = [text.get_text() for text in texts]
    correct_text_string = ['%.2f'%num for num in dist[1]]
    assert np.array_equal(dist[1], heights)
    assert np.array_equal(np.array([0.6]*len(dist[1])), alphas)
    assert np.allclose(correct_colours, colours, atol=1e-2)
    assert np.array_equal(correct_text_positions, text_positions)
    assert text_string == correct_text_string
    plt.close(fig=fig)


def test_plot_individual_conditional_expectation():
    """
    Tests ICE plotting.

    Tests
    :func:`fatf.vis.feature_influence.plot_individual_conditional_expectation`
    function.
    """
    feature_name = 'some feature'
    class_index = 1
    class_name = 'middle'

    figure, axis = fvfi.plot_individual_conditional_expectation(
        FAKE_ICE_ARRAY, FAKE_LINESPACE, class_index, False, feature_name, 
        class_name)

    assert isinstance(figure, plt.Figure)
    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        axis)
    # ...check title
    assert p_title == 'Individual Conditional Expectation'
    # ...check x range
    assert np.array_equal(p_x_range, [FAKE_LINESPACE[0], FAKE_LINESPACE[-1]])
    # ...check x label
    assert p_x_label == feature_name
    # ...check y range
    assert np.array_equal(p_y_range, [-0.05, 1.05])
    # ...check y label
    assert p_y_label == '{} class probability'.format(class_name)

    # Test the line
    assert len(axis.collections) == 1
    l_data, l_colour, l_alpha, l_label, l_width = futv.get_line_data(
        axis.collections[0], is_collection=True)
    assert len(l_data) == FAKE_ICE_ARRAY.shape[0]
    for i, line_array in enumerate(l_data):
        line_data = np.stack(
            [FAKE_LINESPACE, FAKE_ICE_ARRAY[i, :, class_index]], axis=1)
        assert np.array_equal(line_array, line_data)
    assert np.isclose(
        l_colour, np.array([[0.412, 0.412, 0.412, 0.5]]),
        atol=1e-2).all()  # dimgray mapping apparently
    assert l_alpha == 0.5
    assert l_label == 'ICE'
    assert l_width == 1.75

    # Validate plot legend
    legend = [
        i for i in axis.get_children()
        if isinstance(i, matplotlib.legend.Legend)
    ]
    assert len(legend) == 1
    legend_texts = legend[0].get_texts()
    assert len(legend_texts) == 1
    assert legend_texts[0].get_text() == 'ICE'
    plt.close(fig=figure) 

    # Test Categorical ICE
    figure, axis = fvfi.plot_individual_conditional_expectation(
        FAKE_ICE_ARRAY, FAKE_LINESPACE, class_index, True, feature_name, 
        class_name)

    assert isinstance(figure, plt.Figure)
    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        axis)
    # ...check title
    assert p_title == 'Individual Conditional Expectation'
    # ...check x range
    assert np.array_equal(p_x_range, [-0.5, len(FAKE_LINESPACE)+0.5])
    # ...check x label
    assert p_x_label == feature_name
    # ...check y range
    assert np.array_equal(p_y_range, [-0.05, 1.05])
    # ...check y label
    assert p_y_label == '{} class probability'.format(class_name)

    # Test violins
    x_ticklabels = np.array([tick.get_text() for tick in axis.get_xticklabels()])
    assert np.array_equal(x_ticklabels, FAKE_LINESPACE_STRING)
    polygons, lines = axis.collections[:6], axis.collections[6:]
    correct_lines = [
        [np.array([[-0.125, 1.],[0.125, 1.]]), 
         np.array([[1.075, 0.9], [1.325, 0.9]]), 
         np.array([[2.275, 1.], [2.525, 1.]]), 
         np.array([[3.475, 0.9], [3.725, 0.9]]),
         np.array([[4.675, 0.8], [4.925, 0.8]]),
         np.array([[5.875, 0.7],[6.125, 0.7]])],
        [np.array([[-0.125, 0.2], [0.125,  0.2]]),
         np.array([[1.075, 0.3], [1.325, 0.3]]),
         np.array([[2.275, 0.2], [2.525, 0.2]]),
         np.array([[3.475, 0.1], [3.725, 0.1]]),
         np.array([[4.675, 0.], [4.925, 0.]]),
         np.array([[5.875, 0.1], [6.125, 0.1]])],
        [np.array([[0., 0.2], [0., 1.]]),
        np.array([[1.2, 0.3], [1.2, 0.9]]),
        np.array([[2.4, 0.2], [2.4, 1.]]),
        np.array([[3.6, 0.1], [3.6, 0.9]]),
        np.array([[4.8, 0.], [4.8, 0.8]]),
        np.array([[6., 0.1], [6., 0.7]])]]
    for line in zip(lines, correct_lines):
        l_data, l_colour, l_alpha, l_label, l_width = futv.get_line_data(
            line[0], is_collection=True)
        assert np.allclose(l_data, line[1])
        assert np.allclose(l_colour, np.array([0.25, 0.41, 0.88, 1.]), 
                           atol=1e-2)
        assert l_width == 1.75
    plt.close(fig=figure) 

    # Test categorical and string linespace
    figure, axis = fvfi.plot_individual_conditional_expectation(
        FAKE_ICE_ARRAY, FAKE_LINESPACE_CAT, class_index, True, feature_name, 
        class_name)

    assert isinstance(figure, plt.Figure)
    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        axis)
    # ...check title
    assert p_title == 'Individual Conditional Expectation'
    # ...check x range
    assert np.array_equal(p_x_range, [-0.5, len(FAKE_LINESPACE_CAT)+0.5])
    # ...check x label
    assert p_x_label == feature_name
    # ...check y range
    assert np.array_equal(p_y_range, [-0.05, 1.05])
    # ...check y label
    assert p_y_label == '{} class probability'.format(class_name)

    # Test violins
    labels = np.array([text.get_text() for text in axis.get_xticklabels()])
    assert np.array_equal(FAKE_LINESPACE_CAT, labels)
    polygons, lines = axis.collections[:6], axis.collections[6:]
    correct_lines = [
        [np.array([[-0.125, 1.],[0.125, 1.]]), 
         np.array([[1.075, 0.9], [1.325, 0.9]]), 
         np.array([[2.275, 1.], [2.525, 1.]]), 
         np.array([[3.475, 0.9], [3.725, 0.9]]),
         np.array([[4.675, 0.8], [4.925, 0.8]]),
         np.array([[5.875, 0.7],[6.125, 0.7]])],
        [np.array([[-0.125, 0.2], [0.125,  0.2]]),
         np.array([[1.075, 0.3], [1.325, 0.3]]),
         np.array([[2.275, 0.2], [2.525, 0.2]]),
         np.array([[3.475, 0.1], [3.725, 0.1]]),
         np.array([[4.675, 0.], [4.925, 0.]]),
         np.array([[5.875, 0.1], [6.125, 0.1]])],
        [np.array([[0., 0.2], [0., 1.]]),
        np.array([[1.2, 0.3], [1.2, 0.9]]),
        np.array([[2.4, 0.2], [2.4, 1.]]),
        np.array([[3.6, 0.1], [3.6, 0.9]]),
        np.array([[4.8, 0.], [4.8, 0.8]]),
        np.array([[6., 0.1], [6., 0.7]])]]
    for line in zip(lines, correct_lines):
        l_data, l_colour, l_alpha, l_label, l_width = futv.get_line_data(
            line[0], is_collection=True)
        assert np.allclose(l_data, line[1])
        assert np.allclose(l_colour, np.array([0.25, 0.41, 0.88, 1.]), 
                           atol=1e-2)
        assert l_width == 1.75
    plt.close(fig=figure)

    # Test for string data being treated as not categorical
    msg = ('Selected feature is categorical (string-base elements), however '
           'the treat_as_categorical was set to False. Such a combination is '
           'not possible. The feature will be treated as categorical.')
    with pytest.warns(UserWarning) as warning:
        figure, axis = fvfi.plot_individual_conditional_expectation(
            FAKE_ICE_ARRAY, FAKE_LINESPACE_CAT, class_index, False, feature_name,
            class_name)
    assert len(warning) == 1
    assert str(warning[0].message) == msg

    assert isinstance(figure, plt.Figure)
    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        axis)
    # ...check title
    assert p_title == 'Individual Conditional Expectation'
    # ...check x range
    assert np.array_equal(p_x_range, [-0.5, len(FAKE_LINESPACE_CAT)+0.5])
    # ...check x label
    assert p_x_label == feature_name
    # ...check y range
    assert np.array_equal(p_y_range, [-0.05, 1.05])
    # ...check y label
    assert p_y_label == '{} class probability'.format(class_name)

    labels = np.array([text.get_text() for text in axis.get_xticklabels()])
    assert np.array_equal(FAKE_LINESPACE_CAT, labels)
    polygons, lines = axis.collections[:6], axis.collections[6:]
    for line in zip(lines, correct_lines):
        l_data, l_colour, l_alpha, l_label, l_width = futv.get_line_data(
            line[0], is_collection=True)
        assert np.allclose(l_data, line[1])
        assert np.allclose(l_colour, np.array([0.25, 0.41, 0.88, 1.]), 
                           atol=1e-2)
        assert l_width == 1.75
    plt.close(fig=figure)

    # Test with treat_as_categorical=None to infer type
    figure, axis = fvfi.plot_individual_conditional_expectation(
            FAKE_ICE_ARRAY, FAKE_LINESPACE_CAT, class_index, None, feature_name,
            class_name)
    assert len(warning) == 1
    assert str(warning[0].message) == msg

    assert isinstance(figure, plt.Figure)
    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        axis)
    # ...check title
    assert p_title == 'Individual Conditional Expectation'
    # ...check x range
    assert np.array_equal(p_x_range, [-0.5, len(FAKE_LINESPACE_CAT)+0.5])
    # ...check x label
    assert p_x_label == feature_name
    # ...check y range
    assert np.array_equal(p_y_range, [-0.05, 1.05])
    # ...check y label
    assert p_y_label == '{} class probability'.format(class_name)

    labels = np.array([text.get_text() for text in axis.get_xticklabels()])
    assert np.array_equal(FAKE_LINESPACE_CAT, labels)
    polygons, lines = axis.collections[:6], axis.collections[6:]
    for line in zip(lines, correct_lines):
        l_data, l_colour, l_alpha, l_label, l_width = futv.get_line_data(
            line[0], is_collection=True)
        assert np.allclose(l_data, line[1])
        assert np.allclose(l_colour, np.array([0.25, 0.41, 0.88, 1.]), 
                           atol=1e-2)
        assert l_width == 1.75
    plt.close(fig=figure)

    # Test ICE with plot_distribution (just simple histogram)
    dist  = (np.array([0., .2, .4, .6, .8, 1.]), 
             np.array([0.1, 0.2, 0.5, 0.1, 0.1]))
    figure, (axis, dist_axis) = fvfi.plot_individual_conditional_expectation(
        FAKE_ICE_ARRAY, FAKE_LINESPACE, class_index, False, feature_name, 
        class_name, dist, None)

    assert isinstance(figure, plt.Figure)
    # ICE-axis
    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        axis)
    # ...check title
    assert p_title == 'Individual Conditional Expectation'
    # ...check x range
    assert np.array_equal(p_x_range, [FAKE_LINESPACE[0], FAKE_LINESPACE[-1]])
    # ...check x label
    assert p_x_label == feature_name
    # ...check y range
    assert np.array_equal(p_y_range, [-0.05, 1.05])
    # ...check y label
    assert p_y_label == '{} class probability'.format(class_name)

    # Test the line
    assert len(axis.collections) == 1
    l_data, l_colour, l_alpha, l_label, l_width = futv.get_line_data(
        axis.collections[0], is_collection=True)
    assert len(l_data) == FAKE_ICE_ARRAY.shape[0]
    for i, line_array in enumerate(l_data):
        line_data = np.stack(
            [FAKE_LINESPACE, FAKE_ICE_ARRAY[i, :, class_index]], axis=1)
        assert np.array_equal(line_array, line_data)
    assert np.isclose(
        l_colour, np.array([[0.412, 0.412, 0.412, 0.5]]),
        atol=1e-2).all()  # dimgray mapping apparently
    assert l_alpha == 0.5
    assert l_label == 'ICE'
    assert l_width == 1.75

    # Validate plot legend
    legend = [
        i for i in axis.get_children()
        if isinstance(i, matplotlib.legend.Legend)
    ]
    assert len(legend) == 1
    legend_texts = legend[0].get_texts()
    assert len(legend_texts) == 1
    assert legend_texts[0].get_text() == 'ICE'

    # Test plot_distribution axis
    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        dist_axis)
    # ...check title
    assert p_title == ''
    # ...check x range
    assert np.array_equal(p_x_range, [FAKE_LINESPACE[0], FAKE_LINESPACE[-1]])
    # ...check x label
    assert p_x_label == 'Histogram of values for {}'.format(feature_name)
    # ...check y range
    assert np.array_equal(p_y_range, [0., 1.05])
    # ...check y label
    assert p_y_label == ''

    # Test Histogram plot
    bars = dist_axis.patches
    texts = dist_axis.texts
    assert len(bars) == dist[0].shape[0] - 1
    assert len(texts) == dist[0].shape[0] - 1

    heights = np.array([bar.get_height() for bar in bars])
    alphas = np.array([bar.get_alpha() for bar in bars])
    colours = np.array([bar.get_facecolor() for bar in bars])
    correct_colours = np.repeat(
        np.array([0.25, 0.41, 0.88, 0.6])[np.newaxis, :], len(bars), axis=0)
    labels = np.array([text.get_text() for text in dist_axis.get_xticklabels()])
    text_positions = np.array([text.get_position() for text in texts])
    correct_text_positions = np.stack([dist[0][:-1]+0.1,
                                       dist[1]], axis=1)
    text_string = [text.get_text() for text in texts]
    correct_text_string = ['%.2f'%num for num in dist[1]]
    assert np.array_equal(dist[1], heights)
    assert np.array_equal(np.array([0.6]*len(dist[1])), alphas)
    assert np.allclose(correct_colours, colours, atol=1e-2)
    assert np.array_equal(correct_text_positions, text_positions)
    assert text_string == correct_text_string
    plt.close(fig=figure)

    # Test ICE with categorical plot_distribution (to test reording of 
    # plot_distribution to match the order of feature_linespace)
    dist  = (np.array(['c', 'b', 'a', 'd', 'e', 'f']), 
             np.array([0.5, 0.2, 0.1, 0.1, 0.1, 0.1]))
    figure, (axis, dist_axis) = fvfi.plot_individual_conditional_expectation(
        FAKE_ICE_ARRAY, FAKE_LINESPACE_CAT, class_index, True, feature_name, 
        class_name, dist, None)
    correct_dist = [np.array(['a', 'b', 'c', 'd', 'e', 'f']), 
                    np.array([0.1, 0.2, 0.5, 0.1, 0.1, 0.1])]
    assert isinstance(figure, plt.Figure)
    # ICE-axis
    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        axis)
    # ...check title
    assert p_title == 'Individual Conditional Expectation'
    # ...check x range
    assert np.array_equal(p_x_range, [-0.5, 6.5])
    # ...check x label
    assert p_x_label == feature_name
    # ...check y range
    assert np.array_equal(p_y_range, [-0.05, 1.05])
    # ...check y label
    assert p_y_label == '{} class probability'.format(class_name)

    # Test violins
    labels = np.array([text.get_text() for text in axis.get_xticklabels()])
    assert np.array_equal(FAKE_LINESPACE_CAT, labels)
    polygons, lines = axis.collections[:6], axis.collections[6:]
    for line in zip(lines, correct_lines):
        l_data, l_colour, l_alpha, l_label, l_width = futv.get_line_data(
            line[0], is_collection=True)
        assert np.allclose(l_data, line[1])
        assert np.allclose(l_colour, np.array([0.25, 0.41, 0.88, 1.]), 
                           atol=1e-2)
        assert l_width == 1.75

    # Test plot_distribution axis
    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        dist_axis)
    # ...check title
    assert p_title == ''
    # ...check x range
    assert np.array_equal(p_x_range, [-0.5, 6.5])
    # ...check x label
    assert p_x_label == 'Histogram of values for {}'.format(feature_name)
    # ...check y range
    assert np.array_equal(p_y_range, [0., 1.05])
    # ...check y label
    assert p_y_label == ''

    # Test Histogram plot
    bars = dist_axis.patches
    texts = dist_axis.texts
    assert len(bars) == dist[0].shape[0]
    assert len(texts) == dist[0].shape[0]

    heights = np.array([bar.get_height() for bar in bars])
    alphas = np.array([bar.get_alpha() for bar in bars])
    colours = np.array([bar.get_facecolor() for bar in bars])
    correct_colours = np.repeat(
        np.array([0.25, 0.41, 0.88, 0.6])[np.newaxis, :], len(bars), axis=0)
    labels = np.array([text.get_text() for text in dist_axis.get_xticklabels()])
    text_positions = np.array([text.get_position() for text in texts])
    correct_text_positions = np.stack([[0, 1.2, 2.4, 3.6, 4.8, 6],
                                      correct_dist[1]], axis=1)
    text_string = [text.get_text() for text in texts]
    correct_text_string = ['%.2f'%num for num in correct_dist[1]]
    assert np.array_equal(correct_dist[1], heights)
    assert np.array_equal(np.array([0.6]*len(dist[1])), alphas)
    assert np.allclose(correct_colours, colours, atol=1e-2)
    assert np.allclose(correct_text_positions, text_positions, atol=1e-2)
    assert text_string == correct_text_string
    plt.close(fig=figure)


def test_plot_partial_dependence():
    """
    Tests :func:`fatf.vis.feature_influence.plot_partial_dependence` function.
    """
    feature_name = 'some feature'
    class_index = 1
    class_name = 'middle'

    # Test with variance plot
    figure, axis = fvfi.plot_partial_dependence(
        FAKE_PD_ARRAY, FAKE_LINESPACE, class_index, False, FAKE_VARIANCE, False,
        feature_name, class_name)

    assert isinstance(figure, plt.Figure)
    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        axis)
    # ...check title
    assert p_title == 'Partial Dependence'
    # ...check x range
    assert np.array_equal(p_x_range, [FAKE_LINESPACE[0], FAKE_LINESPACE[-1]])
    # ...check x label
    assert p_x_label == feature_name
    # ...check y range
    assert np.array_equal(p_y_range, [-0.05, 1.05])
    # ...check y label
    assert p_y_label == '{} class probability'.format(class_name)

    # Test the line
    assert len(axis.lines) == 3
    l_data, l_colour, l_alpha, l_label, l_width = futv.get_line_data(
        axis.lines[0])
    line_data = np.stack([FAKE_LINESPACE, FAKE_PD_ARRAY[:, class_index]],
                         axis=1)
    assert np.array_equal(l_data, line_data)
    assert l_colour == 'lightsalmon'
    assert l_alpha == 0.6
    assert l_label == 'PD'
    assert l_width == 7

    # Test variance error bar lines
    line_data = np.stack([FAKE_LINESPACE, FAKE_PD_ARRAY[:, class_index] -
                              FAKE_VARIANCE[:, class_index]], axis=1)
    l_data, l_colour, l_alpha, l_label, l_width = futv.get_line_data(
        axis.lines[1])
    assert np.array_equal(l_data, line_data)
    assert l_colour == 'lightsalmon'
    assert l_alpha == 0.6
    assert l_label == '_nolegend_'
    assert l_width == 1.75

    line_data = np.stack([FAKE_LINESPACE, FAKE_PD_ARRAY[:, class_index] +
                              FAKE_VARIANCE[:, class_index]], axis=1)
    l_data, l_colour, l_alpha, l_label, l_width = futv.get_line_data(
        axis.lines[2])
    assert np.array_equal(l_data, line_data)
    assert l_colour == 'lightsalmon'
    assert l_alpha == 0.6
    assert l_label == '_nolegend_'
    assert l_width == 1.75

    assert len(axis.collections) == 1
    # Test variance line collection
    l_data, l_colour, l_alpha, l_label, l_width = futv.get_line_data(
        axis.collections[0], is_collection=True)
    line_data = np.stack([
        np.stack([FAKE_LINESPACE, FAKE_PD_ARRAY[:, class_index] -
                  FAKE_VARIANCE[:, class_index]], axis=1),
        np.stack([FAKE_LINESPACE, FAKE_PD_ARRAY[:, class_index] +
                  FAKE_VARIANCE[:, class_index]], axis=1)])
    l_data = np.stack(l_data, axis=1)
    assert np.array_equal(l_data, line_data)
    assert np.allclose(l_colour, np.array([1., 0.62, 0.47, 0.6]), atol=1e-2)
    assert l_alpha == 0.6
    assert l_label == '_nolegend_'
    assert l_width == 2

    # Validate plot legend
    legend = [
        i for i in axis.get_children()
        if isinstance(i, matplotlib.legend.Legend)
    ]
    assert len(legend) == 1
    legend_texts = legend[0].get_texts()
    assert len(legend_texts) == 2
    assert legend_texts[0].get_text() == 'PD'
    assert legend_texts[1].get_text() == 'Variance'
    plt.close(fig=figure)

    # Test with variance area plot
    figure, axis = fvfi.plot_partial_dependence(
        FAKE_PD_ARRAY, FAKE_LINESPACE, class_index, False, FAKE_VARIANCE, True,
        feature_name, class_name)

    assert isinstance(figure, plt.Figure)
    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        axis)
    # ...check title
    assert p_title == 'Partial Dependence'
    # ...check x range
    assert np.array_equal(p_x_range, [FAKE_LINESPACE[0], FAKE_LINESPACE[-1]])
    # ...check x label
    assert p_x_label == feature_name
    # ...check y range
    assert np.array_equal(p_y_range, [-0.05, 1.05])
    # ...check y label
    assert p_y_label == '{} class probability'.format(class_name)

    # Test the line
    assert len(axis.lines) == 1
    l_data, l_colour, l_alpha, l_label, l_width = futv.get_line_data(
        axis.lines[0])
    line_data = np.stack([FAKE_LINESPACE, FAKE_PD_ARRAY[:, class_index]],
                         axis=1)
    assert np.array_equal(l_data, line_data)
    assert l_colour == 'lightsalmon'
    assert l_alpha == 0.6
    assert l_label == 'PD'
    assert l_width == 7

    assert len(axis.collections) == 1
    # Test variance PolyCollection
    assert np.allclose(axis.collections[0].get_facecolor(), 
                       np.array([1., 0.62, 0.47, 0.3]), atol=1e-2)
    assert np.allclose(axis.collections[0].get_edgecolor(), 
                       np.array([1., 0.62, 0.47, 0.3]), atol=1e-2)
    assert axis.collections[0].get_alpha() == 0.3
    assert axis.collections[0].get_label() == 'Variance'

    # Test the areas
    path = axis.collections[0].get_paths()
    assert len(path) == 1
    assert np.array_equal(path[0].codes, 
                          np.array([1]+[2]*(FAKE_LINESPACE.shape[0]*2+1)+[79]))
    # Construct the polygon path
    first_point = np.array([FAKE_LINESPACE[0], FAKE_PD_ARRAY[0, class_index] + 
                                FAKE_VARIANCE[0, class_index]]).reshape(1, 2)
    last_point = np.array([FAKE_LINESPACE[-1], FAKE_PD_ARRAY[-1, class_index] + 
                           FAKE_VARIANCE[-1, class_index]]).reshape(1, 2)
    var_below = np.stack([FAKE_LINESPACE, FAKE_PD_ARRAY[:, class_index] -
                          FAKE_VARIANCE[:, class_index]], axis=1)
    var_above = np.stack([FAKE_LINESPACE, FAKE_PD_ARRAY[:, class_index] +
                          FAKE_VARIANCE[:, class_index]], axis=1)
    correct_polygon = np.concatenate(
        [first_point, var_below, last_point, np.flip(var_above, 0), 
        first_point], axis=0)
    assert np.array_equal(path[0].vertices, correct_polygon)

    # Validate plot legend
    legend = [
        i for i in axis.get_children()
        if isinstance(i, matplotlib.legend.Legend)
    ]
    assert len(legend) == 1
    legend_texts = legend[0].get_texts()
    assert len(legend_texts) == 2
    assert legend_texts[0].get_text() == 'PD'
    assert legend_texts[1].get_text() == 'Variance'
    plt.close(fig=figure)

    # Test without variance plot
    figure, axis = fvfi.plot_partial_dependence(
        FAKE_PD_ARRAY, FAKE_LINESPACE, class_index, False, None,
        False, feature_name, class_name)

    assert isinstance(figure, plt.Figure)
    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        axis)
    # ...check title

    assert p_title == 'Partial Dependence'
    # ...check x range
    assert np.array_equal(p_x_range, [FAKE_LINESPACE[0], FAKE_LINESPACE[-1]])
    # ...check x label
    assert p_x_label == feature_name
    # ...check y range
    assert np.array_equal(p_y_range, [-0.05, 1.05])
    # ...check y label
    assert p_y_label == '{} class probability'.format(class_name)

    # Test the line
    assert len(axis.lines) == 1
    l_data, l_colour, l_alpha, l_label, l_width = futv.get_line_data(
        axis.lines[0])
    line_data = np.stack([FAKE_LINESPACE, FAKE_PD_ARRAY[:, class_index]],
                         axis=1)
    assert np.array_equal(l_data, line_data)
    assert l_colour == 'lightsalmon'
    assert l_alpha == 0.6
    assert l_label == 'PD'
    assert l_width == 7

    # Validate plot legend
    legend = [
        i for i in axis.get_children()
        if isinstance(i, matplotlib.legend.Legend)
    ]
    assert len(legend) == 1
    legend_texts = legend[0].get_texts()
    assert len(legend_texts) == 1
    assert legend_texts[0].get_text() == 'PD'
    plt.close(fig=figure)

    # Test categorical numerical data without variance
    figure, axis = fvfi.plot_partial_dependence(
        FAKE_PD_ARRAY, FAKE_LINESPACE, class_index, True, None, 
        False, feature_name, class_name)
    assert isinstance(figure, plt.Figure)
    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        axis)
    # ...check title
    assert p_title == 'Partial Dependence'
    # ...check x range
    assert np.array_equal(p_x_range, [-0.5, len(FAKE_LINESPACE)+0.5])
    # ...check x label
    assert p_x_label == feature_name
    # ...check y range
    assert np.array_equal(p_y_range, [0, 1.05])
    # ...check y label
    assert p_y_label == '{} class probability'.format(class_name)

    # Test bar plots
    bars = axis.patches
    heights = np.array([bar.get_height() for bar in bars])
    alphas = np.array([bar.get_alpha() for bar in bars])
    colours = np.array([bar.get_facecolor() for bar in bars])
    correct_colours = np.repeat(
        np.array([0.25, 0.41, 0.88, 0.6])[np.newaxis, :], len(bars), axis=0)
    labels = [text.get_text() for text in axis.get_xticklabels()]
    assert np.array_equal(FAKE_LINESPACE_STRING, labels)
    assert np.array_equal(FAKE_PD_ARRAY[:, 1], heights)
    assert np.array_equal(np.array([0.6]*len(bars)), alphas)
    assert np.allclose(correct_colours, colours, atol=1e-2)
    plt.close(fig=figure)

    # Test categorical string data without variance
    figure, axis = fvfi.plot_partial_dependence(
        FAKE_PD_ARRAY, FAKE_LINESPACE_CAT, class_index, True, None, 
        False, feature_name, class_name)
    assert isinstance(figure, plt.Figure)
    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        axis)
    # ...check title
    assert p_title == 'Partial Dependence'
    # ...check x range
    assert np.array_equal(p_x_range, [-0.5, len(FAKE_LINESPACE_CAT)+0.5])
    # ...check x label
    assert p_x_label == feature_name
    # ...check y range
    assert np.array_equal(p_y_range, [0, 1.05])
    # ...check y label
    assert p_y_label == '{} class probability'.format(class_name)

    # Test bar plots
    bars = axis.patches
    heights = np.array([bar.get_height() for bar in bars])
    alphas = np.array([bar.get_alpha() for bar in bars])
    colours = np.array([bar.get_facecolor() for bar in bars])
    correct_colours = np.repeat(
        np.array([0.25, 0.41, 0.88, 0.6])[np.newaxis, :], len(bars), axis=0)
    labels = np.array([text.get_text() for text in axis.get_xticklabels()])
    assert np.array_equal(FAKE_LINESPACE_CAT, labels)
    assert np.array_equal(FAKE_PD_ARRAY[:, 1], heights)
    assert np.array_equal(np.array([0.6]*len(bars)), alphas)
    assert np.allclose(correct_colours, colours, atol=1e-2)
    plt.close(fig=figure)

    # Test categorical data with variance
    figure, axis = fvfi.plot_partial_dependence(
        FAKE_PD_ARRAY, FAKE_LINESPACE_CAT, class_index, True, FAKE_VARIANCE, 
        False, feature_name, class_name)
    assert isinstance(figure, plt.Figure)
    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        axis)
    # ...check title
    assert p_title == 'Partial Dependence'
    # ...check x range
    assert np.array_equal(p_x_range, [-0.5, len(FAKE_LINESPACE_CAT)+0.5])
    # ...check x label
    assert p_x_label == feature_name
    # ...check y range
    assert np.array_equal(p_y_range, [0, 1.05])
    # ...check y label
    assert p_y_label == '{} class probability'.format(class_name)

    # Test bar plots
    bars = axis.patches
    heights = np.array([bar.get_height() for bar in bars])
    alphas = np.array([bar.get_alpha() for bar in bars])
    colours = np.array([bar.get_facecolor() for bar in bars])
    correct_colours = np.repeat(
        np.array([0.25, 0.41, 0.88, 0.6])[np.newaxis, :], len(bars), axis=0)
    labels = np.array([text.get_text() for text in axis.get_xticklabels()])
    assert np.array_equal(FAKE_LINESPACE_CAT, labels)
    assert np.array_equal(FAKE_PD_ARRAY[:, 1], heights)
    assert np.array_equal(np.array([0.6]*len(bars)), alphas)
    assert np.allclose(correct_colours, colours, atol=1e-2)

    assert len(axis.collections) == 1
    # Test variance line collection
    l_data, l_colour, l_alpha, l_label, l_width = futv.get_line_data(
        axis.collections[0], is_collection=True)
    bar_locs = np.linspace(0, FAKE_LINESPACE_CAT.shape[0], 
                           len(FAKE_LINESPACE_CAT))
    line_data = np.stack([
        np.stack([bar_locs, FAKE_PD_ARRAY[:, class_index] -
                  FAKE_VARIANCE[:, class_index]], axis=1),
        np.stack([bar_locs, FAKE_PD_ARRAY[:, class_index] +
                  FAKE_VARIANCE[:, class_index]], axis=1)])
    l_data = np.stack(l_data, axis=1)
    assert np.array_equal(l_data, line_data)
    assert np.allclose(l_colour, np.array([0., 0., 0., 1.]), atol=1e-2)
    assert l_label == '_nolegend_'
    assert l_width == 1.75
    plt.close(fig=figure)

    # Test categorical data with treat_as_categorical = False
    msg = ('Selected feature is categorical (string-base elements), however '
           'the treat_as_categorical was set to False. Such a combination is '
           'not possible. The feature will be treated as categorical.')
    with pytest.warns(UserWarning) as warning:
        figure, axis = fvfi.plot_partial_dependence(
            FAKE_PD_ARRAY, FAKE_LINESPACE_CAT, class_index, False, None, 
            False, feature_name, class_name)
    assert len(warning) == 1
    assert str(warning[0].message) == msg

    assert isinstance(figure, plt.Figure)
    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        axis)
    # ...check title
    assert p_title == 'Partial Dependence'
    # ...check x range
    assert np.array_equal(p_x_range, [-0.5, len(FAKE_LINESPACE_CAT)+0.5])
    # ...check x label
    assert p_x_label == feature_name
    # ...check y range
    assert np.array_equal(p_y_range, [0, 1.05])
    # ...check y label
    assert p_y_label == '{} class probability'.format(class_name)

    # Test bar plots
    bars = axis.patches
    heights = np.array([bar.get_height() for bar in bars])
    alphas = np.array([bar.get_alpha() for bar in bars])
    colours = np.array([bar.get_facecolor() for bar in bars])
    labels = np.array([text.get_text() for text in axis.get_xticklabels()])
    assert np.array_equal(FAKE_LINESPACE_CAT, labels)
    assert np.array_equal(FAKE_PD_ARRAY[:, 1], heights)
    assert np.array_equal(np.array([0.6]*len(bars)), alphas)
    assert np.allclose(correct_colours, colours, atol=1e-2)
    plt.close(fig=figure)

    # Test with treat_as_categorical=None 
    figure, axis = fvfi.plot_partial_dependence(
            FAKE_PD_ARRAY, FAKE_LINESPACE_CAT, class_index, None, None, 
            False, feature_name, class_name)
    assert len(warning) == 1
    assert str(warning[0].message) == msg

    assert isinstance(figure, plt.Figure)
    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        axis)
    # ...check title
    assert p_title == 'Partial Dependence'
    # ...check x range
    assert np.array_equal(p_x_range, [-0.5, len(FAKE_LINESPACE_CAT)+0.5])
    # ...check x label
    assert p_x_label == feature_name
    # ...check y range
    assert np.array_equal(p_y_range, [0, 1.05])
    # ...check y label
    assert p_y_label == '{} class probability'.format(class_name)

    # Test bar plots
    bars = axis.patches
    heights = np.array([bar.get_height() for bar in bars])
    alphas = np.array([bar.get_alpha() for bar in bars])
    colours = np.array([bar.get_facecolor() for bar in bars])
    labels = np.array([text.get_text() for text in axis.get_xticklabels()])
    assert np.array_equal(FAKE_LINESPACE_CAT, labels)
    assert np.array_equal(FAKE_PD_ARRAY[:, 1], heights)
    assert np.array_equal(np.array([0.6]*len(bars)), alphas)
    assert np.allclose(correct_colours, colours, atol=1e-2)
    plt.close(fig=figure)

    # Test without variance plot
    dist  = (np.array([0., .2, .4, .6, .8, 1.]), 
             np.array([0.1, 0.2, 0.5, 0.1, 0.1]))
    figure, (axis, dist_axis) = fvfi.plot_partial_dependence(
        FAKE_PD_ARRAY, FAKE_LINESPACE, class_index, False, None,
        False, feature_name, class_name, dist)

    assert isinstance(figure, plt.Figure)
    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        axis)
    # ...check title
    assert p_title == 'Partial Dependence'
    # ...check x range
    assert np.array_equal(p_x_range, [FAKE_LINESPACE[0], FAKE_LINESPACE[-1]])
    # ...check x label
    assert p_x_label == feature_name
    # ...check y range
    assert np.array_equal(p_y_range, [-0.05, 1.05])
    # ...check y label
    assert p_y_label == '{} class probability'.format(class_name)

    # Test the line
    assert len(axis.lines) == 1
    l_data, l_colour, l_alpha, l_label, l_width = futv.get_line_data(
        axis.lines[0])
    line_data = np.stack([FAKE_LINESPACE, FAKE_PD_ARRAY[:, class_index]],
                         axis=1)
    assert np.array_equal(l_data, line_data)
    assert l_colour == 'lightsalmon'
    assert l_alpha == 0.6
    assert l_label == 'PD'
    assert l_width == 7

    # Validate plot legend
    legend = [
        i for i in axis.get_children()
        if isinstance(i, matplotlib.legend.Legend)
    ]
    assert len(legend) == 1
    legend_texts = legend[0].get_texts()
    assert len(legend_texts) == 1
    assert legend_texts[0].get_text() == 'PD'
    plt.close(fig=figure)

    # Test plot_distribution axis
    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        dist_axis)
    # ...check title
    assert p_title == ''
    # ...check x range
    assert np.array_equal(p_x_range, [FAKE_LINESPACE[0], FAKE_LINESPACE[-1]])
    # ...check x label
    assert p_x_label == 'Histogram of values for {}'.format(feature_name)
    # ...check y range
    assert np.array_equal(p_y_range, [0., 1.05])
    # ...check y label
    assert p_y_label == ''

    # Test Histogram plot
    bars = dist_axis.patches
    texts = dist_axis.texts
    assert len(bars) == dist[0].shape[0] - 1
    assert len(texts) == dist[0].shape[0] - 1

    heights = np.array([bar.get_height() for bar in bars])
    alphas = np.array([bar.get_alpha() for bar in bars])
    colours = np.array([bar.get_facecolor() for bar in bars])
    correct_colours = np.repeat(
        np.array([0.25, 0.41, 0.88, 0.6])[np.newaxis, :], len(bars), axis=0)
    labels = np.array([text.get_text() for text in dist_axis.get_xticklabels()])
    text_positions = np.array([text.get_position() for text in texts])
    correct_text_positions = np.stack([dist[0][:-1]+0.1,
                                       dist[1]], axis=1)
    text_string = [text.get_text() for text in texts]
    correct_text_string = ['%.2f'%num for num in dist[1]]
    assert np.array_equal(dist[1], heights)
    assert np.array_equal(np.array([0.6]*len(dist[1])), alphas)
    assert np.allclose(correct_colours, colours, atol=1e-2)
    assert np.array_equal(correct_text_positions, text_positions)
    assert text_string == correct_text_string
    plt.close(fig=figure)

    # Test PD with categorical plot_distribution (to test reording of 
    # plot_distribution to match the order of feature_linespace)
    dist  = (np.array(['c', 'b', 'a', 'd', 'e', 'f']), 
             np.array([0.5, 0.2, 0.1, 0.1, 0.1, 0.1]))
    correct_dist = [np.array(['a', 'b', 'c', 'd', 'e', 'f']), 
                    np.array([0.1, 0.2, 0.5, 0.1, 0.1, 0.1])]
    figure, (axis, dist_axis) = fvfi.plot_partial_dependence(
        FAKE_PD_ARRAY, FAKE_LINESPACE_CAT, class_index, True, None, 
        False, feature_name, class_name, dist)
    assert isinstance(figure, plt.Figure)
    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        axis)
    # ...check title
    assert p_title == 'Partial Dependence'
    # ...check x range
    assert np.array_equal(p_x_range, [-0.5, len(FAKE_LINESPACE_CAT)+0.5])
    # ...check x label
    assert p_x_label == feature_name
    # ...check y range
    assert np.array_equal(p_y_range, [0, 1.05])
    # ...check y label
    assert p_y_label == '{} class probability'.format(class_name)

    # Test bar plots
    bars = axis.patches
    heights = np.array([bar.get_height() for bar in bars])
    alphas = np.array([bar.get_alpha() for bar in bars])
    colours = np.array([bar.get_facecolor() for bar in bars])
    correct_colours = np.repeat(
        np.array([0.25, 0.41, 0.88, 0.6])[np.newaxis, :], len(bars), axis=0)
    labels = np.array([text.get_text() for text in axis.get_xticklabels()])
    assert np.array_equal(FAKE_LINESPACE_CAT, labels)
    assert np.array_equal(FAKE_PD_ARRAY[:, 1], heights)
    assert np.array_equal(np.array([0.6]*len(bars)), alphas)
    assert np.allclose(correct_colours, colours, atol=1e-2)

    # Test plot_distribution axis
    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        dist_axis)
    # ...check title
    assert p_title == ''
    # ...check x range
    assert np.array_equal(p_x_range, [-0.5, 6.5])
    # ...check x label
    assert p_x_label == 'Histogram of values for {}'.format(feature_name)
    # ...check y range
    assert np.array_equal(p_y_range, [0., 1.05])
    # ...check y label
    assert p_y_label == ''

    # Test Histogram plot
    bars = dist_axis.patches
    texts = dist_axis.texts
    assert len(bars) == dist[0].shape[0]
    assert len(texts) == dist[0].shape[0]

    heights = np.array([bar.get_height() for bar in bars])
    alphas = np.array([bar.get_alpha() for bar in bars])
    colours = np.array([bar.get_facecolor() for bar in bars])
    correct_colours = np.repeat(
        np.array([0.25, 0.41, 0.88, 0.6])[np.newaxis, :], len(bars), axis=0)
    labels = np.array([text.get_text() for text in dist_axis.get_xticklabels()])
    text_positions = np.array([text.get_position() for text in texts])
    correct_text_positions = np.stack([[0, 1.2, 2.4, 3.6, 4.8, 6],
                                      correct_dist[1]], axis=1)
    text_string = [text.get_text() for text in texts]
    correct_text_string = ['%.2f'%num for num in correct_dist[1]]
    assert np.array_equal(correct_dist[1], heights)
    assert np.array_equal(np.array([0.6]*len(dist[1])), alphas)
    assert np.allclose(correct_colours, colours, atol=1e-2)
    assert np.allclose(correct_text_positions, text_positions, atol=1e-2)
    assert text_string == correct_text_string

    # Test 2-D Partial Dependence
    # Only 2-D PD
    figure, plot_axis = fvfi.plot_partial_dependence(
        FAKE_PD_ARRAY_2D, FAKE_LINESPACE_2D, class_index, False, None, 
        False, None, class_name)
    assert isinstance(figure, plt.Figure)
    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        plot_axis)
    assert p_title == 'Partial Dependence for {}'.format(class_name)
    assert p_x_label == 'feature 0'
    assert np.array_equal(p_x_range, np.array([0., 1.]))
    assert p_y_label == 'feature 1'
    assert np.array_equal(p_y_range, np.array([0.32, 0.5]))
    im = plot_axis.get_children()[-2]
    assert im.cmap.name == 'cividis'
    assert np.array_equal(im.get_extent(), np.array([0.0, 1.0, 0.32, 0.5]))
    assert np.array_equal(im.get_array(), FAKE_PD_ARRAY_2D[:, :, 1])

    # Test 2-D Partial Dependence with PD plot with feature distributions
    dist  = (np.array([0., .2, .4, .6, .8, 1.]), 
             np.array([0.1, 0.2, 0.5, 0.1, 0.1]))
    dist2 = (np.array([0.32, 0.356, 0.392, 0.428, 0.464, 0.5]), 
             np.array([0.2, 0.2, 0.3, 0.1, 0.2]))
    figure, axis = fvfi.plot_partial_dependence(
        FAKE_PD_ARRAY_2D, FAKE_LINESPACE_2D, class_index, False, 
        None, False, [None, 'feature 1'], class_name, [dist, dist2])
    assert isinstance(figure, plt.Figure)
    assert figure._suptitle.get_text() == \
        'Partial Dependence for {}'.format(class_name)
    plot_axis, dist_x, dist_y = axis
    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        plot_axis)
    assert p_title == ''
    assert p_x_label == 'feature 0'
    assert np.array_equal(p_x_range, np.array([0., 1.]))
    assert p_y_label == 'feature 1'
    assert np.array_equal(p_y_range, np.array([0.32, 0.5]))
    im = plot_axis.get_children()[-2]
    assert im.cmap.name == 'cividis'
    assert np.array_equal(im.get_extent(), np.array([0.0, 1.0, 0.32, 0.5]))
    assert np.array_equal(im.get_array(), FAKE_PD_ARRAY_2D[:, :, 1])
    
    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        dist_x)
    assert p_title == ''
    assert p_x_label == 'Histogram of values for feature 0'
    assert np.array_equal(p_x_range, np.array([0., 1.]))
    assert p_y_label == ''
    assert np.array_equal(p_y_range, np.array([0., 1.05]))
    bars = dist_x.patches
    texts = dist_x.texts
    assert len(bars) == dist[0].shape[0] - 1
    assert len(texts) == dist[0].shape[0] - 1

    heights = np.array([bar.get_height() for bar in bars])
    alphas = np.array([bar.get_alpha() for bar in bars])
    colours = np.array([bar.get_facecolor() for bar in bars])
    correct_colours = np.repeat(
        np.array([0.25, 0.41, 0.88, 0.6])[np.newaxis, :], len(bars), axis=0)
    labels = np.array([text.get_text() for text in dist_x.get_xticklabels()])
    text_positions = np.array([text.get_position() for text in texts])
    correct_text_positions = np.stack([dist[0][:-1]+0.1,
                                       dist[1]], axis=1)
    text_string = [text.get_text() for text in texts]
    correct_text_string = ['%.2f'%num for num in dist[1]]
    assert np.array_equal(dist[1], heights)
    assert np.array_equal(np.array([0.6]*len(dist[1])), alphas)
    assert np.allclose(correct_colours, colours, atol=1e-2)
    assert np.array_equal(correct_text_positions, text_positions)
    assert text_string == correct_text_string
    assert len(labels) == 0

    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        dist_y)
    assert p_title == ''
    assert p_y_label == 'Histogram of values for feature 1'
    assert np.array_equal(p_y_range, np.array([0.32, 0.5]))
    assert p_x_label == ''
    assert np.array_equal(p_x_range, np.array([0., 1.05]))
    bars = dist_y.patches
    texts = dist_y.texts
    assert len(bars) == dist2[0].shape[0] - 1
    assert len(texts) == dist2[0].shape[0] - 1

    heights = np.array([bar.get_width() for bar in bars])
    alphas = np.array([bar.get_alpha() for bar in bars])
    colours = np.array([bar.get_facecolor() for bar in bars])
    correct_colours = np.repeat(
        np.array([0.25, 0.41, 0.88, 0.6])[np.newaxis, :], len(bars), axis=0)
    labels = np.array([text.get_text() for text in dist_y.get_yticklabels()])
    text_positions = np.array([text.get_position() for text in texts])
    correct_text_positions = np.stack([dist2[1],
                                       dist2[0][:-1]+0.036/2], axis=1)
    text_string = [text.get_text() for text in texts]
    correct_text_string = ['%.2f'%num for num in dist2[1]]
    assert np.array_equal(dist2[1], heights)
    assert np.array_equal(np.array([0.6]*len(dist2[1])), alphas)
    assert np.allclose(correct_colours, colours, atol=1e-2)
    assert np.allclose(correct_text_positions, text_positions)
    assert text_string == correct_text_string
    assert len(labels) == 0
    plt.close(figure)

    # Test 2-D Partial Dependence with PD plot with feature distributions
    # with one string
    dist  = (np.array([0., .2, .4, .6, .8, 1.]), 
             np.array([0.1, 0.2, 0.5, 0.1, 0.1]))
    dist2 = (np.array(['c', 'b', 'a']), 
             np.array([0.2, 0.3, 0.5]))
    # Test reordering of values in dist to feature_linespace
    correct_dist2 = (np.array(['a', 'b', 'c']), np.array([0.5, 0.3, 0.2]))
    msg = ('Selected feature is categorical (string-base elements), however '
           'the treat_as_categorical was set to False. Such a combination is '
           'not possible. The feature will be treated as categorical.')
    with pytest.warns(UserWarning) as warning:
        figure, axis = fvfi.plot_partial_dependence(
            FAKE_PD_ARRAY_2D, FAKE_LINESPACE_2D_CAT, class_index, False,
            None, False, ['feature 0', None], None, [dist, dist2])
    assert len(warning) == 1
    assert str(warning[0].message) == msg

    assert isinstance(figure, plt.Figure)
    assert figure._suptitle.get_text() == \
        'Partial Dependence for class {}'.format(class_index)
    plot_axis, dist_x, dist_y = axis
    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        plot_axis)
    assert p_title == ''
    assert p_x_label == 'feature 0'
    assert np.array_equal(p_x_range, np.array([0., 1.]))
    assert p_y_label == 'feature 1'
    assert np.array_equal(p_y_range, np.array([-0.5, 3.5]))
    im = plot_axis.get_children()[-2]
    assert im.cmap.name == 'cividis'
    assert np.array_equal(im.get_extent(), np.array([0.0, 1.0, -0.5, 3.5]))
    assert np.array_equal(im.get_array(), FAKE_PD_ARRAY_2D[:, :, 1])

    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        dist_x)
    assert p_title == ''
    assert p_x_label == 'Histogram of values for feature 0'
    assert np.array_equal(p_x_range, np.array([0., 1.]))
    assert p_y_label == ''
    assert np.array_equal(p_y_range, np.array([0., 1.05]))
    bars = dist_x.patches
    texts = dist_x.texts
    assert len(bars) == dist[0].shape[0] - 1
    assert len(texts) == dist[0].shape[0] - 1

    heights = np.array([bar.get_height() for bar in bars])
    alphas = np.array([bar.get_alpha() for bar in bars])
    colours = np.array([bar.get_facecolor() for bar in bars])
    correct_colours = np.repeat(
        np.array([0.25, 0.41, 0.88, 0.6])[np.newaxis, :], len(bars), axis=0)
    labels = np.array([text.get_text() for text in dist_x.get_xticklabels()])
    text_positions = np.array([text.get_position() for text in texts])
    correct_text_positions = np.stack([dist[0][:-1]+0.1,
                                       dist[1]], axis=1)
    text_string = [text.get_text() for text in texts]
    correct_text_string = ['%.2f'%num for num in dist[1]]
    assert np.array_equal(dist[1], heights)
    assert np.array_equal(np.array([0.6]*len(dist[1])), alphas)
    assert np.allclose(correct_colours, colours, atol=1e-2)
    assert np.array_equal(correct_text_positions, text_positions)
    assert text_string == correct_text_string
    assert len(labels) == 0

    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        dist_y)
    assert p_title == ''
    assert p_y_label == 'Histogram of values for feature 1'
    assert np.array_equal(p_y_range, np.array([-0.5, 3.5]))
    assert p_x_label == ''
    assert np.array_equal(p_x_range, np.array([0., 1.05]))
    bars = dist_y.patches
    texts = dist_y.texts
    assert len(bars) == dist2[0].shape[0]
    assert len(texts) == dist2[0].shape[0]

    heights = np.array([bar.get_width() for bar in bars])
    alphas = np.array([bar.get_alpha() for bar in bars])
    colours = np.array([bar.get_facecolor() for bar in bars])
    correct_colours = np.repeat(
        np.array([0.25, 0.41, 0.88, 0.6])[np.newaxis, :], len(bars), axis=0)
    labels = np.array([text.get_text() for text in dist_y.get_yticklabels()])
    text_positions = np.array([text.get_position() for text in texts])
    correct_text_positions = np.stack([correct_dist2[1],
                                       np.array([0., 1.5, 3.])], axis=1)
    text_string = [text.get_text() for text in texts]
    correct_text_string = ['%.2f'%num for num in correct_dist2[1]]
    assert np.array_equal(correct_dist2[1], heights)
    assert np.array_equal(np.array([0.6]*len(dist2[1])), alphas)
    assert np.allclose(correct_colours, colours, atol=1e-2)
    assert np.allclose(correct_text_positions, text_positions)
    assert text_string == correct_text_string
    assert len(labels) == 0
    plt.close(figure)


def test_ice_pd_overlay():
    """
    Tests overlaying PD plot on top of an ICE plot.
    """
    f_name = 'some feature'
    c_index = 1
    c_name = 'middle'

    # Overall with partial dependence variance error bars
    figure, axis = fvfi.plot_individual_conditional_expectation(
        FAKE_ICE_ARRAY, FAKE_LINESPACE, c_index, False, f_name, c_name)
    assert isinstance(figure, plt.Figure)
    assert isinstance(axis, plt.Axes)

    none, axis = fvfi.plot_partial_dependence(FAKE_PD_ARRAY, FAKE_LINESPACE,
                                              c_index, False, FAKE_VARIANCE, 
                                              False, f_name, c_name, None, 
                                              axis)
    assert none is None
    assert isinstance(axis, plt.Axes)

    # Inspect the canvas
    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        axis)
    # ...check title
    assert p_title == ('Individual Conditional Expectation &\nPartial '
                       'Dependence')
    # ...check x range
    assert np.array_equal(p_x_range, [FAKE_LINESPACE[0], FAKE_LINESPACE[-1]])
    # ...check x label
    assert p_x_label == f_name
    # ...check y range
    assert np.array_equal(p_y_range, [-0.05, 1.05])
    # ...check y label
    assert p_y_label == '{} class probability'.format(c_name)

    # Check ICE
    assert len(axis.collections) == 2
    l_data, l_colour, l_alpha, l_label, l_width = futv.get_line_data(
        axis.collections[0], is_collection=True)
    assert len(l_data) == FAKE_ICE_ARRAY.shape[0]
    for i, line_array in enumerate(l_data):
        line_data = np.stack([FAKE_LINESPACE, FAKE_ICE_ARRAY[i, :, c_index]],
                             axis=1)
        assert np.array_equal(line_array, line_data)
    assert np.isclose(
        l_colour, np.array([[0.412, 0.412, 0.412, 0.5]]),
        atol=1e-2).all()  # dimgray mapping apparently
    assert l_alpha == 0.5
    assert l_label == 'ICE'
    assert l_width == 1.75

    # Check PD
    assert len(axis.lines) == 3
    l_data, l_colour, l_alpha, l_label, l_width = futv.get_line_data(
        axis.lines[0])
    line_data = np.stack([FAKE_LINESPACE, FAKE_PD_ARRAY[:, c_index]], axis=1)
    assert np.array_equal(l_data, line_data)
    assert l_colour == 'lightsalmon'
    assert l_alpha == 0.6
    assert l_label == 'PD'
    assert l_width == 7

    # Test variance lines
    line_data = np.stack([FAKE_LINESPACE, FAKE_PD_ARRAY[:, c_index] -
                              FAKE_VARIANCE[:, c_index]], axis=1)
    l_data, l_colour, l_alpha, l_label, l_width = futv.get_line_data(
        axis.lines[1])
    assert np.array_equal(l_data, line_data)
    assert l_colour == 'lightsalmon'
    assert l_alpha == 0.6
    assert l_label == '_nolegend_'
    assert l_width == 1.75

    line_data = np.stack([FAKE_LINESPACE, FAKE_PD_ARRAY[:, c_index] +
                              FAKE_VARIANCE[:, c_index]], axis=1)
    l_data, l_colour, l_alpha, l_label, l_width = futv.get_line_data(
        axis.lines[2])
    
    assert np.array_equal(l_data, line_data)
    assert l_colour == 'lightsalmon'
    assert l_alpha == 0.6
    assert l_label == '_nolegend_'
    assert l_width == 1.75

    # Test variance line collection
    l_data, l_colour, l_alpha, l_label, l_width = futv.get_line_data(
        axis.collections[1], is_collection=True)
    line_data = np.stack([
        np.stack([FAKE_LINESPACE, FAKE_PD_ARRAY[:, c_index] -
                  FAKE_VARIANCE[:, c_index]], axis=1),
        np.stack([FAKE_LINESPACE, FAKE_PD_ARRAY[:, c_index] +
                  FAKE_VARIANCE[:, c_index]], axis=1)])
    l_data = np.stack(l_data, axis=1)
    assert np.array_equal(l_data, line_data)
    assert np.allclose(l_colour, np.array([1., 0.62, 0.47, 0.6]), atol=1e-2)
    assert l_alpha == 0.6
    assert l_label == '_nolegend_'
    assert l_width == 2
    # Validate plot legend
    legend = [
        i for i in axis.get_children()
        if isinstance(i, matplotlib.legend.Legend)
    ]
    assert len(legend) == 1
    legend_texts = legend[0].get_texts()
    assert len(legend_texts) == 3
    assert legend_texts[0].get_text() == 'PD'
    assert legend_texts[1].get_text() == 'ICE'
    assert legend_texts[2].get_text() == 'Variance'
    plt.close(fig=figure)

    # Overlay with partial dependence with variance area
    figure, axis = fvfi.plot_individual_conditional_expectation(
        FAKE_ICE_ARRAY, FAKE_LINESPACE, c_index, False, f_name, c_name)
    assert isinstance(figure, plt.Figure)
    assert isinstance(axis, plt.Axes)

    none, axis = fvfi.plot_partial_dependence(FAKE_PD_ARRAY, FAKE_LINESPACE,
                                              c_index, False, FAKE_VARIANCE,
                                              True, f_name, c_name, None,
                                              axis)
    assert none is None
    assert isinstance(axis, plt.Axes)

    # Inspect the canvas
    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        axis)
    # ...check title
    assert p_title == ('Individual Conditional Expectation &\nPartial '
                       'Dependence')
    # ...check x range
    assert np.array_equal(p_x_range, [FAKE_LINESPACE[0], FAKE_LINESPACE[-1]])
    # ...check x label
    assert p_x_label == f_name
    # ...check y range
    assert np.array_equal(p_y_range, [-0.05, 1.05])
    # ...check y label
    assert p_y_label == '{} class probability'.format(c_name)

    # Check ICE
    assert len(axis.collections) == 2
    l_data, l_colour, l_alpha, l_label, l_width = futv.get_line_data(
        axis.collections[0], is_collection=True)
    assert len(l_data) == FAKE_ICE_ARRAY.shape[0]
    for i, line_array in enumerate(l_data):
        line_data = np.stack([FAKE_LINESPACE, FAKE_ICE_ARRAY[i, :, c_index]],
                             axis=1)
        assert np.array_equal(line_array, line_data)
    assert np.isclose(
        l_colour, np.array([[0.412, 0.412, 0.412, 0.5]]),
        atol=1e-2).all()  # dimgray mapping apparently
    assert l_alpha == 0.5
    assert l_label == 'ICE'
    assert l_width == 1.75

    # Check PD
    assert len(axis.lines) == 1
    l_data, l_colour, l_alpha, l_label, l_width = futv.get_line_data(
        axis.lines[0])
    line_data = np.stack([FAKE_LINESPACE, FAKE_PD_ARRAY[:, c_index]], axis=1)
    assert np.array_equal(l_data, line_data)
    assert l_colour == 'lightsalmon'
    assert l_alpha == 0.6
    assert l_label == 'PD'
    assert l_width == 7

    # Test variance PolyCollection
    assert np.allclose(axis.collections[1].get_facecolor(), 
                       np.array([1., 0.62, 0.47, 0.3]), atol=1e-2)
    assert np.allclose(axis.collections[1].get_edgecolor(), 
                       np.array([1., 0.62, 0.47, 0.3]), atol=1e-2)
    assert axis.collections[1].get_alpha() == 0.3
    assert axis.collections[1].get_label() == 'Variance'

    # Test the areas
    path = axis.collections[1].get_paths()
    assert len(path) == 1
    assert np.array_equal(path[0].codes, 
                          np.array([1]+[2]*(FAKE_LINESPACE.shape[0]*2+1)+[79]))
    # Construct the polygon path
    first_point = np.array([FAKE_LINESPACE[0], FAKE_PD_ARRAY[0, c_index] + 
                                FAKE_VARIANCE[0, c_index]]).reshape(1, 2)
    last_point = np.array([FAKE_LINESPACE[-1], FAKE_PD_ARRAY[-1, c_index] + 
                           FAKE_VARIANCE[-1, c_index]]).reshape(1, 2)
    var_below = np.stack([FAKE_LINESPACE, FAKE_PD_ARRAY[:, c_index] -
                          FAKE_VARIANCE[:, c_index]], axis=1)
    var_above = np.stack([FAKE_LINESPACE, FAKE_PD_ARRAY[:, c_index] +
                          FAKE_VARIANCE[:, c_index]], axis=1)
    correct_polygon = np.concatenate(
        [first_point, var_below, last_point, np.flip(var_above, 0), 
        first_point], axis=0)
    assert np.array_equal(path[0].vertices, correct_polygon)

    # Validate plot legend
    legend = [
        i for i in axis.get_children()
        if isinstance(i, matplotlib.legend.Legend)
    ]
    assert len(legend) == 1
    legend_texts = legend[0].get_texts()
    assert len(legend_texts) == 3
    assert legend_texts[0].get_text() == 'PD'
    assert legend_texts[1].get_text() == 'ICE'
    assert legend_texts[2].get_text() == 'Variance'
    plt.close(fig=figure)
