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
FAKE_PD_ARRAY = np.array([[0.50, 0.50, 0.00],
                          [0.33, 0.33, 0.34],
                          [0.90, 0.07, 0.03],
                          [0.33, 0.33, 0.34],
                          [0.90, 0.07, 0.03],
                          [0.20, 0.30, 0.40]])  # yapf: disable
FAKE_LINESPACE = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
FAKE_LINESPACE_STRING = np.array(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
FAKE_VARIANCE = np.array([[0.20, 0.05, 0.00],
                          [0.10, 0.05, 0.15],
                          [0.05, 0.01, 0.01],
                          [0.26, 0.15, 0.21],
                          [0.08, 0.05, 0.01],
                          [0.15, 0.10, 0.17]])
FAKE_LINESPACE_CAT = np.array(['a', 'b', 'c', 'd', 'e', 'f'])


def test_validate_input():
    """
    Tests :func:`fatf.vis.feature_influence._validate_input`.
    """
    msg = 'test_partial_dependence is not a boolean.'
    with pytest.raises(AssertionError) as exin:
        fvfi._validate_input(None, None, None, None, None, None, None, None,
                             False, 1)
    assert str(exin.value) == msg

    msg = 'variance_area is not a boolean.'
    with pytest.raises(AssertionError) as exin:
        fvfi._validate_input(None, None, None, None, None, None, None, None,
                             1, True)
    assert str(exin.value) == msg

    msg = 'The input array cannot be a structured array.'
    struct_array = np.array([(4, 2), (2, 4)], dtype=[('a', int), ('b', int)])
    with pytest.raises(ValueError) as exin:
        fvfi._validate_input(struct_array, None, None, None, None, None, None,
                             None, False, False)
    assert str(exin.value) == msg

    msg = 'The input array has to be a numerical array.'
    non_numerical_array = np.array([[4, 'a'], [2, 'b']])
    with pytest.raises(ValueError) as exin:
        fvfi._validate_input(non_numerical_array, None, None, None, None, None,
                             None, None, False, False)
    assert str(exin.value) == msg

    numerical_2d_array = np.array([[4, 2], [2, 4], [4, 2]])
    numerical_3d_array = np.array([[[4, 3], [4, 2], [4, 2]],
                                   [[8, 1], [7, 5], [4, 2]],
                                   [[4, 3], [4, 2], [4, 2]],
                                   [[4, 2], [2, 4], [4, 2]]])
    # For Individual Conditional Expectation
    msg = ('plot_individual_condtional_expectation expects a 3-dimensional '
           'array of shape (n_samples, n_steps, n_classes).')
    with pytest.raises(IncorrectShapeError) as exin:
        fvfi._validate_input(numerical_2d_array, None, None, None, None, None,
                             None, None, False, False)
    assert str(exin.value) == msg
    # For Partial Dependence
    msg = ('plot_partial_depenedence expects a 2-dimensional array of shape '
           '(n_steps, n_classes).')
    with pytest.raises(IncorrectShapeError) as exin:
        fvfi._validate_input(numerical_3d_array, None, None, None, None, None,
                             None, None, False, True)
    assert str(exin.value) == msg

    # Linespace
    msg = 'The linespace array cannot be a structured array.'
    with pytest.raises(ValueError) as exin:
        fvfi._validate_input(numerical_2d_array, struct_array, None, None,
                             None, None, None, None, False, True)
    assert str(exin.value) == msg
    #
    msg = ('The linespace array has to be a 1-dimensional array of shape '
           '(n_steps, ).')
    with pytest.raises(IncorrectShapeError) as exin:
        fvfi._validate_input(numerical_3d_array, numerical_2d_array, None,
                             None, None, None, None, None, False, False)
    assert str(exin.value) == msg
    # Linespace vector not matching ICE/ PDP dimensions
    msg = ('The length of the linespace array ({}) does not agree with the '
           'number of linespace steps ({}) in the input array.')
    with pytest.raises(ValueError) as exin:
        fvfi._validate_input(numerical_2d_array, numerical_2d_array[0, :],
                             None, None, None, None, None, None, False, True)
    assert str(exin.value) == msg.format(2, 3)
    with pytest.raises(ValueError) as exin:
        fvfi._validate_input(numerical_3d_array, numerical_2d_array[0, :],
                             None, None, None, None, None, None, False, False)
    assert str(exin.value) == msg.format(2, 3)
    # Variance vector not matching ICE/ PDP dimensions
    msg = ('The length of the variance array ({}) does agree with the number '
           'of linespace steps ({}) in the input array.')
    with pytest.raises(ValueError) as exin:
        fvfi._validate_input(numerical_2d_array, numerical_2d_array[:, 0], 1,
                             None, None, None, None, numerical_2d_array[0, :], 
                             False, True)
    assert str(exin.value) == msg.format(2, 3)
    # variance_area is True but no variance vector supplied
    msg = ('Variance vector has not been given but variance_area has been '
           'given as True. To plot the variance please specify a variance '
           'vector.')
    with pytest.raises(ValueError) as exin:
        fvfi._validate_input(numerical_2d_array, numerical_2d_array[:, 0], 1,
                             None, None, None, None, None, True, True)
    assert str(exin.value) == msg
    # Index
    msg = 'Class index has to be an integer.'
    with pytest.raises(TypeError) as exin:
        fvfi._validate_input(numerical_3d_array, numerical_2d_array[:, 0],
                             None, None, None, None, None, None, False, False)
    assert str(exin.value) == msg
    #
    msg = ('Class index {} is not a valid index for the input array. There '
           'are only {} classes available.')
    with pytest.raises(IndexError) as exin:
        fvfi._validate_input(numerical_3d_array, numerical_2d_array[:, 0], -1,
                             None, None, None, None, None, False, False)
    assert str(exin.value) == msg.format(-1, 2)
    with pytest.raises(IndexError) as exin:
        fvfi._validate_input(numerical_2d_array, numerical_2d_array[:, 0], 2,
                             None, None, None, None, None, False, True)
    assert str(exin.value) == msg.format(2, 2)

    # Feature name
    msg = 'The feature name has to be either None or a string.'
    with pytest.raises(TypeError) as exin:
        fvfi._validate_input(numerical_2d_array, numerical_2d_array[:, 0], 1,
                             None, 42, None, None, None, False, True)
    assert str(exin.value) == msg

    # Class name
    msg = 'The class name has to be either None or a string.'
    with pytest.raises(TypeError) as exin:
        fvfi._validate_input(numerical_3d_array, numerical_2d_array[:, 0], 0,
                             None, None, 42, None, None, False, False)
    assert str(exin.value) == msg

    # Plot axis
    msg = ('The plot axis has to be either None or a matplotlib.pyplot.Axes '
           'type object.')
    with pytest.raises(TypeError) as exin:
        fvfi._validate_input(numerical_2d_array, numerical_2d_array[:, 0], 1,
                             None, 'feature name', None, 42, None, False, True)
    assert str(exin.value) == msg

    # Treat as categorical
    msg = ('treat_as_categorical is not a boolean')
    with pytest.raises(TypeError) as exin:
        fvfi._validate_input(None, None, 1, 10, None, None, None, None,
                             False, True)
    assert str(exin.value) == msg

    # Variance cannot be structured
    msg = ('The variance array cannot be a structured array.')
    with pytest.raises(ValueError) as exin:
        fvfi._validate_input(numerical_2d_array, numerical_2d_array[:, 0],
                             1, False, 'feature name', 'class name', None,
                            struct_array, False, True)
    assert str(exin.value) == msg

    # Variance has to be numerical
    msg = ('The variance array has to be a numerical array.')
    with pytest.raises(ValueError) as exin:
        fvfi._validate_input(numerical_2d_array, numerical_2d_array[:, 0],
                             1, False, 'feature name', 'class name', None,
                            non_numerical_array, False, True)
    assert str(exin.value) == msg

    # All OK
    assert fvfi._validate_input(numerical_2d_array, numerical_2d_array[:, 0],
                                1, False, 'feature name', 'class name', None, 
                                numerical_2d_array[:, 0], True, True)
    fig, my_plot = plt.subplots(1, 1)
    assert fvfi._validate_input(numerical_3d_array, numerical_2d_array[:, 0],
                                1, False, 'feature name', 'class name', my_plot, None,
                                False, False)
    plt.close(fig=fig)

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


def testplot_feature_distribution():
    """
    Tests feature distribution plotting.

    Tests
    :func:`fatf.vis.feature_influence.plot_feature_distribution` function.
    """
    # Invalid plot axis
    msg = ('The plot axis has to be either None or a matplotlib.'
           'pyplot.Axes type object.')
    with pytest.raises(TypeError) as exin:
        fvfi.plot_feature_distribution(np.array([]), None, None, 'plot')
    assert str(exin.value) == msg

    # Invalid feature name
    msg = ('The feature name has to be either None or a string.')
    with pytest.raises(TypeError) as exin:
        fvfi.plot_feature_distribution(np.array([]), None, 12, None)
    assert str(exin.value) == msg

    # Invalid feature distribution type
    msg = ('Feature distribution has to be a list')
    with pytest.raises(TypeError) as exin:
        fvfi.plot_feature_distribution(np.array([]), 0, None, None)
    assert str(exin.value) == msg

    # Invalid feature distribution
    msg = ('Feature distribution has to be a list of length 2 where the first '
           'element is a values array and the second element is a counts '
           'array.')
    with pytest.raises(ValueError) as exin:
        fvfi.plot_feature_distribution(np.array([]), [np.array([])]*4, None,
                                       None)
    assert str(exin.value) == msg

    # List of none np.array as feature distribution
    msg = ('The {} element in feature_distribution array must be of type '
           'np.ndarray.')
    with pytest.raises(TypeError) as exin:
        fvfi.plot_feature_distribution(np.array([]), [np.array([]), 1], None,
                                       None)
    assert str(exin.value) == msg.format(1)

    # Invalid shapes of feature distribution arrays
    msg = ('Values shape {} and counts shape {} do not agree. In order to '
           'define histograms, values has to be of shape '
           '(counts.shape[0]+1, ). In rder to define categorical counts, the '
           'set of values given has to be equal to the set of '
           'feature_linespace. In order to define Gaussian Kernel, values and '
           'counts must be of the same shape.')
    with pytest.raises(ValueError) as exin:
        fvfi.plot_feature_distribution(
            np.array([]), [np.array([1, 2, 3,]), np.array([1])], None, None)
    assert str(exin.value) == msg.format(3, 1)

    # Distribution above 1 should be rejected for histogram
    dist  = [np.array([0., .2, .4, .6, .8, 1.]), 
             np.array([0.1, 0.2, 0.5, 0.1, 1.1,])]
    msg = ('Distribution cannot have value more than 1.0')
    with pytest.raises(ValueError) as exin:
        fvfi.plot_feature_distribution(FAKE_LINESPACE, dist, 'feat')
    assert str(exin.value) == msg

    # Distribution above 1 should be rejected for categorical data
    dist = [np.array([0, 0.2, 0.4, 0.6, 0.8, 1]),
            np.array([0.1, 0.1, 0.3, 0.2, 0.2, 1.1])]
    msg = ('Distribution cannot have value more than 1.0')
    with pytest.raises(ValueError) as exin:
        fvfi.plot_feature_distribution(FAKE_LINESPACE, dist, 'feat')
    assert str(exin.value) == msg


    dist  = [np.array([0., .2, .4, .6, .8, 1.]), 
             np.array([0.1, 0.2, 0.5, 0.1, 0.1])]
    # Without passing axis
    fig, axis = fvfi.plot_feature_distribution(FAKE_LINESPACE, dist, 'feat')
    assert isinstance(fig, plt.Figure)
    p_title, p_x_label, p_x_range, p_y_label, p_y_range = futv.get_plot_data(
        axis)
    # ...check title
    assert p_title == 'Feature Distribution for feat'
    # ...check x range
    assert np.array_equal(p_x_range, [0., 1.])
    # ...check x label
    assert p_x_label == 'feat'
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
    _, axis = fvfi.plot_feature_distribution(
        FAKE_LINESPACE, dist, None, axis)
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
    dist = [np.array([0, 0.2, 0.4, 0.6, 0.8, 1]),
            np.array([0.1, 0.1, 0.3, 0.2, 0.2, 0.1])]
    _, axis = fvfi.plot_feature_distribution(
        FAKE_LINESPACE, dist, None, axis)
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
    dist = [np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
            np.array([0.1, 0.2, 0.4, 0.4, 0.3, 0.2, 0.1, 0.05, 0.05, 0.05])]
    _, axis = fvfi.plot_feature_distribution(
        FAKE_LINESPACE, dist, None, axis)
    assert len(axis.lines) == 1
    l_data, l_colour, l_alpha, l_label, l_width = futv.get_line_data(
        axis.lines[0])
    assert np.array_equal(np.stack(dist, axis=1), l_data)
    assert l_colour == 'royalblue'
    assert l_alpha == 0.6
    assert l_width == 3.0

    # Categorical feature
    fig, axis = plt.subplots()
    dist = [np.array(['a', 'b', 'c', 'd', 'e', 'f']),
            np.array([0.1, 0.1, 0.3, 0.2, 0.2, 0.1])]
    _, axis = fvfi.plot_feature_distribution(
        FAKE_LINESPACE_CAT, dist, None, axis)
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
