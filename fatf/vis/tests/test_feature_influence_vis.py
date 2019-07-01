"""
Tests feature influence (ICE and PD) plotting functions.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
#         Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import pytest

try:
    import matplotlib.legend
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    pytest.skip(
        'Skipping visualisation tests -- matplotlib missing.',
        allow_module_level=True)

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


def test_validate_input():
    """
    Tests :func:`fatf.vis.feature_influence._validate_input`.
    """
    msg = 'test_partial_dependence is not a boolean.'
    with pytest.raises(AssertionError) as exin:
        fvfi._validate_input(None, None, None, None, None, None, 1)
    assert str(exin.value) == msg

    msg = 'The input array cannot be a structured array.'
    struct_array = np.array([(4, 2), (2, 4)], dtype=[('a', int), ('b', int)])
    with pytest.raises(ValueError) as exin:
        fvfi._validate_input(struct_array, None, None, None, None, None, False)
    assert str(exin.value) == msg

    msg = 'The input array has to be a numerical array.'
    non_numerical_array = np.array([[4, 'a'], [2, 'b']])
    with pytest.raises(ValueError) as exin:
        fvfi._validate_input(non_numerical_array, None, None, None, None, None,
                             False)
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
                             False)
    assert str(exin.value) == msg
    # For Partial Dependence
    msg = ('plot_partial_depenedence expects a 2-dimensional array of shape '
           '(n_steps, n_classes).')
    with pytest.raises(IncorrectShapeError) as exin:
        fvfi._validate_input(numerical_3d_array, None, None, None, None, None,
                             True)
    assert str(exin.value) == msg

    # Linespace
    msg = 'The linespace array cannot be a structured array.'
    with pytest.raises(ValueError) as exin:
        fvfi._validate_input(numerical_2d_array, struct_array, None, None,
                             None, None, True)
    assert str(exin.value) == msg
    #
    msg = ('The linespace array has to be a 1-dimensional array of shape '
           '(n_steps, ).')
    with pytest.raises(IncorrectShapeError) as exin:
        fvfi._validate_input(numerical_3d_array, numerical_2d_array, None,
                             None, None, None, False)
    assert str(exin.value) == msg
    #
    msg = 'The linespace array has to be numerical.'
    with pytest.raises(ValueError) as exin:
        fvfi._validate_input(numerical_2d_array, non_numerical_array[0], None,
                             None, None, None, True)
    assert str(exin.value) == msg
    # Linespace vector not matching ICE/ PDP dimensions
    msg = ('The length of the linespace array ({}) does not agree with the '
           'number of linespace steps ({}) in the input array.')
    with pytest.raises(ValueError) as exin:
        fvfi._validate_input(numerical_2d_array, numerical_2d_array[0, :],
                             None, None, None, None, True)
    assert str(exin.value) == msg.format(2, 3)
    with pytest.raises(ValueError) as exin:
        fvfi._validate_input(numerical_3d_array, numerical_2d_array[0, :],
                             None, None, None, None, False)
    assert str(exin.value) == msg.format(2, 3)

    # Index
    msg = 'Class index has to be an integer.'
    with pytest.raises(TypeError) as exin:
        fvfi._validate_input(numerical_3d_array, numerical_2d_array[:, 0],
                             None, None, None, None, False)
    assert str(exin.value) == msg
    #
    msg = ('Class index {} is not a valid index for the input array. There '
           'are only {} classes available.')
    with pytest.raises(IndexError) as exin:
        fvfi._validate_input(numerical_3d_array, numerical_2d_array[:, 0], -1,
                             None, None, None, False)
    assert str(exin.value) == msg.format(-1, 2)
    with pytest.raises(IndexError) as exin:
        fvfi._validate_input(numerical_2d_array, numerical_2d_array[:, 0], 2,
                             None, None, None, True)
    assert str(exin.value) == msg.format(2, 2)

    # Feature name
    msg = 'The feature name has to be either None or a string.'
    with pytest.raises(TypeError) as exin:
        fvfi._validate_input(numerical_2d_array, numerical_2d_array[:, 0], 1,
                             42, None, None, True)
    assert str(exin.value) == msg

    # Class name
    msg = 'The class name has to be either None or a string.'
    with pytest.raises(TypeError) as exin:
        fvfi._validate_input(numerical_3d_array, numerical_2d_array[:, 0], 0,
                             None, 42, None, False)
    assert str(exin.value) == msg

    # Plot axis
    msg = ('The plot axis has to be either None or a matplotlib.pyplot.Axes '
           'type object.')
    with pytest.raises(TypeError) as exin:
        fvfi._validate_input(numerical_2d_array, numerical_2d_array[:, 0], 1,
                             'feature name', None, 42, True)
    assert str(exin.value) == msg

    # All OK
    assert fvfi._validate_input(numerical_2d_array, numerical_2d_array[:, 0],
                                1, 'feature name', 'class name', None, True)
    fig, my_plot = plt.subplots(1, 1)
    assert fvfi._validate_input(numerical_3d_array, numerical_2d_array[:, 0],
                                1, 'feature name', 'class name', my_plot,
                                False)


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
        FAKE_ICE_ARRAY, FAKE_LINESPACE, class_index, feature_name, class_name)

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


def test_plot_partial_dependence():
    """
    Tests :func:`fatf.vis.feature_influence.plot_partial_dependence` function.
    """
    feature_name = 'some feature'
    class_index = 1
    class_name = 'middle'

    figure, axis = fvfi.plot_partial_dependence(
        FAKE_PD_ARRAY, FAKE_LINESPACE, class_index, feature_name, class_name)

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


def test_ice_pd_overlay():
    """
    Tests overlaying PD plot on top of an ICE plot.
    """
    f_name = 'some feature'
    c_index = 1
    c_name = 'middle'

    figure, axis = fvfi.plot_individual_conditional_expectation(
        FAKE_ICE_ARRAY, FAKE_LINESPACE, c_index, f_name, c_name)
    assert isinstance(figure, plt.Figure)
    assert isinstance(axis, plt.Axes)

    none, axis = fvfi.plot_partial_dependence(FAKE_PD_ARRAY, FAKE_LINESPACE,
                                              c_index, f_name, c_name, axis)
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
    assert len(axis.collections) == 1
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

    # Validate plot legend
    legend = [
        i for i in axis.get_children()
        if isinstance(i, matplotlib.legend.Legend)
    ]
    assert len(legend) == 1
    legend_texts = legend[0].get_texts()
    assert len(legend_texts) == 2
    assert legend_texts[0].get_text() == 'PD'
    assert legend_texts[1].get_text() == 'ICE'
