"""
Tests LIME plotting functions.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
#         Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import pytest

import fatf.utils.testing.vis as futv
import fatf.vis.lime as fvl

RED = (1, 0, 0, 1)
GREEN = (0.0, 0.502, 0.0, 1.0)


def test_plot_lime_validation():
    """
    Tests exceptions in the :func:`fatf.vis.lime.plot_lime` function.
    """
    type_error_global = ('The LIME explanation has to be either a dictionary '
                         '(for classification) or a list (for regression).')
    type_error_val = 'One of the explanations is not a list.'
    type_error_tup1 = ('One of the explanation keys is neither an integer nor '
                       'a string.')
    type_error_tup2 = 'One of the explanation values is not a number.'

    with pytest.raises(TypeError) as exin:
        fvl.plot_lime('test')
    assert str(exin.value) == type_error_global

    with pytest.raises(TypeError) as exin:
        fvl.plot_lime({'a': 'test'})
    assert str(exin.value) == type_error_val

    with pytest.raises(TypeError) as exin:
        fvl.plot_lime({'a': [(3, 'string')]})
    assert str(exin.value) == type_error_tup1

    with pytest.raises(TypeError) as exin:
        fvl.plot_lime({'a': [('b', 'string')]})
    assert str(exin.value) == type_error_tup2


def test_plot_lime_classification():
    """
    Tests :func:`fatf.vis.lime.plot_lime` function for classification.
    """
    data = {
        'class0': [('feat0 <= 0.00', -0.415),
                   ('0.50 < feat1 <= 1.00', -0.280),
                   ('0.07 < feat2 <= 0.22', 0.0377),
                   ('0.34 < feat3 <= 0.58', -0.007)],
        'class1': [('0.50 < feat1 <= 1.00', 0.202),
                   ('0.07 < feat2 <= 0.22', -0.076),
                   ('feat0 <= 0.00', 0.019),
                   ('0.34 < feat3 <= 0.58', -0.018)],
        'class2': [('feat0 <= 0.00', 0.395),
                   ('0.50 < feat1 <= 1.00', 0.0775),
                   ('0.07 < feat2 <= 0.22', 0.0392),
                   ('0.34 < feat3 <= 0.58', 0.025)]
    }  # yapf: disable
    classes = sorted(list(data.keys()))
    x_range = [-0.45, 0.43]  # min/max + 0.035 from the data above

    # All the classes have the same feature bounds so the plot should share the
    # y axis -- the first axis has the correct labels, the rest is empty
    y_labels = [x[0] for x in data['class0']]
    widths = [[-0.415, -0.280, 0.0377, -0.007], [0.019, 0.202, -0.076, -0.018],
              [0.395, 0.0775, 0.0392, 0.025]]
    colours = [[RED, RED, GREEN, RED], [GREEN, GREEN, RED, RED],
               [GREEN, GREEN, GREEN, GREEN]]

    fig = fvl.plot_lime(data)
    assert len(fig.axes) == len(classes)

    for axis_index in range(len(fig.axes)):
        bar_data = futv.get_bar_data(fig.axes[axis_index])
        title, x_ticks, x_rng, y_ticks, y_rng, width, colour = bar_data

        # In case the axes are not returned in the right order figure it out.
        # This is needed for Python 3.5
        i = classes.index(title)
        #
        assert title == classes[i]
        #
        for j in x_ticks:
            assert j == ''
        assert len(x_range) == 2
        assert len(x_rng) == 2
        assert abs((x_range[1] - x_range[0]) - (x_rng[1] - x_rng[0])) < 0.02
        #
        if axis_index == 0:
            assert len(y_labels) == len(y_ticks)
            for j in range(len(y_ticks)):
                assert y_labels[j] == y_ticks[j]
        else:
            assert not y_ticks
        assert len(y_rng) == 2
        assert abs(len(data[classes[i]]) - (y_rng[1] - y_rng[0])) < 0.2
        #
        assert len(width) == len(widths[i])
        for j in range(len(width)):
            assert widths[i][j] == width[j]
        #
        assert len(colour) == len(colours[i])
        for j in range(len(colour)):
            assert len(colour[j]) == len(colours[i][j])
            for k in range(len(colour[j])):
                assert abs(colours[i][j][k] - colour[j][k]) < 0.001

    # Test when sharey is False and the yticklabels are unique for each axis
    del data['class1'][2]
    y_labels = [[x[0] for x in data[i]] for i in classes]
    widths = [[x[1] for x in data[i]] for i in classes]
    colours = [[RED, RED, GREEN, RED], [GREEN, RED, RED],
               [GREEN, GREEN, GREEN, GREEN]]

    fig = fvl.plot_lime(data)
    assert len(fig.axes) == len(classes)

    for axis_index in range(len(fig.axes)):
        bar_data = futv.get_bar_data(fig.axes[axis_index])
        title, x_ticks, x_rng, y_ticks, y_rng, width, colour = bar_data

        # In case the axes are not returned in the right order figure it out.
        # This is needed for Python 3.5
        i = classes.index(title)
        #
        assert title == classes[i]
        #
        for j in x_ticks:
            assert j == ''
        assert len(x_range) == 2
        assert len(x_rng) == 2
        assert abs((x_range[1] - x_range[0]) - (x_rng[1] - x_rng[0])) < 0.02
        #
        assert len(y_labels[i]) == len(y_ticks)
        for j in range(len(y_ticks)):
            assert y_labels[i][j] == y_ticks[j]
        assert len(y_rng) == 2
        assert abs(len(data[classes[i]]) - (y_rng[1] - y_rng[0])) < 0.2
        #
        assert len(width) == len(widths[i])
        for j in range(len(width)):
            assert widths[i][j] == width[j]
        #
        # Colours
        assert len(colour) == len(colours[i])
        for j in range(len(colour)):
            assert len(colour[j]) == len(colours[i][j])
            for k in range(len(colour[j])):
                assert abs(colours[i][j][k] - colour[j][k]) < 0.001


def test_plot_lime_regression():
    """
    Tests :func:`fatf.vis.lime.plot_lime` function for regression.
    """
    data = [('feat0 <= 0.00', -0.415),
            ('0.50 < feat1 <= 1.00', -0.280),
            ('0.07 < feat2 <= 0.22', 0.0377),
            ('0.34 < feat3 <= 0.58', -0.007)]  # yapf: disable
    x_range = [-0.45, 0.08]  # min/max + 0.035 from the data above
    y_labels = [x[0] for x in data]
    true_width = [-0.415, -0.280, 0.0377, -0.007]
    true_colour = [RED, RED, GREEN, RED]
    true_title = 'regressor'

    fig = fvl.plot_lime(data)
    assert len(fig.axes) == 1

    bar_data = futv.get_bar_data(fig.axes[0])
    title, x_ticks, x_rng, y_ticks, y_rng, width, colour = bar_data
    assert true_title == title
    #
    for j in x_ticks:
        assert j == ''
    assert len(x_range) == 2
    assert len(x_rng) == 2
    assert abs((x_range[1] - x_range[0]) - (x_rng[1] - x_rng[0])) < 0.05
    #
    assert len(y_labels) == len(y_ticks)
    for j in range(len(y_ticks)):
        assert y_labels[j] == y_ticks[j]
    assert len(y_rng) == 2
    assert abs(len(data) - (y_rng[1] - y_rng[0])) < 0.2
    #
    assert len(width) == len(true_width)
    for j in range(len(width)):
        assert true_width[j] == width[j]
    #
    # Colours
    assert len(colour) == len(true_colour)
    for j in range(len(colour)):
        assert len(colour[j]) == len(true_colour[j])
        for k in range(len(colour[j])):
            assert abs(true_colour[j][k] - colour[j][k]) < 0.001
