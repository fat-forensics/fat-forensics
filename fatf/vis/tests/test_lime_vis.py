"""
Tests LIME plotting functions.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
#         Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import pytest

try:
    import matplotlib
except ImportError:  # pragma: no cover
    pytest.skip(
        'Skipping visualisation tests -- matplotlib missing.',
        allow_module_level=True)
else:
    del matplotlib

import fatf.utils.testing.vis as futv
import fatf.vis.lime as fvl

RED = (1, 0, 0, 1)
GREEN = (0.0, 0.502, 0.0, 1.0)


def test_plot_lime_validation(caplog):
    """
    Tests exceptions in the :func:`fatf.vis.lime.plot_lime` function.
    """
    type_error_global = 'The surrogate explanation has to be a dictionary.'
    type_error_key_type = ('Each value of the surrogate explanation must '
                           'either be a dictionary or a number.')
    type_error_key = 'One of the class names is not a string.'
    type_error_val = 'One of the explanations is not a dictionary.'
    type_error_tup1 = 'One of the explanation keys is not a string.'
    type_error_tup2 = 'One of the explanation values is not a number.'

    value_error_global = 'The surrogate explanation is an empty dictionary.'
    value_error = 'One of the explanations is an empty dictionary.'

    assert len(caplog.records) == 0

    with pytest.raises(TypeError) as exin:
        fvl.plot_lime('test')
    assert str(exin.value) == type_error_global

    with pytest.raises(ValueError) as exin:
        fvl.plot_lime({})
    assert str(exin.value) == value_error_global

    with pytest.raises(TypeError) as exin:
        fvl.plot_lime({'': None})
    assert str(exin.value) == type_error_key_type

    with pytest.raises(TypeError) as exin:
        fvl.plot_lime({None: {}})
    assert str(exin.value) == type_error_key

    with pytest.raises(ValueError) as exin:
        fvl.plot_lime({'a': {}})
    assert str(exin.value) == value_error

    with pytest.raises(TypeError) as exin:
        fvl.plot_lime({'a': {None: 'test'}})
    assert str(exin.value) == type_error_tup1

    with pytest.raises(TypeError) as exin:
        fvl.plot_lime({'b': {'a': 'test'}})
    assert str(exin.value) == type_error_tup2

    with pytest.raises(TypeError) as exin:
        fvl.plot_lime({'a': {'c': 7}, 'b': None})
    assert str(exin.value) == type_error_val

    assert len(caplog.records) == 0


def test_plot_lime_classification(caplog):
    """
    Tests :func:`fatf.vis.lime.plot_lime` function for classification.
    """
    assert len(caplog.records) == 0
    logger_info = ('The explanations cannot share the y-axis as they use '
                   'different sets of interpretable features.')
    data_dict = {
        'class0': {'feat0 <= 0.00': -0.415,
                   '0.50 < feat1 <= 1.00': -0.280,
                   '0.07 < feat2 <= 0.22': 0.0377,
                   '0.34 < feat3 <= 0.58': -0.007},
        'class1': {'0.50 < feat1 <= 1.00': 0.202,
                   '0.07 < feat2 <= 0.22': -0.076,
                   'feat0 <= 0.00': 0.019,
                   '0.34 < feat3 <= 0.58': -0.018},
        'class2': {'feat0 <= 0.00': 0.395,
                   '0.50 < feat1 <= 1.00': 0.0775,
                   '0.07 < feat2 <= 0.22': 0.0392,
                   '0.34 < feat3 <= 0.58': 0.025}
    }  # yapf: disable
    classes = sorted(data_dict.keys())
    x_range = [-0.45, 0.43]  # min/max + 0.035 from the data above

    # All the classes have the same feature bounds so the plot should share the
    # y axis -- the first axis has the correct labels, the rest is empty
    y_labels = sorted(data_dict['class0'].keys())
    widths = [[0.0377, -0.007, -0.280, -0.415], [-0.076, -0.018, 0.202, 0.019],
              [0.0392, 0.025, 0.0775, 0.395]]
    colours = [[GREEN, RED, RED, RED], [RED, RED, GREEN, GREEN],
               [GREEN, GREEN, GREEN, GREEN]]
    widths_dict = []
    colours_dict = []
    for i, _ in enumerate(classes):
        widths_dict_ = []
        colours_dict_ = []
        for label in y_labels:
            y_labels_index = y_labels.index(label)
            widths_dict_.append(widths[i][y_labels_index])
            colours_dict_.append(colours[i][y_labels_index])
        widths_dict.append(widths_dict_)
        colours_dict.append(colours_dict_)

    # Test for a dictionary
    fig = fvl.plot_lime(data_dict)
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
        assert abs(len(data_dict[classes[i]]) - (y_rng[1] - y_rng[0])) < 0.2
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
    del data_dict['class1']['feat0 <= 0.00']
    y_labels = [sorted(data_dict[i].keys()) for i in classes]
    widths = [[data_dict[c][j] for j in y_labels[i]]
              for i, c in enumerate(classes)]
    colours = [[GREEN, RED, RED, RED], [RED, RED, GREEN],
               [GREEN, GREEN, GREEN, GREEN]]

    assert len(caplog.records) == 0
    fig = fvl.plot_lime(data_dict)
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == 'INFO'
    assert caplog.records[0].getMessage() == logger_info

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
        assert abs(len(data_dict[classes[i]]) - (y_rng[1] - y_rng[0])) < 0.2
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

    assert len(caplog.records) == 1


def test_plot_lime_regression(caplog):
    """
    Tests :func:`fatf.vis.lime.plot_lime` function for regression.
    """
    assert len(caplog.records) == 0
    data = {'feat0 <= 0.00': -0.415,
            '0.50 < feat1 <= 1.00': -0.280,
            '0.07 < feat2 <= 0.22': 0.0377,
            '0.34 < feat3 <= 0.58': -0.007}  # yapf: disable
    x_range = [-0.45, 0.08]  # min/max + 0.035 from the data above
    y_labels = sorted(data.keys())
    true_width = [0.0377, -0.007, -0.280, -0.415]
    true_colour = [GREEN, RED, RED, RED]
    true_title = ''

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

    assert len(caplog.records) == 0
