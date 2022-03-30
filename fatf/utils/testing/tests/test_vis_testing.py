"""
Tests visualisation helper functions for tests.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import pytest

try:
    import matplotlib.collections
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    pytest.skip(
        'Skipping visualisation testing tests -- missing matplotlib.',
        allow_module_level=True)

import importlib
import sys

import numpy as np

import fatf.utils.testing.imports as futi
import fatf.utils.testing.vis as futv


def test_import_when_missing():
    """
    Tests importing :mod:`fatf.utils.testing.vis` module sans matplotlib.
    """
    assert 'fatf.utils.testing.vis' in sys.modules
    exception_msg = ('Visualisation testing helper functions require '
                     'matplotlib Python module, which is not installed '
                     'on your system.')
    with futi.module_import_tester('matplotlib', when_missing=True):
        with pytest.raises(ImportError) as excinfo:
            importlib.reload(futv)
        assert str(excinfo.value) == exception_msg
    assert 'fatf.utils.testing.vis' in sys.modules


def test_get_plot_data():
    """
    Tests importing :mod:`fatf.utils.testing.vis.get_plot_data` function.
    """
    true_title = 'my title'
    true_x_label = 'x label'
    true_x_range = [-0.42, 12.34]
    true_y_label = 'y label'
    true_y_range = [-7, 7]

    plot_figure, plot_axis = plt.subplots(1, 1)
    plot_axis.set_title(true_title)
    plot_axis.set_xlim(true_x_range)
    plot_axis.set_xlabel(true_x_label)
    plot_axis.set_ylim(true_y_range)
    plot_axis.set_ylabel(true_y_label)

    title, x_label, x_range, y_label, y_range = futv.get_plot_data(plot_axis)

    assert true_title == title
    assert true_x_label == x_label
    assert true_y_label == y_label
    assert np.array_equal(true_y_range, y_range)
    assert np.array_equal(true_x_range, x_range)


def test_get_line_data():
    """
    Tests importing :mod:`fatf.utils.testing.vis.get_line_data` function.
    """
    # Test collection
    true_data = [[[0, 1], [0, 1]], [[4, 3], [0, 5]]]
    true_label = 'my label'
    true_colour = 'green'
    true_alpha = 0.5
    true_width = 7

    plot_figure, plot_axis = plt.subplots(1, 1)
    line_collection = matplotlib.collections.LineCollection(
        true_data,
        label=true_label,
        color=true_colour,
        alpha=true_alpha,
        linewidth=true_width)
    plot_axis.add_collection(line_collection)

    assert len(plot_axis.collections) == 1
    data, colour, alpha, label, width = futv.get_line_data(
        plot_axis.collections[0], is_collection=True)

    assert np.array_equal(true_data, data)
    assert np.allclose([[0.0, 0.5, 0.0, 0.5]], colour, atol=1e-2)
    assert true_alpha == alpha
    assert true_label == label
    assert true_width == width

    # Test a line
    true_data_x = [0, 1, 2, 3, 4]
    true_data_y = [5, 10, 15, 10, 5]

    plot_figure, plot_axis = plt.subplots(1, 1)
    plot_axis.plot(
        true_data_x,
        true_data_y,
        color=true_colour,
        linewidth=true_width,
        alpha=true_alpha,
        label=true_label)

    assert len(plot_axis.lines) == 1
    data, colour, alpha, label, width = futv.get_line_data(
        plot_axis.lines[0], is_collection=False)

    assert data.shape == (5, 2)
    assert np.array_equal(true_data_x, data[:, 0])
    assert np.array_equal(true_data_y, data[:, 1])
    assert true_colour == colour
    assert true_alpha == alpha
    assert true_label == label
    assert true_width == width

    data, colour, alpha, label, width = futv.get_line_data(plot_axis.lines[0])

    assert data.shape == (5, 2)
    assert np.array_equal(true_data_x, data[:, 0])
    assert np.array_equal(true_data_y, data[:, 1])
    assert true_colour == colour
    assert true_alpha == alpha
    assert true_label == label
    assert true_width == width


def test_get_bar_data():
    """
    Tests importing :mod:`fatf.utils.testing.vis.get_bar_data` function.
    """
    true_title = 'my title'
    true_y_tick_names = ['a', 'b', 'c']
    true_y_range = [0, 1, 2]
    true_bar_widths = [0.25, 0.7, 0.5]
    true_colours = ['red', 'green', 'red']
    true_colours_rgb = [(1, 0, 0, 1), (0.0, 0.502, 0.0, 1.0), (1, 0, 0, 1)]

    plot_figure, plot_axes = plt.subplots(1, 2, sharey=True, sharex=True)
    for axis in plot_axes:
        axis.barh(
            true_y_range, true_bar_widths, align='center', color=true_colours)
        axis.set_yticks(true_y_range)
        axis.set_yticklabels(true_y_tick_names)
        axis.set_title(true_title)

    for i in range(2):
        axis = plot_axes[i]
        tpl = futv.get_bar_data(axis)
        title, x_names, x_range, y_names, y_range, widths, colours = tpl

        assert true_title == title
        #
        for j in x_names:
            assert j == ''
        assert len(x_range) == 2
        assert abs(max(true_bar_widths) - (x_range[1] - x_range[0])) < 0.1
        #
        if i == 0:
            assert np.array_equal(true_y_tick_names, y_names)
        assert len(y_range) == 2
        assert abs(len(true_y_range) - (y_range[1] - y_range[0])) < 0.1
        #
        assert np.array_equal(true_bar_widths, widths)
        #
        assert len(true_colours_rgb) == len(colours)
        for j in range(len(true_colours_rgb)):
            assert len(true_colours_rgb[j]) == len(colours[j])
            for k in range(len(true_colours_rgb[j])):
                assert abs(true_colours_rgb[j][k] - colours[j][k]) < 0.001
