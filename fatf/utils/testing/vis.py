"""
The :mod:`fatf.utils.testing.vis` module holds visualisation testing functions.

This module holds functions that are of great help when testing visualisations
implemented in :mod:`fatf.vis` module. **This module requires the
``matplotlib`` package to be installed.**
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
#         Alex Hepburn <ah13558@bristol.ac.uk>
# License: new BSD

from typing import List, Tuple, Union

import numpy as np

try:
    import matplotlib.lines
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError('Visualisation testing helper functions require '
                      'matplotlib Python module, which is not installed '
                      'on your system.')

__all__ = ['get_plot_data', 'get_line_data', 'get_bar_data']


def get_plot_data(
        plot_axis: plt.Axes) -> Tuple[str, str, List[float], str, List[float]]:
    """
    Extracts plot's title, x-axis name and range and y-axis name and range.

    Parameters
    ----------
    plot_axis : matplotlib.pyplot.Axes
        A matplotlib axis from which all of the aforementioned information will
        be extracted.

    Returns
    -------
    plot_title : string
        Plot's title.
    plot_x_label : string
        Label of the plot's x-axis.
    plot_x_range : List[Number]
        Range of the plot's x-axis.
    plot_y_label : string
        Label of the plot's y-axis.
    plot_y_range : List[Number]
        Range of the plot's y-axis.
    """
    assert isinstance(plot_axis, plt.Axes), 'Must be a matplotlib axis.'

    plot_title = plot_axis.get_title()
    plot_x_label = plot_axis.xaxis.get_label_text()
    plot_x_range = plot_axis.xaxis.get_view_interval()
    plot_y_label = plot_axis.yaxis.get_label_text()
    plot_y_range = plot_axis.yaxis.get_view_interval()

    return plot_title, plot_x_label, plot_x_range, plot_y_label, plot_y_range


def get_line_data(
        line_plot: Union[matplotlib.lines.Line2D,  # yapf: disable
                         matplotlib.collections.LineCollection],
        is_collection: bool = False
) -> Tuple[Union[np.ndarray, List[np.ndarray]], str, float, str, float]:
    """
    Extracts line's data array, colour, alpha channel, label and width.

    Parameters
    ----------
    line_plot : Union[matplotlib.lines.Line2D, \
matplotlib.collections.LineCollection]
        A matplotlib line object extracted from a plot's axis.
    is_collection : boolean, optional (default=False)
        If ``True``, the ``line_plot`` will be treated as a ``LineCollection``.
        Otherwise, it will be treated as a ``Line2D``.

    Returns
    -------
    line_data : Union[numpy.ndarray, List[numpy.ndarray]]
        For ``Line2D`` this will be a numpy array representing x-axis and
        y-axis values used to interpolate a line. On the other hand, for a
        ``LineCollection`` this will be a list of numpy arrays, each
        representing a single line in the collection.
    line_colour : string
        Line colour.
    line_alpha : float
        Line transparency expressed as the alpha channel.
    line_label : string
        Line label used for the plot's legend.
    line_width : float
        Line width.
    """
    assert isinstance(line_plot, (matplotlib.lines.Line2D,
                                  matplotlib.collections.LineCollection)), \
        'Must be a line.'

    if is_collection:
        line_data = line_plot.get_segments()
    else:
        line_data = line_plot.get_xydata()
    line_colour = line_plot.get_color()
    line_alpha = line_plot.get_alpha()
    line_label = line_plot.get_label()
    line_width = line_plot.get_linewidth()

    return line_data, line_colour, line_alpha, line_label, line_width


def get_bar_data(
        plot_axis: plt.Axes
) -> Tuple[str, List[str], List[float], List[str], List[float], List[float],
           List[Tuple[float, float, float, float]]]:
    """
    Extracts plot's title, x-axis name and range and y-axis name and range.

    Parameters
    ----------
    plot_axis : matplotlib.pyplot.Axes
        A matplotlib axis from which all of the aforementioned information will
        be extracted.

    Returns
    -------
    plot_title : string
        Plot's title.
    plot_x_tick_names : List[string]
        Tick labels of the plot's x-axis.
    plot_x_range : List[Number]
        Range of the plot's x-axis.
    plot_y_tick_names : List[string]
        Tick labels of the plot's y-axis.
    plot_y_range : List[Number]
        Range of the plot's y-axis.
    plot_bar_width : List[Number]
        Bar width of every bar in the plot.
    plot_bar_colours : List[Tuple[float, float, float, float]]
        Bar colour of every bar in the plot. This is represented as an (r, g,
        b, alpha) tuple.
    """
    assert isinstance(plot_axis, plt.Axes), 'Must be a matplotlib axis.'

    plot_title = plot_axis.get_title()
    plot_x_tick_names = [
        x.get_text() for x in plot_axis.xaxis.get_ticklabels()
    ]
    plot_x_range = plot_axis.xaxis.get_view_interval()
    plot_y_tick_names = [
        y.get_text() for y in plot_axis.yaxis.get_ticklabels()
    ]
    plot_y_range = plot_axis.yaxis.get_view_interval()
    plot_bar_width = [ybar.get_width() for ybar in plot_axis.patches]
    plot_bar_colours = [ybar.get_facecolor() for ybar in plot_axis.patches]

    return (plot_title, plot_x_tick_names, plot_x_range, plot_y_tick_names,
            plot_y_range, plot_bar_width, plot_bar_colours)
