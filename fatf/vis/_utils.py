import matplotlib 
import matplotlib.pyplot as plt 
import numpy as np

from numbers import Number 
from typing import Tuple, List

def _get_bar_plot_data(plot_axis: plt.Axes
                   ) -> Tuple[str, str, List[Number], str, List[Number], 
                              List[Number]]:
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
        Tick labels of the plot's x-axis.
    plot_x_range : List[Number]
        Range of the plot's x-axis.
    plot_y_label : string
        Tick labels of the plot's y-axis.
    plot_y_range : List[Number]
        Range of the plot's y-axis.
    plot_bar_width : List[Number]
        Bar width for each bar in the plot.
    """
    assert isinstance(plot_axis, plt.Axes), 'Must be None or matplotlib axis.'

    plot_title = plot_axis.get_title()
    plot_x_label = [x.get_text() for x in plot_axis.xaxis.get_ticklabels()]
    plot_x_range = plot_axis.xaxis.get_view_interval()
    plot_y_label = [x.get_text() for x in plot_axis.yaxis.get_ticklabels()]
    plot_y_range = plot_axis.yaxis.get_view_interval()
    plot_bar_width = [x.get_width() for x in plot_axis.patches]

    return (plot_title, plot_x_label, plot_x_range, plot_y_label, plot_y_range,
            plot_bar_width)
