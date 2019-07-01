"""
The :mod:`fatf.vis.feature_influence` module visualises feature influence.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
#         Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

from numbers import Number
from typing import List, Optional, Tuple, Union

import matplotlib.collections

import matplotlib.pyplot as plt
import numpy as np

import fatf.utils.array.validation as fuav

from fatf.exceptions import IncorrectShapeError

__all__ = ['plot_individual_conditional_expectation',
           'plot_partial_dependence']  # yapf: disable


def _validate_input(ice_pdp_array: np.ndarray,
                    feature_linespace: np.ndarray,
                    class_index: int,
                    feature_name: Union[None, str],
                    class_name: Union[None, str],
                    plot_axis: Union[None, plt.Axes],
                    test_partial_dependence: bool = False) -> bool:
    """
    Validates input parameters for ICE and PD plotting functions.

    Validates input parameters for
    :func:`fatf.vis.feature_influence.plot_individual_conditional_expectation`
    and :func:`fatf.vis.feature_influence.plot_partial_dependence` functions.

    Parameters
    ----------
    ice_pdp_array : numpy.ndarray
        An array that contains ICE or PD calculations.
    feature_linespace : numpy.ndarray
        An array that contains the values for which the selected feature was
        sampled.
    class_index : integer
        The index of the class for which the plot will be created.
    feature_name : string or None
        The name of the feature for which ICE or PD was originally calculated.
    class_name : string or None
        The name of the class that ``class_index`` parameter points to.
    plot_axis : matplotlib.pyplot.Axes or None
        A matplotlib axis object to plot on top of.
    test_partial_dependence : boolean
        Whether to treat the input array as PD or ICE calculation result.

    Raises
    ------
    IncorrectShapeError
        The ICE or the PD array has a wrong number of dimensions (3 and 2
        respectively). The feature linespace array has a wrong number of
        dimensions -- 1 is expected.
    IndexError
        The class index is invalid for the input array.
    TypeError
        The class index is not an integer; the feature name is not a string or
        a ``None``; the class name is not a string or a ``None``; the plot axis
        is not a matplotlib.pyplot.Axes type object or a ``None``.
    ValueError
        The input array is structured or is not numerical. The linespace array
        is structured, not numerical or its length does not agree with the
        number of steps in the input array.

    Returns
    -------
    input_is_valid : boolean
        ``True`` if the input is valid, ``False`` otherwise.
    """
    # pylint: disable=too-many-arguments,too-many-branches
    input_is_valid = False

    assert isinstance(test_partial_dependence, bool), \
        'test_partial_dependence is not a boolean.'

    if fuav.is_structured_array(ice_pdp_array):
        raise ValueError('The input array cannot be a structured array.')
    if not fuav.is_numerical_array(ice_pdp_array):
        raise ValueError('The input array has to be a numerical array.')

    if test_partial_dependence:
        if len(ice_pdp_array.shape) != 2:
            raise IncorrectShapeError('plot_partial_depenedence expects a '
                                      '2-dimensional array of shape (n_steps, '
                                      'n_classes).')
    else:
        if len(ice_pdp_array.shape) != 3:
            raise IncorrectShapeError('plot_individual_condtional_expectation '
                                      'expects a 3-dimensional array of shape '
                                      '(n_samples, n_steps, n_classes).')

    if fuav.is_structured_array(feature_linespace):
        raise ValueError('The linespace array cannot be a structured array.')
    if not fuav.is_1d_array(feature_linespace):
        raise IncorrectShapeError('The linespace array has to be a '
                                  '1-dimensional array of shape (n_steps, ).')
    if not fuav.is_numerical_array(feature_linespace):
        raise ValueError('The linespace array has to be numerical.')
    if feature_linespace.shape[0] != ice_pdp_array.shape[-2]:
        raise ValueError('The length of the linespace array ({}) does not '
                         'agree with the number of linespace steps ({}) in '
                         'the input array.'.format(feature_linespace.shape[0],
                                                   ice_pdp_array.shape[-2]))

    # Is the index valid for the array
    if not isinstance(class_index, int):
        raise TypeError('Class index has to be an integer.')
    if class_index < 0 or class_index >= ice_pdp_array.shape[-1]:
        raise IndexError('Class index {} is not a valid index for the '
                         'input array. There are only {} classes '
                         'available.'.format(class_index,
                                             ice_pdp_array.shape[-1]))

    if feature_name is not None and not isinstance(feature_name, str):
        raise TypeError('The feature name has to be either None or a string.')

    if class_name is not None and not isinstance(class_name, str):
        raise TypeError('The class name has to be either None or a string.')

    if plot_axis is not None and not isinstance(plot_axis, plt.Axes):
        raise TypeError('The plot axis has to be either None or a matplotlib.'
                        'pyplot.Axes type object.')

    input_is_valid = True
    return input_is_valid


def _prepare_a_canvas(
        plot_title: str,
        plot_axis: Union[None, plt.Axes],
        class_index: int,
        class_name: Union[None, str],
        feature_name: Union[None, str],
        x_range: List[float]
) -> Tuple[Union[plt.Figure, None], plt.Axes]:  # yapf: disable
    """
    Prepares a matplotlib axis (canvas) for ICE and PDP plotting.

    If the ``plot_axis`` parameter is ``None`` a new matplotlib axis is
    created with the provided name and axes labels. Otherwise, the title of the
    provided ``plot_axis`` is extended with the provided title and the axes
    labels are overwritten if either ``class_name`` or ``feature_name`` is
    given.

    Parameters
    ----------
    plot_title : string
        The title of the plot. If an axis is already provided, this title will
        be appended to the current title of the ``plot_axis``.
    plot_axis : matplotlib.pyplot.Axes, optional (default=None)
        Either ``None`` to create a new matplotlib axis or an axis object to
        plot on top of. In the latter case the ``plot_title`` will be appended
        to the title of the current plot and the axes names will be overwritten
        if they are provided -- ``class_name`` and ``feature_name`` -- (are not
        ``None``).
    class_index : integer
        The index of the class for which the plot will be created. Used to
        generate the y-axis label if ``class_name`` is not provided (is
        ``None``).
    class_name : string, optional (default=None)
        The name of the class that ``class_index`` parameter points to. If
        ``None``, the class name will be the same as the class index. It is
        used to generate a name for the y-axis.
    feature_name : string, optional (default=None)
        The name of the feature for which ICE or PD was originally calculated.
        It is used to generate a name for the x-axis.
    x_range : List[Number]
        A list of 2 numbers where the first one determines the minimum of the
        x-axis range and the second one determines the maximum of the x-axis
        range.

    Raises
    ------
    ValueError
        If the ``plot_axis`` attribute is provided, and is not ``None`` this
        exception will be raised if the range of either of the axes does not
        agree with the range of the axes of the ``plot_axis`` plot.

    Returns
    -------
    plot_figure : Union[matplotlib.pyplot.Figure, None]
        A matplotlib figure that holds the ``plot_axis`` axis. This parameter
        is ``None`` when the user passed in ``plot_axis`` attribute, otherwise,
        when a blank plot is created, this is a figure object holding the plot
        axis (``plot_axis``).
    plot_axis : matplotlib.pyplot.Axes
        A matplotlib axes with x- and y-axes description and a plot title.
    """
    # pylint: disable=too-many-arguments,too-many-branches
    assert isinstance(plot_title, str), 'Must be string.'
    assert plot_axis is None or isinstance(plot_axis, plt.Axes), \
        'Must be None or matplotlib axis.'
    assert isinstance(class_index, int), 'Must be integer.'
    assert class_name is None or isinstance(class_name, str), \
        'Must be None or string.'
    assert feature_name is None or isinstance(feature_name, str), \
        'Must be None or string.'
    assert isinstance(x_range, list), 'Must be list.'
    assert len(x_range) == 2, 'x_range should only contain 2 numbers.'
    assert isinstance(x_range[0], Number) and isinstance(x_range[1], Number), \
        'Both elements of x_range should be numbers.'
    assert x_range[0] < x_range[1], \
        'The first element of x_range should be smaller than the second one.'

    if plot_axis is None:
        if class_name is None:
            class_name = '{} (class index)'.format(class_index)
        if feature_name is None:
            feature_name = "Selected Feature's Linespace"

        plot_figure, plot_axis = plt.subplots(1, 1)
        plot_axis.set_title(plot_title)
        plot_axis.set_xlim(x_range)
        plot_axis.set_xlabel(feature_name)
        plot_axis.set_ylim(np.array([-0.05, 1.05]))
        plot_axis.set_ylabel('{} class probability'.format(class_name))
    else:
        plot_figure = None
        # Feature range should be the same
        current_x_range = plot_axis.xaxis.get_view_interval()
        if not np.array_equal(current_x_range, x_range):
            raise ValueError('The x-axis range of the plot given in the '
                             'plot_axis parameter differs from the x-axis '
                             'range of this plot.')
        current_y_range = plot_axis.yaxis.get_view_interval()
        if not np.array_equal(current_y_range, [-0.05, 1.05]):
            raise ValueError('The y-axis range of the plot given in the '
                             'plot_axis parameter differs from the y-axis '
                             'range of this plot.')

        # Extend plot title
        if plot_title:
            current_title = plot_axis.get_title()
            plot_axis.set_title('{} &\n{}'.format(current_title, plot_title))

        # What about axes names
        if feature_name is None:
            # Only name if it is empty
            current_x_label = plot_axis.xaxis.get_label_text()
            if not current_x_label:
                plot_axis.set_xlabel("Selected Feature's Linespace")
        else:
            # Overwrite x description
            plot_axis.set_xlabel(feature_name)

        if class_name is None:
            # Only name if it is empty
            current_y_label = plot_axis.yaxis.get_label_text()
            if not current_y_label:
                plot_axis.set_ylabel(
                    '{} (class index) class probability'.format(class_index))
        else:
            # Overwrite y description
            plot_axis.set_ylabel('{} class probability'.format(class_name))

    return plot_figure, plot_axis


def plot_individual_conditional_expectation(
        ice_array: np.ndarray,
        feature_linespace: np.ndarray,
        class_index: int,
        feature_name: Optional[str] = None,
        class_name: Optional[str] = None,
        plot_axis: Optional[plt.Axes] = None
) -> Tuple[Union[plt.Figure, None], plt.Axes]:
    """
    Plots Individual Conditional Expectation for a selected class.

    Parameters
    ----------
    ice_array : numpy.ndarray
        An array of (n_samples, n_steps, n_classes) shape with Individual
        Conditional Expectation calculation results for every target class for
        the desired spectrum of the selected feature. This should be the output
        of the :func:`fatf.transparency.models.feature_influence.\
individual_conditional_expectation` function.
    feature_linespace : numpy.ndarray
        A one-dimensional array -- (steps_number, ) -- with the values for
        which the selected feature was sampled when the dataset was evaluated
        for a predictive model. This should be the output of the
        :func:`fatf.transparency.models.feature_influence.\
individual_conditional_expectation` function.
    class_index : integer
        The index of the class for which ICE will be plotted, taken from the
        original dataset.
    feature_name : string, optional (default=None)
        The name of the feature for which ICE was originally calculated.
    class_name : string, optional (default=None)
        The name of the class that ``class_index`` parameter points to. If
        ``None``, the class name will be the same as the class index.
    plot_axis : matplotlib.pyplot.Axes, optional (default=None)
        A matplotlib axes on which the ICE will be plotted. This is useful if
        one wants to overlay multiple ICE plot on top of each other. If
        ``None``, a new axes will be created.

    Raises
    ------
    IncorrectShapeError
        The ICE or the PD array has a wrong number of dimensions (3 and 2
        respectively). The feature linespace array has a wrong number of
        dimensions -- 1 is expected.
    IndexError
        The class index is invalid for the input array.
    TypeError
        The class index is not an integer; the feature name is not a string or
        a ``None``; the class name is not a string or a ``None``; the plot axis
        is not a matplotlib.pyplot.Axes type object or a ``None``.
    ValueError
        The input array is structured or is not numerical. The linespace array
        is structured, not numerical or its length does not agree with the
        number of steps in the input array. If the ``plot_axis`` attribute is
        provided, and is not ``None`` this exception will be raised if the
        range of either of the axes does not agree with the range of the axes
        of the ``plot_axis`` plot.

    Returns
    -------
    plot_figure : Union[matplotlib.pyplot.Figure, None]
        A matplotlib figure that holds the ``plot_axis`` axis. This parameter
        is ``None`` when the user passed in ``plot_axis`` attribute, otherwise,
        when a blank plot is created, this is a figure object holding the plot
        axis (``plot_axis``).
    plot_axis : matplotlib.pyplot.Axes
        A matplotlib axes with the ICE plot.
    """
    # pylint: disable=too-many-arguments
    assert _validate_input(ice_array, feature_linespace, class_index,
                           feature_name, class_name, plot_axis,
                           False), 'Input is invalid.'

    plot_title = 'Individual Conditional Expectation'
    x_range = [feature_linespace[0], feature_linespace[-1]]
    plot_figure, plot_axis = _prepare_a_canvas(
        plot_title, plot_axis, class_index, class_name, feature_name, x_range)

    lines = np.zeros((ice_array.shape[0], ice_array.shape[1], 2),
                     dtype=ice_array.dtype)
    lines[:, :, 1] = ice_array[:, :, class_index]
    lines[:, :, 0] = feature_linespace

    line_collection = matplotlib.collections.LineCollection(
        lines, label='ICE', color='dimgray', alpha=0.5)
    plot_axis.add_collection(line_collection)
    plot_axis.legend()

    return plot_figure, plot_axis


def plot_partial_dependence(pd_array: np.ndarray,
                            feature_linespace: np.ndarray,
                            class_index: int,
                            feature_name: Optional[str] = None,
                            class_name: Optional[str] = None,
                            plot_axis: Optional[plt.Axes] = None
                            ) -> Tuple[Union[plt.Figure, None], plt.Axes]:
    """
    Plots Partial Dependence for a selected class.

    This function raises the same exceptions and errors as the
    :func:`fatf.vis.feature_influence.plot_individual_conditional_expectation`
    function. Please consult its documentation for the exact list or errors and
    exceptions and their description.

    Parameters
    ----------
    pd_array : numpy.ndarray
        An array of (n_steps, n_classes) shape with Partial Dependence
        calculation results for every target class for the desired spectrum of
        the selected feature. This should be the output of the
        :func:`fatf.transparency.models.feature_influence.partial_dependence`
        or :func:`fatf.transparency.models.feature_influence.\
partial_dependence_ice` function.
    feature_linespace : numpy.ndarray
        A one-dimensional array -- (steps_number, ) -- with the values for
        which the selected feature was sampled when the dataset was evaluated
        for a predictive model. This should be the output of the
        :func:`fatf.transparency.models.feature_influence.\
individual_conditional_expectation` function.
    class_index : integer
        The index of the class for which PD will be plotted, taken from the
        original dataset.
    feature_name : string, optional (default=None)
        The name of the feature for which PD was originally calculated.
    class_name : string, optional (default=None)
        The name of the class that ``class_index`` parameter points to. If
        ``None``, the class name will be the same as the class index.
    plot_axis : matplotlib.pyplot.Axes, optional (default=None)
        A matplotlib axes on which the PD will be plotted. This is useful if
        one wants to overlay PD onto an ICE plot. If ``None``, a new axes will
        be created.

    Returns
    -------
    plot_figure : Union[matplotlib.pyplot.Figure, None]
        A matplotlib figure that holds the ``plot_axis`` axis. This parameter
        is ``None`` when the user passed in ``plot_axis`` attribute, otherwise,
        when a blank plot is created, this is a figure object holding the plot
        axis (``plot_axis``).
    plot_axis : matplotlib.pyplot.Axes
        A matplotlib axes with the PD plot.
    """
    # pylint: disable=too-many-arguments
    assert _validate_input(pd_array, feature_linespace, class_index,
                           feature_name, class_name, plot_axis,
                           True), 'Input is invalid.'

    plot_title = 'Partial Dependence'
    x_range = [feature_linespace[0], feature_linespace[-1]]
    plot_figure, plot_axis = _prepare_a_canvas(
        plot_title, plot_axis, class_index, class_name, feature_name, x_range)

    plot_axis.plot(
        feature_linespace,
        pd_array[:, class_index],
        color='lightsalmon',
        linewidth=7,
        alpha=0.6,
        label='PD')
    plot_axis.legend()

    return plot_figure, plot_axis
