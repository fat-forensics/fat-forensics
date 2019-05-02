"""
Partial Dependence and Individual Conditional Expectation plotting functions.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
#         Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

from numbers import Number
from typing import List, Optional, Tuple, Union
import warnings

import matplotlib.collections
from matplotlib import gridspec

import matplotlib.pyplot as plt
import numpy as np

import fatf.utils.array.validation as fuav

from fatf.exceptions import IncorrectShapeError

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

__all__ = ['plot_individual_conditional_expectation',
           'plot_feature_distribution',
           'plot_partial_dependence']  # yapf: disable


def _validate_input(ice_pdp_array: np.ndarray,
                    feature_linespace: Union[np.ndarray, Tuple[np.ndarray, ...]],
                    class_index: int,
                    treat_as_categorical: Union[None, bool, List[bool]],
                    feature_name: Union[None, str, List[str]],
                    class_name: Union[None, str],
                    plot_axis: Union[None, plt.Axes],
                    variance: Union[None, np.ndarray],
                    variance_area: Union[None, bool],
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
    treat_as_categorical : boolean
        Whether or not to treat the feature as categorical
    feature_name : string or None
        The name of the feature for which ICE or PD was originally calculated.
    class_name : string or None
        The name of the class that ``class_index`` parameter points to.
    plot_axis : matplotlib.pyplot.Axes or None
        A matplotlib axis object to plot on top of.
    variance : numpy.ndarray
        An array that contains variance in prediction values used in PD plot.
        This will only be validated if test_partial_dependence = True.
    variance_area : boolean
        Whether to plot variance as area or error bars. If true, the area
        will be plotted, else error bars will be used.
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

    if not isinstance(test_partial_dependence, bool):
        raise TypeError('test_partial_dependence is not a boolean.')

    if variance_area is not None and not isinstance(variance_area, bool):
        raise TypeError('variance_area is not a boolean.')

    if not isinstance(treat_as_categorical, list):
        treat_as_categorical = [treat_as_categorical]

    if len(treat_as_categorical) > 2:
        raise ValueError('Too many values given for treat_as_categorical.')
    for categorical in treat_as_categorical:
        if (not isinstance(categorical, bool) and categorical is not None):
            raise TypeError('treat_as_categorical has to either be None, a '
                            'boolean or a list of None and booleans.')

    if fuav.is_structured_array(ice_pdp_array):
        raise ValueError('The input array cannot be a structured array.')

    if not fuav.is_numerical_array(ice_pdp_array):
        raise ValueError('The input array has to be a numerical array.')

    if not isinstance(feature_name, list):
        feature_name = [feature_name]
    if len(feature_name) > 2:
        raise ValueError('Too many values given for feature_name.')

    if not isinstance(feature_linespace, tuple):
        feature_linespace = [feature_linespace]

    if len(feature_linespace) != 1 and len(feature_linespace) != 2:
        raise ValueError('feature_linespace must be a tuple of numpy.ndarray '
                         'with max length 2 for use in 2 feature partial '
                         'dependence and individual conditional expectation.')
    for linespace in feature_linespace:
        if not isinstance(linespace, np.ndarray):
            raise TypeError('feature_linespace must be a tuple of '
                            'numpy.ndarray')
        if fuav.is_structured_array(linespace):
            raise ValueError('The linespace array cannot be a structured '
                             'array.')
        if not fuav.is_1d_array(linespace):
            raise IncorrectShapeError('The linespace array has to be a '
                                      '1-dimensional array of shape '
                                      '(n_steps, ).')
    if test_partial_dependence:
        if len(ice_pdp_array.shape) != 2 and len(ice_pdp_array.shape) != 3:
            raise IncorrectShapeError('plot_partial_depenedence expects a '
                                      '2-dimensional array of shape (n_steps, '
                                      'n_classes) or for 2-D partial '
                                      'dependence, a shape of (n_steps_1, '
                                      'n_steps_2, n_classes).')
        if len(ice_pdp_array.shape) == 3:
            if len(feature_linespace) != 2:
                raise ValueError('A 3-dimensional partial dependence array '
                                 'was provided but only one feature linespace. '
                                 'A 3-dimensional array can only be used in '
                                 'plotting 2 feature partial depedence, and '
                                 'as such a tuple of 2 feature linespaces must '
                                 'be given.')
            if feature_name[0] is not None and len(feature_name) != 2:
                raise ValueError('A 3-dimension partial dependence array '
                                 'was provided but only one feature name. '
                                 'In order to use feature names you must '
                                 'provide a list of length 2 containing '
                                 'strings or None.')
        if isinstance(variance, np.ndarray):
            if len(ice_pdp_array.shape) == 3:
                msg = ('A variance array was provided but a 3-dimensional '
                       'array was also given. To plot a 2-feature partial '
                       'depenedence, variance cannot be used. The variance '
                       'array will be ignored.')
                warnings.warn(msg, category=UserWarning)
            if fuav.is_structured_array(variance):
                raise ValueError('The variance array cannot be a structured '
                                 'array.')
            if not fuav.is_numerical_array(variance):
                raise ValueError('The variance array has to be a numerical '
                                 'array.')
            if variance.shape[0] != ice_pdp_array.shape[-2]:
                raise ValueError(
                    'The length of the variance array ({}) does agree with '
                    'the number of linespace steps ({}) in the input '
                    'array.'.format(variance.shape[0], 
                                    ice_pdp_array.shape[-2]))
        else:
            if variance_area:
                raise ValueError('Variance vector has not been given but '
                                 'variance_area has been given as True. To '
                                 'plot the variance please specify a variance '
                                 'vector.')
    else:
        if len(ice_pdp_array.shape) != 3 and len(ice_pdp_array.shape) != 4:
            raise IncorrectShapeError('plot_individual_condtional_expectation '
                                      'expects a 3-dimensional array of shape '
                                      '(n_samples, n_steps, n_classes) or for '
                                      '2-D individual conditional expectation, '
                                      'a shape of (n_samples, n_steps_1, '
                                      'n_steps_2, n_classes).')
        if len(ice_pdp_array.shape) == 4:
            if len(feature_linespace) != 2:
                raise ValueError('A 4-dimensional individual conditional '
                                 'expectation array was given but only one '
                                 'feature linespace. A 4-dimensional array '
                                 'can only be used in plotting 2 feature '
                                 'individual conditional expectation, and as '
                                 'such a tuple of 2 feature linespaces must '
                                 'be given.')
            if feature_name[0] is not None and len(feature_name) != 2:
                raise ValueError('A 4-dimension individual conditional '
                                'expectation array was provided but only '
                                'one feature name. In order to use feature '
                                'names you must provide a list of length 2 '
                                'containing strings or None.')
        else:
            if len(feature_linespace) > 1:
                raise ValueError('A 3-dimenionsal individual conditional '
                                 'expectation array was given but 2 feature '
                                 'linespaces. To plot 2 feature individual '
                                 'conditional expectation, a 4-dimensional '
                                 'array must be provided.')
    #if not fuav.is_numerical_array(feature_linespace):
    #    raise ValueError('The linespace array has to be numerical.')
    for linespace, index in zip(feature_linespace, [-2, -3]):
        if linespace.shape[0] != ice_pdp_array.shape[index]:
            raise ValueError('The length of the linespace array ({}) does not '
                             'agree with the number of linespace steps ({}) '
                             'in the input array.'.format(
                                linespace.shape[0], ice_pdp_array.shape[index]))

    # Is the index valid for the array
    if not isinstance(class_index, int):
        raise TypeError('Class index has to be an integer.')
    if class_index < 0 or class_index >= ice_pdp_array.shape[-1]:
        raise IndexError('Class index {} is not a valid index for the '
                         'input array. There are only {} classes '
                         'available. For plotting data computed using a '
                         'regression model, use '
                         'class_index=0.'.format(class_index,
                                                 ice_pdp_array.shape[-1]))
    for feature in feature_name:
        if (feature is not None and not isinstance(feature, str)):
            raise TypeError('The feature name has to be either None or a '
                            'string or a list of strings.')

    if class_name is not None and not isinstance(class_name, str):
        raise TypeError('The class name has to be either None or a string.')

    if plot_axis is not None and not isinstance(plot_axis, plt.Axes):
        raise TypeError('The plot axis has to be either None or a matplotlib.'
                        'pyplot.Axes type object.')

    input_is_valid = True
    return input_is_valid


def _validate_feature(feature_distribution: Tuple[np.ndarray, np.ndarray],
                      treat_as_categorical: bool,
                      feature_name: Union[None, str],
                      feature_linespace: Union[None, np.ndarray],
                      test_feature_linespace: bool,
                      orientation: Union[None, str],
                      plot_axis: Union[None, plt.Axes]) -> bool:
    """
    Validates input for feature distribution function.

    Validates input parameters for
    :func:`fatf.vis.feature_influence.plot_feature_distribution`

    Parameters
    ----------
    feature_distribution : List[numpy.ndarray]
        A list of numpy.ndarray 
    treat_as_categorical : boolean
        Whether or not to treat the feature as categorical
    feature_name : string or None
        The name of the feature for which ICE or PD was originally calculated.
    feature_linespace : numpy.ndarray
        An array that contains the values for which the selected feature was
        sampled.
    test_feature_linespace : boolean
        Whether to not to test if feature_linespace agrees with 
    Returns
    -------
    input_is_valid : boolean
        ``True`` if the input is valid, ``False`` otherwise.
    """
    input_is_valid = False

    if plot_axis is not None and not isinstance(plot_axis, plt.Axes):
        raise TypeError('The plot axis has to be either None or a matplotlib.'
                        'pyplot.Axes type object.')
    
    if feature_name is not None and not isinstance(feature_name, str):
        raise TypeError('The feature name has to be either None or a string.')

    if not isinstance(test_feature_linespace, bool):
        raise TypeError('test_feature_linespace is not a boolean.')

    if not isinstance(treat_as_categorical, bool):
        raise TypeError('treat_as_categorical is not a boolean.')

    if orientation is not None and not isinstance(orientation, str):
        raise TypeError('Orientation must be either None or a string with '
                        'value \'vertical\' or \'horizontal\'.')

    if orientation is not None and orientation not in \
            ['vertical', 'horizontal']:
        raise ValueError('Orientation must be either \'vertical\' or '
                         '\'horizontal\'.')

    if not isinstance(feature_distribution, tuple):
        raise TypeError('Feature distribution has to be a tuple.')


    if len(feature_distribution) != 2:
        raise ValueError('Feature distribution has to be a tuple of length 2 '
                         'where the first element is a values array and the '
                         'second element is a counts array.')

    for i in range(2):
        if not isinstance(feature_distribution[i], np.ndarray):
            raise TypeError('The {} element in feature_distribution array '
                            'must be of type np.ndarray.'.format(i))
        if fuav.is_structured_array(feature_distribution[i]):
            raise TypeError('The {} element in feature_distribution array '
                            'cannot be a structured array.'.format(i))

    values, counts = feature_distribution
    if not fuav.is_numerical_array(counts):
        raise ValueError('The 1 element of feature_distribution has to be a '
                         'numerical array.')
    if treat_as_categorical:
        if values.shape[0] != counts.shape[0]:
            raise ValueError('For categorical data, values and counts array '
                             'must be of the same shape.')
        else:
            if counts.max() > 1.0:
                raise ValueError('Distribution cannot have value more than '
                                 '1.0')
    else:
        # Histogram or bar plot
        if (values.shape[0] == counts.shape[0] + 1):
            if counts.max() > 1.0:
                raise ValueError('Distribution cannot have value more than '
                                 '1.0')
        # KDE needs to have values.shape[0] == counts.shape[0]
        elif not values.shape[0] == counts.shape[0]:
            raise ValueError('Values shape {} and counts shape {} do not '
                             'agree. In order to define histograms, values has '
                             'to be of shape (counts.shape[0]+1, ). In order '
                             'to define Gaussian Kernel, values and counts '
                             'must be of the same shape.'.format(
                             values.shape[0], counts.shape[0]))
    # test_feature_linespace only called from PD and ICE plotting functions
    # so feature_linespace has already been validated.
    if test_feature_linespace and treat_as_categorical:
        # Need to test if treat_as_Categorical then value should be identical
        # to feature_linespace
        if not set(values) == set(feature_linespace):
            raise ValueError('To plot the feature distribution of categorical '
                             'features, the values array in '
                             'feature_distribution[0] must contain all values '
                             'of feature_linespace in it.')
    input_is_valid = True
    return input_is_valid


def _prepare_a_canvas(
        plot_title: str,
        plot_axis: Union[None, plt.Axes],
        class_index: int,
        class_name: Union[None, str],
        feature_name: Union[None, str, List[Union[str, None]]],
        x_range: List[Number],
        plot_distribution: bool = False,
        is_2d: bool = False,
        y_range: Optional[List[Number]] = None,
        plot_axis_3d: Optional[bool] = False,
) -> Tuple[Union[plt.Figure, None], Union[plt.Axes, Tuple[plt.Axes]]]: # yapf: disable
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
    plot_distribution: boolean (default=False)
        Whether to have another axis for plotting feature distribution
        underneath PD or ICE plot.
    is_2d: boolean (default=False)
        Whether the function is being called from 2-D ICE and PD.
    y_range: List[Number] (default=None)
        A list of 2 numbers where the first one determines the minimum of the
        y-axis range and the second one determines the maximum of the y-axis
        range. Only used when is_2d is True for setting y-axis of main axis
        and feature distribution is being plotted on y-axis.
    plot_axis_3d: boolean (default=False)
        A boolean to specify whether the `plot_axis` should be a 3-D axis for
        use in 2-feature individual conditional expectation. Should only be
        true when `_prepare_a_canvas` is called from :func:`fatf.vis.
        feature_influence._plot_individual_conditional_expectation_2d`
        function.
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
        A matplotlib axes with x- and y-axes description and a plot title
        or a tuple of axes where plot_axis[0] is the axis for plotting PD
        or ICE and plot_axis[1] is the axis for plotting feature distribution.
    """
    # pylint: disable=too-many-arguments,too-many-branches
    assert isinstance(plot_title, str), 'Must be string.'
    assert plot_axis is None or isinstance(plot_axis, plt.Axes), \
        'Must be None or matplotlib axis.'
    assert isinstance(class_index, int), 'Must be integer.'
    assert class_name is None or isinstance(class_name, str), \
        'Must be None or string.'
    assert feature_name is None or isinstance(feature_name, str) \
        or isinstance(feature_name, list), 'Must be None or string or list.'
    assert isinstance(x_range, list), 'Must be list.'
    assert len(x_range) == 2, 'x_range should only contain 2 numbers.'
    assert isinstance(x_range[0], Number) and isinstance(x_range[1], Number), \
        'Both elements of x_range should be numbers.'
    assert x_range[0] < x_range[1], (  # type: ignore
        'The first element of x_range should be smaller than the second one.')

    if plot_axis is None:
        if class_name is None:
            class_name = '{} (class index)'.format(class_index)
        if is_2d and not isinstance(feature_name, list):
            if feature_name is None:
                feature_name = ['feature 0', 'feature 1']
            elif isinstance(feature_name, str):
                feature_name = [feature_name]
                feature_name.append('feature 1')
        else:
            if feature_name is None:
                feature_name = 'Selected Feature\'s Linespace'
        projection = '3d' if plot_axis_3d else None
        plot_figure = plt.figure()
        if plot_distribution:
            if is_2d:
                gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], 
                                       height_ratios = [1, 4])
                plot_axis = plot_figure.add_subplot(gs[1, 0],frameon = False,
                                                    projection=projection)
                dist_axis_x = plot_figure.add_subplot(
                    gs[0,0], sharex=plot_axis, frameon=True)
                dist_axis_y = plot_figure.add_subplot(
                    gs[1, 1], sharey=plot_axis, frameon=True)
                dist_axis_x.set_xlim(x_range)
                dist_axis_x.set_ylim([-0.05, 1.05])
                dist_axis_y.set_ylim(y_range)
                dist_axis_y.set_xlim([-0.05, 1.05])
            else:
                plot_figure, (plot_axis, dist_axis) = plt.subplots(
                    2, 1, gridspec_kw = {'height_ratios':[4, 1]})
                dist_axis.set_xlabel('Distribution of Feature')
                dist_axis.set_xlim(x_range)
                dist_axis.patch.set_visible(False)
        else:
            plot_axis = plot_figure.add_subplot(111, projection=projection)
        plot_axis.set_title(plot_title)
        plot_axis.set_xlim(x_range)
        if is_2d:
            plot_axis.set_xlabel(feature_name[0])
            plot_axis.set_ylabel(feature_name[1])
            plot_axis.set_ylim(y_range)
        else:
            plot_axis.set_xlabel(feature_name)
            plot_axis.set_ylim(np.array([-0.05, 1.05]))
            plot_axis.set_ylabel('{} class probability'.format(class_name))
        if plot_axis_3d:
            plot_axis.set_zlabel('Instances')
            plot_axis.set_zticklabels([])
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

    if plot_distribution:
        if is_2d:
            plot_axis = (plot_axis, dist_axis_x, dist_axis_y)
        else:
            plot_axis = (plot_axis, dist_axis)

    return plot_figure, plot_axis


def plot_feature_distribution(
        feature_distribution : List[np.ndarray],
        treat_as_categorical: bool = False,
        feature_name: Optional[str] = None,
        orientation: Optional[str] = 'vertical',
        plot_axis: Optional[plt.Axes] = None
) -> Tuple[Union[plt.Figure, None], plt.Axes]:
    """
    Plots a feature distribution.

    For  exceptions raised by this function please see the documentation of 
    :func:`fatf.vis.feature_influence._validate_feature`

    explain different configurations of plotting the feature distribution


    Parameters
    ----------
    feature_linespace : numpy.ndarray
        A one-dimensional array -- (steps_number, ) -- with the values for
        which the selected feature was sampled when the dataset was evaluated
        for a predictive model. This should be the output of the :func:`fatf.
        transparency.models.feature_influence.
        individual_conditional_expectation` function.
    feature_distribution : List[numpy.ndarray]
        A list of length 2 where they are [values, counts]. The first
        array is of shape (samples,) or (samples+1,). It contains x-axis data
        points either for histogram, gaussian kde or unique values of the array
        The second element of the list should be probability densities for each
        value in the first array. This should be the output of the :func:`fatf.
        transparency.models.compute_feature_distribution`.
    feature_name : string, optional (default=None)
        The name of the feature for which feature distribution originally
        calculated. It is used to generate a name for the x-axis if plot_axis
        is None.
    orientation : string, optionl (default='vertical')
        Whether to plot distribution as vertical or horizontal. This is mainly
        used for 2-D ICE and PD plots.
    plot_axis : matplotlib.pyplot.Axes, optional (default=None)
        A matplotlib axes on which the feature distribution will be plotted.
        This is useful to have feature distribution as a subplot of an ICE
        or PD plot which can be achieved by passing the `feature_distribution`
        parameter to the funcions :func:`fatf.transparency.models.
        plot_individual_conditional_expectation` or :func:`fatf.transparency.
        models.plot_partial_dependence`.
        If  ``None``, a new axes will be created.

    Returns
    -------
    plot_figure : Union[matplotlib.pyplot.Figure, None]
        A matplotlib figure that holds the ``plot_axis`` axis. This parameter
        is ``None`` when the user passed in ``plot_axis`` attribute, otherwise,
        when a blank plot is created, this is a figure object holding the plot
        axis (``plot_axis``).
    plot_axis : matplotlib.pyplot.Axes
        A matplotlib axes with the feature distribution plot.
    """
    assert _validate_feature(
        feature_distribution, treat_as_categorical, feature_name, None, 
        False, orientation, plot_axis), 'Input is invalid.'
    is_horizontal = True if orientation == 'horizontal' else False
    values, counts = feature_distribution
    x_range = [values[0], values[-1]]
    if plot_axis is None:
        plot_title = 'Feature Distribution' if feature_name is None else \
            'Feature Distribution for {}'.format(feature_name)
        plot_figure, plot_axis = _prepare_a_canvas(
            plot_title, plot_axis, 0, None, feature_name, x_range,
            False)
        plot_axis.set_ylabel('Density')
    else:
        plot_figure = None
    if feature_name is None:
        feature_name = 'feature'
    if is_horizontal:
        plot_axis.set_xlim([0, 1.05])
    else:
        plot_axis.set_ylim([0, 1.05])
    if values.shape[0] == counts.shape[0] + 1 or treat_as_categorical:
        # Histogram or categorical data
        plot_values = values
        align = 'edge'
        if treat_as_categorical:
            widths = 0.8
            plot_values = np.linspace(0, values.shape[0], len(values))
            lim = [plot_values[0]-0.5, len(plot_values)+0.5]
            align = 'center'
        else:
            widths = [values[i+1]-values[i] for i in range(len(values)-1)]
            lim = [values[0], values[-1]]
            plot_values = plot_values[:-1]
        if is_horizontal:
            plot_axis.set_ylim(lim)
            bars = plot_axis.barh(plot_values, height=widths, width=counts,
                                  align=align, alpha=0.6, color='royalblue')
            xs = [[bar.get_width(), bar.get_y()+bar.get_height()/2.0,
                 bar.get_width()] for bar in bars]
            ha, va = 'left', 'center'
            if treat_as_categorical:
                plot_axis.set_yticks(plot_values)
                plot_axis.set_yticklabels(list(values))
            plot_axis.set_ylabel(
                'Histogram of values for {}'.format(feature_name),
                rotation=270,labelpad=17)
        else:
            plot_axis.set_xlim(lim)
            bars = plot_axis.bar(plot_values, height=counts, width=widths,
                                 align=align, alpha=0.6, color='royalblue')
            xs = [[bar.get_x() + bar.get_width()/2.0, bar.get_height(), 
                  bar.get_height()] for bar in bars]
            ha, va = 'center', 'bottom'
            if treat_as_categorical:
                plot_axis.set_xticks(plot_values)
                plot_axis.set_xticklabels(list(values))
            plot_axis.set_xlabel(
                'Histogram of values for {}'.format(feature_name))
        for x in xs:
            plot_axis.text(x[0], x[1],
                        '%.2f' % x[2], ha=ha, va=va)
    elif values.shape[0] == counts.shape[0]:
        # KDE
        xlim = [values[0], values[-1]]
        ylim = [-0.05, np.max(counts)+0.05] if np.any(counts > 1.0) else \
               [-0.05, 1.05]
        if orientation == 'horizontal':
            values, counts = counts, values
            xlim, ylim = ylim, xlim
            plot_axis.set_ylabel('KDE fit to {}'.format(feature_name), 
                                 rotation=270, labelpad=17)
        else:
            plot_axis.set_xlabel('KDE fit to {}'.format(feature_name))
        plot_axis.plot(values, counts, linewidth=3, alpha=0.6,
                       color='royalblue')
        plot_axis.set_xlim(xlim)
        plot_axis.set_ylim(ylim)
        plot_axis.patch.set_visible(True)
    plot_axis.grid(True)
    plot_axis.set_axisbelow(True)

    return plot_figure, plot_axis


def _plot_2_feature_distribution(
    feature_distribution: List[Tuple[np.ndarray, np.ndarray]],
    treat_as_categorical: List[bool],
    feature_linespace: Tuple[np.ndarray, np.ndarray],
    feature_name: List[str],
    axis: Tuple[plt.Axes, plt.Axes]
) -> Tuple[plt.Axes, plt.Axes]:
    """
    Plots 2 feature distributions at once and sets axis parameters.

    For use in 2-feature partial dependence and 2-feature individual
    conditional expectation where one is a subplot along the x-axis and the
    other is a subplot along the y axis. Should only be called from
    :func:`fatf.vis.feature_influence._individual_condtional_expectation_2d`
    and  :func:`fatf.vis.feature_influence._partial_depedence_2d`. For
    parameter description see these functions.

    Returns
    ------
    axis: Tuple[plt.Axes, plt.Axes]
        A tuple of length two containing both axis that the feature
        distributions have been plotted on.
    """
    for categorical, distribution, linespace, name, ax, orientation in \
            zip(treat_as_categorical, feature_distribution, 
            feature_linespace, feature_name, axis, 
            ['vertical', 'horizontal']):
        if categorical:
            # Get feature_distribution in same order as feature_linespace
            values, counts = distribution
            xsorted = np.argsort(values)
            ypos = np.searchsorted(values[xsorted], linespace)
            idx = xsorted[ypos]
            values = values[idx]
            counts = counts[idx]
            distribution = (values, counts)
        _, ax = plot_feature_distribution(
            distribution, categorical, name, plot_axis=ax,
            orientation=orientation)
    axis[0].tick_params(axis='x', which='both', bottom=False,
                            top=False, labelbottom=False)
    axis[1].tick_params(axis='y', which='both', bottom=False,
                            top=False, labelleft=False)
    return axis


def _plot_individual_conditional_expectation_1d(
        ice_array: np.ndarray,
        feature_linespace: np.ndarray,
        class_index: int,
        treat_as_categorical: bool = False,
        feature_name: Optional[str] = None,
        class_name: Optional[str] = None,
        feature_distribution : Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        plot_axis: Optional[plt.Axes] = None
) -> Tuple[Union[plt.Figure, None], Union[plt.Axes, Tuple[plt.Axes, ...]]]:
    """
    Plots individual conditional expectation for 1 feature.

    For description of parameters and exceptions view the documentation for
    :func:`fatf.vis.feature_influence.individual_condtional_expectation`
    function.

    Returns
    -------
    plot_figure : Union[matplotlib.pyplot.Figure, None]
        A matplotlib figure that holds the ``plot_axis`` axis. This parameter
        is ``None`` when the user passed in ``plot_axis`` attribute, otherwise,
        when a blank plot is created, this is a figure object holding the plot
        axis (``plot_axis``).
    plot_axis : Union[matplotlib.pyplot.Axes, Tuple[matplotlib.pyplot.Axes, ...]]]
        A matplotlib axes with the ICE plot or a tuple of matplotlib axes of
        length 2 where the first element is the axes with the ICE plot and the
        second axes has the feature distribution.
    """
    plot_distribution = feature_distribution is not None

    if treat_as_categorical:
        x_range = [-0.5, len(feature_linespace)+0.5]
        x_locs = np.linspace(0, feature_linespace.shape[0], 
                             len(feature_linespace))
    else:
        x_range = [feature_linespace[0], feature_linespace[-1]]
    plot_title = ('Individual Conditional Expectation')
    plot_figure, plot_axis = _prepare_a_canvas(
        plot_title, plot_axis, class_index, class_name, feature_name, x_range,
        plot_distribution)

    if plot_distribution:
        if treat_as_categorical:
            # Get feature_distribution in same order as feature_linespace
            values, counts = feature_distribution
            xsorted = np.argsort(values)
            ypos = np.searchsorted(values[xsorted], feature_linespace)
            idx = xsorted[ypos]
            values = values[idx]
            counts = counts[idx]
            feature_distribution = (values, counts)
        (plot_axis, dist_axis) = plot_axis
        _, dist_axis = plot_feature_distribution(
            feature_distribution, treat_as_categorical, feature_name, 
            plot_axis=dist_axis)
    if treat_as_categorical:
        data = ice_array[:, :, class_index]
        parts = plot_axis.violinplot(data, positions=x_locs)
        for key in ['cmaxes', 'cmins', 'cbars']:
            parts[key].set_color('royalblue')
        for part in parts['bodies']:
            part.set_facecolor('royalblue')
            part.set_alpha(0.4)
        plot_axis.set_xticks(x_locs)
        plot_axis.set_xticklabels(feature_linespace)
    else:
        lines = np.zeros((ice_array.shape[0], ice_array.shape[1], 2),
                        dtype=ice_array.dtype)
        lines[:, :, 1] = ice_array[:, :, class_index]
        lines[:, :, 0] = feature_linespace

        line_collection = matplotlib.collections.LineCollection(
            lines, label='ICE', color='dimgray', alpha=0.5)
        plot_axis.add_collection(line_collection)
        plot_axis.legend()

    if plot_distribution:
        plot_axis = [plot_axis, dist_axis]

    if isinstance(plot_figure, plt.Figure):
        plot_figure.tight_layout()

    return plot_figure, plot_axis


def _plot_individual_conditional_expectation_2d(
    ice_array: np.ndarray,
    feature_linespace: np.ndarray,
    class_index: int,
    treat_as_categorical: bool = False,
    feature_name: Optional[str] = None,
    class_name: Optional[str] = None,
    feature_distribution : Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    plot_axis: Optional[plt.Axes] = None
) -> Tuple[Union[plt.Figure, None], Union[plt.Axes, Tuple[plt.Axes, ...]]]:
    """
    Plots individual conditional expectation for 2 features.

    For description of parameters and exceptions view the documentation for
    :func:`fatf.vis.feature_influence.individual_condtional_expectation`
    function.

    Returns
    -------
    plot_figure : Union[matplotlib.pyplot.Figure, None]
        A matplotlib figure that holds the ``plot_axis`` axis. This parameter
        is ``None`` when the user passed in ``plot_axis`` attribute, otherwise,
        when a blank plot is created, this is a figure object holding the plot
        axis (``plot_axis``).
    plot_axis : Union[matplotlib.pyplot.Axes, Tuple[matplotlib.pyplot.Axes, ...]]]
        A matplotlib axes with the ICE plot or a tuple of matplotlib axes of
        length 3 where the first element is the axes with the ICE plot and the
        second axes has the feature distribution for the first feature, and the
        third axes ha the feature distribution for the second feature.
    """
    plot_distribution = feature_distribution is not None

    if len(feature_name) == 1 and feature_name[0] is None:
        feature_name = ['feature {}'.format(0),
                        'feature {}'.format(1)]
    else:
        if feature_name [0] is None:
            feature_name [0] = 'feature {}'.format(0)
        if feature_name[1] is None:
            feature_name[1] = 'feature {}'.format(1)

    if class_name is None:
        class_name = 'class {}'.format(class_index)

    plot_title = 'Individual Conditional Expectation'
    plot_distribution = feature_distribution[0] is not None

    locs, ranges, strides = [], [], []
    for categorical, linespace in zip(treat_as_categorical, feature_linespace):
        if categorical:
            ranges.append([-0.5, len(linespace)+0.5])
            locs.append(np.linspace(0, linespace.shape[0], len(linespace)))
            strides.append(1.5)
        else:
            ranges.append([linespace[0], linespace[-1]])
            locs.append([])
            strides.append(linespace[1]-linespace[0])

    x_range, y_range = ranges[0], ranges[1]
    x_locs, y_locs = locs[0], locs[1]
    x_stride, y_stride = strides[0], strides[1]
    plot_figure, plot_axis = _prepare_a_canvas(
        plot_title, plot_axis, class_index, class_name, feature_name, x_range,
        plot_distribution, True, y_range, True)

    plot_axis.set_zlim([0.5, ice_array.shape[0]+0.5])
    if plot_distribution:
        (plot_axis, dist_axis_x, dist_axis_y) = plot_axis
        plot_axis.set_title('')
        if isinstance(plot_figure, plt.Figure):
            plot_figure.suptitle('Partial Dependence for {}'.format(class_name))
        dist_axis_x, dist_axis_y = _plot_2_feature_distribution(
            feature_distribution, treat_as_categorical, feature_linespace,
            feature_name, (dist_axis_x, dist_axis_y))

    if plot_distribution:
        plot_axis = (plot_axis, dist_axis_x, dist_axis_y)

    img = ice_array[0, :, :, class_index]
    x, y = np.meshgrid(np.linspace(x_range[0],x_range[1],img.shape[0]), np.linspace(y_range[0],y_range[1],img.shape[1]))
    for i in range(0, ice_array.shape[0]):
        Z = (i)*np.ones(x.shape)
        facecolors = plt.cm.cividis(ice_array[i, :, :, class_index], alpha=0.3)
        plot_axis.plot_surface(x, y, Z, rstride=1, cstride=1, 
                           facecolors=facecolors, vmin=0, vmax=1, shade=True)

    #plot_axis.plot_surface(x1, x2, Z, rstride=1, cstride=1,
    #                      cmap=plt.cm.cividis, vmin=0, vmax=1)
    # tight_layout with gridspecs raise a warning that the results might
    # be incorrect. (Problem seen here 
    # https://matplotlib.org/gallery/subplots_axes_and_figures/demo_tight_layout.html)
    if isinstance(plot_figure, plt.Figure):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            plot_figure.tight_layout(rect=[0.05, 0.05, 1, 0.95])

    return plot_figure, plot_axis


def plot_individual_conditional_expectation(
        ice_array: np.ndarray,
        feature_linespace: np.ndarray,
        class_index: int,
        treat_as_categorical: bool = False,
        feature_name: Optional[str] = None,
        class_name: Optional[str] = None,
        feature_distribution : Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        plot_axis: Optional[plt.Axes] = None
) -> Tuple[Union[plt.Figure, None], Union[plt.Axes, Tuple[plt.Axes, ...]]]:
    """
    Plots Individual Conditional Expectation for a selected class.

    For exceptions raised by this function please see the documentation of
    :func:`fatf.vis.feature_influence._prepare_a_canvas` and
    :func:`fatf.vis.feature_influence._validate_input` functions.

    Parameters
    ----------
    ice_array : numpy.ndarray
        An array of (n_samples, n_steps, n_classes) shape with Individual
        Conditional Expectation calculation results for every target class for
        the desired spectrum of the selected feature. This should be the output
        of the :func:`fatf.transparency.models.feature_influence.
        individual_conditional_expectation` function.
    feature_linespace : numpy.ndarray
        A one-dimensional array -- (steps_number, ) -- with the values for
        which the selected feature was sampled when the dataset was evaluated
        for a predictive model. This should be the output of the :func:`fatf.
        transparency.models.feature_influence.
        individual_conditional_expectation` function.
    class_index : integer
        The index of the class for which ICE will be plotted, taken from the
        original dataset. For ICE's computed using a regression model,
        `class_index` should be 0.
    feature_name : string, optional (default=None)
        The name of the feature for which ICE was originally calculated.
    class_name : string, optional (default=None)
        The name of the class that ``class_index`` parameter points to. If
        ``None``, the class name will be the same as the class index.
    feature_distribution : List[numpy.ndarray]
        A list of length 2 where they are [values, counts]. The first
        array is of shape (samples,) or (samples+1,). It contains x-axis data
        points either for histogram, gaussian kde or unique values of the array
        The second element of the list should be probability densities for each
        value in the first array. This should be the output of the :func:`fatf.
        transparency.models.compute_feature_distribution`.
    plot_axis : matplotlib.pyplot.Axes, optional (default=None)
        A matplotlib axes on which the ICE will be plotted. This is useful if
        one wants to overlay multiple ICE plot on top of each other. If
        ``None``, a new axes will be created.

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
                           treat_as_categorical, feature_name, class_name,
                           plot_axis, None, False,
                           False), 'Input is invalid.'
    is_2d = True if len(ice_array.shape) == 4 else False

    if not isinstance(treat_as_categorical, list):
        treat_as_categorical = [treat_as_categorical]
        if len(treat_as_categorical) == 1 and is_2d:
            treat_as_categorical = treat_as_categorical * 2

    if not isinstance(feature_name, list):
        feature_name = [feature_name]

    if not isinstance(feature_distribution, list):
        feature_distribution = [feature_distribution]

    if isinstance(feature_linespace, np.ndarray):
        feature_linespace = (feature_linespace, )

    is_categorical_column = [False if fuav.is_numerical_array(x) 
                             else True if fuav.is_textual_array 
                             else -1 for x in feature_linespace]
    assert -1 not in is_categorical_column, \
        'Must be an array of a base type.' # pragma: nocover

    # If needed, infer the column type.
    for i in range(len(treat_as_categorical)):
        if treat_as_categorical[i] is None:
            treat_as_categorical[i] = is_categorical_column[i]
        elif not treat_as_categorical[i] and is_categorical_column[i]:
            message = ('Selected feature is categorical (string-base elements), '
                    'however the treat_as_categorical was set to False. Such '
                    'a combination is not possible. The feature will be '
                    'treated as categorical.')
            warnings.warn(message, category=UserWarning)
            treat_as_categorical[i] = True

    if is_2d:
        plot_figure, plot_axis = _plot_individual_conditional_expectation_2d(
            ice_array, feature_linespace, class_index, treat_as_categorical,
            feature_name, class_name, feature_distribution, plot_axis)
    else:
        plot_figure, plot_axis = _plot_individual_conditional_expectation_1d(
            ice_array, feature_linespace[0], class_index, 
            treat_as_categorical[0], feature_name[0], class_name,
            feature_distribution[0], plot_axis)

    return plot_figure, plot_axis


def _plot_partial_dependence_1d(
    pd_array: np.ndarray,
    feature_linespace: np.ndarray,
    class_index: int,
    treat_as_categorical: Optional[bool] = None,
    variance: Optional[np.ndarray] = None,
    variance_area: Optional[bool] = False,
    feature_name: Optional[str] = None,
    class_name: Optional[str] = None,
    feature_distribution : Optional[Tuple[np.ndarray, np.ndarray]] = None,
    plot_axis: Optional[plt.Axes] = None
) -> Tuple[Union[plt.Figure, None], Union[plt.Axes, Tuple[plt.Axes, ...]]]:
    """
    Plots partial dependence for 1 feature.

    For description of parameters and exceptions view the documentation for
    :func:`fatf.vis.feature_influence.plot_partial_dependence` function.

    Returns
    -------
    plot_figure : Union[matplotlib.pyplot.Figure, None]
        A matplotlib figure that holds the ``plot_axis`` axis. This parameter
        is ``None`` when the user passed in ``plot_axis`` attribute, otherwise,
        when a blank plot is created, this is a figure object holding the plot
        axis (``plot_axis``).
    plot_axis : Union[matplotlib.pyplot.Axes, Tuple[matplotlib.pyplot.Axes, ...]]]
        A matplotlib axes with the PD plot or a tuple of matplotlib axes of
        length 2 where the first element is the axes with the PD plot and the
        second axes has the feature distribution.
    """
    plot_title = 'Partial Dependence'
    plot_distribution = feature_distribution is not None
    if treat_as_categorical:
        x_range = [-0.5, len(feature_linespace)+0.5]
        x_locs = np.linspace(0, feature_linespace.shape[0], 
                             len(feature_linespace))
    else:
        x_range = [feature_linespace[0], feature_linespace[-1]]
    plot_figure, plot_axis = _prepare_a_canvas(
        plot_title, plot_axis, class_index, class_name, feature_name, x_range,
        plot_distribution)

    if plot_distribution:
        (plot_axis, dist_axis) = plot_axis
        if treat_as_categorical:
            # Get feature_distribution in same order as feature_linespace
            values, counts = feature_distribution
            xsorted = np.argsort(values)
            ypos = np.searchsorted(values[xsorted], feature_linespace)
            idx = xsorted[ypos]
            values = values[idx]
            counts = counts[idx]
            feature_distribution = (values, counts)
        _, dist_axis = plot_feature_distribution(
            feature_distribution, treat_as_categorical, feature_name,
            plot_axis=dist_axis)

    if treat_as_categorical:
        yerr = variance[:, class_index] if isinstance(variance,np.ndarray) \
            else None
        plot_axis.bar(x_locs, pd_array[:, class_index], yerr=yerr, capsize=10,
                      align='center', ecolor='black', alpha=0.6, 
                      color='royalblue')
        plot_axis.set_xticks(x_locs)
        plot_axis.set_xticklabels(feature_linespace)
        plot_axis.set_ylim(np.array([0, 1.05]))
    else:
        plot_axis.plot(
            feature_linespace,
            pd_array[:, class_index],
            color='lightsalmon',
            linewidth=7,
            alpha=0.6,
            label='PD')

        if isinstance(variance, np.ndarray):
            if variance_area:
                plot_axis.fill_between(
                    feature_linespace,
                    pd_array[:, class_index] - variance[:, class_index],
                    pd_array[:, class_index] + variance[:, class_index],
                    alpha=0.3,
                    color='lightsalmon',
                    label='Variance')
            else:
                plot_axis.errorbar(
                    feature_linespace,
                    pd_array[:, class_index],
                    yerr=variance[:, class_index],
                    alpha=0.6,
                    ecolor='lightsalmon',
                    fmt='none',
                    capsize=5,
                    elinewidth=2,
                    markeredgewidth=2,
                    label='Variance')
        plot_axis.legend()

    if plot_distribution:
        plot_axis = [plot_axis, dist_axis]

    if isinstance(plot_figure, plt.Figure):
        plot_figure.tight_layout()

    return plot_figure, plot_axis


def _plot_partial_dependence_2d(
    pd_array: np.ndarray,
    feature_linespace: Tuple[np.ndarray, np.ndarray],
    class_index: int,
    treat_as_categorical: Optional[bool] = None,
    feature_name: Optional[List[str]] = None,
    class_name: Optional[str] = None,
    feature_distribution : Optional[Tuple[np.ndarray, ...]] = None,
    plot_axis: Optional[plt.Axes] = None
) -> Tuple[Union[plt.Figure, None], Union[plt.Axes, Tuple[plt.Axes, ...]]]:
    """
    Plots partial dependence for 2 features at once.

    For description of parameters and exceptions view the documentation for
    :func:`fatf.vis.feature_influence.plot_partial_dependence` function.

    Returns
    -------
    plot_figure : Union[matplotlib.pyplot.Figure, None]
        A matplotlib figure that holds the ``plot_axis`` axis. This parameter
        is ``None`` when the user passed in ``plot_axis`` attribute, otherwise,
        when a blank plot is created, this is a figure object holding the plot
        axis (``plot_axis``).
    plot_axis : Union[matplotlib.pyplot.Axes, Tuple[matplotlib.pyplot.Axes, ...]]]
        A matplotlib axes with the PD plot or a tuple of matplotlib axes of
        length 3 where the first element is the axes with the PD plot and the
        second axes has the feature distribution for the first feature, and the
        third axes ha the feature distribution for the second feature.
    """
    # Preprocess features
    if len(feature_name) == 1 and feature_name[0] is None:
        feature_name = ['feature {}'.format(0),
                        'feature {}'.format(1)]
    else:
        if feature_name [0] is None:
            feature_name [0] = 'feature {}'.format(0)
        if feature_name[1] is None:
            feature_name[1] = 'feature {}'.format(1)

    if class_name is None:
        class_name = 'class {}'.format(class_index)

    plot_title = 'Partial Dependence for {}'.format(class_name)
    plot_distribution = feature_distribution[0] is not None

    locs, ranges = [], []
    for categorical, linespace in zip(treat_as_categorical, feature_linespace):
        if categorical:
            ranges.append([-0.5, len(linespace)+0.5])
            locs.append(np.linspace(0, linespace.shape[0], len(linespace)))
        else:
            ranges.append([linespace[0], linespace[-1]])
            locs.append([])
    x_range, y_range = ranges[0], ranges[1]
    x_locs, y_locs = locs[0], locs[1]

    plot_figure, plot_axis = _prepare_a_canvas(
        plot_title, plot_axis, class_index, class_name, feature_name, x_range,
        plot_distribution, True, y_range)

    if plot_distribution:
        (plot_axis, dist_axis_x, dist_axis_y) = plot_axis
        plot_axis.set_title('')
        if isinstance(plot_figure, plt.Figure):
            plot_figure.suptitle('Partial Dependence for {}'.format(class_name))
        dist_axis_x, dist_axis_y = _plot_2_feature_distribution(
            feature_distribution, treat_as_categorical, feature_linespace,
            feature_name, (dist_axis_x, dist_axis_y))

    im = plot_axis.imshow(pd_array[:, :, class_index], extent=x_range+y_range,
                          origin='lower', cmap=plt.cm.cividis, aspect='auto',
                          vmin=0., vmax=1.)
    plot_figure.subplots_adjust(left=0.80)
    cbar_ax = plot_figure.add_axes([0.05, 0.18, 0.02, 0.5])
    cbar = plot_figure.colorbar(im, cax=cbar_ax, aspect=5)
    cbar.ax.tick_params(labelsize=10)
    cbar_ax.yaxis.set_ticks_position('left')

    if plot_distribution:
        plot_axis = (plot_axis, dist_axis_x, dist_axis_y)

    # tight_layout with gridspecs raise a warning that the results might
    # be incorrect. (Problem seen here 
    # https://matplotlib.org/gallery/subplots_axes_and_figures/demo_tight_layout.html)
    if isinstance(plot_figure, plt.Figure):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            plot_figure.tight_layout(rect=[0.05, 0.05, 1, 0.95])

    return plot_figure, plot_axis


def plot_partial_dependence(
        pd_array: np.ndarray,
        feature_linespace: Tuple[np.ndarray, ...],
        class_index: int,
        treat_as_categorical: Optional[bool] = None,
        variance: Optional[np.ndarray] = None,
        variance_area: Optional[bool] = False,
        feature_name: Optional[str] = None,
        class_name: Optional[str] = None,
        feature_distribution : Optional[Tuple[np.ndarray, ...]] = None,
        plot_axis: Optional[plt.Axes] = None
) -> Tuple[Union[plt.Figure, None], Union[plt.Axes, Tuple[plt.Axes, ...]]]:
    """
    Plots Partial Dependence for a selected class.

    For exceptions raised by this function please see the documentation of
    :func:`fatf.vis.feature_influence._prepare_a_canvas` and
    :func:`fatf.vis.feature_influence._validate_input` functions.

    Parameters
    ----------
    pd_array : numpy.ndarray
        An array of (n_steps, n_classes) shape with Partial Dependence
        calculation results for every target class for the desired spectrum of
        the selected feature. This should be the output of the :func:`fatf.
        transparency.models.feature_influence.partial_dependence` or :func:`
        fatf.transparency.models.feature_influence.partial_dependence_ice`
        function.
    feature_linespace : Tuple[numpy.ndarray, ...]
        A tuple of one-dimensional array -- (steps_number, ) -- with the values
        for which the selected feature was sampled when the dataset was
        evaluated for a predictive model. For plotting 2 feature partial
        dependence, this should be a tuple of length 2. This should be the
        output of the :func:`fatf.transparency.models.feature_influence.
        individual_conditional_expectation` function. 
    class_index : integer
        The index of the class for which PD will be plotted, taken from the
        original dataset. For PD computed using a regression model,
        `class_index` should be 0.
    variance : numpy.ndarray, optional (default=None)
        An array of (n_steps, n_classes) shape with the values for the
        variance of the predictions of data points used to compute Partial
        Dependence. This should be the output of the :func:`fatf.transparency.
        models.feature_influence.partial_dependence` or :func:`
        fatf.transparency.models.feature_influence.partial_dependence_ice`
        function. Variance cannot be used with 2-D partial dependence
    variance_area : boolean, optional (default=False)
        Specifies whether to use error bars method of plotting or solid areas
        to show the variance at each point. If True, the variance will be shown
        by a solid area surrounding the partial dependence line, else then
        error bar method is used.
    feature_name : string, optional (default=None)
        The name of the feature for which PD was originally calculated.
    class_name : string, optional (default=None)
        The name of the class that ``class_index`` parameter points to. If
        ``None``, the class name will be the same as the class index.
    feature_distribution : List[Tuple(numpy.ndarray)]
        A tuple of length 2 where they are (values, counts). The first
        array is of shape (samples,) or (samples+1,). It contains x-axis data
        points either for histogram, gaussian kde or unique values of the array
        The second element of the list should be probability densities for each
        value in the first array. This should be the output of the :func:`fatf.
        transparency.models.compute_feature_distribution`. If 2-D PD then this
        should be a tuple of length 4 where they are (values_1, counts_1,
        values_2, counts_2)
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
    plot_axis : Union[matplotlib.pyplot.Axes, Tuple[matplotlib.pyplot.Axes, ...]]]
        A matplotlib axes with the PD plot or a tuple of matplotlib axes of
        length 2 where the first element is the axes with the PD plot and the
        second axes has the feature distribution for the first feature. Can
        also be a tuple of length 3 where there is an additional axes for the
        feature distribution for the second feature if 2-feature partial
        dependence was plotted.
    """
    # pylint: disable=too-many-arguments
    assert _validate_input(pd_array, feature_linespace, class_index,
                           treat_as_categorical, feature_name, class_name,
                           plot_axis, variance, variance_area,
                           True), 'Input is invalid.'
    is_2d = True if len(pd_array.shape) == 3 else False

    if not isinstance(treat_as_categorical, list):
        treat_as_categorical = [treat_as_categorical]
        if len(treat_as_categorical) == 1 and is_2d:
            treat_as_categorical = treat_as_categorical * 2

    if not isinstance(feature_name, list):
        feature_name = [feature_name]

    if not isinstance(feature_distribution, list):
        feature_distribution = [feature_distribution]

    if isinstance(feature_linespace, np.ndarray):
        feature_linespace = (feature_linespace, )

    is_categorical_column = [False if fuav.is_numerical_array(x) 
                             else True if fuav.is_textual_array 
                             else -1 for x in feature_linespace]
    assert -1 not in is_categorical_column, \
        'Must be an array of a base type.' # pragma: nocover

    # If needed, infer the column type.
    for i in range(len(treat_as_categorical)):
        if treat_as_categorical[i] is None:
            treat_as_categorical[i] = is_categorical_column[i]
        elif not treat_as_categorical[i] and is_categorical_column[i]:
            message = ('Selected feature is categorical (string-base elements), '
                    'however the treat_as_categorical was set to False. Such '
                    'a combination is not possible. The feature will be '
                    'treated as categorical.')
            warnings.warn(message, category=UserWarning)
            treat_as_categorical[i] = True

    if is_2d:
        plot_figure, plot_axis = _plot_partial_dependence_2d(
            pd_array, feature_linespace, class_index, treat_as_categorical,
            feature_name, class_name, feature_distribution, plot_axis)
    else:
        plot_figure, plot_axis = _plot_partial_dependence_1d(
            pd_array, feature_linespace[0], class_index, 
            treat_as_categorical[0], variance, variance_area, feature_name[0],
            class_name, feature_distribution[0], plot_axis)

    return plot_figure, plot_axis
