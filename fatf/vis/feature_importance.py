"""
Functions for calculating feature importance and
Individual Conditional Expectation (ICE)
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
#         Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from fatf.exceptions import IncorrectShapeError


def _check_input(data: np.ndarray,
                 values: np.ndarray,
                 category: int,
                 is_pd: bool = False):
    """
    Checks input variables for both plot_individual_conditional_expectation
    and plot_partial_dependence.

    Parameters
    ----------
    data : numpy.ndarray
        Array that contains probabilities for ICE or PD.
    values : numpy.ndarray
        Containing values of feature tested for ICE or PD.
    category : numpy.ndarray
        Which class to plot probabilities for
    is_pd : boolean
        If true then checks input for use in plot PD else checks for use in
        plot ICE.
    
    Raises
    ------
    IncorrectShapeError
        If data does not have the correct shape for use in specified mode.
    ValueError
        If category is not a valid index in data array or if values array 
        does not contain the same number of entries as steps in data array.
    """
    axis = 3
    msg = ('plot_individual_condtional_expectation expects matrix '
           'of shape [n_samples, steps, n_classes].')
    if is_pd:
        axis = 2
        msg = ('plot_partial_depenedence expects matrix of shape '
               '[steps, n_classes].')
    if axis != len(data.shape):
        raise IncorrectShapeError(msg)
    if category < 0 or category > data.shape[-1]:
        raise ValueError('Category {} is not a valid index for probability '
                         'matrix.'.format(category))
    if values.shape[0] != data.shape[-2]:
        raise ValueError('{} values provided does not match {} steps in '
                         'probability matrix.'.format(values.shape[0], 
                                                      data.shape[-2]))


def plot_individual_conditional_expectiation(
        ice: np.ndarray,
        values: np.ndarray,
        category: int,
        feature_name: str = None,
        category_name: str = None,
        ax: plt.Axes = None,
        plot_pd: bool = False) -> plt.Axes:
    """
    Plots individual conditional expectations for class.

    Parameters
    ----------
    ice : numpy.ndarray
        Shape [n_samples, steps, n_classes] containing probabilities
        outputted from ICE
    values : numpy.array
        Containing values of feature tested for ICE
    category : integer
        Which class to plot probabilities for
    feature_name : string
        Specificy which feature was used to calculate the ICE
    category_name : string
        Name of class chosen. If None then the category_name will be
        the category integer converted to str. Default: None
    ax : matplotlib.pyplot.Axes
        Axes to plot the individual conditional expection on. If None, a new
        axes will be initialised. Default: None
    plot_pd : boolean
        If True then partial dependence line is also plotted. Default: True

    Returns
    -------
    ax : matplotlib.pyplot.Axes
        Figure with individual conditional expectation and partial dependence
        line plotted.
    """
    _check_input(ice, values, category)
    if category_name is None:
        category_name = 'Class {}'.format(str(category))
    if feature_name is None:
        feature_name = 'Feature'
    if ax is None:
        ax = plt.subplot()
        ax.set_xlim([values[0], values[-1]])
        ax.set_ylabel(
            'Probability of belonging to class {}'.format(category_name))
        ax.set_xlabel(feature_name)
        ax.set_title('Individual Conditional Expectation')
    lines = np.zeros((ice.shape[0], ice.shape[1], 2))
    lines[:, :, 1] = ice[:, :, category]
    lines[:, :, 0] = np.tile(values, (ice.shape[0], 1))
    collect = LineCollection(lines, label='Individual Points', color='black',
                             alpha=0.5)
    ax.add_collection(collect)
    mean = np.mean(ice[:, :, category], axis=0)
    if plot_pd is True:
        ax.plot(values, mean, color='yellow', linewidth=10, alpha=0.6, 
                label='Partial Dependence')
    ax.legend()
    return ax


def plot_partial_dependence(
        pd: np.ndarray,
        values: np.ndarray,
        category: int,
        feature_name: str = None,
        category_name: str = None,
        ax: plt.Axes = None,
        plot_pd: bool = False) -> plt.Axes:
    """
    Plots partial dependence for class.

    Parameters
    ----------
    pd : numpy.ndarray
        Shape [steps, n_classes] containing probabilities
        outputted from ICE
    values : numpy.array
        Containing values of feature tested for partial dependence
    category : integer
        Which class to plot probabilities for
    feature_name : string
        Specificy which feature was used to calculate the partial dependence
    category_name : string
        Name of class chosen. If None then the category_name will be
        the category integer converted to str. Default: None
    ax : matplotlib.pyplot.Axes
        Axes to plot the individual conditional expection on. If None, a new
        axes will be initialised. Default: None

    Returns
    -------
    ax : matplotlib.pyplot.Axes
        Figure with individual conditional expectation and partial dependence
        line plotted.
    """
    _check_input(pd, values, category, True)
    if category_name is None:
        category_name = 'Class {}'.format(str(category))
    if feature_name is None:
        feature_name = 'Feature'
    if ax is None:
        ax = plt.subplot()
        ax.set_xlim([values[0], values[-1]])
        ax.set_title('Partial Dependence')
        ax.set_xlabel(feature_name)
        ax.set_ylabel(
            'Probability of belonging to class {}'.format(category_name))
    ax.plot(values, pd[:, category], color='yellow', linewidth=10, alpha=0.6, 
            label='Partial Dependence')
    ax.legend()
    return ax
