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


def plot_individual_conditional_expectiation(
        ice: np.ndarray,
        feature_name: str,
        values: np.ndarray,
        category: int,
        category_name: str = '',
        ax: plt.Axes = None,
        plot_pd: bool = False) -> plt.Axes:
    """
    Plots individual conditional expectations for class.

    Parameters
    ----------
    ice : numpy.ndarray
        Shape [n_samples, steps, n_classes] containing probabilities
        outputted from ICE
    feature_name : string
        Specificy which feature was used to calculate the ICE
    values : numpy.array
        Containing values of feature tested for ICE
    category : integer
        Which class to plot probabilities for
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
    if not category_name:
        category_name = str(category)
    if ax is None:
        ax = plt.subplot(111)
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
        feature_name: str,
        values: np.ndarray,
        category: int,
        category_name: str = '',
        ax: plt.Axes = None,
        plot_pd: bool = False) -> plt.Axes:
    """
    Plots partial dependence for class.

    Parameters
    ----------
    pd : numpy.ndarray
        Shape [steps, n_classes] containing probabilities
        outputted from ICE
    feature_name : string
        Specificy which feature was used to calculate the ICE
    values : numpy.array
        Containing values of feature tested for ICE
    category : integer
        Which class to plot probabilities for
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
    if not category_name:
        category_name = str(category)
    if ax is None:
        ax = plt.subplot(111)
        ax.set_xlim([values[0], values[-1]])
        ax.set_title('Partial Dependence')
        ax.set_xlabel(feature_name)
        ax.set_ylabel(
            'Probability of belonging to class {}'.format(category_name))
    ax.plot(values, pd[:, category], color='yellow', linewidth=10, alpha=0.6, 
            label='Partial Dependence')
    ax.legend()
    return ax
    


if __name__ == '__main__':
    from sklearn import svm
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from fatf.utils.data import datasets
    from fatf.transparency.models import feature_importance as featim
    plt.style.use('ggplot')
    predictor = svm.SVC(probability=True, gamma='scale')
    predictor = KNeighborsClassifier(n_neighbors=15)
    data = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data['data'], 
                                                        data['target'])
    predictor.fit(X_train, y_train)
    i, v = featim.individual_conditional_expectation(X_test, predictor, 
                                                     feature=2)
    ax = plot_individual_conditional_expectiation(
        i, feature_name=data['feature_names'][2], values=v, 
        category_name=data['target_names'][0], category=0, plot_pd=True)
    plt.show()
