"""
Functions for visualising the results of the LIME algorithm.
"""

# Author: Alex Hepburn <ah13558@bristol.ac.uk>
# License: new BSD

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def plot_lime(lime_explained: Dict[str, List[tuple]]) -> plt.Figure:
    """
    Plotting function for dictionary return by explain_instance method in
    Lime class.

    Parameters
    ----------
    lime_explained : Dictionary[string, List[tuple]]
        Dictionary returned from Lime.explain_instance.

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        Figure from matplotlib where it is split into as many subplots as
        there are possible labels in lime_explained object.

    """
    sharey = False
    names = None
    # check if all features are used, all subplots can share y axis
    sets = []
    labels = []
    for key, value in lime_explained.items():
        sets.append({lab[0] for lab in value})
        labels.append(key)
    if all(s == sets[0] for s in sets):
        sharey = True
        fig, axes = plt.subplots(1, len(sets), sharey=sharey, sharex=True)
    else:
        fig, axes = plt.subplots(len(sets), 1, sharex=True)
    fig.suptitle('Local Explanations for classes')
    if sharey: # Make sure all barplots are in the same order if sharing
        names = [lab[0] for lab in lime_explained[labels[0]]]
    # Do the plotting
    for axis, label in zip(axes, labels):
        exp = lime_explained[label]
        vals = [x[1] for x in exp]
        unordered_names = [x[0] for x in exp]
        if sharey:
            # get bars in correct order for sharing y-axis
            ind = [unordered_names.index(item) for item in names]
            vals = [vals[i] for i in ind]
        else:
            names = unordered_names
        colors = ['green' if x > 0 else 'red' for x in vals]
        pos = np.arange(len(exp)) + .5
        axis.barh(pos, vals, align='center', color=colors)
        axis.set_yticks(pos)
        axis.set_yticklabels(names)
        title = str(label)
        axis.set_title(title)
    return fig
