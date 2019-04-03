import matplotlib.pyplot as plt 
from typing import Dict, List
import numpy as np 

def plot_lime(lime_explained: Dict[str, List[tuple]]) -> plt.Figure:
    """Figures to display explainer

    Parameters
    ----
    lime_explained: Dictionary returned from Lime.explain_instance. 

    Returns
    ----
    Figure from matplotlib where it is split into as many subplots as there are 
    possible labels in Dataset.

    """
    sharey = False
    names = None
    # check if all features are used, all subplots can share y axis
    sets = []
    labels = []
    for k, v in lime_explained.items():
        sets.append(set([l[0] for l in v]))
        labels.append(k)
    if all(s==sets[0] for s in sets):
        sharey = True
        f, axs = plt.subplots(1, len(sets), sharey=sharey, sharex=True)
    else:
        f, axs = plt.subplots(len(sets), 1, sharex=True)
    f.suptitle('Local Explanations for classes')
    if sharey: # Make sure all barplots are in the same order if sharing
        names = list(sets[0])
    # Do the plotting
    for ax, label in zip(axs, labels):
        exp = lime_explained[label]
        vals = [x[1] for x in exp]
        unordered_names = [x[0] for x in exp]
        if sharey:
            # get bars in correct order for sharing y-axis
            ind = [unordered_names.index(item) for item in names]
            vals = [vals[i] for i in ind]
        else:
            names = unordered_names
        vals.reverse()
        l = names[::-1]
        colors = ['green' if x > 0 else 'red' for x in vals]
        pos = np.arange(len(exp)) + .5
        ax.barh(pos, vals, align='center', color=colors)
        ax.set_yticks(pos)
        ax.set_yticklabels(l)
        title = str(label)
        ax.set_title(title)
    return f
