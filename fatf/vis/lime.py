"""
The :mod:`fatf.vis.lime` module visualises tabular LIME explanations.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
#         Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

from numbers import Number
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt

__all__ = ['plot_lime']

LimeExplanation = List[Tuple[str, float]]


def plot_lime(
        lime_explanation: Union[LimeExplanation, Dict[str, LimeExplanation]]
) -> plt.Figure:
    """
    Plots a LIME explanation.

    This plotting function is intended for the
    :class:`fatf.transparency.lime.Lime` explainer.

    Parameters
    ----------
    lime_explanation : Dictionary[string, List[Tuple[string, float]]] or \
List[Tuple[string, float]]
        An explanation returned by the ``explain_instance`` method of the LIME
        explainer. For a classifier this will be a dictionary where the keys
        are class names and the values are lists of 2-tuples where the first
        element is an explanatory feature name and the second element is the
        importance of this explanatory feature. On the other hand, for
        regressor explanations this will be simply a list of 2-tuples of the
        same structure as for the classifier.

    Raises
    ------
    TypeError
        The ``lime_explanation`` parameter is not a list (regression) or a
        dictionary (classification). One of the class names is not a string.
        One of the explanatory features is not a string. One of the explanatory
        values is not a number.

    Returns
    -------
    figure : matplotlib.pyplot.Figure
        A matplotlib figure with subplots explaining every label in LIME
        explanation.
    """

    # pylint: disable=too-many-locals,too-many-branches,too-many-statements

    def validate_explanation_list(explanation_list):
        is_valid = False
        if not isinstance(explanation_list, list):
            raise TypeError('One of the explanations is not a list.')
        for key, value in explanation_list:
            if not isinstance(key, str):
                raise TypeError('One of the explanation keys is neither an '
                                'integer nor a string.')
            if not isinstance(value, Number):
                raise TypeError('One of the explanation values is not a '
                                'number.')
        is_valid = True
        return is_valid

    if isinstance(lime_explanation, dict):
        for explanation in lime_explanation.values():
            assert validate_explanation_list(explanation)
    elif isinstance(lime_explanation, list):
        assert validate_explanation_list(lime_explanation)
        # In case the explanation is for a regressor
        lime_explanation = dict(regressor=lime_explanation)
    else:
        raise TypeError('The LIME explanation has to be either a dictionary '
                        '(for classification) or a list (for regression).')

    # Collect all of the class names and their number
    class_names = sorted(list(lime_explanation.keys()))
    class_n = len(class_names)

    # Check whether all of the explanations share the same key names
    share_y = True
    fig_shape = (1, class_n)
    #
    explanation_label_set_old = None
    for class_explanation in lime_explanation.values():
        if explanation_label_set_old is None:
            explanation_label_set_old = {expl[0] for expl in class_explanation}
        else:
            explanation_labels_set = {expl[0] for expl in class_explanation}
            if explanation_label_set_old != explanation_labels_set:
                share_y = False
                fig_shape = (class_n, 1)
                break

    # If sharing y-axis get a common ordering of the explantions
    if share_y:
        name_ordering = [exp[0] for exp in lime_explanation[class_names[0]]]

    figure, axes = plt.subplots(*fig_shape, sharey=share_y, sharex=True)
    figure.suptitle('LIME Explanation')

    # Do the plotting
    for i in range(class_n):
        class_name = class_names[i]
        class_explanation = lime_explanation[class_name]

        # Split a list of pairs into two lists
        exp_names, exp_values = [], []
        for name, value in class_explanation:
            exp_names.append(name)
            exp_values.append(value)

        # Make sure that all bar plots are in the same order if sharing
        if share_y:
            # Get indices of names with respect to the common name ordering
            indices = [exp_names.index(name) for name in name_ordering]
            exp_values = [exp_values[index] for index in indices]

            # Get common name ordering -- this has to be below because of
            # overwriting exp_names, which is used atop
            exp_names = name_ordering

        positions = [i + 0.5 for i in range(len(class_explanation))]
        colours = ['green' if val > 0 else 'red' for val in exp_values]

        if class_n == 1:
            # For a single sub-plot this is not iterable
            axis = axes
        else:
            axis = axes[i]
        axis.barh(positions, exp_values, align='center', color=colours)
        axis.set_yticks(positions)
        axis.set_yticklabels(exp_names)

        axis.set_title('{}'.format(class_name))

    return figure
