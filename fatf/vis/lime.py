"""
The :mod:`fatf.vis.lime` module visualises tabular LIME explanations.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
#         Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import logging

from numbers import Number
from typing import Any, Dict, Set, Union

import matplotlib.pyplot as plt

__all__ = ['plot_lime']

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

BlimeyExplanation = Dict[str, float]


def plot_lime(surrogate_explanation: Union[BlimeyExplanation,
                                           Dict[str, BlimeyExplanation]]
              ) -> plt.Figure:
    """
    Plots an importance-based surrogate explanation, e.g., LIME.

    This plotting function is intended for the :class:`fatf.transparency.\
predictions.surrogate_explainers.TabularBlimeyLime` and :class:`fatf.\
transparency.predictions.surrogate_explainers.TabularBlimeyTree` explainers.
    When multiple explanations are provided, they will share a common y-axis
    if they use the same set of interpretable features.

    Parameters
    ----------
    surrogate_explanation : Dictionary[string, float] or \
Dictionary[string, Dictionary[string, float]]
        An explanation returned by the ``explain_instance`` method of a
        surrogate explainer. For a classifier this will be a dictionary where
        the keys are class names and the values are dictionaries where the key
        is an interpretable feature name and the value is the importance of
        this interpretable feature. For regressor explanations this will be
        a dictionary where the key is an interpretable feature name and the
        value is the importance of this interpretable feature.

        .. versionchanged:: 0.1.0
           Dropped support for LIME explanation format:
           ``List[Tuple[string, float]]`` for regressors and
           ``Dictionary[string, List[Tuple[string, float]]]`` for probabilistic
           classifiers.

        .. versionchanged:: 0.0.2
           Support for surrogate explainer explanations of the form:
           ``Dictionary[string, Dictionary[string, float]]``.

    Raises
    ------
    TypeError
        The ``surrogate_explanation`` parameter is not a dictionary. One of the
        class names is not a string. One of the interpretable feature names is
        not a string. One of the importance values is not a number.
    ValueError
        One of the explanations is an empty dictionary.

    Returns
    -------
    figure : matplotlib.pyplot.Figure
        A matplotlib figure with subplots explaining every label in a surrogate
        explanation.
    """

    # pylint: disable=too-many-locals,too-many-branches,too-many-statements

    def validate(explanation):
        is_valid = False
        if isinstance(explanation, dict):
            if not explanation:
                raise ValueError('One of the explanations is an empty '
                                 'dictionary.')
            for key, value in explanation.items():
                if not isinstance(key, str):
                    raise TypeError('One of the explanation keys is not a '
                                    'string.')
                if not isinstance(value, Number):
                    raise TypeError('One of the explanation values is not a '
                                    'number.')
        else:
            raise TypeError('One of the explanations is not a dictionary.')
        is_valid = True
        return is_valid

    if isinstance(surrogate_explanation, dict):
        if not surrogate_explanation:
            raise ValueError('The surrogate explanation is an empty '
                             'dictionary.')

        example_key = sorted(surrogate_explanation.keys())[0]
        example_explanation = surrogate_explanation[example_key]
        # Nested dictionary for classification
        if isinstance(example_explanation, dict):
            for class_name, explanation in surrogate_explanation.items():
                if not isinstance(class_name, str):
                    raise TypeError('One of the class names is not a string.')
                assert validate(explanation), 'Invalid input.'
            plot_explanation = surrogate_explanation  # type: Any
        # Plain dictionary for regression
        elif isinstance(example_explanation, Number):
            assert validate(surrogate_explanation), 'Invalid input.'
            plot_explanation = {'': surrogate_explanation}
        else:
            raise TypeError('Each value of the surrogate explanation must '
                            'either be a dictionary or a number.')
    else:
        raise TypeError('The surrogate explanation has to be a dictionary.')

    # Collect all of the class names and their number
    class_names = sorted(plot_explanation.keys())
    class_n = len(class_names)

    # Check whether all of the explanations share the same key names
    share_y = True
    fig_shape = (1, class_n)
    #
    explanation_label_set_old = set()  # type: Set[str]
    for class_explanation in plot_explanation.values():
        if not explanation_label_set_old:
            explanation_label_set_old = set(class_explanation.keys())
        else:
            explanation_labels_set = set(class_explanation.keys())
            if explanation_label_set_old != explanation_labels_set:
                share_y = False
                fig_shape = (class_n, 1)
                logger.info('The explanations cannot share the y-axis as they '
                            'use different sets of interpretable features.')
                break

    # If sharing y-axis get a common ordering of the explanations
    if share_y:
        name_ordering = sorted(plot_explanation[class_names[0]].keys())

    figure, axes = plt.subplots(*fig_shape, sharey=share_y, sharex=True)
    figure.suptitle('Surrogate Explanation')

    # Do the plotting
    for i in range(class_n):
        class_name = class_names[i]
        class_explanation = plot_explanation[class_name]

        # Make sure that all bar plots are in the same order if sharing
        if share_y:
            exp_names = name_ordering
        else:
            exp_names = sorted(class_explanation.keys())
        exp_values = [class_explanation[name] for name in exp_names]

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
