"""
Holds tools for building custom explainer objects.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import abc

import numpy as np


class Explainer(abc.ABC):
    """
    A base class for any explainer object implemented in the package.
    """

    def feature_importance(self) -> np.ndarray:
        """
        Computes feature importance.
        """
        raise NotImplementedError('Feature importance not implemented.')

    def model_explanation(self) -> np.ndarray:
        """
        Generates a model explanation.
        """
        raise NotImplementedError('Model explanation (global) not '
                                  'implemented.')

    def prediction_explanation(self) -> np.ndarray:
        """
        Generates a prediction explanation.
        """
        raise NotImplementedError('Prediction explanation (local) not '
                                  'implemented.')
