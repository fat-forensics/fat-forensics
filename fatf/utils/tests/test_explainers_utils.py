"""
Tests explainers tools.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import pytest

import fatf.utils.explainers as fue


class TestExplainer(object):
    """
    Tests :class:`fatf.explainers.Explainer` class.
    """
    explainer = fue.Explainer()

    def test_explainer_class(self):
        """
        Tests :class:`fatf.explainers.Explainer` class initialisation.
        """
        assert self.explainer.__class__.__bases__[0].__name__ == 'ABC'
        assert self.explainer.__class__.__name__ == 'Explainer'

    def test_explainer_class_errors(self):
        """
        Tests :class:`fatf.explainers.Explainer` class unimplemented methods.
        """
        feature_importance = 'Feature importance not implemented.'
        model_explanation = 'Model explanation (global) not implemented.'
        prediction_explanation = ('Prediction explanation (local) not '
                                  'implemented.')

        with pytest.raises(NotImplementedError) as exinf:
            self.explainer.feature_importance()
        assert str(exinf.value) == feature_importance
        with pytest.raises(NotImplementedError) as exinf:
            self.explainer.model_explanation()
        assert str(exinf.value) == model_explanation
        with pytest.raises(NotImplementedError) as exinf:
            self.explainer.prediction_explanation()
        assert str(exinf.value) == prediction_explanation
