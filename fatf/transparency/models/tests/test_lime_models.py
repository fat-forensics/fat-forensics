"""
Tests fatf.transparency.models.lime explainer.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import pytest

import numpy as np

import fatf.transparency.models.lime as ftml
import fatf.transparency.models as ftm

DATA = np.ones((6, 4))


def test_lime():
    """
    Tests :class:`fatf.transparency.models.lime.Lime` class.
    """
    wmsg = ('Disregarding the sample_around_instance parameter -- this LIME '
            'tabular explainer object should only be used to explain a '
            'model. If you are interested in explaining a prediction, please '
            'refer to the fatf.transparency.predictions.lime module.')
    lime = ftml.Lime(DATA)
    assert lime.tabular_explainer.sample_around_instance is False

    with pytest.warns(UserWarning) as w:
        lime = ftm.Lime(DATA, sample_around_instance=True)
    assert len(w) == 1
    assert str(w[0].message) == wmsg
    assert lime.tabular_explainer.sample_around_instance is False

    lime = ftml.Lime(DATA, sample_around_instance=False)
    assert lime.tabular_explainer.sample_around_instance is False

    lime = ftm.Lime(DATA, sample_around_instance=0)
    assert lime.tabular_explainer.sample_around_instance is False

    lime = ftml.Lime(DATA, sample_around_instance='')
    assert lime.tabular_explainer.sample_around_instance is False

    with pytest.warns(UserWarning) as w:
        lime = ftm.Lime(DATA, sample_around_instance='42')
    assert len(w) == 1
    assert str(w[0].message) == wmsg
    assert lime.tabular_explainer.sample_around_instance is False

    with pytest.warns(UserWarning) as w:
        lime = ftml.Lime(DATA, sample_around_instance=42)
    assert len(w) == 1
    assert str(w[0].message) == wmsg
    assert lime.tabular_explainer.sample_around_instance is False
