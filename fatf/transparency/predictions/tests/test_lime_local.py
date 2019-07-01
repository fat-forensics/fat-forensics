"""
Tests fatf.transparency.predictions.lime explainer.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import pytest

try:
    import lime
except ImportError:  # pragma: no cover
    pytest.skip(
        'Skipping predictions lime wrapper tests -- lime missing.',
        allow_module_level=True)
else:
    del lime

import numpy as np

import fatf.transparency.predictions.lime as ftpl

DATA = np.ones((6, 4))


def test_lime():
    """
    Tests :class:`fatf.transparency.predictions.lime.Lime` class.
    """
    wmsg = ('Disregarding the sample_around_instance parameter -- this LIME '
            'tabular explainer object should only be used to explain a '
            'prediction. If you are interested in explaining a model, please '
            'refer to the fatf.transparency.models.lime module.')
    lime = ftpl.Lime(DATA)
    assert lime.tabular_explainer.sample_around_instance is True

    lime = ftpl.Lime(DATA, sample_around_instance=True)
    assert lime.tabular_explainer.sample_around_instance is True

    with pytest.warns(UserWarning) as w:
        lime = ftpl.Lime(DATA, sample_around_instance=False)
    assert len(w) == 1
    assert str(w[0].message) == wmsg
    assert lime.tabular_explainer.sample_around_instance is True

    with pytest.warns(UserWarning) as w:
        lime = ftpl.Lime(DATA, sample_around_instance=0)
    assert len(w) == 1
    assert str(w[0].message) == wmsg
    assert lime.tabular_explainer.sample_around_instance is True

    with pytest.warns(UserWarning) as w:
        lime = ftpl.Lime(DATA, sample_around_instance='')
    assert len(w) == 1
    assert str(w[0].message) == wmsg
    assert lime.tabular_explainer.sample_around_instance is True

    lime = ftpl.Lime(DATA, sample_around_instance='42')
    assert lime.tabular_explainer.sample_around_instance is True

    lime = ftpl.Lime(DATA, sample_around_instance=42)
    assert lime.tabular_explainer.sample_around_instance is True
