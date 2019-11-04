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
    future_warning = (
        'The LIME wrapper will be deprecated in FAT Forensics version '
        '0.0.3. Please consider using the TabularBlimeyLime explainer '
        'class implemented in the fatf.transparency.predictions.'
        'surrogate_explainers module instead. Alternatively, you may '
        'consider building a custom surrogate explainer using the '
        'functionality implemented in FAT Forensics -- see the *Tabular '
        'Surrogates* how-to guide for more details.')

    with pytest.warns(FutureWarning) as w:
        lime = ftpl.Lime(DATA)
    assert len(w) == 1
    assert str(w[0].message) == future_warning
    assert lime.tabular_explainer.sample_around_instance is True

    with pytest.warns(FutureWarning) as w:
        lime = ftpl.Lime(DATA, sample_around_instance=True)
    assert len(w) == 1
    assert str(w[0].message) == future_warning
    assert lime.tabular_explainer.sample_around_instance is True

    with pytest.warns(None) as w:
        lime = ftpl.Lime(DATA, sample_around_instance=False)
    assert len(w) == 2
    assert str(w[0].message) == wmsg
    assert str(w[1].message) == future_warning
    assert lime.tabular_explainer.sample_around_instance is True

    with pytest.warns(None) as w:
        lime = ftpl.Lime(DATA, sample_around_instance=0)
    assert len(w) == 2
    assert str(w[0].message) == wmsg
    assert str(w[1].message) == future_warning
    assert lime.tabular_explainer.sample_around_instance is True

    with pytest.warns(None) as w:
        lime = ftpl.Lime(DATA, sample_around_instance='')
    assert len(w) == 2
    assert str(w[0].message) == wmsg
    assert str(w[1].message) == future_warning
    assert lime.tabular_explainer.sample_around_instance is True

    with pytest.warns(FutureWarning) as w:
        lime = ftpl.Lime(DATA, sample_around_instance='42')
    assert len(w) == 1
    assert str(w[0].message) == future_warning
    assert lime.tabular_explainer.sample_around_instance is True

    with pytest.warns(FutureWarning) as w:
        lime = ftpl.Lime(DATA, sample_around_instance=42)
    assert len(w) == 1
    assert str(w[0].message) == future_warning
    assert lime.tabular_explainer.sample_around_instance is True
