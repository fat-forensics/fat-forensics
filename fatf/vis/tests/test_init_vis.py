"""
Tests the initialisation of fatf visualisation module (:mod:`fatf.vis`).
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import importlib
import sys

import pytest

import fatf.utils.testing.imports as futi
import fatf.vis


def test_import_when_installed():
    """
    Tests importing :mod:`fatf.vis` module with matplotlib_ installed.

    .. _matplotlib: https://matplotlib.org/
    """
    assert 'fatf.vis' in sys.modules
    with futi.module_import_tester('matplotlib', when_missing=False):
        importlib.reload(fatf.vis)
    assert 'fatf.vis' in sys.modules


def test_import_when_missing():
    """
    Tests importing :mod:`fatf.vis` module with matplotlib_ missing.

    .. _matplotlib: https://matplotlib.org/
    """
    assert 'fatf.vis' in sys.modules
    warning_msg = (
        'matplotlib Python module is not installed on your system. '
        'You must install it in order to use fatf.vis functionality. '
        'One possibility is to install matplotlib alongside this package via '
        'visualisation dependencies with: pip install fatf[vis].')
    with futi.module_import_tester('matplotlib', when_missing=True):
        with pytest.warns(ImportWarning) as w:
            importlib.reload(fatf.vis)
        assert str(w[0].message) == warning_msg
    assert 'fatf.vis' in sys.modules
