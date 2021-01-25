"""
Tests initialisation of (:mod:`fatf.transparency.sklearn`) module.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import importlib
import sys

import pytest

import fatf.utils.testing.imports as futi
import fatf.transparency.sklearn


def test_import_when_installed():
    """
    Tests importing :mod:`fatf.transparency.sklearn` with sklearn_ installed.

    .. _sklearn: https://scikit-learn.org/
    """
    assert 'fatf.transparency.sklearn' in sys.modules
    with futi.module_import_tester('sklearn', when_missing=False):
        importlib.reload(fatf.transparency.sklearn)
    assert 'fatf.transparency.sklearn' in sys.modules


def test_import_when_missing():
    """
    Tests importing :mod:`fatf.transparency.sklearn` with sklearn_ missing.

    .. _sklearn: https://scikit-learn.org/
    """
    assert 'fatf.transparency.sklearn' in sys.modules
    exception_msg = (
        'scikit-learn (sklearn) Python module is not installed on your '
        'system. You must install it in order to use '
        'fatf.transparency.sklearn functionality. '
        'One possibility is to install scikit-learn alongside this package '
        'via machine learning dependencies with: pip install '
        'fat-forensics[ml].')
    with futi.module_import_tester('sklearn', when_missing=True):
        with pytest.raises(ImportError) as excinfo:
            importlib.reload(fatf.transparency.sklearn)
        assert str(excinfo.value) == exception_msg
    assert 'fatf.transparency.sklearn' in sys.modules
