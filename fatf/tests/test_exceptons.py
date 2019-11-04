"""
Tests custom warnings, errors and exceptions.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import pytest

import fatf.exceptions


def test_fatfexception():
    """
    Tests :class:`fatf.exceptions.FATFException`.
    """
    default_message = ''
    # Custom exception without a message
    with pytest.raises(fatf.exceptions.FATFException) as exception_info:
        raise fatf.exceptions.FATFException()
    assert str(exception_info.value) == default_message

    # Custom exception without a message
    with pytest.raises(fatf.exceptions.FATFException) as exception_info:
        raise fatf.exceptions.FATFException
    assert str(exception_info.value) == default_message

    # Custom exception with a message
    custom_message = 'Custom message.'
    with pytest.raises(fatf.exceptions.FATFException) as exception_info:
        raise fatf.exceptions.FATFException(custom_message)
    assert str(exception_info.value) == custom_message


def test_incorrectshapeerror():
    """
    Tests :class:`fatf.exceptions.IncorrectShapeError`.
    """
    default_message = ''
    # Custom exception without a message
    with pytest.raises(fatf.exceptions.IncorrectShapeError) as exin:
        raise fatf.exceptions.IncorrectShapeError()
    assert str(exin.value) == default_message

    # Custom exception without a message
    with pytest.raises(fatf.exceptions.IncorrectShapeError) as exin:
        raise fatf.exceptions.IncorrectShapeError
    assert str(exin.value) == default_message

    # Custom exception with a message
    custom_message = 'Custom message.'
    with pytest.raises(fatf.exceptions.IncorrectShapeError) as exin:
        raise fatf.exceptions.IncorrectShapeError(custom_message)
    assert str(exin.value) == custom_message


def test_incompatibleexplainererror():
    """
    Tests :class:`fatf.exceptions.IncompatibleExplainerError`.
    """
    default_message = ''
    # Custom exception without a message
    with pytest.raises(fatf.exceptions.IncompatibleExplainerError) as exin:
        raise fatf.exceptions.IncompatibleExplainerError()
    assert str(exin.value) == default_message

    # Custom exception without a message
    with pytest.raises(fatf.exceptions.IncompatibleExplainerError) as exin:
        raise fatf.exceptions.IncompatibleExplainerError
    assert str(exin.value) == default_message

    # Custom exception with a message
    custom_message = 'Custom message.'
    with pytest.raises(fatf.exceptions.IncompatibleExplainerError) as exin:
        raise fatf.exceptions.IncompatibleExplainerError(custom_message)
    assert str(exin.value) == custom_message


def test_incompatiblemodelerror():
    """
    Tests :class:`fatf.exceptions.IncompatibleModelError`.
    """
    default_message = ''
    # Custom exception without a message
    with pytest.raises(fatf.exceptions.IncompatibleModelError) as exin:
        raise fatf.exceptions.IncompatibleModelError()
    assert str(exin.value) == default_message

    # Custom exception without a message
    with pytest.raises(fatf.exceptions.IncompatibleModelError) as exin:
        raise fatf.exceptions.IncompatibleModelError
    assert str(exin.value) == default_message

    # Custom exception with a message
    custom_message = 'Custom message.'
    with pytest.raises(fatf.exceptions.IncompatibleModelError) as exin:
        raise fatf.exceptions.IncompatibleModelError(custom_message)
    assert str(exin.value) == custom_message


def test_unfittedmodelerror():
    """
    Tests :class:`fatf.exceptions.UnfittedModelError`.
    """
    default_message = ''
    # Custom exception without a message
    with pytest.raises(fatf.exceptions.UnfittedModelError) as exception_info:
        raise fatf.exceptions.UnfittedModelError()
    assert str(exception_info.value) == default_message

    # Custom exception without a message
    with pytest.raises(fatf.exceptions.UnfittedModelError) as exception_info:
        raise fatf.exceptions.UnfittedModelError
    assert str(exception_info.value) == default_message

    # Custom exception with a message
    custom_message = 'Custom message.'
    with pytest.raises(fatf.exceptions.UnfittedModelError) as exception_info:
        raise fatf.exceptions.UnfittedModelError(custom_message)
    assert str(exception_info.value) == custom_message


def test_prefittedmodelerror():
    """
    Tests :class:`fatf.exceptions.PrefittedModelError`.
    """
    default_message = ''
    # Custom exception without a message
    with pytest.raises(fatf.exceptions.PrefittedModelError) as exception_info:
        raise fatf.exceptions.PrefittedModelError()
    assert str(exception_info.value) == default_message

    # Custom exception without a message
    with pytest.raises(fatf.exceptions.PrefittedModelError) as exception_info:
        raise fatf.exceptions.PrefittedModelError
    assert str(exception_info.value) == default_message

    # Custom exception with a message
    custom_message = 'Custom message.'
    with pytest.raises(fatf.exceptions.PrefittedModelError) as exception_info:
        raise fatf.exceptions.PrefittedModelError(custom_message)
    assert str(exception_info.value) == custom_message
