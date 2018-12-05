import pytest

import fatf.exceptions

def test_customexception():
    default_message = ''
    # Custom exception without a message
    with pytest.raises(fatf.exceptions.CustomException) as exception_info:
        raise fatf.exceptions.CustomException()
    assert exception_info.value.message == default_message
    assert str(exception_info.value) == default_message

    # Custom exception without a message
    with pytest.raises(fatf.exceptions.CustomException) as exception_info:
        raise fatf.exceptions.CustomException
    assert exception_info.value.message == default_message
    assert str(exception_info.value) == default_message

    # Custom exception with a message
    custom_message = 'Custom message.'
    with pytest.raises(fatf.exceptions.CustomException) as exception_info:
        raise fatf.exceptions.CustomException(custom_message)
    assert exception_info.value.message == custom_message
    assert str(exception_info.value) == custom_message

def test_customvalueerror():
    default_message = ''
    # Custom exception without a message
    with pytest.raises(fatf.exceptions.CustomValueError) as exception_info:
        raise fatf.exceptions.CustomValueError()
    assert exception_info.value.message == default_message
    assert str(exception_info.value) == default_message

    # Custom exception without a message
    with pytest.raises(fatf.exceptions.CustomValueError) as exception_info:
        raise fatf.exceptions.CustomValueError
    assert exception_info.value.message == default_message
    assert str(exception_info.value) == default_message

    # Custom exception with a message
    custom_message = 'Custom message.'
    with pytest.raises(fatf.exceptions.CustomValueError) as exception_info:
        raise fatf.exceptions.CustomValueError(custom_message)
    assert exception_info.value.message == custom_message
    assert str(exception_info.value) == custom_message

def test_missingimplementationexception():
    default_message = ('This is a default message.\nThis method/function has '
                       'not been implemented yet.'
                      )
    # Custom exception without a message
    with pytest.raises(fatf.exceptions.MissingImplementationException) as \
            exception_info:
        raise fatf.exceptions.MissingImplementationException()
    assert exception_info.value.message == default_message
    assert str(exception_info.value) == default_message

    # Custom exception without a message
    with pytest.raises(fatf.exceptions.MissingImplementationException) as \
            exception_info:
        raise fatf.exceptions.MissingImplementationException
    assert exception_info.value.message == default_message
    assert str(exception_info.value) == default_message

    # Custom exception with a message
    custom_message = 'Custom message.'
    with pytest.raises(fatf.exceptions.MissingImplementationException) as \
            exception_info:
        raise fatf.exceptions.MissingImplementationException(custom_message)
    assert exception_info.value.message == custom_message
    assert str(exception_info.value) == custom_message

def test_incorrectshapeexception():
    default_message = ('This is a default message.\nThis array has incorrect '
                       'shape.'
                      )
    # Custom exception without a message
    with pytest.raises(fatf.exceptions.IncorrectShapeException) as \
            exception_info:
        raise fatf.exceptions.IncorrectShapeException()
    assert exception_info.value.message == default_message
    assert str(exception_info.value) == default_message

    # Custom exception without a message
    with pytest.raises(fatf.exceptions.IncorrectShapeException) as \
            exception_info:
        raise fatf.exceptions.IncorrectShapeException
    assert exception_info.value.message == default_message
    assert str(exception_info.value) == default_message

    # Custom exception with a message
    custom_message = 'Custom message.'
    with pytest.raises(fatf.exceptions.IncorrectShapeException) as \
            exception_info:
        raise fatf.exceptions.IncorrectShapeException(custom_message)
    assert exception_info.value.message == custom_message
    assert str(exception_info.value) == custom_message

def test_incompatiblemodelexception():
    default_message = ('This is a default message.\nThis model is incompatible '
                       'with the desired functionality.'
                      )
    # Custom exception without a message
    with pytest.raises(fatf.exceptions.IncompatibleModelException) as \
            exception_info:
        raise fatf.exceptions.IncompatibleModelException()
    assert exception_info.value.message == default_message
    assert str(exception_info.value) == default_message

    # Custom exception without a message
    with pytest.raises(fatf.exceptions.IncompatibleModelException) as \
            exception_info:
        raise fatf.exceptions.IncompatibleModelException
    assert exception_info.value.message == default_message
    assert str(exception_info.value) == default_message

    # Custom exception with a message
    custom_message = 'Custom message.'
    with pytest.raises(fatf.exceptions.IncompatibleModelException) as \
            exception_info:
        raise fatf.exceptions.IncompatibleModelException(custom_message)
    assert exception_info.value.message == custom_message
    assert str(exception_info.value) == custom_message
