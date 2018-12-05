"""
The :mod:`fatf.exceptions` module includes all custom warnings and error
classes used across FAT-Forensics.
"""

# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: BSD 3 clause

class CustomException(Exception):
    """Base class for exceptions in this module.

    This custom exception inherits form Python's Exception and allows to
    define an optional custom message.

    Args
    ----
    message : str, optional
        A string with the exception message. Defaults to '' (empty string).

    Attributes
    ----------
    message : str
        A string holding the exception message.
    """

    def __init__(self, message: str = '') -> None:
        """Inits CustomException with an empty message unless one is given."""
        self.message = message

    def __str__(self) -> str:
        """Assigns message as a string representation of the exception."""
        return self.message


class CustomValueError(ValueError):
    """Base class for value errors in this module.

    This custom error inherits form Python's ValueError and allows to
    define an optional custom message.

    Args
    ----
    message : str, optional
        A string with the exception message. Defaults to '' (empty string).

    Attributes
    ----------
    message : str
        A string holding the exception message.
    """

    def __init__(self, message: str = '') -> None:
        """Inits CustomValueError with an empty message unless one is given."""
        self.message = message

    def __str__(self) -> str:
        """Assigns message as a string representation of the error."""
        return self.message


class MissingImplementationException(CustomException):
    """Exception raised for unimplemented functionality.

    Args
    ----
    message : str, optional
        A string with the exception message. Defaults to '' (empty string).

    Attributes
    ----------
    message : str
        A string holding the exception message.
    """

    def __init__(self, message: str = '') -> None:
        """Inits MissingImplementationException with an empty message unless
        one is given."""
        if not message:
            self.message = (
                    'This is a default message.\n'
                    'This method/function has not been implemented yet.'
                    )
        else:
            self.message = message


class IncorrectShapeException(CustomException):
    """Exception raised when the shape of an array is not what is expected.

    Args
    ----
    message : str, optional
        A string with the exception message. Defaults to '' (empty string).

    Attributes
    ----------
    message : str
        A string holding the exception message.
    """

    def __init__(self, message: str = '') -> None:
        """Inits IncorrectShapeException with an empty message unless one is
        given."""
        if not message:
            self.message = (
                    'This is a default message.\n'
                    'This array has incorrect shape.'
                    )
        else:
            self.message = message


class IncompatibleModelException(CustomException):
    """Exception raised when a machine learning model is incompatible with the
    functionality of a function or a method.

    This is usually raise if the model lacks fit, predict and, optionally,
    predict_proba methods.

    Args
    ----
    message : str, optional
        A string with the exception message. Defaults to '' (empty string).

    Attributes
    ----------
    message : str
        A string holding the exception message.
    """

    def __init__(self, message: str = '') -> None:
        """Inits IncompatibleModelException with an empty message unless one is
        given."""
        if not message:
            self.message = (
                    'This is a default message.\n'
                    'This model is incompatible with the desired functionality.'
                    )
        else:
            self.message = message
