"""
Holds custom warnings, errors and exceptions.
"""  # yapf: disable
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

__all__ = ['FATFException',
           'MissingImplementationError',
           'IncorrectShapeError',
           'IncompatibleModelError']  # yapf: disable


class FATFException(Exception):
    """
    Base class for FAT-Forensics exceptions (inherits from :class:`Exception`).
    """


class MissingImplementationError(FATFException):
    """
    Exception raised for unimplemented functionality.
    """


class IncorrectShapeError(FATFException):
    """
    Exception raised when the shape of an array is not what is expected.
    """


class IncompatibleModelError(FATFException):
    """
    Exception raised when a model lacks desired functionality.

    For example, it can be raised when a model is expected to output
    probabilities for its prediction but it is does not support this
    functionality.
    """
