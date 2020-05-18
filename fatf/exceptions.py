"""
The :mod:`fatf.exceptions` module holds custom exceptions, errors and warnings.
"""  # yapf: disable
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

__all__ = ['FATFException',
           'IncorrectShapeError',
           'IncompatibleModelError',
           'UnfittedModelError',
           'PrefittedModelError']  # yapf: disable


class FATFException(Exception):
    """
    Base class for FAT Forensics exceptions (inherits from :class:`Exception`).
    """


class IncorrectShapeError(FATFException):
    """
    Exception raised when the shape of an array is not what is expected.
    """


class IncompatibleExplainerError(FATFException):
    """
    Exception raised when an explainer lacks desired functionality.

    .. versionadded:: 0.0.2

    For example, is can be raised when an explainer object does not have an
    ``explain_instance`` method or if the ``explain_instance`` method does not
    have at least one required parameter (an array with a data point to be
    explained).
    """


class IncompatibleModelError(FATFException):
    """
    Exception raised when a model lacks desired functionality.

    For example, it can be raised when a model is expected to output
    probabilities for its predictions but it does not support this
    functionality.
    """


class UnfittedModelError(FATFException):
    """
    Exception raised when a model is unfitted and a fitted one is expected.

    This is usually raised when the model is not fitted and a ``predict`` or
    a ``predict_proba`` method is called.
    """


class PrefittedModelError(FATFException):
    """
    Exception raised when a model is fitted and an unfitted one is expected.

    This is usually raised when a fitted model is tried to be fitted again.
    """
