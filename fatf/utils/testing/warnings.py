"""
The :mod:`fatf.utils.testing.warnings` module holds warnings testing functions.

Functions implemented in this module help to test the code against warning
generation and warning filter settings.

In addition the :mod:`fatf.utils.testing.warnings` module implements three
global variables:

``DEFAULT_WARNINGS``
  A list of default warning filters as of Python 3.7.

``EMPTY_RE``
  A compiled regular expression based on an empty string (``''``).

``EMPTY_RE_I``
  A compiled, case insensitive regular expression based on an empty string
  (``''``).
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import re
import warnings

from typing import Optional, Pattern, Union

__all__ = ['DEFAULT_WARNINGS',
           'EMPTY_RE',
           'EMPTY_RE_I',
           'handle_warnings_filter_pattern',
           'set_default_warning_filters',
           'is_warning_class_displayed']  # yapf: disable

# The default list (reversed, since they are appended) of warning filters as of
# Python 3.7
DEFAULT_WARNINGS = [('ignore', None, ResourceWarning, '', 0),
                    ('ignore', None, ImportWarning, '', 0),
                    ('ignore', '', PendingDeprecationWarning, '', 0),
                    ('ignore', None, DeprecationWarning, '', 0),
                    ('default', None, DeprecationWarning, '__main__', 0)]

EMPTY_RE = re.compile('')
EMPTY_RE_I = re.compile('', re.IGNORECASE)


def handle_warnings_filter_pattern(
        warning_filter_pattern: Union[None, str, Pattern],
        ignore_case: bool = False) -> Pattern:
    """
    Converts a patter within a warning filter into a regex pattern.

    If the input pattern is ``None`` an empty (string) regex pattern is
    returned. A string pattern is complied directly into a regex pattern. If
    the input pattern is already a regex pattern it is returned without any
    modifications.

    Parameters
    ----------
    warning_filter_pattern : Union[None, str, Pattern]
        A pattern (``None``, string or a regex pattern) extracted from a
        warning filter.
    ignore_case : boolean, optional (default=False)
        Should ``re.IGNORECASE`` flag be compiled into the module pattern?
        Defaults to ``False``.

    Raises
    ------
    TypeError
        The ``warning_filter_pattern`` input variable is neither of the
        following types: string, Pattern (``re.compile``) or None.
    ValueError
        The ``warning_filter_pattern`` input variable is a Pattern but its
        status of ``re.IGNORECASE`` flag does not agree with the requirement
        specified by the ``ignore_case`` input variable.

    Returns
    -------
    filter_module_regex : Pattern
        A regular expression pattern corresponding to the input warning filter
        module pattern.
    """
    filter_module_regex = None
    if warning_filter_pattern is None:
        if ignore_case:
            filter_module_regex = EMPTY_RE_I
        else:
            filter_module_regex = EMPTY_RE
    elif isinstance(warning_filter_pattern, str):
        if ignore_case:
            filter_module_regex = re.compile(warning_filter_pattern,
                                             re.IGNORECASE)
        else:
            filter_module_regex = re.compile(warning_filter_pattern)
    elif isinstance(warning_filter_pattern, Pattern):
        ignore_case_error_message = (
            'The input regular expression should {neg} be compiled with '
            're.IGNORECASE flag -- it is imposed by the '
            'warning_filter_pattern input variable.')
        ignore_case_compiled = warning_filter_pattern.flags & 2
        if ignore_case_compiled and ignore_case:
            filter_module_regex = warning_filter_pattern
        elif not ignore_case_compiled and not ignore_case:
            filter_module_regex = warning_filter_pattern
        elif ignore_case_compiled:
            raise ValueError(ignore_case_error_message.format(neg='not'))
        else:
            raise ValueError(ignore_case_error_message.format(neg=''))
    else:
        raise TypeError(
            'The warning filter module pattern should be either a string, a '
            'regular expression pattern or a None type.')
    return filter_module_regex


def set_default_warning_filters() -> None:
    """
    Sets the warning filters to default values (as of Python 3.7).
    """
    warnings.resetwarnings()
    for warning in DEFAULT_WARNINGS:
        warnings.filterwarnings(
            warning[0], category=warning[2], module=warning[3])


def is_warning_class_displayed(warning_class: Warning,
                               warning_module: Optional[str] = None) -> bool:
    """
    Checks whether a warning of a given class will be shown to the user.

    Parameters
    ----------
    warning_class : Warning
        A warning class to be checked.
    warning_module : string, optional (default='fatf.dummy')
        The module string from which the warning is emitted. If not given, this
        defaults to ``fatf.dummy``.

    Returns
    -------
    is_displayed : bool
        True if the warning class will be displayed, False otherwise.
    """
    if warning_module is None:
        warning_module = 'fatf.dummy'
    is_displayed = True

    # Filters for which warnings are displayed
    allowed_warning_filters = ['default', 'error', 'always', 'module', 'once']

    for fltr in warnings.filters:  # type: ignore
        active_warning_filter = fltr[0]
        active_warning_class = fltr[2]
        active_warning_module = handle_warnings_filter_pattern(
            fltr[3], ignore_case=False)
        if (issubclass(warning_class, active_warning_class)  # type: ignore
                and active_warning_module.match(warning_module)):
            if active_warning_filter in allowed_warning_filters:
                is_displayed = True
                break
            else:
                is_displayed = False
                break
    return is_displayed
