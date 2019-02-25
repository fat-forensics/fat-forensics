"""
Functions to help test the code against warning generation.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

from typing import Optional
import warnings

# The default list (reversed, since they are appended) of warning filters as of
# Puthon 3.7
DEFAULT_WARNINGS = [('ignore', None, ResourceWarning, '', 0),
                    ('ignore', None, ImportWarning, '', 0),
                    ('ignore', '', PendingDeprecationWarning, '', 0),
                    ('ignore', None, DeprecationWarning, '', 0),
                    ('default', None, DeprecationWarning, '__main__', 0)]


def set_default_warning_filters() -> None:
    """
    Set the warning filters to default (as of Python 3.7).
    """
    warnings.resetwarnings()
    for warning in DEFAULT_WARNINGS:
        warnings.filterwarnings(
            warning[0], category=warning[2], module=warning[3])


def is_warning_class_displayed(warning_class: Warning,
                               warning_module: Optional[str] = None) -> bool:
    """
    Check whether a warning of a given class will be shown to the user.

    Parameters
    ----------
    warning_class : Warning
        A warning class to be checked.

    Returns
    -------
    is_displayed : bool
        True if the warning class will be displayed, False otherwise.
    warning_module : str
        The module string from which the warning is emitted. If not given, this
        defaults to ``fatf.dummy``.
    """
    if warning_module is None:
        warning_module = 'fatf.dummy'
    is_displayed = True

    # Filters for which warnings are displayed
    allowed_warning_filters = ['default', 'error', 'always', 'module', 'once']

    for active_filter in warnings.filters:  # type: ignore
        active_warning_filter = active_filter[0]
        active_warning_class = active_filter[2]
        active_warning_module = active_filter[3]
        if (issubclass(warning_class, active_warning_class) and  # type: ignore
                active_warning_module.match(warning_module)):
            if active_warning_filter in allowed_warning_filters:
                is_displayed = True
                break
            else:
                is_displayed = False
                break
    return is_displayed
