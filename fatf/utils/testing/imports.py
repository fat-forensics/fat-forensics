"""
The :mod:`fatf.utils.testing.imports` module holds import testing functions.

This module implements functions that help to test the FAT Forensics code that
imports other modules.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import contextlib
import sys

from typing import Iterator

__all__ = ['module_import_tester']


@contextlib.contextmanager
def module_import_tester(module_name: str,
                         when_missing: bool = True) -> Iterator[None]:
    """
    Provides a context for testing imports of installed and missing modules.

    This context can be used to get an environment where a particular module
    (``module_name``) is either not installed -- ``when_missing=True`` -- or
    installed -- ``when_missing=False``. The example below demonstrates a
    possible use case:

    >>> import fatf.utils.testing.imports as futi
    >>> with futi.module_import_tester('a_module', when_missing=True):
    ...     try:
    ...         import a_module
    ...     except ImportError:
    ...         print('Module not found!')
    Module not found!
    >>> with futi.module_import_tester('a_module', when_missing=False):
    ...     import a_module

    In the first example we are making sure that the import will fail by
    providing ``when_missing=True``. On the other hand, the second call ensures
    that the import succeeds.

    .. warning:: Python 3.6 and later will result in
       :class:`ModuleNotFoundError`, however Python 3.5 will raise
       :class:`ImportError`.

    Parameters
    ----------
    module_name : str
        A module name that we want to test given as a string.
    when_missing : bool
        A boolean parameter specifying whether the module named above should be
        available for import or not. Defaults to ``True``.

    Yields
    ------
    A context with the selected module either missing or present.
    """
    action = 0

    # If present and we want to test when missing, remove the module index.
    if when_missing:
        sys_path_backup = sys.path
        sys.path = []
        action = 1
        if module_name in sys.modules:
            module_backup = sys.modules[module_name]
            del sys.modules[module_name]
            action = 2
    # If missing and we want to test when present, fake the module index.
    elif not when_missing and module_name not in sys.modules:
        sys.modules[module_name] = sys.modules['contextlib']
        action = 3

    # Test the import -- context body.
    yield

    # Cleanup context -- restore Python package index.
    if action in (1, 2):
        sys.path = sys_path_backup
        if action == 2:
            sys.modules[module_name] = module_backup
    elif action == 3:
        del sys.modules[module_name]
