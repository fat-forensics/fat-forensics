"""
The :mod:`fatf.utils.tools` module implements general tools for the package.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

from typing import List

__all__ = ['at_least_verion']


def at_least_verion(minimum_requirement: List[int],
                    package_version: List[int]) -> bool:
    """
    Checks if the ``package_version`` satisfies the ``minimum_requirement``.

    Both ``package_version`` and ``minimum_requirement`` are lists of integers
    representing Python package versions, e.g. ``[1, 16, 4]``. This function
    returns ``True`` if the ``package_version`` is at least as high as the
    ``minimum_requirement`` version.

    The ``package_version`` list has to be at least as long as the
    ``minimum_requirement`` list, otherwise a ``ValueError`` exception is
    raised.

    Parameters
    ----------
    minimum_requirement : List[integer]
        The minimum Python package version that is required.
    package_version : list[integer]
        The Python package version to be tested.

    Raises
    ------
    TypeError
        Either of the parameters is not a list or the elements of the lists are
        not integers.
    ValueError
        Raised when either of the input lists is empty and when the
        ``package_version`` list is shorter than the ``minimum_requirement``
        list.

    Returns
    -------
    is_compatible : boolean
        ``True`` if the ``package_version`` satisfies the
        ``minimum_requirement``, ``False`` otherwise.
    """
    # pylint: disable=too-many-branches
    if not isinstance(minimum_requirement, list):
        raise TypeError('minimum_requirement parameter has to be a list.')
    for i, val in enumerate(minimum_requirement):
        if not isinstance(val, int):
            raise TypeError(('{} element ({}) of the minimum_requirement '
                             'list is not an integer.').format(i, val))

    if not isinstance(package_version, list):
        raise TypeError('package_version parameter has to be a list.')
    for i, val in enumerate(package_version):
        if not isinstance(val, int):
            raise TypeError(('{} element ({}) of the package_version list '
                             'is not an integer.').format(i, val))

    # Empty requirement
    if not minimum_requirement:
        raise ValueError('Minimum version for a package is not specified.')
    if not package_version:
        raise ValueError('Current version for a package is not specified.')

    minimum_requirement_len = len(minimum_requirement)
    package_version_len = len(package_version)
    if minimum_requirement_len > package_version_len:
        raise ValueError('The minimum requirement should not be more precise '
                         '(longer) than the current version.')

    is_compatible = True
    for i in range(minimum_requirement_len):
        if package_version[i] < minimum_requirement[i]:
            is_compatible = False
            break
        elif package_version[i] > minimum_requirement[i]:
            assert is_compatible, 'is_compatible should be true -- loop break.'
            is_compatible = True
            break

    return is_compatible
