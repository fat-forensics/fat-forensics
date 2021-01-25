"""
FAT Forensics
=============

FAT Forensics is a Python module integrating a variety of fairness,
accountability (security, privacy) and transparency (explainability,
interpretability) approaches to assess social impact of artificial
intelligence systems.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

from typing import Optional

import logging
import os
import re
import sys
import warnings

# Author and license information
__author__ = 'Kacper Sokol'
__email__ = 'k.sokol@bristol.ac.uk'
__license__ = 'new BSD'

# The current package version
__version__ = '0.1.0'

__all__ = ['setup_warning_filters', 'setup_random_seed']

# Set up logging; enable logging of level INFO and higher
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
_logger_handler = logging.StreamHandler()  # pylint: disable=invalid-name
_logger_formatter = logging.Formatter(  # pylint: disable=invalid-name
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    datefmt='%y-%b-%d %H:%M:%S')
_logger_handler.setFormatter(_logger_formatter)
logger.addHandler(_logger_handler)
if os.environ.get('PYTEST_IN_PROGRESS', None) is None:
    logger.setLevel(logging.INFO)  # pragma: nocover
else:
    logger.setLevel(logging.NOTSET)

# Redirect warnings to the logger module
# logging.captureWarnings(True)
# py_warnings = logging.getLogger('py.warnings')
# py_warnings.addHandler(_logger_handler)
# py_warnings.setLevel(logging.INFO)


def setup_warning_filters():
    """
    Sets up desired warning filters.

    If the warning filters are not specified on the command line or via
    the system variable make sure that :class:`DeprecationWarning` and
    :class:`ImportWarning` raised by this this package always get printed.

    The warning settings used by pytest can be found in pytest.ini, where in
    addition to these two warnings :class:`PendingDeprecationWarning` is
    enabled as well.

    This functionality is tested by test_warnings_emission1() and
    test_warnings_emission2() functions in fatf.tests.test_warning_filters
    module.
    """
    if not sys.warnoptions:
        warnings.filterwarnings(
            'always',
            category=DeprecationWarning,
            module=r'^{0}\.'.format(re.escape(__name__)))
        warnings.filterwarnings(
            'always',
            category=ImportWarning,
            module=r'^{0}\.'.format(re.escape(__name__)))
    else:
        logger.info('External warning filters are being used.')


if 'PYTEST_IN_PROGRESS' not in os.environ:
    setup_warning_filters()  # pragma: no cover


# This function is tested in fatf.tests.test_rngs_seeding
def setup_random_seed(seed: Optional[int] = None) -> None:
    """
    Sets up Python's and numpy's random seed.

    Fixture for the tests to assure globally controllable seeding of random
    number generators in both Python (:func:`random.seed`) and ``numpy``
    (``numpy.random.seed``). The seed is taken either from ``FATF_SEED``
    system variable or from the ``seed`` input parameter; if neither of
    the two is given, it is sampled uniformly from 0--2147483647 range.

    .. note::

       If both ``FATF_SEED`` system variable and ``seed`` input parameter are
       given, the ``seed`` parameter takes the precedence.

    This function loggs (``info``) the origin of the random seed and its value.

    Parameters
    ----------
    seed : integer, optional (default=None)
        .. versionadded:: 0.0.2

        An integer in 0--2147483647 range used to seed Python's and numpy's
        random number generator.

    Raises
    ------
    TypeError
        The ``seed`` input parameter is not an integer.
    ValueError
        The ``seed`` input parameter is outside of the allowed 0--2147483647
        range. The random seed retrieved from the ``FATF_SEED`` system variable
        is either outside of the allowed range or cannot be parsed as an
        integer.
    """
    import numpy as np
    import random

    lower_bound = 0
    upper_bound = 2147483647

    if seed is None:
        # It could have been provided in the environment
        _random_seed_os = os.environ.get('FATF_SEED', None)
        if _random_seed_os is not None:
            # Random seed given as a system variable
            _random_seed_os = _random_seed_os.strip()
            if _random_seed_os.isdigit():
                _random_seed = int(_random_seed_os)
                if _random_seed < lower_bound or _random_seed > upper_bound:
                    raise ValueError('The random seed retrieved from the '
                                     'FATF_SEED system variable ({}) is '
                                     'outside of the allowed 0--2147483647 '
                                     'range.'.format(_random_seed))
                logger.info('Seeding RNGs using the system variable.')
            else:
                raise ValueError('The random seed retrieved from the '
                                 'FATF_SEED system variable ({}) '
                                 'cannot be parsed as a non-negative '
                                 'integer.'.format(_random_seed_os))
        else:
            # No user-defined random seed -- generate randomly
            _random_seed = int(np.random.uniform() * (2**31 - 1))
            logger.info('Seeding RNGs at random.')
    else:
        if isinstance(seed, int):
            if seed < lower_bound or seed > upper_bound:
                raise ValueError('The seed parameter is outside of the '
                                 'allowed 0--2147483647 range.')
            _random_seed = seed
            logger.info('Seeding RNGs using the input parameter.')
        else:
            raise TypeError('The seed parameter is not an integer.')

    logger.info('Seeding RNGs with %r.', _random_seed)
    np.random.seed(_random_seed)
    random.seed(_random_seed)
