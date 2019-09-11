"""
FAT-Forensics
=============

FAT-Forensics is a Python module integrating a variety of fairness,
accountability (security, privacy) and transparency (explainability,
interpretability) approaches to assess social impact of artificial
intelligence systems.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import logging
import os
import re
import sys
import warnings

__all__ = ['setup_warning_filters', 'setup_random_seed']

# Set up logging; enable logging of level INFO and higher
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
_logger_handler = logging.StreamHandler()  # pylint: disable=invalid-name
_logger_formatter = logging.Formatter(  # pylint: disable=invalid-name
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    datefmt='%y-%b-%d %H:%M:%S')
_logger_handler.setFormatter(_logger_formatter)
logger.addHandler(_logger_handler)
logger.setLevel(logging.INFO)

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

# Set the current package version
__version__ = '0.0.1'


# This function is tested in fatf.tests.test_rngs_seeding
def setup_random_seed():
    """
    Sets up Python's and numpy's random seed.

    Fixture for the tests to assure globally controllable seeding of random
    number generators in both Python (:func:`random.seed`) and ``numpy``
    (``numpy.random.seed``). The seed is taken either from ``FATF_SEED``
    system variable; if not given it's sampled uniformly from range
    0--2147483647.
    """
    import numpy as np
    import random

    # It could have been provided in the environment
    _random_seed = os.environ.get('FATF_SEED', None)
    if _random_seed is None:
        _random_seed = np.random.uniform() * (2**31 - 1)
    _random_seed = int(_random_seed)
    logger.info('Seeding RNGs with %r.', _random_seed)
    np.random.seed(_random_seed)
    random.seed(_random_seed)
