"""
Tests random number generator seeding.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import os
import random

import numpy as np

import fatf


def test_random_seed(caplog):
    """
    Tests random number generator seeding when the seed is random.
    """
    fatf.setup_random_seed()
    seed = np.random.get_state()[1][0]
    message = 'Seeding RNGs with {}.'.format(seed)

    # Check logging
    # Check that only one message was logged
    assert len(caplog.records) == 1
    # Check this message's log level
    assert caplog.records[0].levelname == 'INFO'
    # Check that the message matches
    assert caplog.records[0].getMessage() == message

    # Check Python random state
    python_random_seed = random.getstate()
    random.seed(seed)
    assert random.getstate() == python_random_seed
    assert id(random.getstate()) != id(python_random_seed)


def test_osvar_seed(caplog):
    """
    Tests random number generator seeding with a system environment variable.
    """
    seed_int = 42
    seed_int_random = 2147483648
    seed_str = '{}'.format(seed_int)
    message = 'Seeding RNGs with {}.'.format(seed_str)

    os.environ['FATF_SEED'] = seed_str
    fatf.setup_random_seed()

    # Check logging
    # Check that only one message was logged
    assert len(caplog.records) == 1
    # Check this message's log level
    assert caplog.records[0].levelname == 'INFO'
    # Check that the message matches
    assert caplog.records[0].getMessage() == message

    # Pseudo-check the actual seed
    assert random.getstate()[1][0] == seed_int_random
    assert np.random.get_state()[1][0] == seed_int
