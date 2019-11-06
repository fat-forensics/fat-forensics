"""
Tests random number generator seeding.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import os
import random

import pytest

import numpy as np

import fatf


def test_random_seed(caplog):
    """
    Tests random number generator seeding when the seed is random.
    """
    fatf_seed = os.environ.get('FATF_SEED', None)
    if fatf_seed is not None:
        del os.environ['FATF_SEED']  # pragma: nocover
    assert 'FATF_SEED' not in os.environ

    fatf.setup_random_seed()
    seed = np.random.get_state()[1][0]
    message_source = 'Seeding RNGs at random.'
    message_seed = 'Seeding RNGs with {}.'.format(seed)

    # Check logging
    # Check that only one message was logged
    assert len(caplog.records) == 2
    assert caplog.records[0].levelname == 'INFO'
    assert caplog.records[0].getMessage() == message_source
    # Check this message's log level
    assert caplog.records[1].levelname == 'INFO'
    # Check that the message matches
    assert caplog.records[1].getMessage() == message_seed

    # Check Python random state
    python_random_seed = random.getstate()
    random.seed(seed)
    assert random.getstate() == python_random_seed
    assert id(random.getstate()) != id(python_random_seed)

    if fatf_seed is not None:
        os.environ['FATF_SEED'] = fatf_seed  # pragma: nocover
        assert 'FATF_SEED' in os.environ  # pragma: nocover


def test_osvar_seed(caplog):
    """
    Tests random number generator seeding with a system environment variable.
    """
    value_error_range = ('The random seed retrieved from the FATF_SEED system '
                         'variable (2147483648) is outside of the allowed '
                         '0--2147483647 range.')
    value_error_type = ('The random seed retrieved from the FATF_SEED system '
                        'variable (forty-two) cannot be parsed as a '
                        'non-negative integer.')

    seed_int = 42
    seed_int_random = 2147483648
    seed_str = '{}'.format(seed_int)
    message_source = 'Seeding RNGs using the system variable.'
    message_seed = 'Seeding RNGs with {}.'.format(seed_str)

    # Memorise the current state of the system variable
    fatf_seed = os.environ.get('FATF_SEED', None)

    # Check a valid seed
    os.environ['FATF_SEED'] = seed_str
    fatf.setup_random_seed()

    # Check logging
    # Check that only one message was logged
    assert len(caplog.records) == 2
    assert caplog.records[0].levelname == 'INFO'
    assert caplog.records[0].getMessage() == message_source
    # Check this message's log level
    assert caplog.records[1].levelname == 'INFO'
    # Check that the message matches
    assert caplog.records[1].getMessage() == message_seed

    # Pseudo-check the actual seed
    assert random.getstate()[1][0] == seed_int_random
    assert np.random.get_state()[1][0] == seed_int

    # Check invalid seed
    assert len(caplog.records) == 2
    # Not a number
    os.environ['FATF_SEED'] = 'forty-two'
    with pytest.raises(ValueError) as exin:
        fatf.setup_random_seed()
    assert str(exin.value) == value_error_type
    # Outside of the range
    os.environ['FATF_SEED'] = '2147483648'
    with pytest.raises(ValueError) as exin:
        fatf.setup_random_seed()
    assert str(exin.value) == value_error_range
    assert len(caplog.records) == 2

    # Restore the system variable
    if fatf_seed is None:
        del os.environ['FATF_SEED']  # pragma: nocover
    else:
        os.environ['FATF_SEED'] = fatf_seed  # pragma: nocover


def test_seed_seed(caplog):
    """
    Tests random number generator seeding using the ``seed`` input parameter.
    """
    type_error = 'The seed parameter is not an integer.'
    value_error = ('The seed parameter is outside of the allowed '
                   '0--2147483647 range.')

    message_source = 'Seeding RNGs using the input parameter.'
    message_seed = 'Seeding RNGs with 42.'

    assert len(caplog.records) == 0
    with pytest.raises(TypeError) as exin:
        fatf.setup_random_seed('42')
    assert str(exin.value) == type_error
    with pytest.raises(ValueError) as exin:
        fatf.setup_random_seed(-42)
    assert str(exin.value) == value_error
    assert len(caplog.records) == 0

    fatf.setup_random_seed(42)

    assert len(caplog.records) == 2
    assert caplog.records[0].levelname == 'INFO'
    assert caplog.records[0].getMessage() == message_source
    assert caplog.records[1].levelname == 'INFO'
    assert caplog.records[1].getMessage() == message_seed
    assert len(caplog.records) == 2
