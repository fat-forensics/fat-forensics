"""
Test warning filter settings.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import sys
import warnings

import pytest

import fatf
import fatf.utils.testing.warnings

from fatf.utils.testing.warnings import EMPTY_RE


@pytest.mark.parametrize('error_type,error_class',
                         [('Import', ImportWarning),
                          ('Deprecation', DeprecationWarning),
                          ('Pending deprecation', PendingDeprecationWarning)])
def test_warnings_emission1(error_type, error_class):
    """
    Test whether :class:`ImportWarning`, :class:`DeprecationWarning` and
    :class:`PendingDeprecationWarning` warnings are printed. This test is
    executed with ``pytest`` warning filters (cf. ``pytest.ini``).
    """
    message = '{} warning test'.format(error_type)
    with pytest.warns(error_class, match=message) as record:
        warnings.warn(message, error_class)

    # Check that only one warning was raised
    assert len(record) == 1
    # Check that the message matches
    assert record[0].message.args[0] == message
    # Is it being displayed?
    assert fatf.utils.testing.warnings.is_warning_class_displayed(error_class)

    for fltr in warnings.filters:
        warning_matches_module = EMPTY_RE if fltr[3] is None else fltr[3]
        if warning_matches_module is not None:
            assert 'fatf' not in warning_matches_module.pattern


def test_warnings_emission2():
    """
    Test whether :class:`ImportWarning` and :class:`DeprecationWarning`
    warnings are printed. This test is executed with warning filters defined in
    :func:`fatf.setup_warning_filters`.
    """

    def test_record(error_type, error_class, displayed):
        message = '{} warning test'.format(error_type)
        with pytest.warns(error_class, match=message) as record:
            warnings.warn(message, error_class)

        # Check that only one warning was raised
        assert len(record) == 1
        # Check that the message matches
        assert record[0].message.args[0] == message
        # Is it being displayed?
        assert displayed == \
            fatf.utils.testing.warnings.is_warning_class_displayed(error_class)

    fatf.utils.testing.warnings.set_default_warning_filters()
    fatf.setup_warning_filters()

    assert len(warnings.filters) == \
        len(fatf.utils.testing.warnings.DEFAULT_WARNINGS) + 2
    for fltr in warnings.filters[:2]:
        warning_matches_module = EMPTY_RE if fltr[3] is None else fltr[3]
        if warning_matches_module is not None:
            assert 'fatf' in warning_matches_module.pattern

    test_warning = [('Import', ImportWarning),
                    ('Deprecation', DeprecationWarning)]
    for twi, tww in test_warning:
        test_record(twi, tww, True)

    test_record('Pending deprecation', PendingDeprecationWarning, False)


def test_warnings_emission3(caplog):
    """
    Check whether external warning filters are respected. These are supplied
    either via command line flag (e.g. ``python -Wdefault``) or system variable
    (e.g. ``PYTHONWARNINGS="default" python``). Using either of these two
    results in :data:`sys.warnoptions` list not being empty.
    """
    sys.warnoptions = ['default']
    fatf.utils.testing.warnings.set_default_warning_filters()

    fatf.setup_warning_filters()

    # Check logging
    message = 'External warning filters are being used.'
    # Check that only one message was logged
    assert len(caplog.records) == 1
    # Check this message's log level
    assert caplog.records[0].levelname == 'INFO'
    # Check that the message matches
    assert caplog.records[0].getMessage() == message
