"""
Test helper functions for testing warning filters.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import warnings

import fatf.utils.testing.warnings
from fatf.utils.testing.warnings import DEFAULT_WARNINGS


def test_set_default_warning_filters():
    """
    Test setting up default filters.
    """
    fatf.utils.testing.warnings.set_default_warning_filters()

    filters_number = len(DEFAULT_WARNINGS)
    assert len(warnings.filters) == filters_number

    for i in range(filters_number):
        buildin_filter = warnings.filters[i]
        default_filter = DEFAULT_WARNINGS[filters_number - 1 - i]

        # Compare warning action
        assert buildin_filter[0] == default_filter[0]
        # Compare message
        if default_filter[1] is None:
            default_filter_message = ''
        else:
            default_filter_message = default_filter[1]
        assert buildin_filter[1].pattern == default_filter_message
        # Compare warning category
        assert buildin_filter[2] == default_filter[2]
        # Compare module
        assert buildin_filter[3].pattern == default_filter[3]
        # Compare lineno
        assert buildin_filter[4] == default_filter[4]


def test_is_warning_class_displayed():
    """
    Test a function that checks whether a particular warning class is displayed
    based on the available warning filters.
    """
    iwcd = fatf.utils.testing.warnings.is_warning_class_displayed

    # No warning filters -> display
    warnings.resetwarnings()
    assert iwcd(DeprecationWarning)

    # No filter for this particular class -> display
    warnings.filterwarnings('default', category=ImportWarning, module='')
    assert iwcd(DeprecationWarning)

    # A filter that blocks -> do not display
    warnings.resetwarnings()
    warnings.filterwarnings('ignore', category=DeprecationWarning, module='')
    assert not iwcd(DeprecationWarning)

    # A filter that allows -> display
    warnings.resetwarnings()
    warnings.filterwarnings('default', category=DeprecationWarning, module='')
    assert iwcd(DeprecationWarning)
    #
    warnings.resetwarnings()
    warnings.filterwarnings('error', category=DeprecationWarning, module='')
    assert iwcd(DeprecationWarning)
    #
    warnings.resetwarnings()
    warnings.filterwarnings('always', category=DeprecationWarning, module='')
    assert iwcd(DeprecationWarning)
    #
    warnings.resetwarnings()
    warnings.filterwarnings('module', category=DeprecationWarning, module='')
    assert iwcd(DeprecationWarning)
    #
    warnings.resetwarnings()
    warnings.filterwarnings('module', category=DeprecationWarning, module='')
    assert iwcd(DeprecationWarning)

    # A block filter that overwrites another (pass) filter -> do not display
    warnings.resetwarnings()
    warnings.filterwarnings('always', category=DeprecationWarning, module='')
    warnings.filterwarnings('ignore', category=DeprecationWarning, module='')
    assert not iwcd(DeprecationWarning)

    # A pass filter that overwrites another (block) filter -> display
    warnings.resetwarnings()
    warnings.filterwarnings('ignore', category=DeprecationWarning, module='')
    warnings.filterwarnings('always', category=DeprecationWarning, module='')
    assert iwcd(DeprecationWarning)

    # A filter with t namespace
    warnings.resetwarnings()
    warnings.filterwarnings('ignore', category=DeprecationWarning, module='t')
    assert iwcd(DeprecationWarning, 'fatf.test.t')
    assert not iwcd(DeprecationWarning, 't.test')
    warnings.filterwarnings(
        'ignore', category=DeprecationWarning, module='fatf')
    assert not iwcd(DeprecationWarning, 'fatf.test.t')
    warnings.filterwarnings(
        'always', category=DeprecationWarning, module='fatf.test')
    assert not iwcd(DeprecationWarning, 'fatf.t')
    assert iwcd(DeprecationWarning, 'fatf.test.t')
