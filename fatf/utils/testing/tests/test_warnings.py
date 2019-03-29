"""
Tests helper functions for testing warning filters.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import re
import warnings

import pytest

import fatf.utils.testing.warnings as testing_w


def test_handle_warnings_filter_pattern():
    """
    Tests conversion and validation of patterns in a warning filter.

    Message and module parts of a warning filter are checked.
    """

    def assert_correct_pattern(error, error_message, pattern, ignore_case):
        with pytest.raises(error) as value_error:
            testing_w.handle_warnings_filter_pattern(
                pattern, ignore_case=ignore_case)
        assert str(value_error.value) == error_message

    # Test None
    assert testing_w.EMPTY_RE == testing_w.handle_warnings_filter_pattern(
        None, ignore_case=False)
    assert testing_w.EMPTY_RE_I == testing_w.handle_warnings_filter_pattern(
        None, ignore_case=True)

    # Test string
    my_str_pattern = 'my pattern'
    my_re_pattern = re.compile(my_str_pattern)
    my_re_pattern_i = re.compile(my_str_pattern, re.IGNORECASE)
    assert my_re_pattern == testing_w.handle_warnings_filter_pattern(
        my_str_pattern, ignore_case=False)
    assert my_re_pattern_i == testing_w.handle_warnings_filter_pattern(
        my_str_pattern, ignore_case=True)

    # Test re.compile return type
    assert my_re_pattern == testing_w.handle_warnings_filter_pattern(
        my_re_pattern, ignore_case=False)

    assert my_re_pattern_i == testing_w.handle_warnings_filter_pattern(
        my_re_pattern_i, ignore_case=True)
    value_error_message = (
        'The input regular expression should {neg} be compiled with '
        're.IGNORECASE flag -- it is imposed by the warning_filter_pattern '
        'input variable.')
    value_error_message_yes = value_error_message.format(neg='')
    value_error_message_no = value_error_message.format(neg='not')
    #
    assert_correct_pattern(ValueError, value_error_message_yes, my_re_pattern,
                           True)
    assert_correct_pattern(ValueError, value_error_message_no, my_re_pattern_i,
                           False)

    # Test other types: int, list, dict
    type_error_message = (
        'The warning filter module pattern should be either a string, a '
        'regular expression pattern or a None type.')
    assert_correct_pattern(TypeError, type_error_message, 4, False)
    assert_correct_pattern(TypeError, type_error_message, 2, True)
    assert_correct_pattern(TypeError, type_error_message, [4, 2], False)
    assert_correct_pattern(TypeError, type_error_message, [2, 4], True)
    dict_example = {1: '4', 2: '2'}
    assert_correct_pattern(TypeError, type_error_message, dict_example, False)
    assert_correct_pattern(TypeError, type_error_message, dict_example, True)

    # Test other regex flags
    flag_m = re.compile('', re.MULTILINE)
    assert flag_m == testing_w.handle_warnings_filter_pattern(
        flag_m, ignore_case=False)
    assert_correct_pattern(ValueError, value_error_message_yes, flag_m, True)
    #
    flag_i = re.compile('', re.IGNORECASE)
    assert flag_i == testing_w.handle_warnings_filter_pattern(
        flag_i, ignore_case=True)
    assert_correct_pattern(ValueError, value_error_message_no, flag_i, False)
    #
    flag_a = re.compile('', re.ASCII)
    assert flag_a == testing_w.handle_warnings_filter_pattern(
        flag_a, ignore_case=False)
    assert_correct_pattern(ValueError, value_error_message_yes, flag_m, True)
    #
    flag_mi = re.compile('', re.MULTILINE | re.IGNORECASE)
    assert flag_mi == testing_w.handle_warnings_filter_pattern(
        flag_mi, ignore_case=True)
    assert_correct_pattern(ValueError, value_error_message_no, flag_mi, False)
    #
    flag_ma = re.compile('', re.MULTILINE | re.ASCII)
    assert flag_ma == testing_w.handle_warnings_filter_pattern(
        flag_ma, ignore_case=False)
    assert_correct_pattern(ValueError, value_error_message_yes, flag_ma, True)
    #
    flag_ai = re.compile('', re.ASCII | re.IGNORECASE)
    assert flag_ai == testing_w.handle_warnings_filter_pattern(
        flag_ai, ignore_case=True)
    assert_correct_pattern(ValueError, value_error_message_no, flag_ai, False)


def test_set_default_warning_filters():
    """
    Tests setting up default filters.
    """
    testing_w.set_default_warning_filters()

    filters_number = len(testing_w.DEFAULT_WARNINGS)
    assert len(warnings.filters) == filters_number

    for i in range(filters_number):
        builtin_filter = warnings.filters[i]
        default_filter = testing_w.DEFAULT_WARNINGS[filters_number - 1 - i]

        # Compare warning action
        assert builtin_filter[0] == default_filter[0]
        # Compare message
        assert (testing_w.handle_warnings_filter_pattern(
            builtin_filter[1],
            ignore_case=True) == testing_w.handle_warnings_filter_pattern(
                default_filter[1], ignore_case=True))
        # Compare warning category
        assert builtin_filter[2] == default_filter[2]
        # Compare module
        assert (testing_w.handle_warnings_filter_pattern(
            builtin_filter[3],
            ignore_case=False) == testing_w.handle_warnings_filter_pattern(
                default_filter[3], ignore_case=False))
        # Compare lineno
        assert builtin_filter[4] == default_filter[4]


def test_is_warning_class_displayed():
    """
    Tests a function responsible for checking warning filters setup.

    This function tests whether a function responsible for checking whether a
    particular warning class is displayed based on the available warning
    filters behaves as expected.
    """
    # No warning filters -> display
    warnings.resetwarnings()
    assert testing_w.is_warning_class_displayed(DeprecationWarning)

    # No filter for this particular class -> display
    warnings.filterwarnings('default', category=ImportWarning, module='')
    assert testing_w.is_warning_class_displayed(DeprecationWarning)

    # A filter that blocks -> do not display
    warnings.resetwarnings()
    warnings.filterwarnings('ignore', category=DeprecationWarning, module='')
    assert not testing_w.is_warning_class_displayed(DeprecationWarning)

    # A filter that allows -> display
    warnings.resetwarnings()
    warnings.filterwarnings('default', category=DeprecationWarning, module='')
    assert testing_w.is_warning_class_displayed(DeprecationWarning)
    #
    warnings.resetwarnings()
    warnings.filterwarnings('error', category=DeprecationWarning, module='')
    assert testing_w.is_warning_class_displayed(DeprecationWarning)
    #
    warnings.resetwarnings()
    warnings.filterwarnings('always', category=DeprecationWarning, module='')
    assert testing_w.is_warning_class_displayed(DeprecationWarning)
    #
    warnings.resetwarnings()
    warnings.filterwarnings('module', category=DeprecationWarning, module='')
    assert testing_w.is_warning_class_displayed(DeprecationWarning)
    #
    warnings.resetwarnings()
    warnings.filterwarnings('module', category=DeprecationWarning, module='')
    assert testing_w.is_warning_class_displayed(DeprecationWarning)

    # A block filter that overwrites another (pass) filter -> do not display
    warnings.resetwarnings()
    warnings.filterwarnings('always', category=DeprecationWarning, module='')
    warnings.filterwarnings('ignore', category=DeprecationWarning, module='')
    assert not testing_w.is_warning_class_displayed(DeprecationWarning)

    # A pass filter that overwrites another (block) filter -> display
    warnings.resetwarnings()
    warnings.filterwarnings('ignore', category=DeprecationWarning, module='')
    warnings.filterwarnings('always', category=DeprecationWarning, module='')
    assert testing_w.is_warning_class_displayed(DeprecationWarning)

    # A filter with t namespace
    warnings.resetwarnings()
    warnings.filterwarnings('ignore', category=DeprecationWarning, module='t')
    assert testing_w.is_warning_class_displayed(DeprecationWarning,
                                                'fatf.test.t')
    assert not testing_w.is_warning_class_displayed(DeprecationWarning,
                                                    't.test')
    warnings.filterwarnings(
        'ignore', category=DeprecationWarning, module='fatf')
    assert not testing_w.is_warning_class_displayed(DeprecationWarning,
                                                    'fatf.test.t')
    warnings.filterwarnings(
        'always', category=DeprecationWarning, module='fatf.test')
    assert not testing_w.is_warning_class_displayed(DeprecationWarning,
                                                    'fatf.t')
    assert testing_w.is_warning_class_displayed(DeprecationWarning,
                                                'fatf.test.t')
