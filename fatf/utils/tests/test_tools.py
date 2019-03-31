"""
Tests package tools.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import pytest

import fatf.utils.tools as fut


def test_at_least_verion():
    """
    Tests :func:`fatf.utils.tools.at_least_verion` function.
    """
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    # Wrong outer input types
    min_type_error = 'minimum_requirement parameter has to be a list.'
    min_element_type_error = ('{} element ({}) of the minimum_requirement '
                              'list is not an integer.')
    curr_element_type_error = ('{} element ({}) of the package_version list '
                               'is not an integer.')
    current_type_error = 'package_version parameter has to be a list.'
    min_value_error = 'Minimum version for a package is not specified.'
    current_value_error = 'Current version for a package is not specified.'
    length_value_error = ('The minimum requirement should not be more precise '
                          '(longer) than the current version.')
    wrong_outer_types = [None, 0, '0', {}, range(5)]
    wrong_inner_types = [[None, None], ['0', '0'], [1., '0'], [2., 2.],
                         [1. + 0j, 6 + 1j, 2.]]
    partially_wrong_inner_type_1 = [[None, 0], ['0', 0]]
    partially_wrong_inner_type_2 = [[0, '0'], [0, None]]
    correct_inner_types = [[0], [0, 0]]
    empty_inner_type = [[]]

    for i in wrong_outer_types:
        for j in wrong_outer_types:
            with pytest.raises(TypeError) as exin:
                assert fut.at_least_verion(i, j)
            assert str(exin.value) == min_type_error

        for j in wrong_inner_types:
            with pytest.raises(TypeError) as exin:
                assert fut.at_least_verion(i, j)
            assert str(exin.value) == min_type_error
            with pytest.raises(TypeError) as exin:
                assert fut.at_least_verion(j, i)
            assert str(exin.value) == min_element_type_error.format(0, j[0])

        for j in partially_wrong_inner_type_1:
            with pytest.raises(TypeError) as exin:
                assert fut.at_least_verion(i, j)
            assert str(exin.value) == min_type_error
            with pytest.raises(TypeError) as exin:
                assert fut.at_least_verion(j, i)
            assert str(exin.value) == min_element_type_error.format(0, j[0])

        for j in partially_wrong_inner_type_2:
            with pytest.raises(TypeError) as exin:
                assert fut.at_least_verion(i, j)
            assert str(exin.value) == min_type_error
            with pytest.raises(TypeError) as exin:
                assert fut.at_least_verion(j, i)
            assert str(exin.value) == min_element_type_error.format(1, j[1])

        for j in correct_inner_types:
            with pytest.raises(TypeError) as exin:
                assert fut.at_least_verion(i, j)
            assert str(exin.value) == min_type_error
            with pytest.raises(TypeError) as exin:
                assert fut.at_least_verion(j, i)
            assert str(exin.value) == current_type_error

        for j in empty_inner_type:
            with pytest.raises(TypeError) as exin:
                assert fut.at_least_verion(i, j)
            assert str(exin.value) == min_type_error
            with pytest.raises(TypeError) as exin:
                assert fut.at_least_verion(j, i)
            assert str(exin.value) == current_type_error

    for i in wrong_inner_types:
        for j in wrong_inner_types:
            with pytest.raises(TypeError) as exin:
                assert fut.at_least_verion(i, j)
            assert str(exin.value) == min_element_type_error.format(0, i[0])

        for j in partially_wrong_inner_type_1:
            with pytest.raises(TypeError) as exin:
                assert fut.at_least_verion(i, j)
            assert str(exin.value) == min_element_type_error.format(0, i[0])
            with pytest.raises(TypeError) as exin:
                assert fut.at_least_verion(j, i)
            assert str(exin.value) == min_element_type_error.format(0, j[0])

        for j in partially_wrong_inner_type_2:
            with pytest.raises(TypeError) as exin:
                assert fut.at_least_verion(i, j)
            assert str(exin.value) == min_element_type_error.format(0, i[0])
            with pytest.raises(TypeError) as exin:
                assert fut.at_least_verion(j, i)
            assert str(exin.value) == min_element_type_error.format(1, j[1])

        for j in correct_inner_types:
            with pytest.raises(TypeError) as exin:
                assert fut.at_least_verion(i, j)
            assert str(exin.value) == min_element_type_error.format(0, i[0])
            with pytest.raises(TypeError) as exin:
                assert fut.at_least_verion(j, i)
            assert str(exin.value) == curr_element_type_error.format(0, i[0])

        for j in empty_inner_type:
            with pytest.raises(TypeError) as exin:
                assert fut.at_least_verion(i, j)
            assert str(exin.value) == min_element_type_error.format(0, i[0])
            with pytest.raises(TypeError) as exin:
                assert fut.at_least_verion(j, i)
            assert str(exin.value) == curr_element_type_error.format(0, i[0])

    for i in partially_wrong_inner_type_1:
        for j in partially_wrong_inner_type_1:
            with pytest.raises(TypeError) as exin:
                assert fut.at_least_verion(i, j)
            assert str(exin.value) == min_element_type_error.format(0, i[0])

        for j in partially_wrong_inner_type_2:
            with pytest.raises(TypeError) as exin:
                assert fut.at_least_verion(i, j)
            assert str(exin.value) == min_element_type_error.format(0, i[0])
            with pytest.raises(TypeError) as exin:
                assert fut.at_least_verion(j, i)
            assert str(exin.value) == min_element_type_error.format(1, j[1])

        for j in correct_inner_types:
            with pytest.raises(TypeError) as exin:
                assert fut.at_least_verion(i, j)
            assert str(exin.value) == min_element_type_error.format(0, i[0])
            with pytest.raises(TypeError) as exin:
                assert fut.at_least_verion(j, i)
            assert str(exin.value) == curr_element_type_error.format(0, i[0])

        for j in empty_inner_type:
            with pytest.raises(TypeError) as exin:
                assert fut.at_least_verion(i, j)
            assert str(exin.value) == min_element_type_error.format(0, i[0])
            with pytest.raises(TypeError) as exin:
                assert fut.at_least_verion(j, i)
            assert str(exin.value) == curr_element_type_error.format(0, i[0])

    for i in partially_wrong_inner_type_2:
        for j in partially_wrong_inner_type_2:
            with pytest.raises(TypeError) as exin:
                assert fut.at_least_verion(i, j)
            assert str(exin.value) == min_element_type_error.format(1, i[1])

        for j in correct_inner_types:
            with pytest.raises(TypeError) as exin:
                assert fut.at_least_verion(i, j)
            assert str(exin.value) == min_element_type_error.format(1, i[1])
            with pytest.raises(TypeError) as exin:
                assert fut.at_least_verion(j, i)
            assert str(exin.value) == curr_element_type_error.format(1, i[1])

        for j in empty_inner_type:
            with pytest.raises(TypeError) as exin:
                assert fut.at_least_verion(i, j)
            assert str(exin.value) == min_element_type_error.format(1, i[1])
            with pytest.raises(TypeError) as exin:
                assert fut.at_least_verion(j, i)
            assert str(exin.value) == curr_element_type_error.format(1, i[1])

    assert len(correct_inner_types) == 2, \
        'Testing correct_inner_types just for 2 elements.'
    assert fut.at_least_verion(correct_inner_types[0], correct_inner_types[1])
    with pytest.raises(ValueError) as exin:
        assert fut.at_least_verion(correct_inner_types[1],
                                   correct_inner_types[0])
    assert str(exin.value) == length_value_error
    for i in correct_inner_types:
        assert fut.at_least_verion(i, i)

        for j in empty_inner_type:
            with pytest.raises(ValueError) as exin:
                assert fut.at_least_verion(i, j)
            assert str(exin.value) == current_value_error
            with pytest.raises(ValueError) as exin:
                assert fut.at_least_verion(j, i)
            assert str(exin.value) == min_value_error

    for i in empty_inner_type:
        for j in empty_inner_type:
            with pytest.raises(ValueError) as exin:
                assert fut.at_least_verion(i, j)
            assert str(exin.value) == min_value_error

    # Correct outer, correct inner, different lengths
    # Current shorter than required and satisfies
    assert fut.at_least_verion([1], [1, 4, 2])
    assert fut.at_least_verion([1, 6], [1, 6, 0])
    # Current shorter than required and does not satisfy
    assert not fut.at_least_verion([1], [0, 4, 2])
    # Current longer than required and satisfies
    with pytest.raises(ValueError) as exin:
        assert fut.at_least_verion([1, 0, 0], [1, 0])
    assert str(exin.value) == length_value_error
    # Current longer than required and does not satisfy
    with pytest.raises(ValueError) as exin:
        assert fut.at_least_verion([1, 4, 2], [0, 5])
    assert str(exin.value) == length_value_error

    # Correct outer, correct inner, same lengths
    # Correct
    assert fut.at_least_verion([1, 4, 2], [1, 4, 2])
    assert fut.at_least_verion([1, 6, 2], [1, 7, 1])
    assert fut.at_least_verion([1, 6, 2], [2, 0, 0])
    # Incorrect
    assert not fut.at_least_verion([2, 0, 0], [1, 9, 9])
