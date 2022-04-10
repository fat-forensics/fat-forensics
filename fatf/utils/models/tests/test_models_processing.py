"""
Tets the :mod:`fatf.utils.model.processing` module.

.. versionadded:: 0.1.1

Functions and classes for testing model processing functionality.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import pytest

import numpy as np

import fatf.utils.array.tools as fuat
import fatf.utils.models.processing as fump

from fatf.exceptions import IncorrectShapeError


def test_batch_data_exceptions():
    """Tests :func:`fatf.utils.model.processing.batch_data` exceptions."""
    array_1d = np.array([0, 1, 2])
    err_msg = 'The data array must be 2-dimensional.'
    with pytest.raises(IncorrectShapeError) as exin:
        fump.batch_data(array_1d)
    assert str(exin.value) == err_msg

    array_2d = np.array([[0, 1, 2], [4, 5, 6]])

    err_msg = 'The batch size must be an integer.'
    with pytest.raises(TypeError) as exin:
        fump.batch_data(array_2d, 'int')
    assert str(exin.value) == err_msg
    with pytest.raises(TypeError) as exin:
        fump.batch_data(array_2d, 33.0)
    assert str(exin.value) == err_msg

    err_msg = 'The batch size must be larger than 0.'
    with pytest.raises(ValueError) as exin:
        fump.batch_data(array_2d, 0)
    assert str(exin.value) == err_msg
    with pytest.raises(ValueError) as exin:
        fump.batch_data(array_2d, batch_size=-1)
    assert str(exin.value) == err_msg

    err_msg = 'The transformation function must be a callable object.'
    with pytest.raises(TypeError) as exin:
        fump.batch_data(array_2d, transformation_fn='callable')
    assert str(exin.value) == err_msg
    with pytest.raises(TypeError) as exin:
        fump.batch_data(array_2d, 5, transformation_fn='callable')
    assert str(exin.value) == err_msg
    with pytest.raises(TypeError) as exin:
        fump.batch_data(array_2d, 5, 'callable')
    assert str(exin.value) == err_msg

    wrong_callable = lambda _param1, _param2: _param1 + _param2  # noqa: E731
    err_msg = ('The transformation function must have only one required '
               'parameter; now it has 2.')
    with pytest.raises(RuntimeError) as exin:
        fump.batch_data(array_2d, 5, wrong_callable)
    assert str(exin.value) == err_msg


def test_batch_data():
    """Tests :func:`fatf.utils.model.processing.batch_data` functionality."""
    array = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    transform = lambda _a: _a.sum(axis=0)  # noqa: E731
    array_slices = [np.array([[0, 1, 2], [3, 4, 5]]), np.array([[6, 7, 8]])]
    array_slices_ = [np.array([3, 5, 7]), np.array([6, 7, 8])]

    struct_dtype = [('a', np.int8), ('b', np.float64), ('c', np.int8)]
    array_struct = np.array([(0, 1.5, 2), (3, 4.5, 5), (6, 7, 8)],
                            dtype=struct_dtype)
    transform_ = lambda _a: fuat.structured_to_unstructured(  # noqa: E731
        _a).sum(axis=0)
    array_slices_struct = [
        np.array([(0, 1.5, 2), (3, 4.5, 5)], dtype=struct_dtype),
        np.array([(6, 7, 8)], dtype=struct_dtype)
    ]
    array_slices_struct_ = [np.array([3, 6, 7]), np.array([6, 7, 8])]

    itr = fump.batch_data(array, 2)
    for slice, slice_ in zip(itr, array_slices):
        assert np.array_equal(slice, slice_)
    itr = fump.batch_data(array, 2, transform)
    for slice, slice_ in zip(itr, array_slices_):
        assert np.array_equal(slice, slice_)

    itr = fump.batch_data(array_struct, 2)
    for slice, slice_ in zip(itr, array_slices_struct):
        assert np.array_equal(slice, slice_)
    itr = fump.batch_data(array_struct, 2, transform_)
    for slice, slice_ in zip(itr, array_slices_struct_):
        assert np.array_equal(slice, slice_)
