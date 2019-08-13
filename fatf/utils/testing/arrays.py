"""
The :mod:`fatf.utils.testing.arrays` module holds a variety of numpy arrays.

This module holds a collection of (classic and structured) numpy arrays of
various types.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import numpy as np

__all__ = ['NUMERICAL_NP_ARRAY',
           'NOT_NUMERICAL_NP_ARRAY',
           'WIDE_NP_ARRAY',
           'NUMERICAL_STRUCTURED_ARRAY',
           'NOT_NUMERICAL_STRUCTURED_ARRAY',
           'WIDE_STRUCTURED_ARRAY',
           'BASE_NP_ARRAY',
           'NOT_BASE_NP_ARRAY',
           'BASE_STRUCTURED_ARRAY',
           'NOT_BASE_STRUCTURED_ARRAY']  # yapf: disable

NUMERICAL_NP_ARRAY = np.array([
    [True, 1],
    [-1, 1.0],
    [1 + 1j, False],
    [1 + 1j, np.nan],
    [np.inf, -np.inf]])  # yapf: disable
NOT_NUMERICAL_NP_ARRAY = np.array([
    [True, 1],
    [-1, 1.0],
    [1 + 1j, False],
    [1 + 1j, np.nan],
    [np.inf, -np.inf],
    [object(), 7],
    [9, None],
    ['a', u'b']])  # yapf: disable
WIDE_NP_ARRAY = np.array([
    [True, 1, 0],
    [-1, 1.0, 4],
    [np.nan, np.inf, -np.inf],
    [1 + 1j, False, 2]])  # yapf: disable
NUMERICAL_STRUCTURED_ARRAY = np.array([
    (1.0, 1.0 + 1j),
    (1, 1 + 1j),
    (np.nan, -1 + 1j),
    (np.inf, -1 + 1j),
    (-np.inf, -1 + 1j),
    (-1, -1 + 1j)], dtype=[('numbers', '<f8'),
                           ('complex', '<c16')])  # yapf: disable
NOT_NUMERICAL_STRUCTURED_ARRAY = np.array([
    (True, 'a'),
    (1, 'b'),
    (-1, 'c'),
    (1.0, 'd'),
    (1 + 1j, 'e'),
    (False, 'f'),
    (np.nan, 'g'),
    (np.inf, 'h'),
    (-np.inf, 'i')], dtype=[('numerical', 'c8'),
                            ('categorical', 'U1')])  # yapf: disable
WIDE_STRUCTURED_ARRAY = np.array([
    (1.0, 1.0 + 1j, np.nan),
    (np.inf, 1 + 1j, 6),
    (-1, -1 + 1j, -np.inf)], dtype=[('numbers', '<f8'),
                                    ('complex', '<c16'),
                                    ('anybody', '<f8')])  # yapf: disable
BASE_NP_ARRAY = np.array([
    [True, 1],
    [-1, 1.0],
    [1 + 1j, False],
    [1 + 1j, np.nan],
    [np.inf, -np.inf],
    ['a', u'b']])  # yapf: disable  # pylint: disable=too-many-function-args
NOT_BASE_NP_ARRAY = np.array([
    [True, np.timedelta64(366, 'D')],  # pylint: disable=too-many-function-args
    [-1, 1.0],  # type: ignore
    [1 + 1j, np.datetime64('2005-02-25')],  # type: ignore
    [1 + 1j, np.nan],  # type: ignore
    [np.inf, -np.inf],
    ['a', u'b'],  # type: ignore
    [object(), 7],  # type: ignore
    [9, None]])  # yapf: disable
BASE_STRUCTURED_ARRAY = np.array([
    (True, 'a'),
    (1, 'b'),
    (-1, 'c'),
    (1.0, 'd'),
    (1 + 1j, 'e'),
    (False, 'f'),
    (np.nan, 'g'),
    (np.inf, 'h'),
    (-np.inf, 'i')], dtype=[('numerical', 'c8'),
                            ('categorical', 'U1')])  # yapf: disable
NOT_BASE_STRUCTURED_ARRAY = np.array([
    (True, object(), 'a'),
    (1, None, 'b'),
    (-1, None, 'c'),
    (1.0, None, 'd'),
    (1 + 1j, None, 'e'),
    (False, None, 'f'),
    (np.nan, None, 'g'),
    (np.inf, None, 'h'),
    (-np.inf, object(), 'i')], dtype=[('numerical', 'c8'),
                                      ('object', 'object'),
                                      ('categorical', 'U1')])  # yapf: disable
