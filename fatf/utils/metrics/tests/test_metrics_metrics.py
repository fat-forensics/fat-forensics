"""
Holds custom distance functions used for FAT Forensics examples and testing.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import numpy as np
import pytest

import fatf.utils.metrics.metrics as fumm

CMA = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]])
CMA_BIN = np.array([[3, 11], [7, 5]])


def test_multiclass_true_positive_rate():
    """
    Tests :func:`fatf.utils.metrics.metrics.multiclass_true_positive_rate`.
    """
    mtpr_0 = fumm.multiclass_true_positive_rate(CMA, 0)
    assert mtpr_0 == pytest.approx(0.333, abs=1e-3)
    mtpr_1 = fumm.multiclass_true_positive_rate(CMA, 1)
    assert mtpr_1 == 0.5
    mtpr_2 = fumm.multiclass_true_positive_rate(CMA, 2)
    assert mtpr_2 == pytest.approx(0.333, abs=1e-3)


def test_multiclass_true_negative_rate():
    """
    Tests :func:`fatf.utils.metrics.metrics.multiclass_true_negative_rate`.
    """
    type_error = 'The strict parameter has to be a boolean.'
    with pytest.raises(TypeError) as exi:
        fumm.multiclass_true_negative_rate(CMA, 0, 'one')
    assert str(exi.value) == type_error

    metric = pytest.approx(5 / 7)
    mtpr_0_n = fumm.multiclass_true_negative_rate(CMA, 0)
    assert mtpr_0_n == metric
    mtpr_0_n = fumm.multiclass_true_negative_rate(CMA, 0, False)
    assert mtpr_0_n == metric
    #
    metric = pytest.approx(4 / 6)
    mtpr_1_n = fumm.multiclass_true_negative_rate(CMA, 1)
    assert mtpr_1_n == metric
    mtpr_1_n = fumm.multiclass_true_negative_rate(CMA, 1, False)
    assert mtpr_1_n == metric
    #
    metric = pytest.approx(5 / 7)
    mtpr_2_n = fumm.multiclass_true_negative_rate(CMA, 2)
    assert mtpr_2_n == metric
    mtpr_2_n = fumm.multiclass_true_negative_rate(CMA, 2, False)
    assert mtpr_2_n == metric

    metric = pytest.approx(3 / 7)
    mtpr_0_p = fumm.multiclass_true_negative_rate(CMA, 0, True)
    assert mtpr_0_p == metric
    metric = pytest.approx(2 / 6)
    mtpr_1_p = fumm.multiclass_true_negative_rate(CMA, 1, True)
    assert mtpr_1_p == metric
    metric = pytest.approx(3 / 7)
    mtpr_2_p = fumm.multiclass_true_negative_rate(CMA, 2, True)
    assert mtpr_2_p == metric


def test_multiclass_false_positive_rate():
    """
    Tests :func:`fatf.utils.metrics.metrics.multiclass_false_positive_rate`.
    """
    mtpr_0 = fumm.multiclass_false_positive_rate(CMA, 0)
    assert mtpr_0 == pytest.approx(2 / 7, abs=1e-3)
    mtpr_1 = fumm.multiclass_false_positive_rate(CMA, 1)
    assert mtpr_1 == pytest.approx(2 / 6, abs=1e-3)
    mtpr_2 = fumm.multiclass_false_positive_rate(CMA, 2)
    assert mtpr_2 == pytest.approx(2 / 7, abs=1e-3)


def test_multiclass_false_negative_rate():
    """
    Tests :func:`fatf.utils.metrics.metrics.multiclass_false_negative_rate`.
    """
    mtpr_0 = fumm.multiclass_false_negative_rate(CMA, 0)
    assert mtpr_0 == pytest.approx(2 / 3, abs=1e-3)
    mtpr_1 = fumm.multiclass_false_negative_rate(CMA, 1)
    assert mtpr_1 == pytest.approx(2 / 4, abs=1e-3)
    mtpr_2 = fumm.multiclass_false_negative_rate(CMA, 2)
    assert mtpr_2 == pytest.approx(2 / 3, abs=1e-3)


def test_true_positive_rate():
    """
    Tests :func:`fatf.utils.metrics.metrics.true_positive_rate`.
    """
    mtpr = fumm.true_positive_rate(CMA_BIN)
    assert mtpr == pytest.approx(3 / 10, abs=1e-3)


def test_true_negative_rate():
    """
    Tests :func:`fatf.utils.metrics.metrics.true_negative_rate`.
    """
    mtpr = fumm.true_negative_rate(CMA_BIN)
    assert mtpr == pytest.approx(5 / 16, abs=1e-3)


def test_false_negative_rate():
    """
    Tests :func:`fatf.utils.metrics.metrics.false_negative_rate`.
    """
    mtpr = fumm.false_negative_rate(CMA_BIN)
    assert mtpr == pytest.approx(7 / 10, abs=1e-3)


def test_false_positive_rate():
    """
    Tests :func:`fatf.utils.metrics.metrics.false_positive_rate`.
    """
    mtpr = fumm.false_positive_rate(CMA_BIN)
    assert mtpr == pytest.approx(11 / 16, abs=1e-3)


def test_multiclass_positive_predictive_value():
    """
    :func:`fatf.utils.metrics.metrics.multiclass_positive_predictive_value`.
    """
    mtpr_0 = fumm.multiclass_positive_predictive_value(CMA, 0)
    assert mtpr_0 == pytest.approx(1 / 3, abs=1e-3)
    mtpr_1 = fumm.multiclass_positive_predictive_value(CMA, 1)
    assert mtpr_1 == pytest.approx(2 / 4, abs=1e-3)
    mtpr_2 = fumm.multiclass_positive_predictive_value(CMA, 2)
    assert mtpr_2 == pytest.approx(1 / 3, abs=1e-3)


def test_multiclass_negative_predictive_value():
    """
    :func:`fatf.utils.metrics.metrics.multiclass_negative_predictive_value`.
    """
    type_error = 'The strict parameter has to be a boolean.'
    with pytest.raises(TypeError) as exi:
        fumm.multiclass_negative_predictive_value(CMA, 0, 'one')
    assert str(exi.value) == type_error

    metric = pytest.approx(5 / 7)
    mtpr_0_n = fumm.multiclass_negative_predictive_value(CMA, 0)
    assert mtpr_0_n == metric
    mtpr_0_n = fumm.multiclass_negative_predictive_value(CMA, 0, False)
    assert mtpr_0_n == metric
    #
    metric = pytest.approx(4 / 6)
    mtpr_1_n = fumm.multiclass_negative_predictive_value(CMA, 1)
    assert mtpr_1_n == metric
    mtpr_1_n = fumm.multiclass_negative_predictive_value(CMA, 1, False)
    assert mtpr_1_n == metric
    #
    metric = pytest.approx(5 / 7)
    mtpr_2_n = fumm.multiclass_negative_predictive_value(CMA, 2)
    assert mtpr_2_n == metric
    mtpr_2_n = fumm.multiclass_negative_predictive_value(CMA, 2, False)
    assert mtpr_2_n == metric

    metric = pytest.approx(3 / 7)
    mtpr_0_p = fumm.multiclass_negative_predictive_value(CMA, 0, True)
    assert mtpr_0_p == metric
    metric = pytest.approx(2 / 6)
    mtpr_1_p = fumm.multiclass_negative_predictive_value(CMA, 1, True)
    assert mtpr_1_p == metric
    metric = pytest.approx(3 / 7)
    mtpr_2_p = fumm.multiclass_negative_predictive_value(CMA, 2, True)
    assert mtpr_2_p == metric


def test_positive_predictive_value():
    """
    :func:`fatf.utils.metrics.metrics.positive_predictive_value`.
    """
    mtpr = fumm.positive_predictive_value(CMA_BIN)
    assert mtpr == pytest.approx(3 / 14, abs=1e-3)


def test_negative_predictive_value():
    """
    :func:`fatf.utils.metrics.metrics.negative_predictive_value`.
    """
    mtpr = fumm.negative_predictive_value(CMA_BIN)
    assert mtpr == pytest.approx(5 / 12, abs=1e-3)


def test_accuracy():
    """
    Tests :func:`fatf.utils.metrics.metrics.accuracy` function.
    """
    acc = fumm.accuracy(CMA)
    assert acc == pytest.approx(4 / 10, abs=1e-3)

    acc = fumm.accuracy(CMA_BIN)
    assert acc == pytest.approx(8 / 26, abs=1e-3)


def test_multiclass_treatment():
    """
    Tests :func:`fatf.utils.metrics.metrics.multiclass_treatment` function.
    """
    mtpr_0 = fumm.multiclass_treatment(CMA, 0)
    assert mtpr_0 == pytest.approx(2 / 6, abs=1e-3)
    mtpr_1 = fumm.multiclass_treatment(CMA, 1)
    assert mtpr_1 == pytest.approx(2 / 6, abs=1e-3)
    mtpr_2 = fumm.multiclass_treatment(CMA, 2)
    assert mtpr_2 == pytest.approx(2 / 6, abs=1e-3)


def test_treatment():
    """
    Tests :func:`fatf.utils.metrics.metrics.treatment` function.
    """
    mtpr = fumm.treatment(CMA_BIN)
    assert mtpr == pytest.approx(11 / 18, abs=1e-3)
