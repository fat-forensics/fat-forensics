"""
Tests implementations of predictions fairness measures.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import pytest

import numpy as np

from fatf.exceptions import IncorrectShapeError

import fatf.fairness.predictions.measures as ffpm
import fatf.utils.models as fum


def test_counterfactual_fairness():
    """
    Tests :func:`fatf.fairness.predictions.metrics.counterfactual_fairness`.

    The ``CounterfactualExplainer`` class is already tested. These are just
    basic integration tests.
    """
    dataset_struct = np.array([('a', 'a@a.com', 35.70, '0011'),
                               ('b', 'b@a.com', 22.22, '0011'),
                               ('c', 'c@a.com', 11.11, '0011'),
                               ('d', 'd@b.com', 41.27, '0011'),
                               ('e', 'e@b.com', 12.57, '1100'),
                               ('f', 'f@c.com', 05.33, '1100'),
                               ('g', 'g@d.com', 17.29, '1100')],
                              dtype=[('name', 'U1'), ('email', 'U7'),
                                     ('q', float), ('postcode', 'U4')])
    target = np.array(
        ['good', 'good', 'bad', 'mediocre', 'bad', 'mediocre', 'good'])
    knn_struct = fum.KNN(k=1)
    knn_struct.fit(dataset_struct, target)

    cfs, cfs_dist, cfs_pred = ffpm.counterfactual_fairness(
        dataset_struct[2],
        ['q', 'postcode'],
        #
        model=knn_struct,
        dataset=dataset_struct)

    t_cfs = np.array([('c', 'c@a.com', 6.33, '0011'),
                      ('c', 'c@a.com', 7.33, '1100')],
                     dtype=dataset_struct.dtype)
    t_dist = np.array(2 * [4.78])
    t_pred = np.array(2 * ['mediocre'])
    assert np.array_equal(cfs, t_cfs)
    assert np.allclose(cfs_dist, t_dist)
    assert np.array_equal(cfs_pred, t_pred)


def test_counterfactual_fairness_check():
    """
    Tests counterfactual fairness check for a prediction.

    Tests
    :func:`fatf.fairness.predictions.metrics.counterfactual_fairness_check`
    function.
    """
    incorrect_shape_cf = ('The unfair counterfactuals parameter has to be a '
                          '2-dimensional numpy array.')
    incorrect_shape_dist = ('The distances parameter has to be a '
                            '1-dimensional array.')
    value_error_dist = 'The distances array has to be purely numerical.'
    type_error_threshold = 'The threshold parameter has to be a number.'
    runtime_error = ('Either of the two is required to run this function: '
                     'unfair_counterfactuals parameter or both distances and '
                     'threshold parameters.')

    struct_dtype = [('a', int), ('b', int), ('c', int)]

    one_d_array = np.array([1, 2])
    one_d_array_str = np.array(['1', '2'])
    two_d_array = np.array([[1, 2, 3], [4, 5, 6]])
    two_d_array_struct = np.array([(1, 2, 3), (4, 5, 6)], dtype=struct_dtype)

    with pytest.raises(RuntimeError) as exin:
        ffpm.counterfactual_fairness_check()
    assert str(exin.value) == runtime_error
    with pytest.raises(RuntimeError) as exin:
        ffpm.counterfactual_fairness_check(distances='d')
    assert str(exin.value) == runtime_error
    with pytest.raises(RuntimeError) as exin:
        ffpm.counterfactual_fairness_check(threshold='d')
    assert str(exin.value) == runtime_error

    with pytest.raises(IncorrectShapeError) as exin:
        ffpm.counterfactual_fairness_check(unfair_counterfactuals=one_d_array)
    assert str(exin.value) == incorrect_shape_cf

    with pytest.raises(IncorrectShapeError) as exin:
        ffpm.counterfactual_fairness_check(distances=two_d_array, threshold=42)
    assert str(exin.value) == incorrect_shape_dist

    with pytest.raises(ValueError) as exin:
        ffpm.counterfactual_fairness_check(
            distances=one_d_array_str, threshold=42)
    assert str(exin.value) == value_error_dist

    with pytest.raises(TypeError) as exin:
        ffpm.counterfactual_fairness_check(
            distances=one_d_array, threshold='42')
    assert str(exin.value) == type_error_threshold

    # Test functionality
    assert ffpm.counterfactual_fairness_check(
        unfair_counterfactuals=two_d_array)
    assert not ffpm.counterfactual_fairness_check(
        unfair_counterfactuals=np.ndarray((0, 0)))
    assert ffpm.counterfactual_fairness_check(
        unfair_counterfactuals=two_d_array_struct)
    assert not ffpm.counterfactual_fairness_check(
        unfair_counterfactuals=np.ndarray((0, ), dtype=struct_dtype))

    assert not ffpm.counterfactual_fairness_check(
        distances=one_d_array, threshold=7)
    assert ffpm.counterfactual_fairness_check(
        distances=one_d_array, threshold=1)

    # Test unfair_counterfactuals and incomplete distance/threshold
    assert ffpm.counterfactual_fairness_check(
        unfair_counterfactuals=two_d_array, distances='a')
    assert ffpm.counterfactual_fairness_check(
        unfair_counterfactuals=two_d_array, threshold='a')

    # Test precedence
    assert not ffpm.counterfactual_fairness_check(
        unfair_counterfactuals=two_d_array, distances=one_d_array, threshold=7)
